"""Train and evaluate the resume-job compatibility model and the candidate ranker.

Usage:  python train_matching.py

Training data
-------------
The Kaggle dataset has labelled resumes but no job descriptions or match labels, so:
  * job descriptions are SYNTHETIC: generated from skills mined from TRAINING resumes of each
    job category (train and test JDs draw from disjoint halves of each category's skill pool);
  * relevance labels are WEAK, derived from the resume's dataset category vs the JD's category:
        3 = same category, 2 = closely related category, 1 = same broad family, 0 = unrelated.
    They are NOT recruiter judgments.
  * resumes are split by candidate (80/20), JDs by id, so no resume or JD is shared by train/test.

Outputs (models/): compat.joblib, taxonomy.json, match_tfidf.joblib,
                   experiment_report.json, experiment_report.md
"""
import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matching
import parsing
from utils import clean_text

SEED = 42
POOL = 40  # candidates per job

FAMILIES = {
    "tech": ["INFORMATION-TECHNOLOGY", "ENGINEERING", "AUTOMOBILE", "AVIATION", "CONSTRUCTION"],
    "business": ["ACCOUNTANT", "FINANCE", "BANKING", "BUSINESS-DEVELOPMENT", "CONSULTANT", "SALES", "BPO"],
    "creative": ["HR", "PUBLIC-RELATIONS", "DIGITAL-MEDIA", "ARTS", "DESIGNER", "APPAREL"],
    "service": ["HEALTHCARE", "FITNESS", "CHEF", "TEACHER", "ADVOCATE", "AGRICULTURE"],
}
FAMILY_OF = {c: f for f, cs in FAMILIES.items() for c in cs}
CLOSE = {frozenset(p) for p in [
    ("ACCOUNTANT", "FINANCE"), ("FINANCE", "BANKING"), ("ACCOUNTANT", "BANKING"),
    ("BUSINESS-DEVELOPMENT", "SALES"), ("BUSINESS-DEVELOPMENT", "CONSULTANT"),
    ("DESIGNER", "ARTS"), ("DESIGNER", "DIGITAL-MEDIA"), ("DIGITAL-MEDIA", "PUBLIC-RELATIONS"),
    ("ENGINEERING", "AUTOMOBILE"), ("ENGINEERING", "CONSTRUCTION"), ("ENGINEERING", "AVIATION"),
    ("INFORMATION-TECHNOLOGY", "ENGINEERING"), ("HEALTHCARE", "FITNESS"), ("BPO", "SALES"),
    ("HR", "PUBLIC-RELATIONS"),
]}


def relevance(res_cat, job_cat):
    if res_cat == job_cat:
        return 3
    if frozenset((res_cat, job_cat)) in CLOSE:
        return 2
    if FAMILY_OF.get(res_cat) == FAMILY_OF.get(job_cat):
        return 1
    return 0


# ----------------------------------------------------------------- JD generator
EDU_PHRASE = {0: "", 1: "a high school diploma or equivalent", 2: "a bachelor's degree",
              3: "a master's degree", 4: "a PhD"}
TEMPLATES = [
    "{title}\nWe are looking for a {title_l} to join our team. The ideal candidate has {years}+ years "
    "of experience{edu}.\nRequired skills: {req}.\nPreferred skills: {pref}.",
    "Job opening: {title}\nRequirements: {years}+ years of relevant experience{edu}. Strong skills in "
    "{req}. Familiarity with {pref} is a plus.",
    "{title} wanted\nThis role relies on {req}. Minimum {years} years in a similar role{edu}. "
    "Nice to have: {pref}.",
]


def category_stats(train_df, skills_by_row):
    """Skills / experience / education / titles per category, from TRAINING resumes only."""
    n_all = len(train_df)
    df_all = Counter()
    for sk in skills_by_row:
        df_all.update(sk)
    stats = {}
    for cat, grp in train_df.groupby("Category"):
        idx = [train_df.index.get_loc(i) for i in grp.index]
        df_c = Counter()
        for i in idx:
            df_c.update(skills_by_row[i])
        n_c = len(idx)
        scored = []
        for sk, c in df_c.items():
            if c >= 3 and len(sk) > 2:
                p_c, p = c / n_c, df_all[sk] / n_all
                scored.append((p_c * np.log(p_c / p), sk))
        ranked = [sk for _, sk in sorted(scored, reverse=True)[:40]]
        stats[cat] = {
            "skills": ranked,
            "years": np.percentile(grp["years"], [25, 50, 75]).tolist(),
            "edu": int(grp["edu"].mode().iloc[0]),
            "titles": [t.title() for t in grp["title"] if 1 <= len(t.split()) <= 6],
        }
    return stats


def make_jds(stats, cats, n_per_cat, parity, rng):
    jds = []
    for cat in cats:
        st = stats[cat]
        pool = st["skills"][parity::2] or st["skills"]
        for _ in range(n_per_cat):
            k_req = min(rng.randint(4, 6), len(pool))
            picked = rng.sample(pool, min(len(pool), k_req + rng.randint(1, 3)))
            req, pref = picked[:k_req], picked[k_req:] or picked[:1]
            if rng.random() < 0.15:  # a little cross-domain noise
                other = stats[rng.choice(cats)]["skills"]
                if other:
                    pref.append(rng.choice(other))
            years = int(min(10, max(1, round(rng.choice(st["years"]) * 0.4))))
            edu = st["edu"] if rng.random() > 0.25 else max(0, st["edu"] - 1)
            title = rng.choice(st["titles"]) if st["titles"] else cat.title()
            text = rng.choice(TEMPLATES).format(
                title=title, title_l=title.lower(), years=years,
                edu=(", " + EDU_PHRASE[edu]) if edu else "",
                req=", ".join(req), pref=", ".join(pref))
            jds.append({"cat": cat, "text": text})
    return jds


def build_pairs(jd_cats, res_idx_by_cat, rng, pool=POOL):
    ri, ji, y, cats = [], [], [], []
    all_idx = [(c, i) for c, v in res_idx_by_cat.items() for i in v]
    for j, jc in enumerate(jd_cats):
        same = list(res_idx_by_cat.get(jc, []))
        related = [(c, i) for c, i in all_idx if c != jc and relevance(c, jc) >= 1]
        others = [(c, i) for c, i in all_idx if c != jc and relevance(c, jc) == 0]
        chosen = [(jc, i) for i in rng.sample(same, min(10, len(same)))]
        chosen += rng.sample(related, min(10, len(related)))
        chosen += rng.sample(others, min(pool - len(chosen), len(others)))
        for c, i in chosen:
            ri.append(i); ji.append(j); y.append(relevance(c, jc)); cats.append(c)
    return np.array(ri), np.array(ji), np.array(y), np.array(cats)


# ---------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def binary_metrics(y_bin, p):
    return {"roc_auc": float(roc_auc_score(y_bin, p)),
            "pr_auc": float(average_precision_score(y_bin, p)),
            "brier": float(brier_score_loss(y_bin, np.clip(p, 0, 1))),
            "ece": ece(y_bin, np.clip(p, 0, 1))}


def dcg(rels):
    return float(np.sum((2.0 ** np.asarray(rels) - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ranking_metrics(groups, y, score):
    out = defaultdict(list)
    for g in np.unique(groups):
        m = groups == g
        yy, ss = y[m], score[m]
        if len(yy) < 5 or yy.max() == 0:
            continue
        order = np.argsort(-ss, kind="stable")
        ranked = yy[order]
        ideal = np.sort(yy)[::-1]
        for k in (5, 10):
            out[f"ndcg@{k}"].append(dcg(ranked[:k]) / dcg(ideal[:k]))
        rel_total = (yy >= 2).sum()
        if rel_total:
            out["recall@10"].append((ranked[:10] >= 2).sum() / rel_total)
        out["precision@5"].append((ranked[:5] >= 2).mean())
    return {k: float(np.mean(v)) for k, v in out.items()}


def group_sizes(groups):
    _, counts = np.unique(groups, return_counts=True)
    return counts


def fit_clf(X, y):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, min_child_samples=30,
                              random_state=SEED, verbose=-1).fit(X, y)


def fit_ranker(X, y, groups):
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                          num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          min_child_samples=30, random_state=SEED, verbose=-1
                          ).fit(X, y, group=group_sizes(groups))


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Resume.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--jd-train", type=int, default=30, help="JDs per category (train)")
    ap.add_argument("--jd-test", type=int, default=12, help="JDs per category (test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(SEED)
    t0 = time.time()

    from train import ensure_data
    ensure_data(args.data)
    df = pd.read_csv(args.data).dropna(subset=["Resume_str", "Category"]).drop_duplicates("Resume_str")
    df = df[df["Resume_str"].map(clean_text).str.len() > 50].reset_index(drop=True)
    print(f"{len(df)} resumes")

    # ---- split resumes by candidate
    tr_i, te_i = train_test_split(np.arange(len(df)), test_size=0.2, stratify=df["Category"],
                                  random_state=SEED)
    train_raw = df.loc[tr_i, "Resume_str"].tolist()

    # ---- taxonomy + tf-idf (fit on training resumes only)
    taxonomy = parsing.mine_taxonomy(train_raw)
    parsing.save_taxonomy(taxonomy, os.path.join(args.out, "taxonomy.json"))
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.9,
                            sublinear_tf=True, max_features=30000)
    tfidf.fit([clean_text(t) for t in train_raw])
    joblib.dump(tfidf, os.path.join(args.out, "match_tfidf.joblib"))
    print(f"taxonomy: {len(set(taxonomy.values()))} skills")

    # ---- embeddings (cached)
    key = hashlib.md5("".join(df["Resume_str"]).encode("utf-8", "ignore")).hexdigest()
    cache = os.path.join(os.path.dirname(args.data) or ".", "resume_embeddings.npz")
    emb = None
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key:
            emb = z["emb"]
    if emb is None:
        print("embedding resumes (one-off, cached)...")
        emb = matching.embed_documents([clean_text(t) for t in df["Resume_str"]])
        np.savez(cache, key=key, emb=emb)
    print(f"embeddings ready ({time.time() - t0:.0f}s)")

    R = matching.build_profiles(df["Resume_str"], tfidf, taxonomy, parsing.parse_resume, emb=emb)
    df["years"], df["edu"] = R.years, R.edu
    df["title"] = [parsing.resume_title(t) for t in df["Resume_str"]]

    # ---- synthetic JDs + graded labels
    cats = sorted(df["Category"].unique())
    train_df = df.loc[tr_i]
    stats = category_stats(train_df, [R.skills[i] for i in tr_i])
    jd_train = make_jds(stats, cats, args.jd_train, 0, rng)
    jd_test = make_jds(stats, cats, args.jd_test, 1, rng)
    print("sample JD:\n" + jd_train[0]["text"] + "\n")
    parse_job = lambda t, tx: parsing.parse_job(t, tx)
    J_tr = matching.build_profiles([j["text"] for j in jd_train], tfidf, taxonomy, parse_job)
    J_te = matching.build_profiles([j["text"] for j in jd_test], tfidf, taxonomy, parse_job)

    by_cat = lambda idx: {c: [i for i in idx if df.at[i, "Category"] == c] for c in cats}
    ri_tr, ji_tr, y_tr, rc_tr = build_pairs([j["cat"] for j in jd_train], by_cat(tr_i), rng)
    ri_te, ji_te, y_te, rc_te = build_pairs([j["cat"] for j in jd_test], by_cat(te_i), rng)
    X_tr = matching.pair_features(R, J_tr, ri_tr, ji_tr)
    X_te = matching.pair_features(R, J_te, ri_te, ji_te)
    print(f"pairs: train {len(y_tr)} ({len(jd_train)} JDs), test {len(y_te)} ({len(jd_test)} JDs); "
          f"label dist test {dict(Counter(y_te.tolist()))}")
    yb_tr, yb_te = (y_tr >= 2).astype(int), (y_te >= 2).astype(int)

    # JD-disjoint calibration split inside train
    cal_jds = set(rng.sample(range(len(jd_train)), len(jd_train) // 5))
    cal_mask = np.array([j in cal_jds for j in ji_tr])
    fit_mask = ~cal_mask

    report = {
        "data": {"resumes": len(df), "train_resumes": len(tr_i), "test_resumes": len(te_i),
                 "train_jds": len(jd_train), "test_jds": len(jd_test),
                 "train_pairs": int(len(y_tr)), "test_pairs": int(len(y_te)),
                 "test_label_distribution": {int(k): int(v) for k, v in sorted(Counter(y_te.tolist()).items())},
                 "taxonomy_size": len(set(taxonomy.values())),
                 "label_definition": "3 same category; 2 closely related; 1 same broad family; 0 unrelated. "
                                     "Weak labels derived from dataset categories; JDs are synthetic. "
                                     "'Compatible' = relevance >= 2.",
                 "split": "resumes by candidate (80/20), JDs by id, JD skill pools disjoint between train/test"},
        "embedding_model": matching.EMB_MODEL,
    }

    # ---- 1. baselines vs models (compatibility prediction + ranking)
    F = {f: i for i, f in enumerate(matching.FEATURES)}
    results = {}

    def evaluate(name, score_te, groups=ji_te):
        results[name] = {**binary_metrics(yb_te, score_te), **ranking_metrics(groups, y_te, score_te)}
        r = results[name]
        print(f"{name:34s} AUC {r['roc_auc']:.3f}  PR-AUC {r['pr_auc']:.3f}  NDCG@10 {r['ndcg@10']:.3f}")

    evaluate("Baseline: keyword overlap", X_te[:, F["skill_cov"]])
    evaluate("Baseline: TF-IDF cosine", X_te[:, F["tfidf_cos"]])
    evaluate("Baseline: embedding cosine", X_te[:, F["emb_cos"]])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, yb_tr)
    evaluate("Logistic Regression (all features)", lr.predict_proba(X_te)[:, 1])

    clf = fit_clf(X_tr[fit_mask], yb_tr[fit_mask])
    raw_cal, raw_te = clf.predict_proba(X_tr[cal_mask])[:, 1], clf.predict_proba(X_te)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, yb_tr[cal_mask])
    p_te = iso.predict(raw_te)
    evaluate("LightGBM classifier (uncalibrated)", raw_te)
    evaluate("LightGBM classifier (calibrated)", p_te)

    ranker = fit_ranker(X_tr, y_tr, ji_tr)
    evaluate("LightGBM LambdaRank ranker", ranker.predict(X_te))
    report["model_comparison"] = results
    report["calibration"] = {"before": binary_metrics(yb_te, raw_te), "after": binary_metrics(yb_te, p_te)}
    bins = np.linspace(0, 1, 11)
    report["reliability_curve"] = [
        {"bin": f"{lo:.1f}-{hi:.1f}", "mean_predicted": float(p_te[m].mean()),
         "observed_rate": float(yb_te[m].mean()), "n": int(m.sum())}
        for lo, hi in zip(bins[:-1], bins[1:])
        for m in [(p_te >= lo) & (p_te < hi if hi < 1 else p_te <= hi)] if m.any()]

    # ---- 2. ablation: remove one feature group at a time
    ablation = {}
    for g, cols in [("(none - full model)", [])] + list(matching.GROUPS.items()):
        keep = [i for f, i in F.items() if f not in cols]
        c = fit_clf(X_tr[:, keep], yb_tr)
        rk = fit_ranker(X_tr[:, keep], y_tr, ji_tr)
        ablation[g] = {**binary_metrics(yb_te, c.predict_proba(X_te[:, keep])[:, 1]),
                       "ndcg@10": ranking_metrics(ji_te, y_te, rk.predict(X_te[:, keep]))["ndcg@10"]}
        print(f"ablation - {g:22s} AUC {ablation[g]['roc_auc']:.3f}  NDCG@10 {ablation[g]['ndcg@10']:.3f}")
    report["ablation"] = ablation

    # ---- 3. feature importance (gain) + explanation sanity
    gain = clf.booster_.feature_importance("gain")
    report["feature_importance"] = {matching.FEATURES[i]: float(gain[i] / gain.sum())
                                    for i in np.argsort(-gain)}

    # ---- 4. error analysis
    fp, fn = Counter(), Counter()
    jd_cat_te = np.array([jd_test[j]["cat"] for j in ji_te])
    for rc, jc, yb, p in zip(rc_te, jd_cat_te, yb_te, p_te):
        if p >= 0.5 and not yb:
            fp[f"{rc} resume vs {jc} job"] += 1
        if p < 0.5 and yb:
            fn[f"{rc} resume vs {jc} job"] += 1
    report["error_analysis"] = {
        "false_positive_rate": float(((p_te >= 0.5) & (yb_te == 0)).sum() / max((yb_te == 0).sum(), 1)),
        "false_negative_rate": float(((p_te < 0.5) & (yb_te == 1)).sum() / max(yb_te.sum(), 1)),
        "top_false_positives": fp.most_common(8),
        "top_false_negatives": fn.most_common(8),
        "mean_probability_by_relevance": {int(k): float(p_te[y_te == k].mean()) for k in range(4)},
    }

    # ---- 5. bias check: counterfactual identity text must not change scores
    variants = {"baseline": "", "female name": "Emily Walsh (she/her) ", "male name": "Jamal Washington (he/him) ",
                "south-asian name": "Priya Sharma ", "east-asian name": "Wei Chen ", "older (born 1962)": "Born 1962 "}
    sample = rng.sample(range(len(ri_te)), 120)
    base_idx = [ri_te[k] for k in sample]
    jd_idx = np.array([ji_te[k] for k in sample])
    probs = {}
    for name, prefix in variants.items():
        texts = [prefix + df.at[i, "Resume_str"] for i in base_idx]
        Rv = matching.build_profiles(texts, tfidf, taxonomy, parsing.parse_resume)
        Xv = matching.pair_features(Rv, J_te, np.arange(len(texts)), jd_idx)
        probs[name] = iso.predict(clf.predict_proba(Xv)[:, 1])
    base_p = probs["baseline"]
    report["bias_check"] = {
        "method": "Prefix each of 120 test resumes with identity text (name / pronouns / birth year) and "
                  "compare compatibility probabilities. Protected attributes are never model inputs.",
        "variants": {n: {"mean_abs_change": float(np.abs(p - base_p).mean()),
                         "max_abs_change": float(np.abs(p - base_p).max()),
                         "decision_flips": int(((p >= 0.5) != (base_p >= 0.5)).sum())}
                     for n, p in probs.items() if n != "baseline"},
    }
    for n, v in report["bias_check"]["variants"].items():
        print(f"bias check - {n:20s} mean |dP| {v['mean_abs_change']:.4f}  flips {v['decision_flips']}")

    # ---- save
    joblib.dump({"clf": clf, "calibrator": iso, "ranker": ranker, "features": matching.FEATURES},
                os.path.join(args.out, "compat.joblib"))
    with open(os.path.join(args.out, "experiment_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, os.path.join(args.out, "experiment_report.md"))
    print(f"done in {time.time() - t0:.0f}s -> {args.out}/")


def write_markdown(r, path):
    L = ["# Experiment report", "", "## Data and labels", ""]
    d = r["data"]
    L += [f"- {d['resumes']} resumes ({d['train_resumes']} train / {d['test_resumes']} test, split by candidate)",
          f"- {d['train_jds']} train JDs / {d['test_jds']} test JDs (**synthetic**), {d['train_pairs']} / {d['test_pairs']} pairs",
          f"- Split: {d['split']}", f"- Labels: {d['label_definition']}",
          f"- Embedding model: `{r['embedding_model']}`; skill taxonomy: {d['taxonomy_size']} skills mined from training resumes", "",
          "## Baselines vs models (held-out test set)", "",
          "| Model | ROC-AUC | PR-AUC | Brier | ECE | NDCG@5 | NDCG@10 | Recall@10 | P@5 |", "|---|---|---|---|---|---|---|---|---|"]
    for n, m in r["model_comparison"].items():
        L.append(f"| {n} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | "
                 f"{m['ndcg@5']:.3f} | {m['ndcg@10']:.3f} | {m['recall@10']:.3f} | {m['precision@5']:.3f} |")
    L += ["", "## Ablation (remove one feature group)", "", "| Removed | ROC-AUC | PR-AUC | NDCG@10 |", "|---|---|---|---|"]
    for n, m in r["ablation"].items():
        L.append(f"| {n} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['ndcg@10']:.3f} |")
    c = r["calibration"]
    L += ["", "## Calibration", "",
          f"Brier {c['before']['brier']:.4f} -> {c['after']['brier']:.4f}, ECE {c['before']['ece']:.4f} -> {c['after']['ece']:.4f} (isotonic).",
          "", "## Error analysis", ""]
    e = r["error_analysis"]
    L += [f"- False-positive rate {e['false_positive_rate']:.3f}, false-negative rate {e['false_negative_rate']:.3f}",
          f"- Mean predicted probability by relevance level: {e['mean_probability_by_relevance']}",
          "- Most common false positives: " + "; ".join(f"{k} ({v})" for k, v in e["top_false_positives"]),
          "- Most common false negatives: " + "; ".join(f"{k} ({v})" for k, v in e["top_false_negatives"]),
          "", "## Bias check", "", r["bias_check"]["method"], "",
          "| Variant | Mean abs change | Max abs change | Decision flips |", "|---|---|---|---|"]
    for n, v in r["bias_check"]["variants"].items():
        L.append(f"| {n} | {v['mean_abs_change']:.4f} | {v['max_abs_change']:.4f} | {v['decision_flips']} |")
    L += ["", "## Feature importance (share of gain)", ""]
    L += [f"- {k}: {v:.3f}" for k, v in list(r["feature_importance"].items())[:10]]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
