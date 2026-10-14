import json
import os
import subprocess
import sys

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import matching
import parsing
from utils import clean_text, extract_text

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
P = lambda name: os.path.join(MODELS, name)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024


def ensure_trained():
    for artifact, script in (("model.joblib", "train.py"), ("compat.joblib", "train_matching.py")):
        if not os.path.exists(P(artifact)):
            print(f"Missing models/{artifact} - running {script} (first run only, takes several minutes)...")
            subprocess.run([sys.executable, os.path.join(BASE, script)], cwd=BASE, check=True)


ensure_trained()

classifier = joblib.load(P("model.joblib"))           # resume -> job category (TF-IDF + SVM)
cls_tfidf = classifier.named_steps["tfidf"]
classes = classifier.classes_
taxonomy = parsing.load_taxonomy(P("taxonomy.json"))
match_tfidf = joblib.load(P("match_tfidf.joblib"))
bundle = joblib.load(P("compat.joblib"))              # compatibility classifier + calibrator + ranker
compat, calibrator, ranker = bundle["clf"], bundle["calibrator"], bundle["ranker"]
F = {f: i for i, f in enumerate(matching.FEATURES)}
matching.embed_texts(["warm up"])                     # load the embedding model once at start-up


# ---------------------------------------------------------------- helpers
def top_categories(proba, n=3):
    idx = np.argsort(proba)[::-1][:n]
    return [{"category": str(classes[i]), "confidence": round(float(proba[i]) * 100, 1)} for i in idx]


def missing_keywords(resume_clean, job_clean, n=8):
    """Highest-weighted job terms (TF-IDF) that never appear in the resume."""
    terms = cls_tfidf.get_feature_names_out()
    jv = cls_tfidf.transform([job_clean]).toarray()[0]
    rv = cls_tfidf.transform([resume_clean]).toarray()[0]
    out = []
    for i in np.argsort(jv)[::-1]:
        if jv[i] <= 0 or len(out) >= n:
            break
        if rv[i] == 0 and " " not in terms[i]:
            out.append(str(terms[i]))
    return out


def read_resume(upload):
    raw = extract_text(upload.filename, upload.read())
    if len(clean_text(raw)) < 50:
        raise ValueError("Could not extract enough text from the resume.")
    return raw


def dimensions(x):
    """Four human-readable dimensions. These are feature values, not learned scores."""
    return {
        "required_skills": round(float(x[F["skill_cov"]]) * 100) if x[F["req_skill_count"]] > 0 else None,
        "semantic_relevance": round(float(np.clip(x[F["emb_cos"]], 0, 1)) * 100),
        "experience": round(float(min(x[F["exp_ratio"]], 1.0)) * 100) if x[F["job_years"]] > 0 else None,
        "education": round(float(x[F["edu_meets"]]) * 100) if x[F["job_edu"]] > 0 else None,
    }


def compat_probability(X):
    return calibrator.predict(compat.predict_proba(X)[:, 1])


def category_info(raw):
    proba = classifier.predict_proba([clean_text(raw)])[0]
    return top_categories(proba, 3)


# --------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/metrics")
def metrics():
    out = {}
    if os.path.exists(P("metrics.json")):
        out["classifier"] = json.load(open(P("metrics.json")))
    if os.path.exists(P("experiment_report.json")):
        out["matching"] = json.load(open(P("experiment_report.json")))
    return jsonify(out)


@app.post("/api/analyze")
def analyze():
    job_text = request.form.get("job_description", "").strip()
    upload = request.files.get("resume")
    if upload is None or not upload.filename:
        return jsonify({"error": "Please upload a resume."}), 400
    try:
        raw = read_resume(upload)
    except Exception as e:
        return jsonify({"error": f"Could not read the resume: {e}"}), 400

    cats = category_info(raw)
    profile = parsing.parse_resume(raw, taxonomy)
    result = {
        "predicted_category": cats[0],
        "top_categories": cats,
        "profile": {k: profile[k] for k in ("skills", "education", "years_experience", "certifications")},
        "resume_skills": profile["skills"],
    }

    if job_text:
        R = matching.build_profiles([raw], match_tfidf, taxonomy, parsing.parse_resume)
        J = matching.build_profiles([job_text], match_tfidf, taxonomy, parsing.parse_job)
        X = matching.pair_features(R, J, [0], [0])
        score = float(compat_probability(X)[0])
        job = parsing.parse_job(job_text, taxonomy)
        have = set(profile["skills"])
        result.update({
            "job_category": category_info(job_text)[0],
            "match_score": round(score * 100, 1),
            "dimensions": dimensions(X[0]),
            "job_requirements": {k: job[k] for k in ("skills", "education", "years_experience")},
            "matched_skills": [s for s in job["skills"] if s in have],
            "missing_skills": [s for s in job["skills"] if s not in have],
            "missing_keywords": missing_keywords(clean_text(raw), clean_text(job_text)),
            "explanation": matching.explain(compat.booster_, X[0]),
        })
    return jsonify(result)


@app.post("/api/rank")
def rank():
    job_text = request.form.get("job_description", "").strip()
    uploads = [u for u in request.files.getlist("resumes") if u.filename]
    if not job_text:
        return jsonify({"error": "Paste a job description to rank candidates against."}), 400
    if len(uploads) < 2:
        return jsonify({"error": "Upload at least two resumes to rank."}), 400
    if len(uploads) > 30:
        return jsonify({"error": "Please upload at most 30 resumes at a time."}), 400

    raws, names, errors = [], [], []
    for u in uploads:
        try:
            raws.append(read_resume(u))
            names.append(u.filename)
        except Exception as e:
            errors.append(f"{u.filename}: {e}")
    if len(raws) < 2:
        return jsonify({"error": "Fewer than two resumes could be read. " + "; ".join(errors)}), 400

    R = matching.build_profiles(raws, match_tfidf, taxonomy, parsing.parse_resume)
    J = matching.build_profiles([job_text], match_tfidf, taxonomy, parsing.parse_job)
    X = matching.pair_features(R, J, np.arange(len(raws)), np.zeros(len(raws), dtype=int))
    probs = compat_probability(X)
    rank_scores = ranker.predict(X)
    job = parsing.parse_job(job_text, taxonomy)

    candidates = []
    for i in np.argsort(-rank_scores, kind="stable"):
        have = R.skills[i]
        expl = matching.explain(compat.booster_, X[i], top=2)
        candidates.append({
            "filename": names[i],
            "match_score": round(float(probs[i]) * 100, 1),
            "ranker_score": round(float(rank_scores[i]), 3),
            "dimensions": dimensions(X[i]),
            "years_experience": float(R.years[i]),
            "matched_skills": [s for s in job["skills"] if s in have],
            "missing_skills": [s for s in job["skills"] if s not in have],
            "role": category_info(raws[i])[0],
            "explanation": expl,
        })
    for pos, c in enumerate(candidates, 1):
        c["rank"] = pos
    return jsonify({"candidates": candidates, "job_requirements": {
        k: job[k] for k in ("skills", "education", "years_experience")}, "warnings": errors})


if __name__ == "__main__":
    app.run(debug=False)
