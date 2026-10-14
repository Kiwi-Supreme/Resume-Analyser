"""Semantic embeddings + resume/job pair features shared by training and the web app."""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from scipy import sparse

import parsing
from utils import clean_text

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 180
MAX_CHUNKS = 4

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(EMB_MODEL)
    return _encoder


def embed_texts(texts, batch_size=64):
    return get_encoder().encode(list(texts), batch_size=batch_size,
                                normalize_embeddings=True, show_progress_bar=False)


def embed_documents(texts, batch_size=64):
    """Long documents are split into word chunks; chunk embeddings are averaged."""
    chunks, owner = [], []
    for i, t in enumerate(texts):
        words = str(t).split()
        pieces = [" ".join(words[j:j + CHUNK_WORDS])
                  for j in range(0, max(len(words), 1), CHUNK_WORDS)][:MAX_CHUNKS]
        chunks.extend(pieces or [""])
        owner.extend([i] * len(pieces or [""]))
    emb = embed_texts(chunks, batch_size)
    owner = np.array(owner)
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    np.add.at(out, owner, emb)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


# --------------------------------------------------------------- profiles
class Profiles:
    """Column-oriented features for a set of resumes or job descriptions."""

    def __init__(self, emb, tfidf, skills, years, edu):
        self.emb, self.tfidf = emb, tfidf
        self.skills, self.years, self.edu = skills, years, edu

    def __len__(self):
        return len(self.skills)

    def take(self, idx):
        idx = np.asarray(idx)
        return Profiles(self.emb[idx], self.tfidf[idx],
                        [self.skills[i] for i in idx], self.years[idx], self.edu[idx])


def build_profiles(raw_texts, tfidf_vectorizer, taxonomy, parser, emb=None):
    raw_texts = list(raw_texts)
    parsed = [parser(t, taxonomy) for t in raw_texts]
    if emb is None:
        emb = embed_documents([clean_text(t) for t in raw_texts])
    tfidf = tfidf_vectorizer.transform([clean_text(t) for t in raw_texts])
    return Profiles(
        emb, tfidf,
        [set(p["skills"]) for p in parsed],
        np.array([p["years_experience"] for p in parsed], dtype=np.float32),
        np.array([p["education_level"] for p in parsed], dtype=np.float32),
    )


# ------------------------------------------------------------ pair features
GROUPS = {
    "embedding": ["emb_cos"],
    "tfidf": ["tfidf_cos"],
    "skills": ["skill_cov", "skill_matched", "skill_missing", "skill_jaccard",
               "req_skill_count", "resume_skill_count"],
    "experience": ["resume_years", "job_years", "exp_ratio", "exp_gap", "exp_meets"],
    "education": ["resume_edu", "job_edu", "edu_gap", "edu_meets"],
}
FEATURES = [f for g in GROUPS.values() for f in g]

FEATURE_LABELS = {
    "emb_cos": "Semantic similarity (embeddings)",
    "tfidf_cos": "Keyword similarity (TF-IDF)",
    "skill_cov": "Required-skill coverage",
    "skill_matched": "Number of matched skills",
    "skill_missing": "Number of missing skills",
    "skill_jaccard": "Skill overlap (Jaccard)",
    "req_skill_count": "Skills required by the job",
    "resume_skill_count": "Skills found in resume",
    "resume_years": "Candidate's years of experience",
    "job_years": "Years of experience required",
    "exp_ratio": "Experience vs requirement",
    "exp_gap": "Experience surplus / shortfall",
    "exp_meets": "Meets experience requirement",
    "resume_edu": "Candidate's education level",
    "job_edu": "Education level required",
    "edu_gap": "Education surplus / shortfall",
    "edu_meets": "Meets education requirement",
}


def pair_features(R, J, ri, ji):
    """Feature matrix for pairs (R[ri[k]], J[ji[k]]). Returns array (n_pairs, len(FEATURES))."""
    ri, ji = np.asarray(ri), np.asarray(ji)
    emb_cos = np.einsum("ij,ij->i", R.emb[ri], J.emb[ji])
    tfidf_cos = np.asarray(R.tfidf[ri].multiply(J.tfidf[ji]).sum(axis=1)).ravel()

    n = len(ri)
    cov, matched, missing, jacc, req_n, res_n = (np.zeros(n) for _ in range(6))
    for k in range(n):
        rs, js = R.skills[ri[k]], J.skills[ji[k]]
        inter = len(rs & js)
        matched[k], missing[k] = inter, len(js) - inter
        cov[k] = inter / len(js) if js else 0.0
        jacc[k] = inter / len(rs | js) if (rs | js) else 0.0
        req_n[k], res_n[k] = len(js), len(rs)

    ry, jy = R.years[ri], J.years[ji]
    ratio = np.where(jy > 0, np.minimum(ry / np.maximum(jy, 1e-9), 3.0), 1.0)
    re_, je = R.edu[ri], J.edu[ji]

    cols = {
        "emb_cos": emb_cos, "tfidf_cos": tfidf_cos,
        "skill_cov": cov, "skill_matched": matched, "skill_missing": missing,
        "skill_jaccard": jacc, "req_skill_count": req_n, "resume_skill_count": res_n,
        "resume_years": ry, "job_years": jy, "exp_ratio": ratio, "exp_gap": ry - jy,
        "exp_meets": (ry >= jy).astype(float),
        "resume_edu": re_, "job_edu": je, "edu_gap": re_ - je,
        "edu_meets": (re_ >= je).astype(float),
    }
    return np.column_stack([cols[f] for f in FEATURES]).astype(np.float32)


def explain(booster, x_row, top=4):
    """Per-feature contributions (LightGBM TreeSHAP, log-odds) for one pair."""
    contrib = booster.predict(x_row.reshape(1, -1), pred_contrib=True)[0][:-1]
    order = np.argsort(contrib)
    def item(i):
        return {"feature": FEATURES[i], "label": FEATURE_LABELS[FEATURES[i]],
                "value": round(float(x_row[i]), 3), "impact": round(float(contrib[i]), 3)}
    positive = [item(i) for i in order[::-1] if contrib[i] > 0.01][:top]
    negative = [item(i) for i in order if contrib[i] < -0.01][:top]
    return {"positive": positive, "negative": negative}
