"""Resume / job-description parsing: skill taxonomy, education, experience, certifications."""
import datetime
import json
import os
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from utils import SKILLS as SEED_SKILLS
from utils import clean_text

EDU_NAMES = {0: "Not specified", 1: "High school / diploma / associate",
             2: "Bachelor's", 3: "Master's", 4: "PhD"}

ALIASES = {
    "ms excel": "excel", "microsoft excel": "excel", "ms word": "microsoft word",
    "ms office": "microsoft office", "js": "javascript", "ml": "machine learning",
    "ai": "artificial intelligence", "postgres": "postgresql", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "powerbi": "power bi", "k8s": "kubernetes",
    "reactjs": "react", "nodejs": "node", "node js": "node",
}

BLACKLIST = set("""excellent strong good skilled experience experienced ability proficient knowledge
years other various including company name city state work working team player highly motivated
detail oriented self hard able basic advanced skills skill professional results driven
    reliable dependable adaptable bilingual english enterprise ads client clients
    january february march april may june july august september october november december""".split())

SKILLS_START = re.compile(r"\b(skills|highlights|core qualifications|qualifications)\b", re.I)
SKILLS_END = re.compile(
    r"\b(experience|education|accomplishments|work history|certifications?|additional information|"
    r"interests|affiliations|languages|professional experience)\b", re.I)


# ------------------------------------------------------------------ taxonomy
def _skill_candidates(raw):
    """Phrases listed in a resume's skills/highlights section."""
    m = SKILLS_START.search(raw)
    if not m:
        return set()
    section = raw[m.end(): m.end() + 1800]
    end = SKILLS_END.search(section)
    if end:
        section = section[: end.start()]
    out = set()
    for item in re.split(r"\s{2,}|\n|,|;|•|·|\||:|/|\(|\)", section):
        words = clean_text(item).split()
        if not 1 <= len(words) <= 3:
            continue
        if any(len(w) < 2 for w in words):
            continue
        if words[0] in ENGLISH_STOP_WORDS or words[-1] in ENGLISH_STOP_WORDS:
            continue
        if any(w in BLACKLIST for w in (words[0], words[-1])) and len(words) == 1:
            continue
        if words[0] in BLACKLIST or words[-1] in BLACKLIST:
            continue
        out.add(" ".join(words))
    return out


def mine_taxonomy(raw_resumes, min_df=8, max_df_frac=0.3):
    """Data-driven skill taxonomy: phrases that recur in the skills sections of many resumes."""
    df = Counter()
    for raw in raw_resumes:
        df.update(_skill_candidates(raw))
    cap = max_df_frac * len(raw_resumes)
    vocab = {p for p, c in df.items() if min_df <= c <= cap}
    vocab |= {s for s in SEED_SKILLS if len(s) > 1}
    taxonomy = {p: ALIASES.get(p, p) for p in vocab}
    taxonomy.update({a: c for a, c in ALIASES.items() if c in vocab or a in vocab})
    return taxonomy


def save_taxonomy(taxonomy, path):
    with open(path, "w") as f:
        json.dump(taxonomy, f)


def load_taxonomy(path):
    with open(path) as f:
        return json.load(f)


def extract_skills(text, taxonomy):
    """Canonical skills found in free text (1-3 gram lookup in the taxonomy)."""
    tokens = clean_text(text).split()
    found = set()
    i = 0
    while i < len(tokens):  # longest match wins, no overlapping matches
        for n in (3, 2, 1):
            gram = " ".join(tokens[i:i + n])
            if len(tokens[i:i + n]) == n and gram in taxonomy:
                found.add(taxonomy[gram])
                i += n
                break
        else:
            i += 1
    return found


# ----------------------------------------------------------------- education
_EDU = [
    (4, re.compile(r"\b(ph\.?\s?d|doctorate|doctor of)\b", re.I)),
    (3, re.compile(r"\b(master'?s?|mba|m\.?tech|m\.?eng)\b", re.I)),
    (3, re.compile(r"\b(M\.?S\.?c?|M\.?A\.?)\.?\s+(in|of|degree)\b")),
    (2, re.compile(r"\b(bachelor'?s?|b\.?tech|b\.?eng|bba)\b", re.I)),
    (2, re.compile(r"\b(B\.?S\.?c?|B\.?A\.?)\.?\s+(in|of|degree)\b")),
    (1, re.compile(r"\b(associate'?s?\s+(degree|of|in)|high school|diploma|ged)\b", re.I)),
]


def education_level(text):
    level = 0
    for lvl, pat in _EDU:
        if pat.search(text):
            level = max(level, lvl)
    return level


# ---------------------------------------------------------------- experience
_YEARS_STATED = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)\b", re.I)
_RANGE = re.compile(
    r"\b((?:19|20)\d{2})\b\W{0,10}?(?:(?:to|until)\b|[-\u2013\u2014])\W{0,10}?"
    r"(?:\d{1,2}/|[A-Za-z]{3,9}\.?\s+)?((?:19|20)\d{2}|current|present|now|today)\b", re.I)


def experience_years(text):
    """Years of experience: max(stated 'N years', union of dated job ranges)."""
    now = datetime.date.today().year
    stated = [int(n) for n in _YEARS_STATED.findall(text) if 0 < int(n) <= 45]
    intervals = []
    for a, b in _RANGE.findall(text):
        start = int(a)
        end = now if b.lower() in ("current", "present", "now", "today") else int(b)
        if 1960 <= start <= end <= now:
            intervals.append((start, end))
    covered = 0
    last_end = None
    for s, e in sorted(intervals):
        if last_end is None or s > last_end:
            covered += e - s
            last_end = e
        elif e > last_end:
            covered += e - last_end
            last_end = e
    return float(min(max(max(stated, default=0), covered), 45))


def certifications(raw, limit=8):
    out = []
    for seg in re.split(r"\s{2,}|\n", raw):
        seg = seg.strip()
        if 6 < len(seg) <= 120 and re.search(r"certif|licens", seg, re.I):
            out.append(seg)
    return out[:limit]


# ------------------------------------------------------------------ profiles
def resume_title(raw):
    first = re.split(r"\s{2,}|\n", raw.strip(), maxsplit=1)[0]
    return " ".join(first.split()[:8])


def parse_resume(raw, taxonomy):
    return {
        "title": resume_title(raw),
        "skills": sorted(extract_skills(raw, taxonomy)),
        "education_level": education_level(raw),
        "education": EDU_NAMES[education_level(raw)],
        "years_experience": experience_years(raw),
        "certifications": certifications(raw),
    }


def job_title(text):
    first = re.split(r"[\n.:]| - | — ", text.strip(), maxsplit=1)[0]
    return " ".join(first.split()[:12])


def parse_job(text, taxonomy):
    m = _YEARS_STATED.search(text)
    years = float(m.group(1)) if m and int(m.group(1)) <= 25 else 0.0
    return {
        "title": job_title(text),
        "skills": sorted(extract_skills(text, taxonomy)),
        "education_level": education_level(text),
        "education": EDU_NAMES[education_level(text)],
        "years_experience": years,
    }
