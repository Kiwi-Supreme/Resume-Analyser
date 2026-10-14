# Resume Analyser: Resume Analysis and Job Compatibility Prediction Using Machine Learning

Upload a resume and a job description. The app predicts the resume's job category, scores resume-job compatibility, ranks multiple candidates for a role, and explains every score. Flask backend, plain HTML/CSS/JavaScript frontend.

## Features

- **Analyze one resume:** predicted role (top 3, with confidence), compatibility score against a job description, matched and missing skills, missing keywords, and a "Why this score?" panel.
- **Rank candidates:** upload 2 to 30 resumes for one job and get an ordered shortlist with scores and per-candidate explanations.
- **Model report:** baselines, ablation, calibration, error analysis and the bias check, shown inside the app.
- Accepts PDF, DOCX and TXT resumes.

## ML components

| # | Component | Techniques |
|---|---|---|
| 1 | **Resume parsing & skill extraction** | Skill taxonomy *mined from the dataset's own "Skills" sections*, longest-match extraction, education level, years of experience (stated + union of dated job ranges), certifications |
| 2 | **Role classification** | TF-IDF + Linear SVM (5 models compared with 5-fold CV, tuned with GridSearchCV), 24 categories |
| 3 | **Semantic matching** | `all-MiniLM-L6-v2` sentence embeddings (chunked, averaged), TF-IDF and keyword-overlap baselines |
| 4 | **Supervised compatibility prediction** | LightGBM on 18 resume-job pair features, isotonic probability calibration, Logistic Regression baseline |
| 5 | **Learning-to-rank** | LightGBM LambdaRank, ranks candidates for a job |
| 6 | **Explainability & evaluation** | TreeSHAP contributions per prediction, ablations, calibration, error analysis, counterfactual bias check |

The displayed score is the **calibrated probability from the trained compatibility model**. The four dimension bars (required skills, semantic relevance, experience, education) are measured feature values, not separate learned outputs. It is not a probability of being hired.

## Training data: read this first

The Kaggle dataset ([snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), 2,481 resumes, 24 categories) has labelled resumes but **no job descriptions and no match labels**. So:

- Job descriptions are **synthetic**, generated from skills mined from *training* resumes of each category.
- Relevance labels are **weak labels derived from resume categories**: 3 = same category, 2 = closely related, 1 = same broad family, 0 = unrelated. They are *not* recruiter judgments.
- To limit leakage, resumes are split by candidate, JDs by id, and train/test JDs draw from disjoint skill pools.

Treat the compatibility and ranking metrics as results on this weak-label benchmark, not as real-world hiring accuracy.

## Results

**Role classifier** (24 classes, held-out test set): Linear SVM, **71.4% accuracy, 0.67 macro-F1** (random guessing is about 4%).

**Compatibility and ranking** (held-out candidates and jobs):

| Model | ROC-AUC | NDCG@10 |
|---|---|---|
| Keyword overlap (baseline) | 0.694 | 0.645 |
| TF-IDF cosine (baseline) | 0.769 | 0.643 |
| Embedding cosine (baseline) | 0.759 | 0.626 |
| Logistic Regression | 0.807 | 0.670 |
| **LightGBM classifier (calibrated)** | 0.804 | **0.699** |
| LambdaRank ranker | 0.802 | 0.677 |

- Removing the embedding features causes the largest drop in ROC-AUC (0.806 to 0.779).
- Probabilities are well calibrated (expected calibration error about 0.018).
- Bias check: adding names, pronouns or a birth year to resumes changes scores by about 1% on average.

Full tables (ablation, calibration, error analysis, bias check) are in [`models/experiment_report.md`](models/experiment_report.md). Regenerate with `python train_matching.py`.

## Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py          # open http://127.0.0.1:5000
```
First run downloads the dataset (public, no Kaggle login) and the embedding model and trains everything. The one-off embedding step takes several minutes on CPU; later runs reuse the cache. Retrain manually with `python train.py` and `python train_matching.py`.

## Presentation

[`Resume_Analyser.pptx`](Resume_Analyser.pptx) is the project presentation: problem, method, data, EDA, features, results, demo screenshots, conclusion and references.

## Structure
```
train.py             role classifier training + evaluation
train_matching.py    synthetic JDs, pairs, compatibility model, ranker, experiments
parsing.py           skill taxonomy, education / experience / certification parsing
matching.py          embeddings, pair features, SHAP explanations
utils.py             file text extraction, cleaning
app.py               Flask API: /api/analyze, /api/rank, /api/metrics
static/              index.html, style.css, script.js
models/              metrics.json, experiment_report.{json,md}, taxonomy.json, confusion_matrix.png
Resume_Analyser.pptx project presentation
```

## Limitations and future work
- **Real labels:** the job descriptions are synthetic and labels are category-derived. Training on recruiter-labelled resume-job pairs and hiring outcomes is the main next step.
- **Skill coverage:** the skill list is mined from this dataset, so niche or very new skills may be missed. Linking to an ontology such as ESCO or O*NET would help.
- **Parsing:** scanned (image-only) PDFs are not supported; OCR and named-entity recognition for titles and projects are planned.
- **Fairness:** protected attributes are never used as features, but names or identity text can shift scores slightly (see the bias check). Name masking and formal audits are future work, and scores should not be used alone to rank real candidates.
- **Deployment:** cloud hosting, user accounts and a feedback loop for retraining.

## Author

Built by [Kiwi-Supreme](https://github.com/Kiwi-Supreme).
