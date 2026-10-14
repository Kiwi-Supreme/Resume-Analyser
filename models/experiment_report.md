# Experiment report

## Data and labels

- 2481 resumes (1984 train / 497 test, split by candidate)
- 720 train JDs / 288 test JDs (**synthetic**), 28800 / 11520 pairs
- Split: resumes by candidate (80/20), JDs by id, JD skill pools disjoint between train/test
- Labels: 3 same category; 2 closely related; 1 same broad family; 0 unrelated. Weak labels derived from dataset categories; JDs are synthetic. 'Compatible' = relevance >= 2.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`; skill taxonomy: 419 skills mined from training resumes

## Baselines vs models (held-out test set)

| Model | ROC-AUC | PR-AUC | Brier | ECE | NDCG@5 | NDCG@10 | Recall@10 | P@5 |
|---|---|---|---|---|---|---|---|---|
| Baseline: keyword overlap | 0.694 | 0.500 | 0.206 | 0.128 | 0.703 | 0.645 | 0.507 | 0.703 |
| Baseline: TF-IDF cosine | 0.769 | 0.648 | 0.284 | 0.283 | 0.681 | 0.643 | 0.512 | 0.683 |
| Baseline: embedding cosine | 0.759 | 0.622 | 0.186 | 0.108 | 0.673 | 0.626 | 0.502 | 0.684 |
| Logistic Regression (all features) | 0.807 | 0.690 | 0.154 | 0.012 | 0.711 | 0.670 | 0.535 | 0.717 |
| LightGBM classifier (uncalibrated) | 0.805 | 0.690 | 0.155 | 0.023 | 0.712 | 0.674 | 0.540 | 0.720 |
| LightGBM classifier (calibrated) | 0.804 | 0.672 | 0.155 | 0.018 | 0.739 | 0.699 | 0.559 | 0.740 |
| LightGBM LambdaRank ranker | 0.802 | 0.689 | 0.203 | 0.191 | 0.718 | 0.677 | 0.541 | 0.725 |

## Ablation (remove one feature group)

| Removed | ROC-AUC | PR-AUC | NDCG@10 |
|---|---|---|---|
| (none - full model) | 0.806 | 0.692 | 0.677 |
| embedding | 0.779 | 0.652 | 0.650 |
| tfidf | 0.789 | 0.658 | 0.654 |
| skills | 0.795 | 0.680 | 0.664 |
| experience | 0.804 | 0.689 | 0.676 |
| education | 0.803 | 0.691 | 0.677 |

## Calibration

Brier 0.1546 -> 0.1546, ECE 0.0233 -> 0.0177 (isotonic).

## Error analysis

- False-positive rate 0.095, false-negative rate 0.504
- Mean predicted probability by relevance level: {0: 0.20551274988314971, 1: 0.24090805603073126, 2: 0.3620689874312273, 3: 0.5519741461385075}
- Most common false positives: PUBLIC-RELATIONS resume vs ARTS job (14); INFORMATION-TECHNOLOGY resume vs BPO job (13); INFORMATION-TECHNOLOGY resume vs AUTOMOBILE job (12); BUSINESS-DEVELOPMENT resume vs BANKING job (12); INFORMATION-TECHNOLOGY resume vs CONSULTANT job (12); CONSULTANT resume vs ACCOUNTANT job (10); DIGITAL-MEDIA resume vs ARTS job (10); ADVOCATE resume vs FITNESS job (10)
- Most common false negatives: AGRICULTURE resume vs AGRICULTURE job (94); APPAREL resume vs APPAREL job (93); CONSULTANT resume vs CONSULTANT job (84); HEALTHCARE resume vs HEALTHCARE job (84); AVIATION resume vs AVIATION job (79); ARTS resume vs ARTS job (78); BANKING resume vs BANKING job (74); FITNESS resume vs FITNESS job (72)

## Bias check

Prefix each of 120 test resumes with identity text (name / pronouns / birth year) and compare compatibility probabilities. Protected attributes are never model inputs.

| Variant | Mean abs change | Max abs change | Decision flips |
|---|---|---|---|
| female name | 0.0103 | 0.1300 | 2 |
| male name | 0.0096 | 0.1562 | 1 |
| south-asian name | 0.0078 | 0.1089 | 2 |
| east-asian name | 0.0054 | 0.1066 | 1 |
| older (born 1962) | 0.0060 | 0.1089 | 2 |

## Feature importance (share of gain)

- tfidf_cos: 0.512
- emb_cos: 0.194
- skill_jaccard: 0.063
- resume_skill_count: 0.054
- resume_years: 0.029
- skill_cov: 0.026
- exp_gap: 0.016
- job_years: 0.016
- exp_ratio: 0.015
- job_edu: 0.015
