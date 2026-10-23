# Live Chase Probability Model Card

## Purpose

This model estimates whether the batting side will complete a successful chase **after the recorded delivery has finished**. It is a historical-data experiment, not an official live score feed or a guarantee of the match result.

## Training and Evaluation

- Source: `csv/ipl_ball_by_ball_processed.csv`, dates 2008-04-18 through 2026-05-31.
- Target: `is_chasing_winner`; unlabeled ties and no-results are excluded.
- Excluded examples: super overs, DLS/shortened matches, innings-ending states, missing labels, and invalid state fields.
- Train: 2008-2019, 82,661 states across 717 matches.
- Model selection: 2020-2021, 13,251 states across 115 matches.
- Final estimator fit after selection: 2008-2021, 95,912 states across 832 matches.
- Probability calibration: 2022-2023, 16,976 states across 145 matches.
- Final test: 2024-2026, 23,814 states across 208 matches.
- All states from a match remain in one date-based partition. Every match has equal total weight within its partition so longer matches do not dominate the metrics.
- Candidate models: logistic regression and random forest. Logistic regression was selected using the lowest match-weighted Brier score on 2020-2021.
- Calibration: sigmoid/Platt-style logistic mapping fitted on the separate 2022-2023 calibration period. The 2024-2026 test period was not used for model fitting, selection, or calibration.

## Held-Out Test Results

Metrics below are match-weighted. Accuracy, precision, recall, and F1 use a 0.5 probability threshold.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prior baseline | 0.558 | 0.558 | 1.000 | 0.716 | 0.500 | 0.247 | 0.687 |
| Logistic regression, uncalibrated | 0.788 | 0.887 | 0.711 | 0.789 | 0.900 | 0.146 | 0.454 |
| Logistic regression, sigmoid calibrated | 0.795 | 0.880 | 0.732 | 0.799 | 0.900 | 0.139 | 0.426 |

The held-out metrics improve over this baseline. This is encouraging evidence for the selected historical evaluation, not proof that performance will remain the same on future seasons. Delivery states within a match are correlated even though each match is kept in a single split and receives equal total metric weight.

The saved confusion matrix counts delivery states, not matches. It is therefore not directly comparable to the match-weighted metric values in the table.

## Input Contract

The model accepts batting team, bowling team, venue, target, current score, wickets lost, and legal balls bowled. Derived inputs include runs needed, wickets remaining, balls remaining, current run rate, and required run rate. It rejects invalid targets, finished chases, invalid wickets/balls, and empty team or venue names.

Inputs describe the state after a completed delivery. `match_winner`, `is_chasing_winner`, final scores, innings-ending flags, and other future/result fields are not model features. The model's `wickets_remaining` uses the standard 10-wicket innings limit. Its manual state validation assumes a 120-legal-ball innings; DLS/shortened matches are excluded.

## Explainability and Limitations

`reports/live_chase_feature_importance.csv` contains absolute logistic-regression coefficient magnitudes after preprocessing. These magnitudes indicate model association, not causal impact; absolute values omit coefficient direction and can be affected by one-hot category frequency. Categorical venue/team coefficients can be unstable for rare categories, and unseen categories are ignored by the encoder.

The model excludes tied/no-result outcomes, super overs, and DLS/shortened matches. It does not use recent momentum features because their timing was not yet independently verified. The calibration curve and confusion matrix are saved under `reports/`. Manual inputs are hypothetical unless copied from a verified historical state.

## Reproduce

From the project root on Windows:

```powershell
.\.venv\Scripts\python.exe -m src.train_chase_model
.\.venv\Scripts\python.exe -m pytest -q
```

The trained artifact is `models/live_chase_probability.joblib`; the model version and feature schema are checked when loaded by `src/chase_predictor.py`.