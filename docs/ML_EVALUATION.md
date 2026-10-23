# Machine Learning Audit & Model Evaluation Report

## 1. Executive Summary

This report documents the machine learning models developed within the IPL INSIGHT platform, their feature formulations, chronological evaluation protocols, leakage prevention measures, and empirical test results.

Two distinct models are evaluated:
1. **Pre-Match Winner Classifier**: An experimental model attempting to predict match outcomes before the toss.
2. **Live Chase Probability Model**: A calibrated in-play classifier estimating the probability of a successful run chase after any completed delivery in the second innings.

---

## 2. Chronological Validation & Leakage Prevention Protocol

A fundamental flaw in naive sports analytics is random train/test splitting (`train_test_split(shuffle=True)`). Random splitting causes catastrophic data leakage in sports because:
- Delivery states within the same match share identical venue conditions, pitch deterioration, and match momentum.
- A model trained on over 5 of a match will overfit when predicting over 6 of the same match.
- Information from future calendar seasons leaks into past predictions.

To guarantee zero leakage:
1. **Match-Level Partitioning**: Every delivery row from a given match is assigned to exactly one partition.
2. **Strict Chronological Ordering**:
   - **Fit / Training Period**: 2008 through 2021 (832 matches, 95,912 chase states)
   - **Probability Calibration Period**: 2022 through 2023 (145 matches, 16,976 chase states)
   - **Held-Out Test Period**: 2024 through 2026 (208 matches, 23,814 chase states)
3. **Feature Timing Integrity**: Features describe only the state immediately after a completed legal or extra delivery. Eventual match outcomes, final innings totals, and innings-ending flags are strictly excluded from candidate features.
4. **Equal Match Weighting**: Because matches vary in ball count, evaluation metrics are match-weighted so longer matches do not disproportionately skew metrics.

---

## 3. Model 1: Pre-Match Winner Classifier Audit

### 3.1 Objective & Target Definition
- **Target**: `team1_won` (1 if team 1 won, 0 if lost). Ties and no-results excluded.
- **Candidate Features**: `team1`, `team2`, `venue`, rolling historical win rates, head-to-head records, venue first-innings averages, and venue chasing win rates prior to match date.

### 3.2 Held-Out Empirical Test Results
Data source: `reports/model_evaluation.csv` (Test Split: 2024–2026 matches)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Brier Score |
|---|---|---|---|---|---|---|
| **Majority Baseline** | 44.60% | 44.60% | 1.000 | 0.617 | **0.500** | 0.251 |
| **Logistic Regression** | 45.54% | 40.37% | 0.463 | 0.431 | **0.440** | 0.290 |

### 3.3 Critical Findings & The Safety Gate
- **Failure to Beat Baseline**: The pre-match logistic regression model achieves an ROC-AUC of **0.440**, which is worse than a random baseline (0.500).
- **Domain Root Cause**: Cricket T20 match outcomes before the toss are dominated by high stochastic variance, team selection variations, day-of pitch behavior, and toss impact. Pre-match aggregate team features lack sufficient signal to overcome this variance.
- **Production Safety Gate (`src/future_predictions.py`)**: 
  Instead of deploying an inaccurate model or using an LLM to hallucinate predictions, the platform enforces an automated safety gate:
  `if model_auc <= baseline_auc: return {"enabled": False, "reason": "Model did not beat held-out baseline."}`
  Future-match predictions are **strictly disabled** with an honest, transparent explanation to the user.

---

## 4. Model 2: Live Chase Probability Model Audit

### 4.1 Objective & Target Definition
- **Prediction Point**: Immediately after any completed delivery in regular second innings.
- **Target**: `is_chasing_winner` (1 if the batting side completes the chase, 0 if they fall short).
- **Exclusions**: Super overs, DLS rain-shortened matches, terminal states (already finished), and unlabeled matches.
- **Input Features**:
  - `batting_team`, `bowling_team`, `venue` (one-hot encoded)
  - `target_runs`, `current_score`, `legal_balls_bowled`, `current_wickets`
  - Derived: `runs_needed`, `balls_remaining`, `wickets_remaining`, `current_run_rate`, `required_run_rate`

### 4.2 Held-Out Empirical Test Results
Data source: `reports/live_chase_evaluation.csv` (Test Split: 2024–2026 matches, 23,814 states across 208 matches, match-weighted)

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Brier Score | Log Loss |
|---|---|---|---|---|---|---|---|
| **Prior Baseline** | 55.77% | 55.77% | 1.000 | 0.716 | 0.500 | 0.2467 | 0.6865 |
| **Logistic Regression (Uncalibrated)** | 78.81% | 88.66% | 0.711 | 0.789 | 0.8995 | 0.1461 | 0.4535 |
| **Logistic Regression (Sigmoid Calibrated)** | **79.46%** | **87.98%** | **0.732** | **0.799** | **0.8995** | **0.1389** | **0.4260** |

### 4.3 Why Calibrated Logistic Regression Outperformed Random Forest
On the validation split (2020–2021):
- Logistic Regression achieved validation Brier score of **0.1331** (ROC-AUC 0.8928).
- Random Forest achieved validation Brier score of **0.1400** (ROC-AUC 0.8807).
Logistic regression demonstrated superior calibration and lower log-loss, preventing the overconfident probability spikes typical of decision tree ensembles in extreme game states.

---

## 5. Model Interpretability & Factor Impact

Feature importance is computed via signed log-odds coefficients and absolute magnitude ranking (`reports/live_chase_feature_importance.csv`):

| Rank | Model Factor | Description / Directional Impact |
|---|---|---|
| 1 | `required_run_rate` | Primary negative factor. As RRR climbs above 10.0, chase win probability drops exponentially. |
| 2 | `wickets_remaining` | Primary positive factor. Having 7+ wickets in hand strongly elevates chase likelihood even when RRR is high. |
| 3 | `current_run_rate` | Positive momentum indicator reflecting batting scoring rhythm. |
| 4 | `balls_remaining` | Interaction term providing runway for consolidation. |
| 5 | `venue` | Category coefficients reflect venue-specific chasing advantages (e.g. Wankhede vs Chepauk). |

In the Prediction Lab, the user is presented with both the headline percentage and the specific local factors driving that state.
