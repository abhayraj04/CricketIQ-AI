# IPL Analytics & AI Intelligence Platform — Implementation Plan

## 1. Executive Summary & Audit Baseline

This document provides a comprehensive audit of the existing codebase, baseline test results, data schema, model performance, architectural gaps, and the step-by-step implementation plan to elevate this project into a production-grade, portfolio-ready sports analytics and AI platform.

### Baseline Test Execution
- **Test Framework**: Pytest 9.1.1 on Python 3.14.0 (Windows)
- **Baseline Test Suite**: 46 passed in 128.37s
- **Test Files Audited**:
  - `tests/test_analytics_metrics.py`: 5 passed
  - `tests/test_chase_features.py`: 11 passed
  - `tests/test_chase_predictor.py`: 13 passed
  - `tests/test_dashboard_navigation.py`: 1 passed (Streamlit AppTest across all 9 pages)
  - `tests/test_match_views.py`: 2 passed
  - `tests/test_prediction_lab_app.py`: 2 passed
  - `tests/test_team_registry.py`: 7 passed
  - `tests/test_time_splits.py`: 3 passed
  - `tests/test_ui_theme.py`: 2 passed

---

## 2. Existing Architecture & Feature Inventory

### Core Modules
| Module | Location | Status | Description |
|---|---|---|---|
| `app.py` | Root | Working | Streamlit entry point, theme toggle (Dark/Light), sidebar filters (years, teams, venues, stage), page router. |
| `src/data_loader.py` | `src/` | Working | SQLite query loader with Streamlit `@st.cache_data`. |
| `src/dashboard_views.py` | `src/` | Working | Renders Overview, Team Analytics, Player Analytics, Venue & Toss, Ball by Ball. |
| `src/match_views.py` | `src/` | Working | Match search, scorecard breakdown, over-by-over chart, batter/bowler figures. |
| `src/prediction_views.py` | `src/` | Working | Live chase prediction (hypothetical & historical), future match mode safety gate. |
| `src/report_views.py` | `src/` | Working | Model performance comparison, data quality checks, schema inspection. |
| `src/metrics.py` | `src/` | Working | Pure analytics metric calculation functions. |
| `src/ui_theme.py` | `src/` | Working | Dark/Light theme CSS variables, Plotly theme templates. |
| `src/branding.py` & `src/brand_views.py` | `src/` | Working | Team identity badge, player initials fallback. |
| `src/team_registry.py` | `src/` | Working | Team alias registry, future eligibility validator. |
| `src/future_predictions.py` | `src/` | Working | Safety gate disabling future prediction when model ROC-AUC <= baseline. |
| `src/chase_features.py` | `src/` | Working | Feature engineering and leakage exclusion for second innings chase states. |
| `src/chase_model.py` & `src/chase_predictor.py` | `src/` | Working | Calibrated logistic regression chase model inference and feature contributions. |
| `src/clean_data.py` | `src/` | Working | Pipeline cleaning raw CSVs into processed data and generating quality report. |
| `src/load_to_sqlite.py` | `src/` | Working | Batch loads processed data into `ipl_analytics.sqlite`. |

---

## 3. Data Inventory & Schema

| Table / Source File | Size | Grain | Primary / Natural Key | Description |
|---|---|---|---|---|
| `ipl_matches_summary` | 1,243 rows x 59 cols | One row per match | `match_id` | Match-level facts, dates (2008-04-18 to 2026-05-31), teams, tosses, venue, outcomes, player of match. |
| `ipl_ball_by_ball_processed` | 295,732 rows x 108 cols | One row per delivery | `match_id` + `innings` + `ball_seq_idx` | Granular delivery events, runs, extras, wickets, cumulative chase state. |
| `ipl_player_profiles` | 811 rows x 41 cols | One row per player | `player_name` | Career aggregated statistics, batting/bowling styles, roles, rates. |
| `ipl_player_summary` | 811 rows x 41 cols | Duplicate of profiles | `player_name` | Auxiliary duplicate; omitted from fact joins. |

### Verified Data Constraints:
- Delivery run reconciliation: 381,529 batter runs + 20,209 extras = 401,738 total runs.
- 1,218 recorded winners, 16 ties, 9 no-results.
- Dates range from 2008 through 2026. Calendar year derived from dates is used for chronological splits rather than the unreliable raw season label.
- Missing values represent true absence of event (e.g. bowler wicket on run-out, or bowler economy when zero overs bowled).

---

## 4. Model Inventory & Baseline Metrics

### Model 1: Pre-Match Winner Classifier (`models/team1_win_prediction.joblib`)
- **Target**: `team1_won` (binary: 1 if team1 won, 0 if lost). Unlabeled ties/no-results excluded.
- **Features**: Categorical (`team1`, `team2`, `venue`), Historical rolling form & win rates (`team1_win_pct_prior`, `h2h_team1_win_pct`, `venue_chase_win_pct`, etc.).
- **Evaluation Split**: Chronological. Train (< 2022), Validation (2022-2023), Test (2024-2026).
- **Actual Baseline vs Candidate Test Metrics**:
  - Majority Baseline: Accuracy = 44.60%, ROC-AUC = 0.500, Brier = 0.251
  - Logistic Regression: Accuracy = 45.54%, ROC-AUC = 0.440, Brier = 0.290
- **Assessment**: The pre-match model **fails to outperform the random baseline** on held-out test data (ROC-AUC 0.440 < 0.500).
- **Safety Gate**: Properly disabled for live and future inference with explicit user explanation.

### Model 2: Live Second-Innings Chase Probability (`models/live_chase_probability.joblib`)
- **Target**: `is_chasing_winner` (1 if batting side in 2nd innings successfully chases target).
- **Exclusions**: Super overs, DLS/shortened matches, terminal states, missing labels.
- **Splits**: Fit through 2021 (95,912 states, 832 matches), Calibrated 2022-2023 (16,976 states, 145 matches), Test 2024-2026 (23,814 states, 208 matches). Whole matches kept within a single split.
- **Actual Test Metrics (Match-Weighted)**:
  - Prior Baseline: Accuracy = 55.77%, ROC-AUC = 0.500, Brier = 0.247, Log Loss = 0.687
  - Logistic Regression (Uncalibrated): Accuracy = 78.81%, ROC-AUC = 0.8995, Brier = 0.146, Log Loss = 0.453
  - Logistic Regression (Sigmoid Calibrated): Accuracy = 79.46%, ROC-AUC = 0.8995, Brier = 0.1389, Log Loss = 0.4260
- **Assessment**: Highly reliable calibrated model that substantially outperforms baseline on held-out chronological test matches.

---

## 5. Current Gaps & Limitations

1. **AI Assistant Integration Missing**:
   - No conversational cricket assistant currently exists in the application.
   - No LLM provider integration, prompt templates, or deterministic SQL/analytics grounding layer.
   - Missing "Explain This Match" button on Match Explorer.
   - Missing "Explain This Prediction" button on Prediction Lab.
   - Missing AI Team and Player Performance Reports.
2. **Page & Navigation Expansion**:
   - Need a dedicated 10th page: **AI Intelligence Center** that unifies the AI Cricket Analyst, grounded report generation, and configuration instructions.
   - Missing interactive Radar Chart / multi-metric player comparison in Player Analytics.
   - Overview page lacks an AI executive summary card.
3. **Security & Configuration**:
   - `.env` and Streamlit secrets are not explicitly ignored in `.gitignore`.
   - Need `.env.example` with clear instructions for configuring OpenAI or OpenAI-compatible providers without hardcoding keys.
   - Need strict SQL injection safeguards, query restrictions (read-only, whitelist tables, parameterization).
4. **Branding & Visual Polish**:
   - Official verified franchise colors can be integrated into the team branding configuration to enhance badges and chart themes while respecting copyright and noting asset attribution in `docs/ASSET_SOURCES.md`.

---

## 6. Planned Implementation Phases

### Phase 1: Audit and Baseline (Completed)
- Document current state, models, and run baseline pytest suite.
- Create `docs/IMPLEMENTATION_PLAN.md`.

### Phase 2: Design System & Visual Architecture
- Refine `.stApp` and CSS styles: typography, card borders, subtle shadows, consistent padding, last updated indicator, and accessible high-contrast Dark/Light palette.
- Enhance KPI cards and empty states across views.
- Ensure responsive layouts and clean navigation.

### Phase 3: Analytics Page Enhancements
- **Overview**: Add AI Executive Summary snippet, season records, and win-pattern trends.
- **Team Analytics**: Add season rankings, head-to-head records, chase vs defend breakdown, venue performance.
- **Player Analytics**: Add multi-metric radar comparison, role benchmarks, top percentile highlights.
- **Match Explorer**: Add "Explain This Match" with automated data retrieval and turning point extraction.
- **Venue & Toss**: Add sample-size reliability indicators and scatter charts for toss advantage.
- **Ball by Ball**: Add phase-wise run distributions (Powerplay, Middle, Death) and cumulative run-rate progression.
- **Prediction Lab**: Add "Explain This Prediction" breaking down local feature contributions into plain-English reasoning.
- **Model Performance**: Dual reporting for Pre-match (honest baseline failure) and Live-chase (calibrated success).
- **Data Quality**: Enhanced anomaly inspection and automated remediation guidance.

### Phase 4: AI Cricket Analyst & Security Engine
- Build `src/ai/`:
  - `src/ai/config.py`: Safe environment & secrets loading, provider settings, cost limits.
  - `src/ai/provider.py`: Lightweight OpenAI-compatible REST client using Python `requests` with zero extra dependencies; includes Mock Provider for local/offline testing.
  - `src/ai/query_engine.py`: Strictly read-only, parameterized query builder with whitelist table enforcement and SQL mutation rejection.
  - `src/ai/service.py`: Context grounding orchestrator ensuring numbers come from deterministic execution before LLM interpretation.
  - `src/ai/explainers.py`: Domain-specific generators for match summaries, prediction explanations, and player/team scouting reports.
- Create Page 10: **AI Intelligence Center** in `app.py` with chat interface, quick query pills, session history, clear chat, and API setup instructions.

### Phase 5: Team Branding & Eligibility
- Update `config/team_branding.json` with verified franchise colors (e.g., CSK #F9CD05, MI #004BA0, KKR #3A225D, RCB #DA1818, etc.) while documenting citations in `docs/ASSET_SOURCES.md`.
- Ensure badge fallbacks remain clean and consistent.

### Phase 6: Testing & Quality Assurance
- Add unit tests in `tests/test_ai_assistant.py` covering:
  - Missing API key behavior & graceful fallback.
  - Mock provider integration.
  - SQL injection attack prevention.
  - Database mutation rejection (`DROP`, `DELETE`, `UPDATE`, `INSERT`).
  - Strict table whitelisting and query timeouts.
  - Evidence retrieval and grounding verification.
- Add tests in `tests/test_advanced_analytics.py` for new analytical and explainer components.
- Run full pytest suite and verify 100% pass rate.

### Phase 7: Documentation & Deployment Readiness
- Create/update:
  - `README.md`: Flagship project presentation, architecture, quick start, deployment guide.
  - `docs/ARCHITECTURE.md`: Complete system architecture and data flow diagrams.
  - `docs/AI_ASSISTANT.md`: AI engine architecture, security safeguards, query whitelist.
  - `docs/ML_EVALUATION.md`: In-depth analysis of pre-match vs live-chase models.
  - `docs/DATA_DICTIONARY.md`: Full schema and column definitions.
  - `docs/ASSET_SOURCES.md`: License, fair-use, and source attribution documentation.
  - `.env.example`: Secure environment configuration template.
  - Update `.gitignore` to prevent credential leakage.
