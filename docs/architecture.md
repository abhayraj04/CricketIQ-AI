# System Architecture — IPL INSIGHT Platform

## 1. High-Level Architecture Overview

`IPL INSIGHT` is structured as a decoupled, multi-tier sports analytics, machine learning, and AI platform. It emphasizes factual data grounding, reproducibility, zero-leakage ML pipelines, and strict read-only query security.

```
+-------------------------------------------------------------------------+
|                              PRESENTATION TIER                          |
|                       Streamlit Modern SaaS Interface                   |
|  - Executive Overview      - Team Analytics         - Player Analytics  |
|  - Match Explorer          - Venue & Toss           - Ball-by-Ball      |
|  - Prediction Lab          - Model Performance      - Data Quality      |
|  - AI Intelligence Center  - Plotly Visualizations  - Dark/Light Theme  |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                               SERVICE TIER                              |
|  +---------------------------+       +-------------------------------+  |
|  |       AI Service Layer    |       |     Analytics Service Layer   |  |
|  | - AICricketAnalyst        |       | - metrics.py                  |  |
|  | - Context Grounding       |       | - dashboard_views.py          |  |
|  | - Match & Pred Explainers |       | - brand_views.py              |  |
|  | - Scouting Dossiers       |       | - data_loader.py (st.cache)   |  |
|  +-------------+-------------+       +---------------+---------------+  |
|                |                                     |                  |
|                v                                     v                  |
|  +---------------------------+       +-------------------------------+  |
|  |   ReadOnly Query Engine   |       |       ML Inference Engine     |  |
|  | - SQL Whitelist & Limits  |       | - Live Chase Calibrated Model |  |
|  | - Mutation Blockers       |       | - Feature Engineering Pipeline|  |
|  | - Parameterized Queries   |       | - Pre-Match Safety Gate       |  |
|  +-------------+-------------+       +---------------+---------------+  |
+----------------|-------------------------------------|------------------+
                 |                                     |
                 v                                     v
+-------------------------------------------------------------------------+
|                                DATA TIER                                |
|  - SQLite Database (`data/processed/ipl_analytics.sqlite` in URI mode)  |
|  - Processed CSV Records (`ipl_matches_summary`, `deliveries`, `player`)|
|  - Calibrated Model Artifact (`models/live_chase_probability.joblib`)   |
|  - Verified Team Registry & Brand Configurations (`config/`)            |
+-------------------------------------------------------------------------+
```

---

## 2. Component Breakdown

### 2.1 Presentation Tier (`app.py`, `src/*_views.py`, `assets/styles.css`)
- **Framework**: Streamlit 1.66+ with responsive grid layout and `@st.cache_data` / `@st.cache_resource`.
- **Styling**: `assets/styles.css` with CSS custom properties defining Dark Mode (`#0b1220`) and Light Mode (`#f3f6fa`), Barlow Condensed display headers, and DM Sans body typography.
- **Visualizations**: Plotly Express & Plotly Graph Objects configured via `src/ui_theme.py` to match the active theme palette automatically.
- **Pages (10 Core Modules)**:
  1. `Overview`: High-level metrics, tournament timeline, run volume, toss influence, AI briefing.
  2. `Team analytics`: Franchise comparisons, head-to-head records, defend vs chase splits.
  3. `Player analytics`: Directory search, batting & bowling rates, multi-attribute radar comparison, AI player scouting dossiers.
  4. `Match explorer`: Match search, scorecards, over-by-over momentum charts, batter/bowler figures, "Explain This Match" debrief.
  5. `Venue & toss`: Bat-first vs chase win rates, sample reliability thresholds, toss bias analysis.
  6. `Ball by ball`: Phase scoring (Powerplay, Middle, Death), dot-ball ratios, wicket patterns.
  7. `Prediction Lab`: Live chase probability inference, local logistic regression contributions, "Explain This Prediction" debrief, and future matchup safety gate.
  8. `Model performance`: Held-out validation tables, ROC-AUC, Brier score, calibration curve, confusion matrix, baseline comparison.
  9. `Data quality`: Missing value audit, duplicate key checks, run reconciliation, calendar date integrity.
  10. `AI Intelligence Center`: Conversational cricket analyst chat, verified audit trails, automated dossiers, safe provider settings.

---

### 2.2 AI & Query Engine (`src/ai/`)
- **`config.py`**: Reads `AI_API_KEY`, `OPENAI_API_KEY`, `AI_MODEL`, and provider options from environment variables, Streamlit secrets, or temporary session state. Never exposes API keys in logs or exceptions.
- **`provider.py`**:
  - `OpenAICompatibleProvider`: Zero-dependency REST client utilizing Python `requests` with strict 30s timeouts, payload capping, and error mapping. Works with OpenAI, Groq, Ollama, OpenRouter, and local LLM endpoints.
  - `MockProvider`: Offline deterministic provider used for unit tests and local testing without incurring API token costs.
- **`query_engine.py`**:
  - Opens SQLite with `file:{path}?mode=ro` (read-only URI mode).
  - Enforces SQL security: blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `ATTACH`, etc.
  - Limits execution to parameterized `SELECT` queries across audited tables only.
- **`service.py`**:
  - Parses intent, queries deterministic metrics, checks grounding, and provides a structured JSON/markdown response.
  - Features an automated **deterministic fallback mode** that synthesizes grounded tables even when no LLM key is configured.
- **`explainers.py`**:
  - `explain_historical_match`: Formats match details, key batting/bowling figures, and turning point overs (15+ runs or 2+ wickets).
  - `explain_prediction`: Breaks down required run rate, balls remaining, wickets in hand, and signed log-odds contributions into plain-English reasoning.
  - `generate_player_intelligence_report`: Generates scouting dossiers with phase breakdowns and pace vs spin splits.

---

### 2.3 Machine Learning Pipeline (`src/chase_*.py`, `src/train_*.py`)
- **Data Hygiene**: Delivery rows are partitioned strictly by calendar date (`date < 2020` for fit, `2020-2023` for calibration, `2024-2026` for held-out testing).
- **Leakage Prevention**: All delivery states from the same match reside in the exact same chronological partition. Post-match outcomes, future balls, final innings scores, and terminal balls are strictly excluded.
- **Calibration**: Sigmoid (Platt-scale) logistic calibration fitted on separate calibration years ensures predicted probabilities match observed empirical win rates.

---

### 2.4 Data Tier (`data/`, `config/`)
- `ipl_analytics.sqlite`: Normalized tables containing 1,243 matches, 295,732 ball deliveries, and 811 player profiles.
- `team_branding.json`: Versioned team registry with short names, verified franchise era aliases, and future season eligibility gates.
- `player_images.json`: Image mapping registry enforcing fallback to neutral initials rather than unlicensed web scraping.
