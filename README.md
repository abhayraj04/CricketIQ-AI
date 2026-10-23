# CricketMind AI — IPL Analytics, Machine Learning & Intelligence Platform

> **Tagline:** Every Match. Every Metric. Smarter Insights.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Pytest-57%20Passed-brightgreen.svg)](#-testing--automated-verification)
[![AI Engine](https://img.shields.io/badge/AI_Analyst-Grounded_LLM_%2B_SQL-purple.svg)](#-grounded-ai-cricket-analyst)
[![Data Parity](https://img.shields.io/badge/Parity-100%25_Verified-success.svg)](#-data-architecture--auditing)

---

## 📌 Executive Overview

**CricketMind AI** is an enterprise-grade sports analytics, machine learning, and artificial intelligence platform spanning every Indian Premier League season from **2008 through 2026**.

Engineered as a flagship data science portfolio project, CricketMind AI unites **mathematically reconciled historical data** (1,243 matches, 295,732 deliveries, 811 player profiles), an **out-of-time calibrated machine learning inference engine**, and a **deterministic retrieval-grounded conversational AI Cricket Analyst**.

Every metric is calculated dynamically from the underlying database, models enforce chronological temporal splits to prevent data leakage, and AI responses are bounded by parameterized, read-only SQL queries with zero mutation risks.

---

## 📸 Platform Interface & Screenshots

*(Capture and insert screenshots from your local or deployed application)*

| Executive Overview & AI Briefing | In-Play Chase Probability Lab |
|:---:|:---:|
| *(Add `docs/screenshots/overview.png`)* | *(Add `docs/screenshots/prediction_lab.png`)* |

| Conversational Cricket Analyst | Multi-Attribute Player Radar |
|:---:|:---:|
| *(Add `docs/screenshots/ai_analyst.png`)* | *(Add `docs/screenshots/player_radar.png`)* |

---

## 🚀 Ten Core Application Modules

1. **Executive Overview**: High-level tournament metrics, annual run and dismissal trajectories, toss influence trends, and an automated data-grounded AI executive briefing.
2. **Team Analytics**: Season-by-season win percentages, chase vs defend breakdowns, toss decision correlations, head-to-head records, and verified franchise era identity mappings.
3. **Player Analytics**: Searchable career directory, batting/bowling efficiency rates, multi-attribute player radar comparison (Strike Rate, Average, Boundary %, Scoring Volume, Dot-Ball Control), and on-demand AI scouting dossiers.
4. **Match Explorer**: Granular historical match search, ball-by-ball scorecards, over-by-over momentum charts, batter/bowler figures, and the **"Explain This Match"** AI tactical debrief.
5. **Venue & Toss Analysis**: Venue-specific run rates, chasing success ratios, toss advantage scatter distributions, and sample-size reliability gates (< 10 match warning).
6. **Ball-by-Ball Analysis**: Over-by-over run rate progressions, phase scoring breakdown (Powerplay, Middle, Death), and dismissal distribution patterns.
7. **Prediction Lab**: Live second-innings chase probability powered by an out-of-time calibrated classifier with signed log-odds factor impact and the **"Explain This Prediction"** interpretability debrief.
8. **Model Performance**: Held-out test metrics, ROC-AUC, Brier score, Platt calibration curves, confusion matrix, and chronological split auditing.
9. **Data Quality & Methodology**: Automated reconciliation audit (381,529 batter runs + 20,209 extras = 401,738 total runs), duplicate checks, and season label integrity validation.
10. **AI Intelligence Center**: Conversational cricket assistant with prompt suggestion pills, grounded audit trails, custom franchise/player dossier generator, and secure provider configuration.

---

## 🤖 Grounded AI Cricket Analyst

- **Deterministic Grounding**: The LLM is never trusted to perform raw arithmetic, count matches, or invent cricket facts. Queries trigger parameterized read-only SQLite executions first; the LLM merely translates factual numbers into structured prose.
- **Audit Trails**: Every AI response includes an expandable "Retrieved Evidence" drawer displaying the exact JSON payload extracted from SQLite.
- **Zero-Mutation Security**: Queries run via SQLite URI read-only mode (`file:...?mode=ro`). Prohibited mutation keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`) are blocked by regex before execution.
- **Zero Setup / $0 Cost Fallback**: Operates fully in **Deterministic Grounded Mode** when no API key is provided, formatting structured markdown tables directly from SQLite without raising exceptions.

---

## 🎯 Machine Learning & The Safety Gate

### 1. Live Chase Model (`models/live_chase_probability.joblib`)
- **Task**: Predict probability of a successful run chase after any completed second-innings delivery.
- **Leakage Prevention**: All delivery states from the same match remain in one chronological partition. Train (2008–2021), Calibration (2022–2023), Held-Out Test (2024–2026).
- **Held-Out Test Results**: **ROC-AUC: 0.8995**, **Accuracy: 79.46%**, **Brier Score: 0.1389** (beats baseline of 0.2467).

### 2. Pre-Match Model & The Safety Gate
- Experimental pre-match logistic regression achieved held-out ROC-AUC of **0.440** (failing to beat the 0.500 baseline).
- **The Safety Gate**: Rather than deploying an inaccurate model or using an LLM to hallucinate pre-match odds, future match predictions are **strictly gated and disabled** with transparent reasoning.

---

## 📂 Architecture & Technical Documentation

For deep technical dives, consult the dedicated documentation in [`docs/`](docs/):

- [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) — Comprehensive technical audit, risk analysis, and resolutions.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Multi-tier system architecture and data flows.
- [`docs/AI_ASSISTANT.md`](docs/AI_ASSISTANT.md) — Conversational agent grounding, query engine, and security.
- [`docs/ML_EVALUATION.md`](docs/ML_EVALUATION.md) — In-depth machine learning audit and baseline comparisons.
- [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — Schema definitions, grain specifications, and constraints.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Step-by-step Streamlit Community Cloud deployment guide.
- [`docs/ASSET_SOURCES.md`](docs/ASSET_SOURCES.md) — License, fair use, and BCCI trademark attribution.

---

## 🛠️ Quick Start & Local Execution

### 1. Clone & Setup
```powershell
# Clone the repository
git clone https://github.com/<your-username>/cricketmind-ai.git
cd cricketmind-ai

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔑 AI Provider Configuration (Optional)

The application is fully functional in **Deterministic Grounded Mode** without an API key.

To enable conversational LLM prose:
1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Edit `.env` with your API credentials:
   ```env
   AI_PROVIDER=openai
   AI_API_KEY=sk-proj-your-api-key
   AI_MODEL=gpt-4o-mini
   AI_BASE_URL=https://api.openai.com/v1
   ```
3. Alternatively, enter a temporary session key in the **AI Intelligence Center** tab in the UI.

---

## 🧪 Testing & Automated Verification

Run the full automated test suite (57 unit, integration, and UI tests):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Verified test coverage:
- **AI Engine**: SQL injection prevention, mutation blocking (`DROP`, `DELETE`, `UPDATE`), query whitelisting, timeout handling, and deterministic fallback.
- **Analytics**: Head-to-head calculations, innings scorecards, over aggregation, and rate derivations.
- **Machine Learning**: Temporal split integrity, feature timing validation, and calibrated probability normalization.
- **UI & Navigation**: Headless Streamlit AppTest running all 10 pages in both Dark and Light appearance modes.

---

## 🚀 Streamlit Community Cloud Deployment

1. Initialize git inside the project directory:
   ```powershell
   git init
   git add .
   git commit -m "feat: release CricketMind AI production platform"
   git branch -M main
   git remote add origin https://github.com/<your-username>/cricketmind-ai.git
   git push -u origin main
   ```
2. Open [share.streamlit.io](https://share.streamlit.io) and deploy `app.py`.
3. *(Optional)* Add `AI_API_KEY` to Streamlit Cloud Secrets.
4. Detailed troubleshooting and memory guides are in [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## 📜 Intellectual Property & Fair Use Disclaimer

CricketMind AI is an independent academic, analytical, and data science research project. It is not affiliated with, endorsed by, or sponsored by the Board of Control for Cricket in India (BCCI) or the Indian Premier League (IPL). All cricket team names and player identities are used strictly under descriptive fair-use principles for statistical analysis. Copyrighted logos and portraits are not bundled; accessible short-name badges and player initials are generated dynamically.
