# Production Deployment Guide: Streamlit Community Cloud

This guide provides step-by-step instructions to deploy **CricketMind AI — IPL Analytics, Machine Learning & Intelligence Platform** to Streamlit Community Cloud with zero configuration friction.

---

## 1. Pre-Deployment Readiness Checklist

Before pushing to GitHub, verify that your local workspace meets these requirements:

- [x] **Git Repository Isolated**: Verified that Git root is `C:/Users/abhay/IPL_MATCH_AND_DASHBOARD` and not your personal user directory.
- [x] **File Size Compliant**: All files committed to Git are under GitHub's 100 MB limit (the 160 MB SQLite and 186 MB CSV files are excluded in `.gitignore`; the 9.1 MB compressed Parquet file is retained).
- [x] **Secrets Excluded**: `.env`, `.env.*`, and `.streamlit/secrets.toml` are in `.gitignore`.
- [x] **Self-Healing Database**: `app.py` automatically initializes `ipl_analytics.sqlite` from `parquet/` in ~3 seconds on first startup if missing.
- [x] **Dependencies Declared**: `requirements.txt` contains all runtime packages (`streamlit`, `pandas`, `plotly`, `scikit-learn`, `requests`, `pyarrow`).
- [x] **Theme Pinned**: `.streamlit/config.toml` enforces the dark theme and headless server settings.
- [x] **Automated Tests Passing**: All 57 tests pass with zero errors.

---

## 2. Step 1: Create a GitHub Repository

1. Open your browser and navigate to [GitHub: New Repository](https://github.com/new).
2. Set the repository name to `cricketmind-ai` (or `ipl-analytics-platform`).
3. Set visibility to **Public** (required for free Streamlit Community Cloud deployment).
4. **Do not** initialize with a README, `.gitignore`, or license (these already exist locally).
5. Click **Create repository**.

---

## 3. Step 2: Push Local Workspace to GitHub

Open PowerShell in `C:\Users\abhay\IPL_MATCH_AND_DASHBOARD` and run:

```powershell
# 1. Verify you are inside the project folder
cd C:\Users\abhay\IPL_MATCH_AND_DASHBOARD

# 2. Stage verified project files (large CSVs and secrets will be ignored automatically)
git add .

# 3. Create initial commit
git commit -m "feat: release CricketMind AI production platform"

# 4. Point to your new GitHub repository (replace with your GitHub username)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/cricketmind-ai.git

# 5. Push to the main branch
git branch -M main
git push -u origin main
```

---

## 4. Step 3: Deploy on Streamlit Community Cloud

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/) using your GitHub account.
2. Click the **"New app"** button in the upper right corner.
3. Select the repository: `<YOUR_GITHUB_USERNAME>/cricketmind-ai`.
4. Set the **Branch** to `main`.
5. Set the **Main file path** to `app.py`.
6. Set the **App URL** to a custom subdomain (e.g. `cricketmind-ai.streamlit.app`).

---

## 5. Step 4: Configure Streamlit Cloud Secrets (Optional)

CricketMind AI works **100% out-of-the-box in Deterministic Grounded Mode** without any API key.

If you wish to enable conversational prose via an OpenAI or compatible provider:
1. In the Streamlit Cloud deployment dialog, click **"Advanced settings"** $\rightarrow$ **"Secrets"**.
2. Paste your configuration:
   ```toml
   AI_PROVIDER = "openai"
   AI_API_KEY = "sk-proj-your-actual-api-key"
   AI_MODEL = "gpt-4o-mini"
   AI_BASE_URL = "https://api.openai.com/v1"
   ```
3. Click **Save** and then click **Deploy!**

---

## 6. Step 5: Verification of Public Deployment

Once the deployment completes (~1 to 2 minutes):
1. **Initial Boot**: The app will display a short spinner: *"Initializing analytics database from verified sources..."* as it builds `ipl_analytics.sqlite` from the 9.1 MB Parquet file.
2. **Page Navigation**: Test navigating between all 10 pages in the sidebar (Executive Overview, Team Analytics, Player Analytics, Match Explorer, Venue & Toss, Ball by Ball, Prediction Lab, Model Performance, Data Quality, AI Intelligence Center).
3. **Theme Toggle**: Verify switching between Dark and Light appearance in the sidebar.
4. **Prediction Lab**: Verify that the live chase probability gauge calculates in-play odds.
5. **AI Intelligence Center**: Test a sample query like *"Compare Chennai and Mumbai across all seasons"*. If secrets were configured, it will return an LLM synthesis; if not, it will return a deterministic markdown breakdown.

---

## 7. Troubleshooting Common Cloud Failures

| Symptom / Log Error | Root Cause | Solution |
|---|---|---|
| `git push rejected: File is larger than 100 MB` | Tried to commit `ipl_analytics.sqlite` or `ipl_ball_by_ball_processed.csv`. | Run `git rm --cached data/processed/ipl_analytics.sqlite`, ensure `.gitignore` contains `data/processed/*.sqlite`, and commit again. |
| `ModuleNotFoundError: No module named 'pyarrow'` | Missing Parquet engine on cloud container. | Ensure `pyarrow>=15.0,<28` is listed in `requirements.txt`. |
| `ModuleNotFoundError: No module named 'requests'` | Missing REST client on cloud container. | Ensure `requests>=2.31,<3` is listed in `requirements.txt`. |
| `FileNotFoundError: ipl_analytics.sqlite` | SQLite database was omitted from Git to save bandwidth. | The app handles this automatically via `ensure_database_ready()`. If customized, run `python src/load_to_sqlite.py`. |
| `StreamlitAPIException: st.session_state has no key` | State accessed before filter initialization. | Fixed in `app.py`: all default filter keys are populated before view rendering. |
| Memory Limit Warning (> 1 GB) | Loading all 295k deliveries repeatedly into memory. | Fixed via Streamlit caching (`@st.cache_data`) in `src/data_loader.py` and lazy-loading ball-by-ball rows per selected match. |
