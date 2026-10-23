# AI Cricket Analyst & Intelligence Engine

## 1. Overview & Core Philosophy

The **AI Cricket Analyst** embedded in IPL INSIGHT is engineered around a core tenet: **Deterministic Grounding**. 
Large language models (LLMs) are notorious for hallucinating statistics, fabricating match results, and making arithmetic errors. In this platform, **the LLM is never trusted to perform raw arithmetic, count matches, or invent cricket facts**.

Instead, the system follows a deterministic retrieval-augmented pattern:
1. **Intent Recognition**: Maps user queries to approved analytical intents (head-to-head records, top batters/bowlers, venue chasing dynamics, match turning points, model performance, or data quality).
2. **Deterministic SQL/Pandas Execution**: Retrieves and computes exact metrics from the verified SQLite database (`data/processed/ipl_analytics.sqlite`) using parameterized read-only queries.
3. **Grounding Context Packaging**: Packages calculated numbers and sample sizes into a factual evidence packet.
4. **Synthesized Explanation**: When an AI provider is configured, the LLM receives the evidence and is instructed to generate an executive debrief without modifying any numbers. When no provider is configured, the system gracefully falls back to structured markdown tables.

---

## 2. Architecture & File Structure

```
src/ai/
├── __init__.py           # Unified module interface
├── config.py             # Environment & secrets manager (API keys, timeouts, tokens)
├── provider.py           # Provider implementations (OpenAI REST client & Mock Provider)
├── query_engine.py       # Secure, read-only parameterized SQLite query engine
├── service.py            # AICricketAnalyst orchestrator & intent router
└── explainers.py         # Match explainers, prediction explainers, & player dossiers
```

---

## 3. Security Safeguards

### 3.1 Strict Read-Only Database Mode
- The SQLite connection is initialized using URI mode: `file:{database_path}?mode=ro`.
- The operating system and SQLite driver reject any write or disk mutation attempt.

### 3.2 SQL Mutation Blocker
- All incoming queries pass through regex validation before execution:
  ```python
  FORBIDDEN_SQL_PATTERNS = re.compile(
      r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|EXEC|VACUUM|GRANT|REVOKE)\b",
      re.IGNORECASE,
  )
  ```
- Any query containing prohibited keywords immediately raises `SecurityViolationError`.
- Multiple statements chained via semicolons (`;`) are strictly blocked to prevent stacked SQL injection attacks.

### 3.3 Whitelisted Tables & Max Limits
- Queries are restricted to the three audited tables:
  - `ipl_matches_summary`
  - `ipl_player_profiles`
  - `ipl_ball_by_ball_processed`
- All queries enforce a hard limit of `max_rows=100` to prevent memory exhaustion and runaway payload sizes.

### 3.4 Zero Secret Leakage
- API keys are never written to logs, console output, exception tracebacks, or browser-visible markup.
- `config.masked_key` formats keys as `sk-...1234` for safe administrative confirmation.
- `.env`, `.env.*`, and `.streamlit/secrets.toml` are excluded in `.gitignore`.

---

## 4. Grounded Features

### 4.1 Conversational Chat (`src/ai_views.py`)
- Accessible via the **AI Intelligence Center** tab.
- Supports interactive multi-turn dialogue with prompt suggestion pills.
- Includes an **"Audit Trail"** expander under every response showing the exact structured JSON retrieved from SQLite.

### 4.2 "Explain This Match" (`src/match_views.py`)
- Synthesizes match facts on demand:
  - Match winner, victory margin, toss decision, and Player of the Match.
  - Top 3 batting contributions (runs, balls, 4s, 6s).
  - Top 3 bowling spells (wickets, runs conceded, legal balls).
  - Turning point overs: highlights overs with **15+ runs** or **2+ dismissals**.

### 4.3 "Explain This Prediction" (`src/prediction_views.py`)
- Translates the live chase model's statistical output into actionable coaching insights:
  - Explains the pressure created by the required run rate vs current run rate.
  - Highlights the signed log-odds contribution of top model factors.
  - Communicates uncertainty, sample bounds, and reminds the user that empirical models express likelihood, not certainty.

### 4.4 Automated Player Scouting Dossier (`src/ai_views.py` & `src/dashboard_views.py`)
- Generates a comprehensive scouting dossier:
  - Career volume and productivity (runs, average, strike rate).
  - Boundary efficiency (% runs in boundaries).
  - Phase breakdown (Powerplay SR vs Death Overs SR).
  - Tactical matchups (Strike Rate vs Pace vs Spin).
  - Bowling efficiency (economy, average, strike rate).
  - Downloadable as a formatted text dossier.

---

## 5. Configuration & Offline Capability

The system operates seamlessly in two modes:

### Mode A: Full LLM Intelligence (Active)
Configure any OpenAI-compatible provider:
```bash
# In .env file:
AI_PROVIDER=openai
AI_API_KEY=sk-proj-your-api-key
AI_MODEL=gpt-4o-mini
AI_BASE_URL=https://api.openai.com/v1
```
Compatible with OpenAI (`gpt-4o-mini`), Groq (`llama-3.1-70b-versatile`), OpenRouter, or local Ollama instances (`http://localhost:11434/v1`).

### Mode B: Deterministic Fallback (Zero Setup / $0 Cost)
- If no API key is provided, the platform automatically switches to **Deterministic Grounded Mode**.
- Rather than throwing an error or showing a broken chat, it queries the database and formats publication-ready markdown tables directly.
- All core analytical dashboards, model performance metrics, and prediction calculations remain 100% functional.
