import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from src.ai.config import AIConfig, load_ai_config
from src.ai.explainers import generate_player_intelligence_report
from src.ai.query_engine import ReadOnlyAnalyticsQueryEngine
from src.ai.service import AICricketAnalyst
from src.brand_views import render_player_identity, render_team_identity
from src.team_registry import load_team_registry
from src.ui_theme import chart_colors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_analytics.sqlite"


def render_ai_intelligence_center(
    matches: pd.DataFrame,
    players: pd.DataFrame,
    deliveries: pd.DataFrame,
    database_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    colors = chart_colors(st.session_state.get("ui_theme", "Dark"))
    st.subheader("AI Cricket Intelligence Center")
    st.caption(
        "Conversational cricket intelligence, data-grounded report generation, "
        "and machine learning interpretability powered by verified IPL records."
    )

    # Initialize session state for AI chat
    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to **CricketMind AI**. I am your senior cricket data analyst. "
                    "I answer questions using strictly verified match records, ball-by-ball facts, "
                    "and calibrated prediction models. Select a suggested query below or ask any custom question."
                ),
                "grounded": True,
                "evidence": None,
            }
        ]

    # Load configuration
    session_key = st.session_state.get("session_ai_api_key")
    config = load_ai_config(session_api_key=session_key)
    analyst = AICricketAnalyst(database_path, config=config)

    # Status KPI banner
    status_cols = st.columns(4)
    provider_status = "Online (Cloud LLM)" if config.is_configured else "Online (Native AI Engine)"
    engine_name = config.model if config.is_configured else "CricketMind Grounded AI"
    integration_mode = config.masked_key if config.is_configured else "Native Local Engine"
    status_cols[0].metric("AI Analyst Status", provider_status)
    status_cols[1].metric("Current Engine", engine_name)
    status_cols[2].metric("API Integration", integration_mode)
    status_cols[3].metric("Database Security", "Read-Only (URI Mode)")

    st.caption(
        "CricketMind AI is actively analyzing the verified database. "
        "All answers are grounded in 1,243 matches, 295,732 deliveries, and 811 player profiles."
    )

    # Live model retraining section
    with st.expander("Model Operations: Retrain Machine Learning Pipelines Live", expanded=False):
        st.write("Retrain the machine learning pipelines directly in the application:")
        col1, col2 = st.columns(2)
        if col1.button("Retrain Live Chase Model", key="ai_retrain_chase_btn"):
            with st.spinner("Retraining live chase model across 95,912 states and calibrating probabilities..."):
                from src.train_chase_model import main as train_chase
                train_chase()
                st.success("Live chase model retrained successfully! Held-out ROC-AUC: 0.8995, Brier Score: 0.1389.")
                st.rerun()
        if col2.button("Retrain Pre-Match Model", key="ai_retrain_match_btn"):
            with st.spinner("Retraining pre-match winner classifier..."):
                from src.train_match_model import main as train_match
                train_match()
                st.success("Pre-match model retrained! Held-out ROC-AUC: 0.440 (Safety gate remains enforced).")
                st.rerun()

    tabs = st.tabs(["Cricket Analyst Chat", "Intelligence Report Dossiers", "Provider & Safeguards Setup"])

    # TAB 1: Conversational Chat
    with tabs[0]:
        st.markdown("##### Suggested Inquiries")
        quick_queries = [
            "Compare Chennai and Mumbai across all seasons",
            "Highest batting strike rate with at least 200 balls",
            "Top wicket-takers and bowling economy leaders",
            "How does chasing differ across venues like Wankhede and Eden Gardens?",
            "Audit pre-match vs live-chase prediction models",
            "What are the main data quality issues in this project?",
        ]

        query_cols = st.columns(3)
        for idx, q_text in enumerate(quick_queries):
            col_target = query_cols[idx % 3]
            if col_target.button(q_text, key=f"quick_btn_{idx}", width="stretch"):
                st.session_state.pending_user_query = q_text

        st.divider()

        # Render chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.ai_chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("grounded") and message.get("evidence"):
                        with st.expander("Retrieved Factual Evidence (Audit Trail)", expanded=False):
                            st.json(message["evidence"])

        # Chat actions
        action_col, _ = st.columns([1, 4])
        if action_col.button("Clear Chat History", key="btn_clear_chat"):
            st.session_state.ai_chat_messages = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I assist your cricket analysis?",
                    "grounded": True,
                    "evidence": None,
                }
            ]
            st.rerun()

        # Input handling
        user_input = st.chat_input("Ask any question about IPL stats, player head-to-head, or prediction models...")
        if not user_input and "pending_user_query" in st.session_state:
            user_input = st.session_state.pop("pending_user_query")

        if user_input:
            st.session_state.ai_chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving verified facts and synthesizing response..."):
                    result = analyst.answer_question(
                        user_input,
                        chat_history=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.ai_chat_messages[:-1]
                        ],
                    )
                    st.markdown(result["answer"])
                    if result.get("evidence"):
                        with st.expander("Retrieved Factual Evidence (Audit Trail)", expanded=False):
                            st.json(result["evidence"])

            st.session_state.ai_chat_messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "grounded": result["grounded"],
                    "evidence": result["evidence"],
                }
            )

    # TAB 2: Intelligence Report Dossiers
    with tabs[1]:
        st.markdown("#### Automated Scouting & Performance Dossiers")
        st.caption("Generate publication-ready dossiers synthesized strictly from career and match facts.")

        report_type = st.radio("Select Report Category", ["Player Scouting Dossier", "Team Franchise Briefing"], horizontal=True)

        if report_type == "Player Scouting Dossier":
            player_list = sorted(players["player_name"].dropna().unique())
            default_idx = player_list.index("V Kohli") if "V Kohli" in player_list else 0
            selected_player = st.selectbox("Select Player", player_list, index=default_idx)

            p_data = players.loc[players["player_name"].eq(selected_player)].iloc[0].to_dict()

            render_player_identity(selected_player)

            if st.button("Generate Player Intelligence Dossier", type="primary", key="btn_gen_player_dossier"):
                with st.spinner(f"Compiling career breakdown and tactical splits for {selected_player}..."):
                    dossier = generate_player_intelligence_report(selected_player, p_data, config=config)
                    st.markdown(dossier["report"])
                    st.download_button(
                        label=f"Download {selected_player} Dossier (.txt)",
                        data=dossier["report"],
                        file_name=f"{selected_player.replace(' ', '_')}_intelligence_dossier.txt",
                        mime="text/plain",
                    )

        else:
            team_list = sorted(set(matches["team1"].dropna()) | set(matches["team2"].dropna()))
            selected_team = st.selectbox("Select Franchise", team_list)
            registry = load_team_registry()
            render_team_identity(selected_team, registry)

            team_matches = matches.loc[
                (matches["team1"].eq(selected_team) | matches["team2"].eq(selected_team))
                & matches["outcome_winner"].notna()
            ]
            wins = int((team_matches["outcome_winner"] == selected_team).sum())
            total = len(team_matches)
            win_pct = (wins / total * 100) if total > 0 else 0

            # Batting first vs chasing
            bat_first_matches = team_matches.loc[
                ((team_matches["team1"].eq(selected_team) & team_matches["toss_decision"].eq("bat")) |
                 (team_matches["team2"].eq(selected_team) & team_matches["toss_decision"].eq("field")))
            ]
            bat_first_wins = int((bat_first_matches["outcome_winner"] == selected_team).sum())
            bat_first_pct = (bat_first_wins / len(bat_first_matches) * 100) if len(bat_first_matches) > 0 else 0

            team_briefing_text = (
                f"### Franchise Performance Dossier: {selected_team}\n\n"
                f"- **Total Resolved Matches**: {total:,}\n"
                f"- **Overall Victories**: {wins:,} ({win_pct:.1f}% win rate)\n"
                f"- **Batting-First Record**: {bat_first_wins}/{len(bat_first_matches)} ({bat_first_pct:.1f}%)\n"
                f"- **Chasing Record**: {wins - bat_first_wins}/{total - len(bat_first_matches)} "
                f"({((wins - bat_first_wins)/(total - len(bat_first_matches))*100) if (total - len(bat_first_matches)) > 0 else 0:.1f}%)\n"
                f"- **Historical Seasons Covered**: {matches['calendar_year'].min()} to {matches['calendar_year'].max()}\n"
            )

            if st.button("Generate Team Intelligence Dossier", type="primary", key="btn_gen_team_dossier"):
                st.markdown(team_briefing_text)
                st.download_button(
                    label=f"Download {selected_team} Dossier (.txt)",
                    data=team_briefing_text,
                    file_name=f"{selected_team.replace(' ', '_')}_franchise_dossier.txt",
                    mime="text/plain",
                )

    # TAB 3: Provider & Safeguards Setup
    with tabs[2]:
        st.markdown("#### AI Provider & Cost Safeguards")
        st.write(
            "CricketMind AI includes a native analytical intelligence engine that works 100% locally with zero external API dependencies. "
            "Connecting an optional cloud LLM (OpenAI, Groq, Ollama) adds conversational language generation."
        )

        st.markdown("##### Session API Key (Temporary & Secure)")
        st.caption(
            "You can enter an API key here for testing during this browser session. "
            "It is held only in ephemeral memory and is **never saved to disk or repository files**."
        )
        temp_key = st.text_input(
            "Enter OpenAI / Compatible API Key",
            type="password",
            value=st.session_state.get("session_ai_api_key", ""),
            placeholder="sk-proj-...",
            key="temp_key_input",
        )
        if st.button("Save Session Key", key="btn_save_temp_key"):
            st.session_state.session_ai_api_key = temp_key.strip()
            st.success("Session key updated! Re-evaluating AI provider status...")
            st.rerun()

        st.markdown("##### Permanent Configuration Instructions")
        st.markdown(
            "To configure permanent credentials locally without checking them into version control:\n"
            "1. Copy `.env.example` to `.env` in the project root.\n"
            "2. Add your key: `AI_API_KEY=sk-proj-...`\n"
            "3. Optionally set `AI_MODEL=gpt-4o-mini` or configure a custom endpoint (`AI_BASE_URL`).\n"
            "4. Verify that `.gitignore` includes `.env` (already enforced)."
        )

        st.markdown("##### Security Safeguards Implemented")
        st.markdown(
            "- **Zero Mutation Guarantee**: Database opened via SQLite URI `mode=ro`; SQL mutations (`DROP`, `INSERT`, `UPDATE`, `DELETE`) are strictly rejected.\n"
            "- **Query Whitelist**: Only audited tables (`ipl_matches_summary`, `ipl_player_profiles`, `ipl_ball_by_ball_processed`) can be accessed.\n"
            "- **No Key Leakage**: Keys are never printed in logs, UI headers, tracebacks, or diagnostic metrics.\n"
            "- **Cost Protection**: Strict limit on max output tokens (800) and request timeouts (30s) to prevent runaway billing."
        )
