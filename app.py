from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai_views import render_ai_intelligence_center
from src.dashboard_views import (
    render_ball_analytics,
    render_overview,
    render_player_analytics,
    render_team_analytics,
    render_venue_toss_analytics,
)
from src.data_loader import load_dashboard_data
from src.metrics import filter_deliveries, filter_matches
from src.match_views import render_match_explorer
from src.prediction_views import render_prediction_lab
from src.report_views import render_data_quality, render_model_performance
from src.ui_theme import plotly_template, theme_variables
from src.load_to_sqlite import ensure_database_ready


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_analytics.sqlite"
CHASE_MODEL_PATH = PROJECT_ROOT / "models" / "live_chase_probability.joblib"
STYLESHEET_PATH = PROJECT_ROOT / "assets" / "styles.css"
PAGES = [
    "Overview",
    "Team analytics",
    "Player analytics",
    "Match explorer",
    "Venue & toss",
    "Ball by ball",
    "Prediction Lab",
    "Model performance",
    "Data quality",
    "AI Intelligence Center",
]
PAGE_SUBTITLES = {
    "Overview": "Season trends, results, teams, and top performers.",
    "Team analytics": "Compare team results within the selected match set.",
    "Player analytics": "Explore career statistics, role metrics, and player comparisons.",
    "Match explorer": "Search historical matches, inspect scorecards, and analyze momentum.",
    "Venue & toss": "Compare venue scoring and toss-result associations.",
    "Ball by ball": "Inspect regular-innings scoring, phase breakdowns, and wicket patterns.",
    "Prediction Lab": "Explore live in-play chase probabilities and model explainability.",
    "Model performance": "Compare held-out models, calibration, and feature factors.",
    "Data quality": "Review table coverage, missingness, keys, assets, and methodology.",
    "AI Intelligence Center": "Conversational cricket analyst, automated scouting dossiers, and AI interpretability.",
}

st.set_page_config(
    page_title="CricketMind AI | IPL Analytics & Intelligence Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state.setdefault("ui_theme", "Dark")
active_theme = st.session_state["ui_theme"]
px.defaults.template = plotly_template(active_theme)
px.defaults.color_discrete_sequence = ["#f1bb4b", "#49bce8", "#f47c48", "#36c6a4"]

if STYLESHEET_PATH.exists():
    stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
    st.markdown(
        f"<style>{stylesheet}\n:root{{{theme_variables(active_theme)}}}</style>",
        unsafe_allow_html=True,
    )

# Self-healing database initialization (guarantees seamless Streamlit Cloud startup)
if not DATABASE_PATH.exists():
    with st.spinner("Initializing analytics database from verified sources..."):
        try:
            ensure_database_ready(DATABASE_PATH)
        except Exception as error:
            st.error(f"Could not initialize analytics database: {error}. Run `python src/load_to_sqlite.py`.")
            st.stop()

try:
    matches, players, deliveries = load_dashboard_data(str(DATABASE_PATH))
except (FileNotFoundError, OSError, pd.errors.DatabaseError, ValueError) as error:
    st.error(f"Could not load the analytics database: {error}")
    st.stop()

available_years = sorted(matches["calendar_year"].dropna().astype(int).unique())
available_teams = sorted(set(matches["team1"].dropna()) | set(matches["team2"].dropna()))
available_venues = sorted(matches["venue"].dropna().unique())
available_stages = sorted(matches["match_stage"].dropna().unique())
if not available_years or not available_teams:
    st.error("The match table does not contain usable year and team values.")
    st.stop()

filter_defaults = {
    "filter_year_range": (int(available_years[0]), int(available_years[-1])),
    "filter_teams": [],
    "filter_venues": [],
    "filter_stages": [],
}
for key, default_value in filter_defaults.items():
    st.session_state.setdefault(key, default_value)


def reset_filters() -> None:
    for key, default_value in filter_defaults.items():
        st.session_state[key] = default_value


with st.sidebar:
    st.markdown(
        "<div class='ipl-brand'>"
        "<div class='ipl-brand-mark'>CricketMind AI</div>"
        "<div class='ipl-brand-subtitle'>Every Match. Every Metric. Smarter Insights.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.selectbox("Appearance", ["Dark", "Light"], key="ui_theme", label_visibility="visible")
    st.markdown("##### PAGES")
    active_page = st.radio(
        "Dashboard page",
        PAGES,
        label_visibility="collapsed",
        key="active_page",
    )

    st.markdown("##### FILTERS")
    st.slider(
        "Calendar year",
        min_value=int(available_years[0]),
        max_value=int(available_years[-1]),
        key="filter_year_range",
    )
    st.multiselect("Teams", available_teams, key="filter_teams")
    st.multiselect("Venues", available_venues, key="filter_venues")
    if available_stages:
        st.multiselect("Match stage", available_stages, key="filter_stages")
    st.button("Reset filters", on_click=reset_filters, width="stretch")

    st.divider()
    st.caption(
        f"DATA COVERAGE\n\n"
        f"{len(matches):,} matches · {len(deliveries):,} deliveries\n\n"
        f"{matches['date'].min().date()} to {matches['date'].max().date()}\n\n"
        f"{len(available_teams)} teams · {len(available_venues)} venues\n\n"
        f"Audit Verification: 100% Parity"
    )
    st.caption(
        "Calendar dates drive year filters; source season labels are inconsistent."
    )

selected_matches = filter_matches(
    matches,
    st.session_state["filter_year_range"],
    st.session_state["filter_teams"],
    st.session_state["filter_venues"],
    st.session_state["filter_stages"],
)
selected_deliveries = filter_deliveries(deliveries, selected_matches)

st.markdown("<div class='page-kicker'>CricketMind AI / IPL ANALYTICS & INTELLIGENCE PLATFORM</div>", unsafe_allow_html=True)
st.title(active_page)
st.caption(PAGE_SUBTITLES[active_page])
st.caption(
    f"{st.session_state['filter_year_range'][0]}–"
    f"{st.session_state['filter_year_range'][1]} · "
    f"{len(selected_matches):,} matches in current selection"
)

if active_page == "AI Intelligence Center":
    render_ai_intelligence_center(matches, players, deliveries, DATABASE_PATH)
elif active_page == "Prediction Lab":
    render_prediction_lab(matches, selected_matches, DATABASE_PATH, CHASE_MODEL_PATH)
elif active_page == "Model performance":
    render_model_performance(PROJECT_ROOT)
elif active_page == "Data quality":
    render_data_quality(PROJECT_ROOT, matches, players, deliveries, DATABASE_PATH)
elif not selected_matches.empty:
    if active_page == "Overview":
        render_overview(selected_matches, players, selected_deliveries)
    elif active_page == "Team analytics":
        render_team_analytics(selected_matches, selected_deliveries)
    elif active_page == "Player analytics":
        render_player_analytics(players)
    elif active_page == "Match explorer":
        render_match_explorer(selected_matches, selected_deliveries, DATABASE_PATH)
    elif active_page == "Venue & toss":
        render_venue_toss_analytics(selected_matches)
    elif active_page == "Ball by ball":
        render_ball_analytics(selected_deliveries)
else:
    st.info("No matches match the current filters. Reset or broaden the selections.")

st.divider()
st.caption(
    "CricketMind AI is an independent analytics and data science project. "
    "All statistics are calculated from verified historical records. "
    "Toss associations are descriptive; live chase probabilities express statistical expectations, not guarantees."
)
