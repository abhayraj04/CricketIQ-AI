from pathlib import Path
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ai.explainers import explain_prediction
from src.chase_features import TARGET_COLUMN
from src.chase_model import local_feature_contributions
from src.chase_predictor import (
    load_chase_model,
    make_chase_state,
    predict_chase_probability,
)
from src.brand_views import render_team_identity
from src.data_loader import load_chase_examples
from src.future_predictions import future_prediction_status
from src.team_registry import load_team_registry
from src.ui_theme import chart_colors, plotly_template


@st.cache_resource(show_spinner=False)
def _cached_chase_model(model_path: str) -> dict:
    return load_chase_model(model_path)


def _manual_state(
    matches: pd.DataFrame,
    state_key: str = "manual_chase_state",
) -> dict[str, object]:
    teams = sorted(set(matches["team1"].dropna()) | set(matches["team2"].dropna()))
    venues = sorted(matches["venue"].dropna().unique())
    if len(teams) < 2 or not venues:
        raise ValueError("At least two teams and one venue are required for a scenario.")

    first_row = st.columns(3)
    batting_team = first_row[0].selectbox("Batting team", teams, key=f"{state_key}_batting")
    bowling_options = [team for team in teams if team != batting_team]
    bowling_team = first_row[1].selectbox(
        "Bowling team",
        bowling_options,
        key=f"{state_key}_bowling",
    )
    venue = first_row[2].selectbox("Venue", venues, key=f"{state_key}_venue")

    score_column, target_column = st.columns(2)
    target_score = target_column.number_input(
        "Target score",
        min_value=1,
        max_value=350,
        value=175,
        step=1,
        key=f"{state_key}_target",
    )
    current_score = score_column.number_input(
        "Current score",
        min_value=0,
        max_value=349,
        value=min(120, int(target_score) - 1),
        step=1,
        key=f"{state_key}_score",
    )
    wickets_column, balls_column = st.columns(2)
    wickets_lost = wickets_column.slider(
        "Wickets lost",
        min_value=0,
        max_value=9,
        value=4,
        key=f"{state_key}_wickets",
    )
    legal_balls = balls_column.slider(
        "Legal balls bowled",
        min_value=0,
        max_value=119,
        value=60,
        key=f"{state_key}_legal_balls",
    )
    return {
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue,
        "target_score": int(target_score),
        "current_score": int(current_score),
        "wickets_lost": int(wickets_lost),
        "legal_balls_bowled": int(legal_balls),
    }


def _historical_state(
    matches: pd.DataFrame,
    database_path: str,
) -> tuple[dict[str, object] | None, str | None]:
    try:
        examples = load_chase_examples(database_path)
    except (OSError, ValueError, pd.errors.DatabaseError) as error:
        st.error(f"Could not load historical chase states: {error}")
        return None, None

    selected_match_ids = matches["match_id"].astype("string").dropna().unique()
    filtered_examples = examples.loc[
        examples["match_id"].isin(selected_match_ids)
    ].copy()
    if filtered_examples.empty:
        st.info("No eligible historical chase states match the current sidebar filters.")
        return None, None

    match_options = (
        filtered_examples.sort_values("date", ascending=False)
        .drop_duplicates("match_id")
        .reset_index(drop=True)
    )
    match_labels = {
        row["match_id"]: (
            f"{row['date'].date()} | {row['batting_team']} vs {row['bowling_team']} "
            f"| {row['venue']}"
        )
        for _, row in match_options.iterrows()
    }
    selected_match_id = st.selectbox(
        "Historical chase",
        match_options["match_id"].tolist(),
        format_func=lambda match_id: match_labels[match_id],
        key="prediction_historical_match",
    )
    chase_states = (
        filtered_examples.loc[filtered_examples["match_id"].eq(selected_match_id)]
        .sort_values("ball_seq_idx")
        .reset_index(drop=True)
    )
    if chase_states.empty:
        st.info("No eligible state is available for the selected match.")
        return None, None

    def format_state(state_index: int) -> str:
        state = chase_states.iloc[state_index]
        over = int(state.get("ball_seq_idx", state_index + 1))
        return (
            f"After delivery {over}: {int(state['current_score'])}/"
            f"{int(state['wickets_lost'])}, target {int(state['target_score'])}"
        )

    state_index = st.select_slider(
        "Select state after delivery",
        options=list(range(len(chase_states))),
        value=len(chase_states) - 1,
        format_func=format_state,
        key="prediction_historical_state",
    )
    selected_state = chase_states.iloc[state_index]
    st.caption(
        f"Historical state from {selected_state['date'].date()}; "
        "prediction point is after this delivery and before the next one."
    )
    return (
        {
            "batting_team": selected_state["batting_team"],
            "bowling_team": selected_state["bowling_team"],
            "venue": selected_state["venue"],
            "target_score": int(selected_state["target_score"]),
            "current_score": int(selected_state["current_score"]),
            "wickets_lost": int(selected_state["wickets_lost"]),
            "legal_balls_bowled": int(selected_state["legal_balls_bowled"]),
        },
        str(selected_state[TARGET_COLUMN]) if TARGET_COLUMN in selected_state else None,
    )


def render_future_match_mode() -> None:
    st.subheader("Future match prediction")
    st.caption("Eligibility comes only from the season-specific registry; historical CSV teams are never substituted.")
    registry = load_team_registry()
    current_year = date.today().year
    future_seasons = sorted(
        int(season)
        for season in registry.get("seasons", {})
        if int(season) > current_year
    )
    if not future_seasons:
        future_seasons = [current_year + 1]

    season = st.selectbox("Season", future_seasons, key="future_prediction_season")
    status = future_prediction_status(
        registry,
        season,
        Path(__file__).resolve().parents[1] / "reports" / "model_evaluation.csv",
        Path(__file__).resolve().parents[1] / "models" / "team1_win_prediction.joblib",
        today=date.today(),
    )
    season_data = registry.get("seasons", {}).get(str(season), {})
    source_url = season_data.get("source", "https://www.iplt20.com/teams")
    st.caption(
        f"Participation last verified: {season_data.get('last_verified', 'not recorded')} · "
        f"Source: {source_url}"
    )
    if "2026" in registry.get("seasons", {}):
        with st.expander("Verified current-season teams · 2026", expanded=False):
            st.caption("Initial badges are used because official logo reuse is not authorized in this project.")
            team_registry = load_team_registry()
            for team_name in registry["seasons"]["2026"]["teams"]:
                render_team_identity(team_name, team_registry)

    if status["teams"]:
        team_columns = st.columns(2)
        team_columns[0].selectbox(
            "Team A",
            status["teams"],
            key=f"future_team_a_{season}",
            disabled=not status["enabled"],
        )
        team_columns[1].selectbox(
            "Team B",
            status["teams"],
            key=f"future_team_b_{season}",
            disabled=not status["enabled"],
        )
    else:
        st.info("No team selector is shown because this season has no verified participant list.")

    if not status["enabled"]:
        st.warning(str(status["reason"]))
        st.caption(
            "Both scheduled fixtures and hypothetical matchups remain unavailable until "
            "season participants and a reliable pre-match model are verified."
        )
        return

    st.info(
        "The season registry is eligible, but future pre-match inference is not enabled "
        "until its model and inputs pass validation."
    )


def render_prediction_lab(
    matches: pd.DataFrame,
    selected_matches: pd.DataFrame,
    database_path: str | Path,
    model_path: str | Path,
) -> None:
    colors = chart_colors(st.session_state.get("ui_theme", "Dark"))
    st.subheader("Live chase probability")
    st.caption(
        "Estimates whether the batting team will complete the chase from a state "
        "recorded after a delivery. This is not an official live feed."
    )

    try:
        bundle = _cached_chase_model(str(model_path))
    except (OSError, ValueError, KeyError) as error:
        st.error(str(error))
        st.code(".\\.venv\\Scripts\\python.exe -m src.train_chase_model", language="powershell")
        return

    test_metrics = bundle.get("test_metrics", {}).get("selected_model", {})
    model_info = st.columns(4)
    model_info[0].metric("Model", bundle["model_name"])
    model_info[1].metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', float('nan')):.3f}")
    model_info[2].metric("Test Brier", f"{test_metrics.get('brier_score', float('nan')):.3f}")
    model_info[3].metric("Test period", f"{bundle['test_start'][:4]}–2026")

    st.caption(
        f"Trained on {bundle['model_fit_matches']:,} matches through 2021; "
        f"calibrated on {bundle['calibration_matches']:,} matches from 2022–2023."
    )
    with st.expander("Model limitations and evaluation", expanded=False):
        st.write(
            "The model excludes super overs, DLS/shortened matches, terminal states, "
            "and tied/no-result outcomes. Test metrics are match-weighted over delivery "
            "states; states within one match are correlated. These probabilities are "
            "historical estimates, not guarantees."
        )
        st.write(f"Evaluation details: {bundle.get('evaluation_weighting', 'Not recorded')}.")

    prediction_mode = st.radio(
        "Prediction mode",
        ["Live chase probability", "Future match prediction"],
        horizontal=True,
        key="prediction_mode",
    )
    if prediction_mode == "Future match prediction":
        render_future_match_mode()
        return

    state_source = st.radio(
        "Prediction input",
        ["Hypothetical scenario", "Historical chase state"],
        horizontal=True,
        key="prediction_state_source",
    )
    state = None
    actual_label = None
    if state_source == "Hypothetical scenario":
        st.caption("This is a manually entered hypothetical state, not live match information.")
        state = _manual_state(matches)
    else:
        state, actual_label = _historical_state(selected_matches, str(database_path))

    if state is None:
        return

    try:
        feature_row = make_chase_state(state)
    except ValueError as error:
        st.error(f"Invalid match state: {error}")
        return

    state_metrics = st.columns(5)
    state_metrics[0].metric("Runs needed", f"{int(feature_row.iloc[0]['runs_needed'])}")
    state_metrics[1].metric("Wickets remaining", f"{int(feature_row.iloc[0]['wickets_remaining'])}")
    state_metrics[2].metric("Balls remaining", f"{int(feature_row.iloc[0]['balls_remaining'])}")
    current_rate = feature_row.iloc[0]["current_run_rate"]
    state_metrics[3].metric("Current run rate", f"{current_rate:.2f}" if pd.notna(current_rate) else "N/A")
    state_metrics[4].metric("Required run rate", f"{feature_row.iloc[0]['required_run_rate']:.2f}")

    if not st.button("Estimate chase probability", type="primary", width="stretch"):
        return

    try:
        prediction = predict_chase_probability(state, bundle)
        contributions = local_feature_contributions(
            bundle["model"],
            bundle["calibrator"],
            feature_row,
        )
    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Prediction failed: {error}")
        return

    st.markdown("#### Model estimate")
    probability_columns = st.columns(2)
    probability_columns[0].metric(
        f"{state['batting_team']} chase success",
        f"{prediction['chasing_team_probability']:.1%}",
    )
    probability_columns[1].metric(
        f"{state['bowling_team']} win",
        f"{prediction['bowling_team_probability']:.1%}",
    )

    probability_data = pd.DataFrame(
        {
            "Outcome": [f"{state['batting_team']} chase succeeds", f"{state['bowling_team']} wins"],
            "Probability": [
                prediction["chasing_team_probability"] * 100,
                prediction["bowling_team_probability"] * 100,
            ],
        }
    )
    left, right = st.columns([1, 1.4])
    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction["chasing_team_probability"] * 100,
                number={"suffix": "%", "font": {"color": colors["text"]}},
                title={"text": "Chase success", "font": {"color": colors["muted"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": colors["muted"]},
                    "bar": {"color": colors["gold"]},
                    "bgcolor": colors["surface"],
                    "bordercolor": colors["line"],
                    "steps": [
                        {"range": [0, 35], "color": colors["line"]},
                        {"range": [35, 65], "color": colors["surface"]},
                        {"range": [65, 100], "color": colors["surface-raised"]},
                    ],
                },
            )
        )
        gauge.update_layout(template=plotly_template(st.session_state.get("ui_theme", "Dark")), height=260, margin={"l": 20, "r": 20, "t": 45, "b": 10})
        st.plotly_chart(gauge, width="stretch")
    with right:
        st.plotly_chart(
            px.bar(
                probability_data,
                x="Probability",
                y="Outcome",
                orientation="h",
                color="Outcome",
                color_discrete_sequence=[colors["gold"], colors["blue"]],
                range_x=[0, 100],
                labels={"Probability": "Estimated probability (%)"},
            ).update_layout(
                template=plotly_template(st.session_state.get("ui_theme", "Dark")),
                showlegend=False,
                margin={"l": 10, "r": 10, "t": 25, "b": 10},
            ),
            width="stretch",
        )

    st.markdown("#### Factors influencing this estimate")
    st.caption(
        "Signed contributions to the calibrated model log-odds. Positive values push "
        "toward a successful chase; negative values push away. Association is not causation."
    )
    explanation = contributions.head(8).copy()
    explanation["feature"] = explanation["feature"].str.replace(
        r"^(numeric__|categorical__)", "", regex=True
    )
    explanation = explanation.rename(
        columns={"feature": "Model factor", "log_odds_contribution": "Log-odds contribution"}
    )
    st.dataframe(explanation, width="stretch", hide_index=True)

    # AI Prediction Explanation
    with st.expander("Explain This Prediction (AI Interpretability)", expanded=False):
        if st.button("Generate Prediction Explanation with AI", key="btn_explain_prediction_ai"):
            with st.spinner("Synthesizing model factors, required rate pressure, and uncertainty bounds..."):
                contrib_dict = dict(zip(explanation["Model factor"], explanation["Log-odds contribution"]))
                pred_expl = explain_prediction(state, feature_row, prediction, contrib_dict)
                st.markdown(pred_expl["explanation"])

                def _discuss_in_ai():
                    st.session_state["pending_user_query"] = (
                        f"Analyze this live chase prediction: {state['batting_team']} chasing {state['target_score']} "
                        f"against {state['bowling_team']}. Current score is {state['current_score']}/{state['wickets_lost']} "
                        f"in {int(state['legal_balls_bowled']//6)}.{int(state['legal_balls_bowled']%6)} overs."
                    )
                    st.session_state["active_page"] = "AI Intelligence Center"

                st.button(
                    "💬 Chat with AI About This State in Intelligence Center",
                    on_click=_discuss_in_ai,
                    key="btn_chat_ai_chase",
                    help="Jump directly to AI Intelligence Center with this match state preloaded",
                )
        else:
            st.caption("Click to translate mathematical feature weights and run-rate dynamics into plain-English reasoning.")

    if actual_label is not None and state_source == "Historical chase state":
        st.caption("The historical outcome is intentionally not shown alongside the prediction.")
