from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai.explainers import explain_historical_match
from src.brand_views import render_player_identity, render_team_identity
from src.metrics import innings_scorecard, over_scorecard
from src.team_registry import load_team_registry
from src.ui_theme import chart_colors, plotly_template

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_analytics.sqlite"


def filter_match_records(
    matches: pd.DataFrame,
    search: str = "",
    result_filter: str = "All",
    ascending: bool = False,
) -> pd.DataFrame:
    filtered = matches.copy()
    search = search.strip()
    if search:
        searchable = ["team1", "team2", "venue", "outcome_winner", "player_of_match"]
        search_mask = pd.Series(False, index=filtered.index)
        for column in searchable:
            search_mask |= filtered[column].astype("string").str.contains(
                search,
                case=False,
                na=False,
                regex=False,
            )
        search_mask |= filtered["date"].astype("string").str.contains(
            search,
            case=False,
            na=False,
            regex=False,
        )
        filtered = filtered.loc[search_mask]

    if result_filter == "Winner recorded":
        filtered = filtered.loc[
            filtered["outcome_winner"].notna()
            & filtered["outcome_winner"].astype("string").str.strip().ne("")
        ]
    elif result_filter == "Tie":
        filtered = filtered.loc[
            filtered["is_tie"].fillna(False).astype(bool)
            | filtered["outcome_result"].eq("tie")
        ]
    elif result_filter == "No result":
        filtered = filtered.loc[
            filtered["is_no_result"].fillna(False).astype(bool)
            | filtered["outcome_result"].eq("no result")
        ]
    elif result_filter != "All":
        raise ValueError(f"Unknown result filter: {result_filter}")

    return filtered.sort_values("date", ascending=ascending)


def render_match_explorer(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    database_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    colors = chart_colors(st.session_state.get("ui_theme", "Dark"))
    st.subheader("Match explorer")
    st.caption("One row per match. Delivery details are aggregated only after selecting a match.")

    controls = st.columns([1.5, 1, 1])
    search = controls[0].text_input("Search team, venue, or player", key="match_search").strip()
    result_filter = controls[1].selectbox(
        "Recorded result",
        ["All", "Winner recorded", "Tie", "No result"],
        key="match_result_filter",
    )
    sort_order = controls[2].selectbox(
        "Sort matches",
        ["Newest first", "Oldest first"],
        key="match_sort_order",
    )

    ascending = sort_order == "Oldest first"
    filtered = filter_match_records(matches, search, result_filter, ascending)
    st.caption(f"{len(filtered):,} matches in current selection")

    display_columns = [
        "date", "calendar_year", "match_stage", "team1", "team2", "venue",
        "toss_winner", "toss_decision", "outcome_winner", "outcome_by_runs",
        "outcome_by_wickets", "player_of_match",
    ]
    st.dataframe(filtered[display_columns], width="stretch", hide_index=True)
    st.download_button(
        "Download filtered matches",
        data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="ipl_filtered_matches.csv",
        mime="text/csv",
        width="stretch",
        disabled=filtered.empty,
    )

    if filtered.empty:
        st.info("No matches match the current search and filters.")
        return

    st.markdown("#### Match detail")
    labels = {
        row["match_id"]: (
            f"{row['date'].date()} | {row['team1']} vs {row['team2']} | {row['venue']}"
        )
        for _, row in filtered.iterrows()
    }
    selected_match_id = st.selectbox(
        "Select match",
        filtered["match_id"].tolist(),
        format_func=lambda match_id: labels[match_id],
        key="match_detail_id",
    )
    match = filtered.loc[filtered["match_id"].eq(selected_match_id)].iloc[0]
    if match["outcome_winner"]:
        winner_text = str(match["outcome_winner"])
    elif match["outcome_result"] == "tie":
        winner_text = "Tie (super-over result is not in outcome_winner)"
    elif match["outcome_result"] == "no result":
        winner_text = "No result"
    else:
        winner_text = "Result unavailable"
    team_identity_columns = st.columns(2)
    registry = load_team_registry()
    with team_identity_columns[0]:
        render_team_identity(match["team1"], registry)
    with team_identity_columns[1]:
        render_team_identity(match["team2"], registry)
    st.markdown(
        f"{match['date'].date()} · {match['venue']} · **Result:** {winner_text}"
    )
    if pd.notna(match["player_of_match"]):
        st.caption("Player of the Match")
        render_player_identity(str(match["player_of_match"]))

    # AI Match Explanation section
    with st.expander("Explain This Match (AI Tactical Summary & Turning Points)", expanded=False):
        if st.button("Generate Match Explanation", key=f"btn_explain_{selected_match_id}"):
            with st.spinner("Synthesizing match scorecard, turning points, and key spells..."):
                explanation = explain_historical_match(str(selected_match_id), database_path)
                st.markdown(explanation["summary"])
        else:
            st.caption("Click to synthesize key partnerships, decisive spells, and high-momentum overs.")

    scorecard = innings_scorecard(deliveries, selected_match_id)
    if scorecard.empty:
        st.info("No regular-innings delivery scorecard is available for this match.")
        return

    st.markdown("##### Innings scorecard")
    st.dataframe(scorecard, width="stretch", hide_index=True)
    innings_options = scorecard["innings"].astype(int).tolist()
    selected_innings = st.selectbox(
        "Analyze innings",
        innings_options,
        format_func=lambda innings: f"Innings {innings}",
        key="match_detail_innings",
    )
    inning_deliveries = deliveries.loc[
        deliveries["match_id"].eq(selected_match_id)
        & pd.to_numeric(deliveries["innings"], errors="coerce").eq(selected_innings)
        & pd.to_numeric(deliveries["is_super_over"], errors="coerce").fillna(0).eq(0)
    ]
    overs = over_scorecard(deliveries, selected_match_id)
    overs = overs.loc[overs["innings"].eq(selected_innings)]
    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.markdown("##### Runs and wickets by over")
        if not overs.empty:
            over_chart = px.bar(
                overs,
                x="over_number",
                y="runs",
                color="wickets",
                color_continuous_scale=[colors["blue"], colors["orange"]],
                hover_data=["wickets"],
                labels={"over_number": "Over", "runs": "Runs", "wickets": "Wickets"},
            ).update_layout(template=plotly_template(st.session_state.get("ui_theme", "Dark")))
            st.plotly_chart(over_chart, width="stretch")
    with detail_columns[1]:
        st.markdown("##### Batter contributions")
        batter_summary = (
            inning_deliveries.groupby("batter", as_index=False)
            .agg(
                runs=("runs_batter", "sum"),
                fours=("is_four", "sum"),
                sixes=("is_six", "sum"),
            )
            .sort_values("runs", ascending=False)
        )
        st.dataframe(batter_summary, width="stretch", hide_index=True)

    bowler_summary = (
        inning_deliveries.groupby("bowler", as_index=False)
        .agg(
            runs_conceded=("bowler_runs_conceded", "sum"),
            wickets=("is_bowler_wicket", "sum"),
            legal_balls=("is_legal_delivery", "sum"),
        )
        .sort_values(["wickets", "runs_conceded"], ascending=[False, True])
    )
    st.markdown("##### Bowling figures")
    st.dataframe(bowler_summary, width="stretch", hide_index=True)
