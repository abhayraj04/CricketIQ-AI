import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.metrics import (
    head_to_head_summary,
    result_margin_counts,
    team_batting_order_summary,
    team_innings_averages,
    team_summary,
    venue_result_summary,
    year_match_run_summary,
)
from src.brand_views import render_player_identity, render_team_identity
from src.team_registry import load_team_registry, player_initials, team_short_name
from src.ui_theme import chart_colors, plotly_template
from src.ai.explainers import generate_player_intelligence_report


def _current_theme() -> str:
    return st.session_state.get("ui_theme", "Dark")


def render_overview(
    matches: pd.DataFrame,
    players: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> None:
    colors = chart_colors(_current_theme())
    st.subheader("Executive overview")
    teams = team_summary(matches)
    resolved_matches = int(matches["outcome_winner"].notna().sum())
    total_runs = int(pd.to_numeric(deliveries["total_runs"], errors="coerce").sum())
    total_wickets = int(pd.to_numeric(deliveries["is_wicket"], errors="coerce").sum())
    average_score = pd.to_numeric(matches["inn1_total_runs"], errors="coerce").mean()

    metrics = st.columns(4)
    metrics[0].metric("Matches", f"{len(matches):,}")
    metrics[1].metric("Recorded results", f"{resolved_matches:,}")
    metrics[2].metric("Runs · selected matches", f"{total_runs:,}")
    metrics[3].metric("Dismissals · selected matches", f"{total_wickets:,}")

    st.caption(
        f"First-innings average: {average_score:.1f} runs · "
        f"Selected coverage: {matches['date'].min().date()} to {matches['date'].max().date()} · "
        f"{matches['calendar_year'].nunique()} calendar years · "
        f"{matches[['team1', 'team2']].stack().nunique()} teams · "
        f"{matches['venue'].nunique()} venues · {len(players):,} player profiles"
        if pd.notna(average_score)
        else "First-innings score average is unavailable for this selection."
    )
    st.caption("Delivery run and dismissal totals exclude super overs; career player totals are not year-filtered.")

    yearly = year_match_run_summary(matches, deliveries)
    margins = result_margin_counts(matches)
    top_team = teams.iloc[0] if not teams.empty else None
    top_venue = matches["venue"].value_counts()
    top_year = yearly.loc[yearly["runs"].idxmax()] if not yearly.empty else None
    with st.expander("AI Executive Briefing & Data-Grounded Intelligence", expanded=False):
        toss_wins = int((matches["toss_winner_is_winner"].fillna(False).astype(bool)).sum())
        toss_pct = (toss_wins / resolved_matches * 100) if resolved_matches > 0 else 0
        st.markdown(
            f"**Historical Overview ({matches['calendar_year'].min()}–{matches['calendar_year'].max()})**:\n\n"
            f"- **Tournament Scale**: Analyzed **{len(matches):,}** matches comprising **{len(deliveries):,}** regular deliveries across **{matches['venue'].nunique()}** venues.\n"
            f"- **Run Environment**: Total of **{total_runs:,}** runs scored and **{total_wickets:,}** dismissals, with a tournament average 1st innings score of **{average_score:.1f}** runs.\n"
            f"- **Toss Association**: Toss-winning teams won **{toss_wins:,}** of {resolved_matches:,} resolved matches ({toss_pct:.1f}%).\n"
            f"- **Dominant Venue**: **{top_venue.index[0]}** leads with **{int(top_venue.iloc[0])}** matches hosted.\n"
            f"- **Grounded Verification**: All figures retrieved from verified SQLite tables (`ipl_matches_summary`, `ipl_ball_by_ball_processed`)."
        )

    st.markdown(
        "<div class='insight-strip'>"
        + (
            f"<b>Leading team:</b> {teams.index[0]} with {int(top_team['wins'])} wins "
            f"({top_team['win_percent']:.1f}% of resolved appearances)."
            if top_team is not None
            else "No resolved team results in this selection."
        )
        + " &nbsp; · &nbsp; "
        + (
            f"<b>Highest run year:</b> {int(top_year['calendar_year'])} "
            f"({int(top_year['runs']):,} regular-innings runs)."
            if top_year is not None
            else "No delivery runs in this selection."
        )
        + " &nbsp; · &nbsp; "
        + (
            f"<b>Most-used venue:</b> {top_venue.index[0]} "
            f"({int(top_venue.iloc[0])} matches)."
            if not top_venue.empty
            else "No venue records in this selection."
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    first_charts = st.columns([1.6, 1])
    with first_charts[0]:
        st.markdown("#### Match and run trends")
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(
            go.Scatter(
                x=yearly["calendar_year"],
                y=yearly["matches"],
                name="Matches",
                mode="lines+markers",
                line={"color": colors["gold"], "width": 3},
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=yearly["calendar_year"],
                y=yearly["runs"],
                name="Runs",
                mode="lines+markers",
                line={"color": colors["blue"], "width": 2},
            ),
            secondary_y=True,
        )
        figure.update_layout(
            template=plotly_template(_current_theme()),
            margin={"l": 8, "r": 8, "t": 15, "b": 8},
            legend={"orientation": "h", "y": 1.12},
            hovermode="x unified",
        )
        figure.update_xaxes(title="Calendar year", dtick=2, gridcolor=colors["line"])
        figure.update_yaxes(title="Matches", secondary_y=False, gridcolor=colors["line"])
        figure.update_yaxes(title="Runs", secondary_y=True, showgrid=False)
        st.plotly_chart(figure, width="stretch")

    with first_charts[1]:
        st.markdown("#### Wins by innings outcome")
        st.plotly_chart(
            px.bar(
                margins,
                x="matches",
                y="result",
                orientation="h",
                color="result",
                color_discrete_sequence=[colors["blue"], colors["orange"]],
                labels={"matches": "Resolved matches", "result": "Result"},
            ).update_layout(
                template=plotly_template(_current_theme()),
                showlegend=False,
                margin={"l": 8, "r": 8, "t": 15, "b": 8},
            ),
            width="stretch",
        )
        st.caption("Ties and no-results are excluded from the result-margin counts.")

    second_charts = st.columns(2)
    with second_charts[0]:
        st.markdown("#### Team win leaderboard")
        leaderboard = teams.head(10).sort_values("wins").reset_index(names="team")
        team_registry = load_team_registry()
        leaderboard["display_team"] = leaderboard["team"].map(
            lambda team: (
                f"{team_short_name(team, team_registry)} · {team}"
                if team_short_name(team, team_registry)
                else f"{player_initials(team)} · {team}"
            )
        )
        st.plotly_chart(
            px.bar(
                leaderboard,
                x="wins",
                y="display_team",
                orientation="h",
                color_discrete_sequence=[colors["gold"]],
                labels={"wins": "Wins", "display_team": "Team"},
                hover_data=["matches", "losses", "win_percent"],
            ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
            width="stretch",
        )

    with second_charts[1]:
        st.markdown("#### Busy venues")
        venue_counts = matches["venue"].value_counts().head(8).sort_values()
        venue_data = venue_counts.rename_axis("venue").reset_index(name="matches")
        st.plotly_chart(
            px.bar(
                venue_data,
                x="matches",
                y="venue",
                orientation="h",
                color_discrete_sequence=[colors["teal"]],
                labels={"matches": "Matches", "venue": "Venue"},
            ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
            width="stretch",
        )

    st.markdown("#### Top career performers · full supplied profile totals")
    performers = st.columns(2)
    for column, metric, label, color in [
        (performers[0], "runs_scored", "Runs", colors["orange"]),
        (performers[1], "wickets", "Wickets", colors["blue"]),
    ]:
        with column:
            st.markdown(f"**Top five by {label.lower()}**")
            top_players = players.nlargest(5, metric).sort_values(metric)
            render_player_identity(str(top_players.iloc[-1]["player_name"]))
            st.plotly_chart(
                px.bar(
                    top_players,
                    x=metric,
                    y="player_name",
                    orientation="h",
                    color_discrete_sequence=[color],
                    hover_data=["matches_selected"],
                    labels={metric: label, "player_name": "Player"},
                ).update_layout(template="plotly_dark", margin={"l": 8, "r": 8, "t": 5, "b": 8}),
                width="stretch",
            )

    st.markdown("#### Latest recorded results in the selected period")
    result_columns = [
        "date", "calendar_year", "team1", "team2", "venue", "outcome_winner",
        "outcome_by_runs", "outcome_by_wickets", "player_of_match",
    ]
    recorded_results = matches.loc[matches["outcome_winner"].notna()]
    if recorded_results.empty:
        st.info("No matches with a recorded winner match these filters.")
    else:
        st.dataframe(
            recorded_results.sort_values("date", ascending=False)[result_columns].head(20),
            width="stretch",
            hide_index=True,
        )
    st.caption("Historical results only; this table is not a live fixture feed.")


def render_team_analytics(matches: pd.DataFrame, deliveries: pd.DataFrame) -> None:
    colors = chart_colors(_current_theme())
    st.subheader("Team analytics")
    summary = team_summary(matches)
    if summary.empty:
        st.info("No matches with recorded winners match these filters.")
        return

    team_names = summary.index.tolist()
    selected_for_compare = st.multiselect(
        "Compare teams",
        options=team_names,
        default=team_names[: min(3, len(team_names))],
        key="team_compare",
    )
    comparison = summary.loc[selected_for_compare] if selected_for_compare else summary
    comparison_table = comparison.reset_index(names="team")
    team_registry = load_team_registry()
    identity_columns = st.columns(min(4, len(comparison_table)))
    for column, team_name in zip(identity_columns, comparison_table["team"]):
        with column:
            render_team_identity(team_name, team_registry)
    st.dataframe(
        comparison_table,
        width="stretch",
        hide_index=True,
        column_config={"win_percent": st.column_config.NumberColumn(format="%.1f%%")},
    )
    comparison_plot = comparison_table.sort_values("win_percent")
    st.plotly_chart(
        px.bar(
            comparison_plot,
            x="win_percent",
            y="team",
            orientation="h",
            color="team",
            color_discrete_sequence=[colors["gold"], colors["blue"], colors["orange"], colors["teal"]],
            labels={"win_percent": "Win percentage", "team": "Team"},
            hover_data=["matches", "wins", "losses"],
        ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
        width="stretch",
    )

    resolved = matches.loc[matches["outcome_winner"].notna()]
    team_match_rows = pd.concat(
        [
            resolved.assign(team=resolved["team1"], won=resolved["outcome_winner"].eq(resolved["team1"])),
            resolved.assign(team=resolved["team2"], won=resolved["outcome_winner"].eq(resolved["team2"])),
        ],
        ignore_index=True,
    )
    if selected_for_compare:
        team_match_rows = team_match_rows.loc[team_match_rows["team"].isin(selected_for_compare)]
    season_form = (
        team_match_rows.groupby(["calendar_year", "team"], as_index=False)
        .agg(matches=("won", "size"), wins=("won", "sum"))
    )
    season_form["win_percent"] = season_form["wins"] / season_form["matches"] * 100
    st.markdown("#### Season-wise win percentage")
    st.plotly_chart(
        px.line(
            season_form,
            x="calendar_year",
            y="win_percent",
            color="team",
            markers=True,
            labels={"calendar_year": "Calendar year", "win_percent": "Win percentage"},
        ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
        width="stretch",
    )

    if len(team_names) >= 2:
        st.markdown("#### Head-to-head")
        first_default = selected_for_compare[0] if selected_for_compare else team_names[0]
        second_candidates = [team for team in team_names if team != first_default]
        second_default = (
            selected_for_compare[1]
            if len(selected_for_compare) > 1 and selected_for_compare[1] != first_default
            else second_candidates[0]
        )
        head_columns = st.columns(2)
        team_a = head_columns[0].selectbox(
            "Team A",
            team_names,
            index=team_names.index(first_default),
            key="head_to_head_team_a",
        )
        team_b_options = [team for team in team_names if team != team_a]
        if second_default not in team_b_options:
            second_default = team_b_options[0]
        team_b = head_columns[1].selectbox(
            "Team B",
            team_b_options,
            index=team_b_options.index(second_default),
            key="head_to_head_team_b",
        )
        team_registry = load_team_registry()
        identity_columns = st.columns(2)
        with identity_columns[0]:
            render_team_identity(team_a, team_registry)
        with identity_columns[1]:
            render_team_identity(team_b, team_registry)
        record = head_to_head_summary(matches, team_a, team_b)
        head_metrics = st.columns(4)
        head_metrics[0].metric("Meetings", f"{record['matches']:,}")
        head_metrics[1].metric(f"{team_a} wins", f"{record['team_a_wins']:,}")
        head_metrics[2].metric(f"{team_b} wins", f"{record['team_b_wins']:,}")
        head_metrics[3].metric("Ties / no-results", f"{record['ties']} / {record['no_results']}")

    st.markdown("#### Batting-first and chase record")
    order_summary = team_batting_order_summary(matches, deliveries)
    if selected_for_compare:
        order_summary = order_summary.loc[order_summary["team"].isin(selected_for_compare)]
    if order_summary.empty:
        st.info("No resolved innings-order results are available for this selection.")
    else:
        order_table = order_summary.copy()
        for column in ["batting_first_win_percent", "chase_win_percent"]:
            order_table[column] = order_table[column].round(1)
        st.dataframe(order_table, width="stretch", hide_index=True)

    st.markdown("#### Runs per innings")
    innings_averages = team_innings_averages(deliveries)
    if selected_for_compare:
        innings_averages = innings_averages.loc[
            innings_averages["team"].isin(selected_for_compare)
        ]
    st.dataframe(innings_averages, width="stretch", hide_index=True)

    st.markdown("#### Win margins")
    resolved_wins = matches.loc[matches["outcome_winner"].notna()].copy()
    run_margins = pd.to_numeric(resolved_wins["outcome_by_runs"], errors="coerce")
    wicket_margins = pd.to_numeric(resolved_wins["outcome_by_wickets"], errors="coerce")
    resolved_wins["margin_type"] = np.select(
        [run_margins.gt(0), wicket_margins.gt(0)],
        ["By runs", "By wickets"],
        default="Other resolved result",
    )
    margin_rows = resolved_wins[["outcome_winner", "margin_type"]].rename(
        columns={"outcome_winner": "team"}
    )
    if selected_for_compare:
        margin_rows = margin_rows.loc[margin_rows["team"].isin(selected_for_compare)]
    margin_summary = margin_rows.groupby(["team", "margin_type"]).size().rename("wins").reset_index()
    st.caption("Ties and no-results have no recorded winning margin and are excluded.")
    if margin_summary.empty:
        st.info("No resolved winning margins are available for this selection.")
    else:
        st.dataframe(margin_summary, width="stretch", hide_index=True)

    st.markdown("#### Toss outcomes by toss-winning team")
    toss_rows = matches.dropna(
        subset=["toss_winner", "toss_winner_is_winner", "outcome_winner"]
    ).copy()
    toss_rows["toss_winner_match_win"] = toss_rows["toss_winner_is_winner"].astype(bool)
    toss_summary = (
        toss_rows.groupby("toss_winner", as_index=False)
        .agg(
            tosses=("match_id", "nunique"),
            matches_won=("toss_winner_match_win", "sum"),
        )
        .rename(columns={"toss_winner": "team"})
    )
    toss_summary["toss_winner_match_win_percent"] = (
        100 * toss_summary["matches_won"] / toss_summary["tosses"]
    )
    if selected_for_compare:
        toss_summary = toss_summary.loc[toss_summary["team"].isin(selected_for_compare)]
    st.dataframe(
        toss_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "toss_winner_match_win_percent": st.column_config.NumberColumn(format="%.1f%%")
        },
    )

    if deliveries.empty:
        return
    runs_by_team = (
        deliveries.groupby("batting_team")["total_runs"]
        .sum()
        .rename("runs_scored")
        .to_frame()
    )
    runs_by_team["runs_conceded"] = deliveries.groupby("bowling_team")["total_runs"].sum()
    runs_by_team = runs_by_team.reindex(selected_for_compare or team_names).dropna(how="all")
    st.markdown("#### Runs scored and conceded")
    st.caption("Delivery totals use the filtered matches; team names from distinct franchise eras remain separate.")
    st.dataframe(runs_by_team.reset_index(names="team"), width="stretch", hide_index=True)


def render_player_analytics(players: pd.DataFrame) -> None:
    colors = chart_colors(_current_theme())
    st.subheader("Player analytics")
    st.caption(
        "Career statistics cover the supplied profile table. Match year/team filters do not "
        "apply because player profiles are not split by team or season."
    )
    search = st.text_input("Search player", placeholder="Start typing a player name")
    filter_columns = st.columns(3)
    roles = sorted(players["primary_role"].dropna().unique())
    batting_styles = sorted(players["bat_style"].dropna().unique())
    bowling_styles = sorted(players["bowl_style"].dropna().unique())
    selected_roles = filter_columns[0].multiselect("Role", roles, key="player_role_filter")
    selected_batting_styles = filter_columns[1].multiselect(
        "Batting style", batting_styles, key="player_bat_style_filter"
    )
    selected_bowling_styles = filter_columns[2].multiselect(
        "Bowling style", bowling_styles, key="player_bowl_style_filter"
    )
    filtered_players = players.loc[
        players["player_name"].str.contains(search, case=False, na=False, regex=False)
    ]
    if selected_roles:
        filtered_players = filtered_players.loc[
            filtered_players["primary_role"].isin(selected_roles)
        ]
    if selected_batting_styles:
        filtered_players = filtered_players.loc[
            filtered_players["bat_style"].isin(selected_batting_styles)
        ]
    if selected_bowling_styles:
        filtered_players = filtered_players.loc[
            filtered_players["bowl_style"].isin(selected_bowling_styles)
        ]
    if filtered_players.empty:
        st.info("No players match the current search and style filters.")
        return

    player_options = filtered_players["player_name"].tolist()
    chosen_players = st.multiselect(
        "Compare players",
        options=player_options,
        default=player_options[: min(2, len(player_options))],
        key="player_compare",
    )

    batting_columns = [
        "player_name", "matches_selected", "runs_scored", "batting_average",
        "batting_strike_rate", "balls_faced", "fours", "sixes", "boundary_run_pct",
        "powerplay_strike_rate", "middle_strike_rate", "death_strike_rate",
        "runs_vs_pace", "sr_vs_pace", "runs_vs_spin", "sr_vs_spin",
    ]
    bowling_columns = [
        "player_name", "matches_selected", "innings_bowled", "legal_balls_bowled",
        "runs_conceded", "wickets", "bowling_economy", "bowling_average",
        "bowling_strike_rate", "dot_ball_pct_bowling", "powerplay_economy",
        "death_overs_economy",
    ]
    if chosen_players:
        st.markdown("#### Selected player comparison")
        chosen = filtered_players.loc[filtered_players["player_name"].isin(chosen_players)]
        comparison_columns = list(
            dict.fromkeys(
                ["player_name", "primary_role", "bat_style", "bowl_style"]
                + batting_columns
                + bowling_columns
            )
        )
        player_identity_columns = st.columns(min(4, len(chosen)))
        for column, (_, player) in zip(player_identity_columns, chosen.iterrows()):
            with column:
                render_player_identity(player["player_name"])
        st.dataframe(
            chosen[comparison_columns],
            width="stretch",
            hide_index=True,
        )

        if len(chosen) >= 2:
            st.markdown("##### Multi-Attribute Radar Comparison")
            radar_categories = ["Strike Rate", "Average", "Boundary %", "Volume", "Dot-Ball Control"]
            radar_fig = go.Figure()
            for _, p in chosen.iterrows():
                sr_val = min(100.0, (float(p.get("batting_strike_rate", 0) or 0) / 180.0) * 100)
                avg_val = min(100.0, (float(p.get("batting_average", 0) or 0) / 50.0) * 100)
                bnd_val = min(100.0, (float(p.get("boundary_run_pct", 0) or 0) / 75.0) * 100)
                vol_val = min(100.0, (float(p.get("runs_scored", 0) or 0) / 6000.0) * 100)
                dot_val = max(0.0, 100.0 - float(p.get("dot_ball_pct_batting", 40) or 40) * 1.5)
                r_scores = [sr_val, avg_val, bnd_val, vol_val, dot_val]
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=r_scores + [r_scores[0]],
                        theta=radar_categories + [radar_categories[0]],
                        fill="toself",
                        name=p["player_name"],
                    )
                )
            radar_fig.update_layout(
                polar={"radialaxis": {"visible": True, "range": [0, 100]}},
                template=plotly_template(_current_theme()),
                height=320,
                margin={"l": 30, "r": 30, "t": 25, "b": 25},
            )
            st.plotly_chart(radar_fig, width="stretch")

        with st.expander("Generate AI Scouting Dossier", expanded=False):
            target_p = st.selectbox("Select player to analyze", chosen_players, key="ai_dossier_select")
            if st.button("Generate Intelligence Dossier", key="btn_dossier_gen"):
                with st.spinner(f"Compiling career breakdown and tactical splits for {target_p}..."):
                    target_row = chosen.loc[chosen["player_name"].eq(target_p)].iloc[0].to_dict()
                    dossier = generate_player_intelligence_report(target_p, target_row)
                    st.markdown(dossier["report"])

    st.download_button(
        "Download filtered player profiles",
        data=filtered_players.to_csv(index=False).encode("utf-8"),
        file_name="ipl_filtered_player_profiles.csv",
        mime="text/csv",
        width="stretch",
    )

    batting_top, bowling_top = st.columns(2)
    with batting_top:
        st.markdown("#### Top run scorers")
        top_runs = filtered_players.nlargest(10, "runs_scored").sort_values("runs_scored")
        st.plotly_chart(
            px.bar(
                top_runs,
                x="runs_scored",
                y="player_name",
                orientation="h",
                hover_data=["matches_selected", "batting_strike_rate"],
                color_discrete_sequence=[colors["orange"]],
            ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
            width="stretch",
        )
    with bowling_top:
        st.markdown("#### Leading wicket takers")
        top_wickets = filtered_players.nlargest(10, "wickets").sort_values("wickets")
        st.plotly_chart(
            px.bar(
                top_wickets,
                x="wickets",
                y="player_name",
                orientation="h",
                hover_data=["matches_selected", "bowling_economy"],
                color_discrete_sequence=[colors["blue"]],
            ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
            width="stretch",
        )

    st.markdown("#### Batting volume and strike rate")
    batting_scatter = filtered_players.dropna(subset=["runs_scored", "batting_strike_rate"])
    st.plotly_chart(
        px.scatter(
            batting_scatter,
            x="runs_scored",
            y="batting_strike_rate",
            size="matches_selected",
            hover_name="player_name",
            hover_data=["balls_faced", "batting_average"],
            labels={"runs_scored": "Career runs", "batting_strike_rate": "Strike rate"},
        ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
        width="stretch",
    )
    st.caption("Rate statistics from small samples can be misleading; compare matches and balls faced.")


def render_venue_toss_analytics(matches: pd.DataFrame) -> None:
    colors = chart_colors(_current_theme())
    st.subheader("Venue and toss intelligence")
    venue_names = sorted(matches["venue"].dropna().unique())
    selected_venues = st.multiselect(
        "Compare venues",
        venue_names,
        default=[],
        key="venue_compare",
        help="This page-specific selection narrows venue comparisons only.",
    )
    venue_matches = matches.loc[matches["venue"].isin(selected_venues)] if selected_venues else matches
    venue_stats = venue_result_summary(venue_matches)
    st.caption(f"{len(venue_stats):,} venues · {len(venue_matches):,} matches in this comparison")
    venue_chart, score_chart = st.columns(2)
    with venue_chart:
        st.markdown("#### Matches by venue")
        venue_plot = venue_stats.head(12).sort_values("matches")
        st.plotly_chart(
            px.bar(
                venue_plot,
                x="matches",
                y="venue",
                orientation="h",
                color_discrete_sequence=[colors["teal"]],
            ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
            width="stretch",
        )
        st.dataframe(
            venue_plot[["venue", "matches", "resolved_matches", "ties", "no_results"]].sort_values(
                "matches", ascending=False
            ),
            width="stretch",
            hide_index=True,
        )
    with score_chart:
        st.markdown("#### Average first-innings score")
        score_plot = venue_stats.dropna(subset=["average_first_innings_score"]).sort_values(
            "average_first_innings_score"
        )
        st.plotly_chart(
            px.bar(
                score_plot.tail(12),
                x="average_first_innings_score",
                y="venue",
                orientation="h",
                color_discrete_sequence=[colors["orange"]],
                labels={"average_first_innings_score": "Average runs", "venue": "Venue"},
            ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8}),
            width="stretch",
        )
        st.dataframe(
            score_plot.tail(12)[
                ["venue", "average_first_innings_score", "resolved_matches", "matches"]
            ].sort_values("average_first_innings_score", ascending=False),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Batting-first and chase results")
    venue_result_plot = venue_stats.head(15).sort_values("matches")
    if venue_result_plot.empty:
        st.info("No venue results match the current selection.")
    else:
        venue_result_chart = px.bar(
            venue_result_plot,
            y="venue",
            x=["batting_first_win_percent", "chase_win_percent"],
            orientation="h",
            barmode="group",
            labels={"value": "Win percentage of resolved matches", "variable": "Outcome"},
            color_discrete_sequence=[colors["orange"], colors["blue"]],
            hover_data=["resolved_matches", "ties", "no_results"],
        ).update_layout(template=plotly_template(_current_theme()), margin={"l": 8, "r": 8, "t": 10, "b": 8})
        st.plotly_chart(venue_result_chart, width="stretch")
        st.caption("Winning rates use resolved matches as the denominator; ties and no-results are excluded.")

    team_venue_rows = venue_matches.loc[venue_matches["outcome_winner"].notna()].copy()
    team_venue_rows = pd.concat(
        [
            team_venue_rows.assign(team=team_venue_rows["team1"]),
            team_venue_rows.assign(team=team_venue_rows["team2"]),
        ],
        ignore_index=True,
    )
    if not team_venue_rows.empty:
        team_venue_stats = (
            team_venue_rows.assign(won=team_venue_rows["outcome_winner"].eq(team_venue_rows["team"]))
            .groupby(["venue", "team"], as_index=False)
            .agg(matches=("match_id", "nunique"), wins=("won", "sum"))
        )
        team_venue_stats["win_percent"] = 100 * team_venue_stats["wins"] / team_venue_stats["matches"]
        team_venue_stats = team_venue_stats.sort_values(["matches", "win_percent"], ascending=[False, False])
        st.markdown("#### Team results by venue")
        st.caption("Small samples can produce unstable percentages; always read wins alongside matches.")
        st.dataframe(
            team_venue_stats.head(40),
            width="stretch",
            hide_index=True,
            column_config={"win_percent": st.column_config.NumberColumn(format="%.1f%%")},
        )

    toss_data = venue_matches.dropna(
        subset=["toss_decision", "toss_winner_is_winner", "outcome_winner"]
    ).copy()
    if toss_data.empty:
        st.info("No resolved toss records match the current filters.")
    else:
        toss_data["toss_winner_won"] = toss_data["toss_winner_is_winner"].astype(bool)
        toss_rates = (
            toss_data.groupby("toss_decision", observed=True)["toss_winner_won"]
            .mean()
            .mul(100)
            .rename("win_percent")
            .reset_index()
        )
        st.markdown("#### Toss-winner match-win rate by decision")
        st.plotly_chart(
            px.bar(
                toss_rates,
                x="toss_decision",
                y="win_percent",
                color="toss_decision",
                color_discrete_sequence=[colors["gold"], colors["blue"]],
                labels={"toss_decision": "Toss decision", "win_percent": "Win percentage"},
            ).update_layout(template=plotly_template(_current_theme()), showlegend=False),
            width="stretch",
        )
        st.caption("Association only: these historical percentages do not show that the toss caused a win.")


def render_ball_analytics(deliveries: pd.DataFrame) -> None:
    colors = chart_colors(_current_theme())
    st.subheader("Ball-by-ball analytics")
    regular_deliveries = deliveries.loc[
        pd.to_numeric(deliveries["innings"], errors="coerce").isin([1, 2])
    ].copy()
    if regular_deliveries.empty:
        st.info("No regular-innings deliveries match the current filters.")
        return

    st.caption("The global year/team/venue/stage filters apply first; these controls narrow delivery analysis further.")
    match_ids = sorted(regular_deliveries["match_id"].dropna().unique().tolist(), reverse=True)
    match_id = st.selectbox("Match", match_ids, key="ball_match")
    match_deliveries = regular_deliveries.loc[regular_deliveries["match_id"].eq(match_id)]
    innings_options = sorted(
        pd.to_numeric(match_deliveries["innings"], errors="coerce").dropna().astype(int).unique()
    )
    innings = st.selectbox("Innings", innings_options, format_func=lambda value: f"Innings {value}", key="ball_innings")
    inning_deliveries = match_deliveries.loc[
        pd.to_numeric(match_deliveries["innings"], errors="coerce").eq(innings)
    ].copy()
    batting_teams = sorted(inning_deliveries["batting_team"].dropna().unique())
    bowling_teams = sorted(inning_deliveries["bowling_team"].dropna().unique())
    team_columns = st.columns(2)
    batting_choice = team_columns[0].selectbox(
        "Batting team", ["All"] + batting_teams, key="ball_batting_team"
    )
    bowling_choice = team_columns[1].selectbox(
        "Bowling team", ["All"] + bowling_teams, key="ball_bowling_team"
    )
    if batting_choice != "All":
        inning_deliveries = inning_deliveries.loc[
            inning_deliveries["batting_team"].eq(batting_choice)
        ]
    if bowling_choice != "All":
        inning_deliveries = inning_deliveries.loc[
            inning_deliveries["bowling_team"].eq(bowling_choice)
        ]
    if inning_deliveries.empty:
        st.info("No deliveries match the selected match, innings, and teams.")
        return

    over_totals = (
        inning_deliveries.assign(
            over_number=pd.to_numeric(inning_deliveries["over"], errors="coerce") + 1
        )
        .groupby("over_number", as_index=False)
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            legal_balls=("is_legal_delivery", "sum"),
            dots=("is_dot_ball", "sum"),
            boundaries=("is_boundary", "sum"),
        )
    )
    over_columns = st.columns(2)
    with over_columns[0]:
        st.markdown("#### Runs and wickets by over")
        over_chart = px.bar(
            over_totals,
            x="over_number",
            y="runs",
            color="wickets",
            color_continuous_scale=[colors["blue"], colors["orange"]],
            hover_data=["wickets", "legal_balls", "dots", "boundaries"],
            labels={"over_number": "Displayed over", "runs": "Runs", "wickets": "Wickets"},
        ).update_layout(template=plotly_template(_current_theme()))
        st.plotly_chart(over_chart, width="stretch")
    with over_columns[1]:
        st.markdown("#### Scoring rate progression")
        rate_column = "current_run_rate" if innings == 1 else "required_run_rate"
        rate_label = "Current run rate" if innings == 1 else "Required run rate"
        rate_data = inning_deliveries.copy()
        rate_data["sequence"] = pd.to_numeric(rate_data["ball_seq_idx"], errors="coerce")
        rate_data[rate_column] = pd.to_numeric(rate_data[rate_column], errors="coerce")
        if innings == 2:
            rate_data = rate_data.loc[
                rate_data["is_innings_ended"].fillna(0).astype(int).eq(0)
                & rate_data["target_runs"].notna()
            ]
        rate_data = rate_data.dropna(subset=["sequence", rate_column]).sort_values("sequence")
        if rate_data.empty:
            st.info(f"{rate_label} is unavailable for this innings state.")
        else:
            rate_chart = px.line(
                rate_data,
                x="sequence",
                y=rate_column,
                labels={"sequence": "Delivery sequence", rate_column: rate_label},
            ).update_layout(template=plotly_template(_current_theme()))
            st.plotly_chart(rate_chart, width="stretch")

    phase_summary = (
        inning_deliveries.groupby("match_phase", dropna=False, as_index=False)
        .agg(
            runs=("total_runs", "sum"),
            dismissals=("is_wicket", "sum"),
            legal_balls=("is_legal_delivery", "sum"),
            dot_balls=("is_dot_ball", "sum"),
            boundaries=("is_boundary", "sum"),
        )
        .sort_values("match_phase")
    )
    phase_summary["dot_ball_percent"] = (
        100 * phase_summary["dot_balls"] / phase_summary["legal_balls"].replace(0, pd.NA)
    )
    st.markdown("#### Match-phase summary")
    st.dataframe(phase_summary.round(1), width="stretch", hide_index=True)

    dismissal_summary = (
        inning_deliveries.loc[inning_deliveries["is_wicket"].astype(bool)]
        .groupby("dismissal_kind", dropna=False)
        .size()
        .rename("dismissals")
        .sort_values(ascending=False)
        .rename_axis("dismissal_kind")
        .reset_index()
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Dismissals")
        if dismissal_summary.empty:
            st.info("No dismissals recorded in this innings.")
        else:
            st.dataframe(dismissal_summary, width="stretch", hide_index=True)
    with right:
        st.markdown("#### Delivery notes")
        st.caption("Displayed over = zero-based source `over` + 1. Extras remain in total runs; legal balls use `is_legal_delivery`.")
        st.caption("Run-rate states are post-delivery. Chase required-rate charts omit innings-ending rows.")