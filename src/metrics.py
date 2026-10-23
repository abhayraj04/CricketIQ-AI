import pandas as pd


def filter_matches(
    matches: pd.DataFrame,
    year_range: tuple[int, int],
    teams: list[str],
    venues: list[str],
    stages: list[str],
) -> pd.DataFrame:
    filtered = matches.loc[
        matches["calendar_year"].between(year_range[0], year_range[1])
    ]

    if teams:
        filtered = filtered.loc[
            filtered["team1"].isin(teams) | filtered["team2"].isin(teams)
        ]
    if venues:
        filtered = filtered.loc[filtered["venue"].isin(venues)]
    if stages:
        filtered = filtered.loc[filtered["match_stage"].isin(stages)]
    return filtered.copy()


def filter_deliveries(
    deliveries: pd.DataFrame,
    selected_matches: pd.DataFrame,
) -> pd.DataFrame:
    """Filter deliveries by match IDs, then exclude super overs."""
    match_ids = selected_matches["match_id"].dropna().unique()
    is_regular_innings = pd.to_numeric(
        deliveries["is_super_over"], errors="coerce"
    ).fillna(0).eq(0)
    return deliveries.loc[
        deliveries["match_id"].isin(match_ids) & is_regular_innings
    ].copy()


def team_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarize resolved match appearances; ties and no-results are excluded."""
    has_winner = matches["outcome_winner"].notna() & matches[
        "outcome_winner"
    ].astype("string").str.strip().ne("")
    resolved = matches.loc[has_winner]
    appearances = pd.concat([resolved["team1"], resolved["team2"]]).value_counts()
    wins = resolved["outcome_winner"].value_counts()

    summary = pd.DataFrame({"matches": appearances, "wins": wins}).fillna(0)
    summary["losses"] = summary["matches"] - summary["wins"]
    summary["win_percent"] = summary["wins"].div(summary["matches"]).mul(100)
    return (
        summary.rename_axis("team")
        .reset_index()
        .sort_values(["wins", "team"], ascending=[False, True])
        .set_index("team")
    )


def result_margin_counts(matches: pd.DataFrame) -> pd.DataFrame:
    """Count resolved wins by margin; ties and no-results are not losses."""
    resolved = matches.loc[matches["outcome_winner"].notna()]
    runs = pd.to_numeric(resolved["outcome_by_runs"], errors="coerce")
    wickets = pd.to_numeric(resolved["outcome_by_wickets"], errors="coerce")
    return pd.DataFrame(
        {
            "result": ["Batting first (by runs)", "Chasing (by wickets)"],
            "matches": [int(runs.gt(0).sum()), int(wickets.gt(0).sum())],
        }
    )


def year_match_run_summary(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> pd.DataFrame:
    """Combine match counts and delivery runs after aggregating each grain."""
    match_counts = matches.groupby("calendar_year").size().rename("matches")
    regular_deliveries = deliveries.loc[
        pd.to_numeric(deliveries["is_super_over"], errors="coerce").fillna(0).eq(0)
    ]
    run_totals = regular_deliveries.groupby("calendar_year")["total_runs"].sum()
    result = pd.concat([match_counts, run_totals.rename("runs")], axis=1).fillna(0)
    return result.reset_index().sort_values("calendar_year")


def head_to_head_summary(
    matches: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> dict[str, int]:
    if team_a == team_b:
        raise ValueError("Choose two different teams for a head-to-head comparison.")
    pair_matches = matches.loc[
        (matches["team1"].eq(team_a) & matches["team2"].eq(team_b))
        | (matches["team1"].eq(team_b) & matches["team2"].eq(team_a))
    ]
    has_winner = pair_matches["outcome_winner"].notna() & pair_matches[
        "outcome_winner"
    ].astype("string").str.strip().ne("")
    result_label = pair_matches["outcome_result"].fillna("").astype("string").str.lower()
    tied = pair_matches.get("is_tie", pd.Series(False, index=pair_matches.index)).fillna(False).astype(bool)
    no_result = pair_matches.get("is_no_result", pd.Series(False, index=pair_matches.index)).fillna(False).astype(bool)
    ties = tied | result_label.eq("tie")
    no_results = no_result | result_label.eq("no result")
    return {
        "matches": int(pair_matches["match_id"].nunique()),
        "team_a_wins": int((has_winner & pair_matches["outcome_winner"].eq(team_a)).sum()),
        "team_b_wins": int((has_winner & pair_matches["outcome_winner"].eq(team_b)).sum()),
        "ties": int((ties & ~no_results).sum()),
        "no_results": int(no_results.sum()),
    }


def regular_innings_deliveries(deliveries: pd.DataFrame) -> pd.DataFrame:
    is_regular = pd.to_numeric(deliveries["is_super_over"], errors="coerce").fillna(0).eq(0)
    innings = pd.to_numeric(deliveries["innings"], errors="coerce").isin([1, 2])
    return deliveries.loc[is_regular & innings].copy()


def innings_scorecard(deliveries: pd.DataFrame, match_id: object) -> pd.DataFrame:
    match_deliveries = regular_innings_deliveries(deliveries)
    match_deliveries = match_deliveries.loc[match_deliveries["match_id"].eq(match_id)]
    if match_deliveries.empty:
        return pd.DataFrame(
            columns=["innings", "batting_team", "runs", "wickets", "legal_balls"]
        )
    scorecard = (
        match_deliveries.groupby(["innings", "batting_team"], as_index=False)
        .agg(
            runs=("total_runs", "sum"),
            wickets=("is_wicket", "sum"),
            legal_balls=("is_legal_delivery", "sum"),
        )
        .sort_values("innings")
    )
    scorecard["overs_bowled"] = scorecard["legal_balls"] / 6
    return scorecard


def over_scorecard(deliveries: pd.DataFrame, match_id: object) -> pd.DataFrame:
    match_deliveries = regular_innings_deliveries(deliveries)
    match_deliveries = match_deliveries.loc[match_deliveries["match_id"].eq(match_id)]
    if match_deliveries.empty:
        return pd.DataFrame(columns=["innings", "over_number", "runs", "wickets"])
    return (
        match_deliveries.assign(
            over_number=pd.to_numeric(match_deliveries["over"], errors="coerce") + 1
        )
        .groupby(["innings", "over_number"], as_index=False)
        .agg(runs=("total_runs", "sum"), wickets=("is_wicket", "sum"))
        .sort_values(["innings", "over_number"])
    )


def team_innings_averages(deliveries: pd.DataFrame) -> pd.DataFrame:
    innings = regular_innings_deliveries(deliveries)
    if innings.empty:
        return pd.DataFrame(
            columns=["team", "innings_batted", "average_runs_scored", "innings_bowled", "average_runs_conceded"]
        )
    innings_totals = (
        innings.groupby(["match_id", "innings", "batting_team", "bowling_team"], as_index=False)
        .agg(runs=("total_runs", "sum"))
    )
    batting = (
        innings_totals.groupby("batting_team", as_index=False)
        .agg(innings_batted=("runs", "size"), average_runs_scored=("runs", "mean"))
        .rename(columns={"batting_team": "team"})
    )
    bowling = (
        innings_totals.groupby("bowling_team", as_index=False)
        .agg(innings_bowled=("runs", "size"), average_runs_conceded=("runs", "mean"))
        .rename(columns={"bowling_team": "team"})
    )
    return batting.merge(bowling, on="team", how="outer").sort_values("team")


def team_batting_order_summary(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize resolved team wins by actual batting-first/chase innings order."""
    resolved = matches.loc[
        matches["outcome_winner"].notna()
        & matches["outcome_winner"].astype("string").str.strip().ne("")
    ]
    first_innings = regular_innings_deliveries(deliveries)
    first_innings = first_innings.loc[
        pd.to_numeric(first_innings["innings"], errors="coerce").eq(1)
    ]
    first_batting_team = (
        first_innings.groupby("match_id")["batting_team"]
        .agg(lambda values: values.iloc[0] if not values.empty else None)
        .rename("first_batting_team")
    )
    resolved = resolved.join(first_batting_team, on="match_id", how="inner")
    if resolved.empty:
        return pd.DataFrame(
            columns=["team", "batting_first_matches", "batting_first_wins", "chase_matches", "chase_wins"]
        )

    team_rows = pd.concat(
        [
            resolved.assign(team=resolved["team1"]),
            resolved.assign(team=resolved["team2"]),
        ],
        ignore_index=True,
    )
    team_rows["won"] = team_rows["outcome_winner"].eq(team_rows["team"])
    team_rows["batted_first"] = team_rows["team"].eq(team_rows["first_batting_team"])
    summary = (
        team_rows.groupby("team")
        .apply(
            lambda group: pd.Series(
                {
                    "batting_first_matches": int(group["batted_first"].sum()),
                    "batting_first_wins": int((group["batted_first"] & group["won"]).sum()),
                    "chase_matches": int((~group["batted_first"]).sum()),
                    "chase_wins": int((~group["batted_first"] & group["won"]).sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    summary["batting_first_win_percent"] = (
        100 * summary["batting_first_wins"] / summary["batting_first_matches"].replace(0, pd.NA)
    )
    summary["chase_win_percent"] = (
        100 * summary["chase_wins"] / summary["chase_matches"].replace(0, pd.NA)
    )
    return summary.sort_values("team")


def venue_result_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Report venue counts and result rates with explicit resolved denominators."""
    data = matches.copy()
    data["runs_margin"] = pd.to_numeric(data["outcome_by_runs"], errors="coerce")
    data["wickets_margin"] = pd.to_numeric(data["outcome_by_wickets"], errors="coerce")
    data["has_winner"] = data["outcome_winner"].notna() & data[
        "outcome_winner"
    ].astype("string").str.strip().ne("")
    data["batting_first_win"] = data["has_winner"] & data["runs_margin"].gt(0)
    data["chase_win"] = data["has_winner"] & data["wickets_margin"].gt(0)
    data["is_tie_record"] = data["is_tie"].fillna(False).astype(bool) | data[
        "outcome_result"
    ].eq("tie")
    data["is_no_result_record"] = data["is_no_result"].fillna(False).astype(bool) | data[
        "outcome_result"
    ].eq("no result")
    summary = (
        data.groupby("venue", as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            resolved_matches=("has_winner", "sum"),
            batting_first_wins=("batting_first_win", "sum"),
            chase_wins=("chase_win", "sum"),
            ties=("is_tie_record", "sum"),
            no_results=("is_no_result_record", "sum"),
            average_first_innings_score=("inn1_total_runs", "mean"),
        )
    )
    denominator = summary["resolved_matches"].replace(0, pd.NA)
    summary["batting_first_win_percent"] = 100 * summary["batting_first_wins"] / denominator
    summary["chase_win_percent"] = 100 * summary["chase_wins"] / denominator
    return summary.sort_values(["matches", "venue"], ascending=[False, True])