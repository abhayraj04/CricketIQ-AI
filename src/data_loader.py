from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from src.chase_features import build_chase_dataset


@st.cache_data(show_spinner="Loading IPL data...")
def load_dashboard_data(
    database_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only the columns needed by the current dashboard views."""
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    with sqlite3.connect(path) as connection:
        matches = pd.read_sql_query(
            "SELECT match_id, calendar_year, date, venue, match_stage, team1, team2, "
            "toss_winner, toss_decision, outcome_winner, outcome_by_runs, "
            "outcome_by_wickets, outcome_result, player_of_match, team1_won, "
            "toss_winner_is_winner, is_tie, is_no_result, is_dls_affected, "
            "is_super_over_match, inn1_total_runs, inn2_target_runs, "
            "team1_playing_xi, team2_playing_xi FROM ipl_matches_summary",
            connection,
            parse_dates=["date"],
        )
        players = pd.read_sql_query(
            "SELECT player_name, player_full_name, matches_selected, runs_scored, "
            "is_overseas, bat_style, bowl_style, is_wicketkeeper, field_pos, "
            "balls_faced, batting_average, batting_strike_rate, fours, sixes, "
            "boundary_run_pct, dot_ball_pct_batting, powerplay_strike_rate, "
            "middle_strike_rate, death_strike_rate, runs_vs_pace, sr_vs_pace, "
            "runs_vs_spin, sr_vs_spin, innings_bowled, legal_balls_bowled, "
            "runs_conceded, wickets, bowling_economy, bowling_strike_rate, "
            "bowling_average, dot_ball_pct_bowling, powerplay_economy, "
            "death_overs_economy, primary_role FROM ipl_player_profiles",
            connection,
        )
        deliveries = pd.read_sql_query(
            "SELECT match_id, calendar_year, innings, over, ball_in_over, ball_seq_idx, "
            "is_super_over, total_runs, runs_batter, runs_extras, extra_wides, "
            "extra_noballs, extra_legbyes, extra_byes, is_legal_delivery, is_dot_ball, "
            "is_non_boundary_run, is_four, is_six, is_boundary, is_wicket, "
            "is_bowler_wicket, player_out, dismissal_kind, batter, bowler, "
            "batting_team, bowling_team, striker_bat_style, bowler_type, "
            "bowler_bowl_style, bowler_runs_conceded, tactical_matchup, match_phase, "
            "partnership_id, partnership_runs, target_runs, current_score, "
            "current_wickets, wickets_in_hand, legal_balls_bowled, balls_remaining, "
            "runs_needed, current_run_rate, required_run_rate, max_innings_balls, "
            "is_innings_ended "
            "FROM ipl_ball_by_ball_processed",
            connection,
        )

    return matches, players, deliveries


@st.cache_data(show_spinner="Loading historical chase states...")
def load_chase_examples(database_path: str) -> pd.DataFrame:
    """Load and validate second-innings states only when Prediction Lab opens."""
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")

    columns = [
        "match_id",
        "date",
        "ball_seq_idx",
        "innings",
        "is_super_over",
        "is_dls_affected",
        "is_shortened_match",
        "is_innings_ended",
        "is_chasing_winner",
        "target_runs",
        "current_score",
        "current_wickets",
        "legal_balls_bowled",
        "max_innings_balls",
        "batting_team",
        "bowling_team",
        "venue",
    ]
    with sqlite3.connect(path) as connection:
        raw_states = pd.read_sql_query(
            f"SELECT {', '.join(columns)} FROM ipl_ball_by_ball_processed "
            "WHERE innings = 2",
            connection,
            parse_dates=["date"],
        )
    return build_chase_dataset(raw_states)