from collections.abc import Collection

import numpy as np
import pandas as pd


TARGET_COLUMN = "is_chasing_winner"
FEATURE_COLUMNS = [
    "batting_team",
    "bowling_team",
    "venue",
    "target_score",
    "current_score",
    "runs_needed",
    "wickets_lost",
    "wickets_remaining",
    "legal_balls_bowled",
    "balls_remaining",
    "current_run_rate",
    "required_run_rate",
]

REQUIRED_COLUMNS = {
    "match_id",
    "date",
    "innings",
    "is_super_over",
    "is_dls_affected",
    "is_shortened_match",
    "is_innings_ended",
    TARGET_COLUMN,
    "target_runs",
    "current_score",
    "current_wickets",
    "legal_balls_bowled",
    "max_innings_balls",
    "batting_team",
    "bowling_team",
    "venue",
}

BOOLEAN_VALUES = {
    "true": True,
    "1": True,
    "1.0": True,
    "yes": True,
    "false": False,
    "0": False,
    "0.0": False,
    "no": False,
}


def _parse_boolean(series: pd.Series, column_name: str) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    present_values = set(values.dropna()) - {""}
    unexpected_values = present_values - BOOLEAN_VALUES.keys()
    if unexpected_values:
        raise ValueError(
            f"Unexpected values in {column_name}: {sorted(unexpected_values)}"
        )
    return values.map(BOOLEAN_VALUES).astype("boolean")


def build_chase_dataset(deliveries: pd.DataFrame) -> pd.DataFrame:
    """Create post-delivery, nonterminal live-chase examples from raw rows."""
    missing_columns = REQUIRED_COLUMNS - set(deliveries.columns)
    if missing_columns:
        raise ValueError(f"Delivery data is missing columns: {sorted(missing_columns)}")

    states = deliveries.copy()
    states["date"] = pd.to_datetime(states["date"], errors="coerce")
    states["innings"] = pd.to_numeric(states["innings"], errors="coerce")

    flag_columns = [
        "is_super_over",
        "is_dls_affected",
        "is_shortened_match",
        "is_innings_ended",
        TARGET_COLUMN,
    ]
    for column_name in flag_columns:
        states[column_name] = _parse_boolean(states[column_name], column_name)

    numeric_columns = [
        "target_runs",
        "current_score",
        "current_wickets",
        "legal_balls_bowled",
        "max_innings_balls",
    ]
    for column_name in numeric_columns:
        states[column_name] = pd.to_numeric(states[column_name], errors="coerce")

    eligible = states.loc[
        states["innings"].eq(2)
        & ~states["is_super_over"].fillna(False)
        & ~states["is_dls_affected"].fillna(False)
        & ~states["is_shortened_match"].fillna(False)
        & ~states["is_innings_ended"].fillna(False)
        & states[TARGET_COLUMN].notna()
    ].copy()

    required_state = [
        "match_id",
        "date",
        "target_runs",
        "current_score",
        "current_wickets",
        "legal_balls_bowled",
        "max_innings_balls",
        "batting_team",
        "bowling_team",
        "venue",
    ]
    eligible = eligible.dropna(subset=required_state)
    eligible = eligible.loc[
        eligible["current_wickets"].between(0, 10)
        & eligible["legal_balls_bowled"].ge(0)
        & eligible["max_innings_balls"].gt(0)
        & eligible["target_runs"].gt(0)
    ].copy()

    label_counts = eligible.groupby("match_id")[TARGET_COLUMN].nunique(dropna=True)
    conflicting_matches = label_counts[label_counts > 1]
    if not conflicting_matches.empty:
        raise ValueError(
            "Chase outcome label changes within match IDs: "
            f"{conflicting_matches.index[:5].tolist()}"
        )

    eligible["target_score"] = eligible["target_runs"]
    eligible["runs_needed"] = (
        eligible["target_score"] - eligible["current_score"]
    ).clip(lower=0)
    eligible["wickets_lost"] = eligible["current_wickets"]
    eligible["wickets_remaining"] = 10 - eligible["wickets_lost"]
    eligible["balls_remaining"] = (
        eligible["max_innings_balls"] - eligible["legal_balls_bowled"]
    )

    eligible = eligible.loc[eligible["balls_remaining"].gt(0)].copy()
    legal_balls = eligible["legal_balls_bowled"].replace(0, np.nan)
    eligible["current_run_rate"] = eligible["current_score"] * 6 / legal_balls
    eligible["required_run_rate"] = (
        eligible["runs_needed"] * 6 / eligible["balls_remaining"]
    )

    eligible["calendar_year"] = eligible["date"].dt.year.astype("int16")
    eligible[TARGET_COLUMN] = eligible[TARGET_COLUMN].astype("int8")
    eligible["match_id"] = eligible["match_id"].astype("string")

    metadata_columns = ["match_id", "date", "calendar_year", "ball_seq_idx"]
    available_metadata = [column for column in metadata_columns if column in eligible]
    output_columns = list(dict.fromkeys(available_metadata + FEATURE_COLUMNS + [TARGET_COLUMN]))
    return eligible[output_columns].reset_index(drop=True)


def load_chase_dataset(csv_path: str, chunksize: int = 50_000) -> pd.DataFrame:
    """Read only the columns needed from the source delivery CSV."""
    source_columns = sorted(REQUIRED_COLUMNS | {"ball_seq_idx"})
    chunks = pd.read_csv(
        csv_path,
        usecols=lambda column: column in source_columns,
        chunksize=chunksize,
        low_memory=False,
    )
    feature_chunks = [build_chase_dataset(chunk) for chunk in chunks]
    if not feature_chunks:
        return pd.DataFrame(columns=["match_id", "date", "calendar_year", *FEATURE_COLUMNS, TARGET_COLUMN])
    dataset = pd.concat(feature_chunks, ignore_index=True)
    labels_per_match = dataset.groupby("match_id")[TARGET_COLUMN].nunique()
    if labels_per_match.gt(1).any():
        raise ValueError("Chase outcome label changes within a match across CSV chunks.")
    dates_per_match = dataset.groupby("match_id")["date"].nunique()
    if dates_per_match.gt(1).any():
        raise ValueError("A match_id maps to multiple dates across CSV chunks.")
    return dataset


def validate_feature_columns(feature_columns: Collection[str]) -> None:
    """Reject target/result fields if someone expands the model input list."""
    forbidden_tokens = (
        "winner",
        "final_",
        "innings_end",
        "is_chasing_winner",
        "match_winner",
        "outcome_",
        "batting_team_won",
    )
    forbidden = [
        column
        for column in feature_columns
        if any(token in column for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"Target leakage columns are forbidden as inputs: {forbidden}")