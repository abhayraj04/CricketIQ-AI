import pandas as pd


def chronological_match_splits(
    dataset: pd.DataFrame,
    validation_start: str = "2020-01-01",
    test_start: str = "2024-01-01",
) -> dict[str, pd.DataFrame]:
    """Split date-ordered states while keeping each match in one partition."""
    required_columns = {"match_id", "date"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing split columns: {sorted(missing_columns)}")
    if dataset["match_id"].isna().any() or dataset["date"].isna().any():
        raise ValueError("Match IDs and dates must be present before splitting.")

    validation_date = pd.Timestamp(validation_start)
    test_date = pd.Timestamp(test_start)
    if validation_date >= test_date:
        raise ValueError("validation_start must be earlier than test_start.")

    dates_per_match = dataset.groupby("match_id")["date"].nunique()
    if dates_per_match.gt(1).any():
        raise ValueError("Every match_id must map to exactly one date.")

    dates = pd.to_datetime(dataset["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("All split dates must be valid dates.")

    splits = {
        "train": dataset.loc[dates < validation_date].copy(),
        "validation": dataset.loc[(dates >= validation_date) & (dates < test_date)].copy(),
        "test": dataset.loc[dates >= test_date].copy(),
    }

    match_sets = {name: set(frame["match_id"]) for name, frame in splits.items()}
    if (
        match_sets["train"] & match_sets["validation"]
        or match_sets["train"] & match_sets["test"]
        or match_sets["validation"] & match_sets["test"]
    ):
        raise RuntimeError("A match_id appears in more than one chronological split.")
    return splits