from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATA_FILES = {
    "ipl_ball_by_ball_processed": "ipl_ball_by_ball_processed.csv",
    "ipl_matches_summary": "ipl_matches_summary.csv",
    "ipl_player_profiles": "ipl_player_profiles.csv",
    "ipl_player_summary": "ipl_player_summary.csv",
}

ID_COLUMNS = {
    "match_id",
    "batter_id",
    "bowler_id",
    "non_striker_id",
    "cricsheet_id",
}

REQUIRED_COLUMNS = {
    "ipl_ball_by_ball_processed": {"match_id", "date", "season", "innings", "ball_seq_idx"},
    "ipl_matches_summary": {"match_id", "date", "season", "outcome_winner"},
    "ipl_player_profiles": {"player_name", "cricsheet_id"},
    "ipl_player_summary": {"player_name", "cricsheet_id"},
}


def is_boolean_column(column_name: str) -> bool:
    """Identify flags by their names so numeric counts are not treated as flags."""
    return (
        column_name.startswith(("is_", "has_"))
        or column_name.endswith(("_won", "_is_winner", "_is_toss_winner"))
    )


def clean_table(table_name: str, file_name: str) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, int]]:
    source_path = RAW_DATA_DIR / file_name
    if not source_path.exists():
        raise FileNotFoundError(f"Expected source file was not found: {source_path}")

    data = pd.read_csv(source_path, low_memory=False)
    missing_columns = REQUIRED_COLUMNS[table_name] - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"{file_name} is missing required columns: {sorted(missing_columns)}"
        )

    original_row_count = len(data)
    duplicate_rows = int(data.duplicated().sum())
    duplicate_keys = 0
    duplicate_profile_ids = 0

    if table_name == "ipl_ball_by_ball_processed":
        key_columns = ["match_id", "innings", "ball_seq_idx"]
        duplicate_keys = int(data.duplicated(subset=key_columns).sum())
    elif table_name == "ipl_matches_summary":
        duplicate_keys = int(data.duplicated(subset=["match_id"]).sum())
    elif table_name in {"ipl_player_profiles", "ipl_player_summary"}:
        duplicate_keys = int(data.duplicated(subset=["player_name"]).sum())
        duplicate_profile_ids = int(data["cricsheet_id"].duplicated().sum())

    # Identifiers are labels, not measurements. Keeping them as text avoids
    # accidental arithmetic and preserves IDs that may contain letters.
    for column_name in ID_COLUMNS.intersection(data.columns):
        data[column_name] = data[column_name].astype("string")

    # Convert known calendar dates and retain the source season label unchanged.
    date_columns = [column for column in ("date", "match_date") if column in data.columns]
    invalid_dates = 0
    for column_name in date_columns:
        parsed_dates = pd.to_datetime(data[column_name], errors="coerce")
        invalid_dates += int((data[column_name].notna() & parsed_dates.isna()).sum())
        data[column_name] = parsed_dates

    # Convert explicit flags, including flags stored as 1.0 and 0.0.
    boolean_values = {
        "true": True,
        "1": True,
        "1.0": True,
        "yes": True,
        "false": False,
        "0": False,
        "0.0": False,
        "no": False,
    }
    for column_name in data.columns:
        if not is_boolean_column(column_name):
            continue
        values = data[column_name].astype("string").str.strip().str.lower()
        non_empty_values = set(values.dropna()) - {""}
        unexpected_values = non_empty_values - boolean_values.keys()
        if unexpected_values:
            raise ValueError(
                f"Unexpected values in flag column {column_name}: "
                f"{sorted(unexpected_values)}"
            )
        data[column_name] = values.map(boolean_values).astype("boolean")

    # Convert a column to numbers only when every nonblank value is numeric.
    excluded_columns = ID_COLUMNS | set(date_columns)
    for column_name in data.columns:
        if column_name in excluded_columns or is_boolean_column(column_name):
            continue
        original_values = data[column_name]
        converted_values = pd.to_numeric(original_values, errors="coerce")
        nonblank_values = original_values.notna() & original_values.astype("string").str.strip().ne("")
        if nonblank_values.any() and converted_values[nonblank_values].notna().all():
            data[column_name] = converted_values

    # Calendar year comes from the actual match date, not the unreliable season label.
    if "date" in data.columns:
        data["calendar_year"] = data["date"].dt.year.astype("Int64")

    quality_rows: list[dict[str, object]] = []
    for column_name in data.columns:
        missing_count = int(data[column_name].isna().sum())
        quality_rows.append(
            {
                "table": table_name,
                "column": column_name,
                "data_type": str(data[column_name].dtype),
                "missing_count": missing_count,
                "missing_percent": round(missing_count / len(data) * 100, 2) if len(data) else 0,
            }
        )

    diagnostics = {
        "rows": original_row_count,
        "columns": len(data.columns),
        "exact_duplicate_rows": duplicate_rows,
        "duplicate_key_rows": duplicate_keys,
        "duplicate_profile_id_rows": duplicate_profile_ids,
        "invalid_dates": invalid_dates,
    }
    return data, quality_rows, diagnostics


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    quality_report: list[dict[str, object]] = []

    for table_name, file_name in DATA_FILES.items():
        cleaned_data, quality_rows, diagnostics = clean_table(table_name, file_name)
        output_path = PROCESSED_DATA_DIR / file_name
        cleaned_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
        quality_report.extend(quality_rows)

        print(
            f"{file_name}: {diagnostics['rows']:,} rows, "
            f"{diagnostics['columns']} columns, "
            f"{diagnostics['exact_duplicate_rows']} exact duplicate rows, "
            f"{diagnostics['duplicate_key_rows']} duplicate key rows, "
            f"{diagnostics['duplicate_profile_id_rows']} duplicate profile ID rows, "
            f"{diagnostics['invalid_dates']} invalid dates"
        )
        print(f"  Saved processed copy: {output_path.relative_to(PROJECT_ROOT)}")

    report_path = PROCESSED_DATA_DIR / "data_quality_report.csv"
    pd.DataFrame(quality_report).to_csv(report_path, index=False)
    print(f"Data quality report saved: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()