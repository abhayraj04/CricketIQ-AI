from pathlib import Path
import sqlite3

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_DATA_DIR = PROJECT_ROOT / "parquet"
DATABASE_PATH = PROCESSED_DATA_DIR / "ipl_analytics.sqlite"
CHUNK_SIZE = 20_000

DATA_FILES = {
    "ipl_ball_by_ball_processed": "ipl_ball_by_ball_processed.csv",
    "ipl_matches_summary": "ipl_matches_summary.csv",
    "ipl_player_profiles": "ipl_player_profiles.csv",
    "ipl_player_summary": "ipl_player_summary.csv",
}

REQUIRED_COLUMNS = {
    "ipl_ball_by_ball_processed": {"match_id", "innings", "total_runs", "date", "calendar_year"},
    "ipl_matches_summary": {"match_id", "team1", "team2", "outcome_winner", "calendar_year"},
    "ipl_player_profiles": {"player_name", "runs_scored", "wickets"},
    "ipl_player_summary": {"player_name", "runs_scored", "wickets"},
}


def load_table(
    connection: sqlite3.Connection,
    table_name: str,
    file_name: str,
) -> int:
    csv_path = PROCESSED_DATA_DIR / file_name
    parquet_path = PARQUET_DATA_DIR / f"{table_name}.parquet"

    # Fast-load directly from compressed parquet if present (optimal for cloud deployments)
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "calendar_year" not in df.columns and "date" in df.columns:
            df["calendar_year"] = pd.to_datetime(df["date"]).dt.year.astype("int16")
        missing_columns = REQUIRED_COLUMNS[table_name] - set(df.columns)
        if missing_columns:
            raise ValueError(f"{parquet_path.name} is missing columns: {sorted(missing_columns)}")
        df.to_sql(table_name, connection, if_exists="replace", index=False, chunksize=25_000)
        return len(df)

    if not csv_path.exists():
        raise FileNotFoundError(f"Neither processed CSV ({csv_path}) nor Parquet ({parquet_path}) was found.")

    column_names = set(pd.read_csv(csv_path, nrows=0).columns)
    if "calendar_year" not in column_names and "date" in column_names:
        column_names.add("calendar_year")

    missing_columns = REQUIRED_COLUMNS[table_name] - column_names
    if missing_columns:
        raise ValueError(
            f"{file_name} is missing required columns: {sorted(missing_columns)}"
        )

    total_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False)
    ):
        if "calendar_year" not in chunk.columns and "date" in chunk.columns:
            chunk["calendar_year"] = pd.to_datetime(chunk["date"]).dt.year.astype("int16")
        chunk.to_sql(
            table_name,
            connection,
            if_exists="replace" if chunk_number == 0 else "append",
            index=False,
            chunksize=CHUNK_SIZE,
        )
        total_rows += len(chunk)

    return total_rows


def ensure_database_ready(target_db_path: Path = DATABASE_PATH) -> Path:
    """Ensure SQLite database exists; builds automatically if missing."""
    target_path = Path(target_db_path)
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target_path)
    try:
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA journal_mode = MEMORY")
        for table_name, file_name in DATA_FILES.items():
            load_table(connection, table_name, file_name)
        connection.commit()
        integrity_result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity_result != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity_result}")
    finally:
        connection.close()
    return target_path


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    try:
        for table_name, file_name in DATA_FILES.items():
            row_count = load_table(connection, table_name, file_name)
            print(f"Loaded {row_count:,} rows into {table_name}")

        integrity_result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity_result != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity_result}")
    finally:
        connection.close()

    print(f"SQLite database ready: {DATABASE_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
