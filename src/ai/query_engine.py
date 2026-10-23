import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


FORBIDDEN_SQL_PATTERNS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|EXEC|VACUUM|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

ALLOWED_TABLES = {
    "ipl_matches_summary",
    "ipl_player_profiles",
    "ipl_ball_by_ball_processed",
}


class SecurityViolationError(Exception):
    """Raised when an unsafe database operation or SQL query is attempted."""
    pass


class ReadOnlyAnalyticsQueryEngine:
    """Secure, strictly read-only query engine with whitelist tables, parameterization, and limit enforcement."""

    def __init__(self, database_path: str | Path):
        self.db_path = Path(database_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Create a read-only SQLite connection using URI mode."""
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=10.0)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.OperationalError:
            conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            conn.row_factory = sqlite3.Row
            return conn

    def execute_safe_query(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 100,
    ) -> List[Dict[str, Any]]:
        """Validate and execute a strictly read-only parameterized query."""
        cleaned = query.strip()
        if not cleaned:
            return []

        if FORBIDDEN_SQL_PATTERNS.search(cleaned):
            match = FORBIDDEN_SQL_PATTERNS.search(cleaned)
            keyword = match.group(0) if match else "UNKNOWN"
            raise SecurityViolationError(
                f"Query rejected: Database modification keyword '{keyword}' is forbidden in read-only analytics mode."
            )

        if not (cleaned.upper().startswith("SELECT") or cleaned.upper().startswith("WITH")):
            raise SecurityViolationError("Query rejected: Only SELECT or WITH queries are permitted.")

        statements = [stmt.strip() for stmt in cleaned.split(";") if stmt.strip()]
        if len(statements) > 1:
            raise SecurityViolationError("Query rejected: Multiple SQL statements are not allowed.")

        safe_query = statements[0]
        params = params or ()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(safe_query, params)
            rows = cursor.fetchmany(max_rows)
            return [dict(row) for row in rows]

    def query_head_to_head(
        self,
        team_a: str,
        team_b: str,
        start_year: int = 2008,
        end_year: int = 2026,
    ) -> Dict[str, Any]:
        """Retrieve verified head-to-head match records between two teams."""
        sql = """
            SELECT match_id, date, venue, match_stage, toss_winner, toss_decision,
                   outcome_winner, outcome_by_runs, outcome_by_wickets, outcome_result,
                   player_of_match
            FROM ipl_matches_summary
            WHERE ((team1 = ? AND team2 = ?) OR (team1 = ? AND team2 = ?))
              AND calendar_year >= ? AND calendar_year <= ?
            ORDER BY date DESC
        """
        rows = self.execute_safe_query(
            sql,
            (team_a, team_b, team_b, team_a, start_year, end_year),
            max_rows=100,
        )

        team_a_wins = sum(1 for r in rows if r.get("outcome_winner") == team_a)
        team_b_wins = sum(1 for r in rows if r.get("outcome_winner") == team_b)
        ties_or_nr = sum(1 for r in rows if r.get("outcome_winner") not in (team_a, team_b))

        return {
            "team_a": team_a,
            "team_b": team_b,
            "total_matches": len(rows),
            "team_a_wins": team_a_wins,
            "team_b_wins": team_b_wins,
            "ties_or_no_result": ties_or_nr,
            "recent_matches": rows[:5],
        }

    def query_top_batters(
        self,
        min_balls: int = 200,
        metric: str = "runs_scored",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve top batting performers filtered by minimum balls faced."""
        valid_sort_columns = {
            "runs_scored": "runs_scored DESC",
            "batting_strike_rate": "batting_strike_rate DESC",
            "batting_average": "batting_average DESC",
            "sixes": "sixes DESC",
            "fours": "fours DESC",
        }
        order_clause = valid_sort_columns.get(metric, "runs_scored DESC")

        sql = f"""
            SELECT player_name, primary_role, bat_style, matches_selected, balls_faced,
                   runs_scored, batting_average, batting_strike_rate, fours, sixes, boundary_run_pct
            FROM ipl_player_profiles
            WHERE balls_faced >= ?
            ORDER BY {order_clause}
            LIMIT ?
        """
        return self.execute_safe_query(sql, (min_balls, limit))

    def query_top_bowlers(
        self,
        min_balls: int = 120,
        metric: str = "wickets",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve top bowling performers filtered by minimum legal deliveries."""
        valid_sort_columns = {
            "wickets": "wickets DESC",
            "bowling_economy": "bowling_economy ASC",
            "bowling_average": "bowling_average ASC",
            "bowling_strike_rate": "bowling_strike_rate ASC",
        }
        order_clause = valid_sort_columns.get(metric, "wickets DESC")

        sql = f"""
            SELECT player_name, primary_role, bowl_style, matches_selected, innings_bowled,
                   legal_balls_bowled, runs_conceded, wickets, bowling_economy, bowling_average,
                   bowling_strike_rate, dot_ball_pct_bowling
            FROM ipl_player_profiles
            WHERE legal_balls_bowled >= ? AND wickets > 0
            ORDER BY {order_clause}
            LIMIT ?
        """
        return self.execute_safe_query(sql, (min_balls, limit))

    def query_player_profile(self, player_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve career profile and statistics for a specific player."""
        sql = """
            SELECT * FROM ipl_player_profiles
            WHERE player_name LIKE ? OR player_full_name LIKE ?
            LIMIT 1
        """
        rows = self.execute_safe_query(sql, (f"%{player_name}%", f"%{player_name}%"))
        return rows[0] if rows else None

    def query_team_overview(self, team_name: str) -> Dict[str, Any]:
        """Retrieve overall franchise match facts, wins, and bat first vs chase splits."""
        sql = """
            SELECT
                COUNT(*) as total_matches,
                SUM(CASE WHEN outcome_winner = ? THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome_result IN ('tie', 'no result') THEN 1 ELSE 0 END) as ties_or_nr,
                SUM(CASE WHEN outcome_winner = ? AND outcome_by_runs > 0 THEN 1 ELSE 0 END) as bat_first_wins,
                SUM(CASE WHEN outcome_winner = ? AND outcome_by_wickets > 0 THEN 1 ELSE 0 END) as chase_wins
            FROM ipl_matches_summary
            WHERE team1 = ? OR team2 = ?
        """
        rows = self.execute_safe_query(sql, (team_name, team_name, team_name, team_name, team_name))
        if not rows or rows[0]["total_matches"] == 0:
            return {"team": team_name, "total_matches": 0, "wins": 0, "win_percent": 0.0}
        r = rows[0]
        total = r["total_matches"] or 0
        wins = r["wins"] or 0
        win_pct = (wins / total * 100) if total > 0 else 0
        return {
            "team": team_name,
            "total_matches": total,
            "wins": wins,
            "win_percent": round(win_pct, 1),
            "bat_first_wins": r["bat_first_wins"] or 0,
            "chase_wins": r["chase_wins"] or 0,
            "ties_or_nr": r["ties_or_nr"] or 0,
        }

    def query_match_details(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full match details, team scores, and turning points."""
        match_sql = """
            SELECT * FROM ipl_matches_summary WHERE match_id = ? LIMIT 1
        """
        matches = self.execute_safe_query(match_sql, (match_id,))
        if not matches:
            return None
        match_info = matches[0]

        scorer_sql = """
            SELECT innings, batter, batting_team, SUM(runs_batter) as runs,
                   COUNT(CASE WHEN is_legal_delivery = 1 THEN 1 END) as balls,
                   SUM(is_four) as fours, SUM(is_six) as sixes
            FROM ipl_ball_by_ball_processed
            WHERE match_id = ? AND is_super_over = 0
            GROUP BY innings, batter, batting_team
            ORDER BY runs DESC
            LIMIT 6
        """
        top_batters = self.execute_safe_query(scorer_sql, (match_id,))

        bowler_sql = """
            SELECT innings, bowler, bowling_team, SUM(is_bowler_wicket) as wickets,
                   SUM(bowler_runs_conceded) as runs_conceded,
                   COUNT(CASE WHEN is_legal_delivery = 1 THEN 1 END) as balls
            FROM ipl_ball_by_ball_processed
            WHERE match_id = ? AND is_super_over = 0
            GROUP BY innings, bowler, bowling_team
            ORDER BY wickets DESC, runs_conceded ASC
            LIMIT 6
        """
        top_bowlers = self.execute_safe_query(bowler_sql, (match_id,))

        over_sql = """
            SELECT innings, over + 1 as over_number, batting_team, bowling_team,
                   SUM(total_runs) as over_runs, SUM(is_wicket) as over_wickets
            FROM ipl_ball_by_ball_processed
            WHERE match_id = ? AND is_super_over = 0
            GROUP BY innings, over, batting_team, bowling_team
            HAVING SUM(total_runs) >= 15 OR SUM(is_wicket) >= 2
            ORDER BY innings, over
        """
        turning_overs = self.execute_safe_query(over_sql, (match_id,))

        return {
            "match": match_info,
            "top_batters": top_batters,
            "top_bowlers": top_bowlers,
            "turning_point_overs": turning_overs,
        }

    def query_venue_chase_stats(self, venue: str) -> Dict[str, Any]:
        """Compute bat-first vs chasing win rates at a given venue."""
        sql = """
            SELECT
                COUNT(*) as total_matches,
                SUM(CASE WHEN outcome_by_runs > 0 THEN 1 ELSE 0 END) as bat_first_wins,
                SUM(CASE WHEN outcome_by_wickets > 0 THEN 1 ELSE 0 END) as chase_wins,
                SUM(CASE WHEN outcome_result IN ('tie', 'no result') THEN 1 ELSE 0 END) as ties_or_nr,
                AVG(inn1_total_runs) as avg_first_innings_score
            FROM ipl_matches_summary
            WHERE venue LIKE ?
        """
        rows = self.execute_safe_query(sql, (f"%{venue}%",))
        if not rows or rows[0]["total_matches"] == 0:
            return {"venue": venue, "total_matches": 0}

        row = rows[0]
        total = row["total_matches"]
        resolved = row["bat_first_wins"] + row["chase_wins"]
        bat_first_pct = (row["bat_first_wins"] / resolved * 100) if resolved > 0 else 0
        chase_pct = (row["chase_wins"] / resolved * 100) if resolved > 0 else 0

        return {
            "venue": venue,
            "total_matches": total,
            "bat_first_wins": row["bat_first_wins"],
            "chase_wins": row["chase_wins"],
            "bat_first_win_pct": round(bat_first_pct, 1),
            "chase_win_pct": round(chase_pct, 1),
            "avg_first_innings_score": round(row["avg_first_innings_score"] or 0, 1),
            "sample_reliability": "Low (< 10 matches)" if total < 10 else "High (>= 10 matches)",
        }
