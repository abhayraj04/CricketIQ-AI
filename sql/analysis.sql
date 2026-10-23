-- Run these queries against data/processed/ipl_analytics.sqlite.
-- The tables use the real column names in the processed CSV files.

-- 1. Match counts by calendar year. Do not group by the inconsistent source season.
SELECT calendar_year, COUNT(*) AS matches
FROM ipl_matches_summary
GROUP BY calendar_year
ORDER BY calendar_year;

-- 2. Team win counts and win percentages among matches with a recorded winner.
WITH appearances AS (
    SELECT team1 AS team
    FROM ipl_matches_summary
    WHERE outcome_winner IS NOT NULL AND TRIM(outcome_winner) <> ''
    UNION ALL
    SELECT team2 AS team
    FROM ipl_matches_summary
    WHERE outcome_winner IS NOT NULL AND TRIM(outcome_winner) <> ''
),
wins AS (
    SELECT outcome_winner AS team, COUNT(*) AS win_count
    FROM ipl_matches_summary
    WHERE outcome_winner IS NOT NULL AND TRIM(outcome_winner) <> ''
    GROUP BY outcome_winner
)
SELECT
    appearances.team,
    COUNT(*) AS resolved_matches,
    COALESCE(wins.win_count, 0) AS wins,
    ROUND(100.0 * COALESCE(wins.win_count, 0) / COUNT(*), 1) AS win_percent
FROM appearances
LEFT JOIN wins ON wins.team = appearances.team
GROUP BY appearances.team
ORDER BY wins DESC, appearances.team;

-- 3. Rank teams by wins using a window function.
WITH team_wins AS (
    SELECT outcome_winner AS team, COUNT(*) AS wins
    FROM ipl_matches_summary
    WHERE outcome_winner IS NOT NULL AND TRIM(outcome_winner) <> ''
    GROUP BY outcome_winner
)
SELECT
    team,
    wins,
    DENSE_RANK() OVER (ORDER BY wins DESC) AS win_rank
FROM team_wins
ORDER BY win_rank, team;

-- 4. Year-over-year change in match counts using LAG.
WITH yearly_matches AS (
    SELECT calendar_year, COUNT(*) AS matches
    FROM ipl_matches_summary
    GROUP BY calendar_year
)
SELECT
    calendar_year,
    matches,
    LAG(matches) OVER (ORDER BY calendar_year) AS previous_year_matches,
    matches - LAG(matches) OVER (ORDER BY calendar_year) AS change_from_previous_year
FROM yearly_matches
ORDER BY calendar_year;

-- 5. Players with the most Player of the Match awards.
SELECT player_of_match, COUNT(*) AS awards
FROM ipl_matches_summary
WHERE player_of_match IS NOT NULL AND TRIM(player_of_match) <> ''
GROUP BY player_of_match
ORDER BY awards DESC, player_of_match
LIMIT 10;

-- 6. Top career run scorers and wicket takers from the supplied profile table.
SELECT player_name, runs_scored, matches_selected
FROM ipl_player_profiles
ORDER BY runs_scored DESC
LIMIT 10;

SELECT player_name, wickets, matches_selected
FROM ipl_player_profiles
ORDER BY wickets DESC
LIMIT 10;

-- 7. Venues with the most matches.
SELECT venue, COUNT(*) AS matches
FROM ipl_matches_summary
GROUP BY venue
ORDER BY matches DESC, venue
LIMIT 12;

-- 8. Batting-first versus chasing wins by the recorded winning margin.
SELECT
    SUM(CASE WHEN CAST(outcome_by_runs AS INTEGER) > 0 THEN 1 ELSE 0 END) AS wins_by_runs,
    SUM(CASE WHEN CAST(outcome_by_wickets AS INTEGER) > 0 THEN 1 ELSE 0 END) AS wins_by_wickets
FROM ipl_matches_summary
WHERE outcome_winner IS NOT NULL AND TRIM(outcome_winner) <> '';

-- 9. Toss-winning team win percentage by toss decision.
SELECT
    toss_decision,
    COUNT(*) AS resolved_matches,
    ROUND(100.0 * AVG(CASE WHEN toss_winner_is_winner = 1 THEN 1.0 ELSE 0.0 END), 1)
        AS toss_winner_win_percent
FROM ipl_matches_summary
WHERE outcome_winner IS NOT NULL
  AND TRIM(outcome_winner) <> ''
  AND toss_decision IS NOT NULL
  AND toss_winner_is_winner IS NOT NULL
GROUP BY toss_decision
ORDER BY toss_decision;

-- 10. Join delivery-level data to match context and summarize first-innings runs.
WITH innings_scores AS (
    SELECT
        match_id,
        calendar_year,
        batting_team,
        SUM(total_runs) AS innings_runs
    FROM ipl_ball_by_ball_processed
    WHERE innings = 1 AND is_super_over = 0
    GROUP BY match_id, calendar_year, batting_team
)
SELECT
    match_summary.calendar_year,
    innings_scores.batting_team,
    COUNT(*) AS innings_count,
    ROUND(AVG(innings_scores.innings_runs), 1) AS average_first_innings_runs
FROM innings_scores
JOIN ipl_matches_summary AS match_summary
    ON match_summary.match_id = innings_scores.match_id
GROUP BY match_summary.calendar_year, innings_scores.batting_team
ORDER BY match_summary.calendar_year, average_first_innings_runs DESC;