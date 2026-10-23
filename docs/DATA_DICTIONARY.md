# Data Dictionary & Schema Documentation

## 1. Overview & Grain Specification

The IPL INSIGHT analytics database (`data/processed/ipl_analytics.sqlite`) is structured around three primary tables:

| Table | Rows | Columns | Grain | Natural Key |
|---|---|---|---|---|
| `ipl_matches_summary` | 1,243 | 59 | One row per match | `match_id` |
| `ipl_ball_by_ball_processed` | 295,732 | 108 | One row per delivery | `match_id` + `innings` + `ball_seq_idx` |
| `ipl_player_profiles` | 811 | 41 | One row per player | `player_name` |

---

## 2. Table: `ipl_matches_summary`

Contains match-level facts, dates, teams, venues, toss details, and official outcomes.

| Column Name | Data Type | Nullable | Description & Analytical Rules |
|---|---|---|---|
| `match_id` | String | No | Primary key. Unique string identifier joining to deliveries. |
| `date` | Timestamp | No | Match calendar date (YYYY-MM-DD). Range: 2008-04-18 to 2026-05-31. |
| `calendar_year` | Integer | No | Derived from `date`. Authoritative field for time filtering. |
| `season` | String | Yes | Source label. Inconsistent (e.g. '2007' covers 2008 dates). Retained for provenance only. |
| `match_stage` | String | Yes | Tournament phase (e.g. Group, Qualifier, Eliminator, Final). |
| `team1` | String | No | Franchise designated as Team 1. |
| `team2` | String | No | Franchise designated as Team 2. |
| `venue` | String | No | Official stadium name. |
| `toss_winner` | String | Yes | Franchise that won the coin toss. |
| `toss_decision` | String | Yes | 'bat' or 'field'. |
| `toss_winner_is_winner` | Boolean | Yes | True if toss-winning team won the match. |
| `outcome_winner` | String | Yes | Name of victorious franchise. Blank/Null for ties and no-results. |
| `outcome_by_runs` | Float | Yes | Victory margin in runs (batting-first victory). |
| `outcome_by_wickets` | Float | Yes | Victory margin in wickets (chasing victory). |
| `outcome_result` | String | Yes | 'tie' or 'no result' when no outright winner was recorded in regular play. |
| `player_of_match` | String | Yes | Official Player of the Match awardee. |
| `inn1_total_runs` | Float | Yes | Total runs scored in first innings. |
| `inn2_target_runs` | Float | Yes | Target score set for second innings. |
| `team1_won` | Integer | Yes | 1 if team1 won, 0 if lost. Null for ties and no-results. |

---

## 3. Table: `ipl_ball_by_ball_processed`

Contains granular delivery events, run breakdowns, dismissals, and cumulative chase states.

| Column Name | Data Type | Nullable | Description & Analytical Rules |
|---|---|---|---|
| `match_id` | String | No | Foreign key linking to `ipl_matches_summary.match_id`. |
| `innings` | Integer | No | Innings number (1 or 2 for regular innings; 3+ for super overs). |
| `over` | Integer | No | 0-based over index (0 to 19). Display cricket over uses `over + 1`. |
| `ball_in_over` | Integer | No | Ball sequence index within the specific over. |
| `ball_seq_idx` | Integer | No | Cumulative delivery index in the match. |
| `batting_team` | String | No | Batting franchise. |
| `bowling_team` | String | No | Fielding / bowling franchise. |
| `batter` | String | No | Striker player name. |
| `bowler` | String | No | Bowler player name. |
| `non_striker` | String | Yes | Non-striker player name. |
| `runs_batter` | Integer | No | Runs off the bat (0 to 6). |
| `runs_extras` | Integer | No | Extras awarded (wides, no-balls, byes, leg-byes). |
| `total_runs` | Integer | No | `runs_batter + runs_extras`. Reconciles to exactly 401,738 total tournament runs. |
| `is_legal_delivery` | Boolean | No | False for wides and no-balls; True for legitimate deliveries. |
| `is_wicket` | Boolean | No | True for any dismissal (14,686 tournament total). |
| `is_bowler_wicket` | Boolean | No | True only for bowler-credited dismissals (excludes run-outs, retired out). Total: 13,482. |
| `player_out` | String | Yes | Player dismissed on this ball. |
| `dismissal_kind` | String | Yes | Mode of dismissal ('caught', 'bowled', 'run out', 'lbw', etc.). |
| `current_score` | Integer | Yes | Cumulative innings runs completed after this delivery. |
| `current_wickets` | Integer | Yes | Cumulative wickets fallen after this delivery. |
| `wickets_in_hand` | Integer | Yes | `10 - current_wickets`. |
| `legal_balls_bowled` | Integer | Yes | Cumulative legal balls completed in innings. |
| `balls_remaining` | Integer | Yes | `max_innings_balls - legal_balls_bowled`. |
| `runs_needed` | Integer | Yes | Target minus current score (second innings). |
| `current_run_rate` | Float | Yes | Current scoring rate per 6 legal balls. |
| `required_run_rate` | Float | Yes | Required scoring rate per 6 remaining legal balls. |
| `is_chasing_winner` | Integer | Yes | Target label for live chase model. 1 if chase succeeded, 0 if failed. |

---

## 4. Table: `ipl_player_profiles`

Contains career totals and pre-calculated performance rates across all covered IPL seasons.

| Column Name | Data Type | Description |
|---|---|---|
| `player_name` | String | Primary display key (e.g. 'V Kohli', 'MS Dhoni', 'JJ Bumrah'). |
| `cricsheet_id` | String | Identifier from source data (note: 3 IDs map to multiple names, so `player_name` is used as primary key). |
| `primary_role` | String | 'Batter', 'Bowler', 'All-Rounder', 'Wicketkeeper Batter'. |
| `bat_style` | String | 'Right hand bat', 'Left hand bat'. |
| `bowl_style` | String | Bowling style classification (e.g. 'Right arm fast', 'Slow left arm orthodox'). |
| `matches_selected`| Integer | Career match appearances. |
| `runs_scored` | Integer | Career aggregate batting runs. |
| `balls_faced` | Integer | Career deliveries faced as batter. |
| `batting_average` | Float | Runs scored per dismissal. |
| `batting_strike_rate`| Float | Runs per 100 balls faced. |
| `fours` / `sixes`| Integer | Career boundary counts. |
| `boundary_run_pct`| Float | Percentage of runs scored in boundaries. |
| `powerplay_strike_rate`| Float | Strike rate during overs 1–6. |
| `death_strike_rate`| Float | Strike rate during overs 17–20. |
| `sr_vs_pace` / `sr_vs_spin`| Float | Strike rate split against pace and spin bowling. |
| `wickets` | Integer | Career bowler-credited dismissals. |
| `bowling_economy`| Float | Runs conceded per 6 legal balls bowled. |
| `bowling_average`| Float | Runs conceded per wicket. |
| `bowling_strike_rate`| Float | Legal balls bowled per wicket. |
