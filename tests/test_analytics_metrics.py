import pandas as pd
import pytest

from src.metrics import (
    head_to_head_summary,
    innings_scorecard,
    over_scorecard,
    team_innings_averages,
    team_batting_order_summary,
    venue_result_summary,
)


def test_head_to_head_keeps_ties_and_no_results_out_of_wins():
    matches = pd.DataFrame(
        [
            {"match_id": "1", "team1": "A", "team2": "B", "outcome_winner": "A", "outcome_result": "", "is_tie": 0, "is_no_result": 0},
            {"match_id": "2", "team1": "B", "team2": "A", "outcome_winner": "B", "outcome_result": "", "is_tie": 0, "is_no_result": 0},
            {"match_id": "3", "team1": "A", "team2": "B", "outcome_winner": None, "outcome_result": "tie", "is_tie": 1, "is_no_result": 0},
            {"match_id": "4", "team1": "B", "team2": "A", "outcome_winner": None, "outcome_result": "no result", "is_tie": 0, "is_no_result": 1},
        ]
    )

    result = head_to_head_summary(matches, "A", "B")

    assert result == {"matches": 4, "team_a_wins": 1, "team_b_wins": 1, "ties": 1, "no_results": 1}
    with pytest.raises(ValueError, match="different teams"):
        head_to_head_summary(matches, "A", "A")


def sample_deliveries():
    return pd.DataFrame(
        [
            {"match_id": "m1", "innings": 1, "over": 0, "batting_team": "A", "bowling_team": "B", "total_runs": 2, "is_wicket": 0, "is_legal_delivery": 1, "is_super_over": 0},
            {"match_id": "m1", "innings": 1, "over": 0, "batting_team": "A", "bowling_team": "B", "total_runs": 4, "is_wicket": 1, "is_legal_delivery": 1, "is_super_over": 0},
            {"match_id": "m1", "innings": 2, "over": 0, "batting_team": "B", "bowling_team": "A", "total_runs": 3, "is_wicket": 0, "is_legal_delivery": 1, "is_super_over": 0},
            {"match_id": "m1", "innings": 3, "over": 0, "batting_team": "A", "bowling_team": "B", "total_runs": 6, "is_wicket": 0, "is_legal_delivery": 1, "is_super_over": 1},
            {"match_id": "m2", "innings": 1, "over": 0, "batting_team": "A", "bowling_team": "B", "total_runs": 12, "is_wicket": 0, "is_legal_delivery": 1, "is_super_over": 0},
        ]
    )


def test_match_scorecards_exclude_super_overs_and_use_display_over_numbers():
    deliveries = sample_deliveries()

    innings = innings_scorecard(deliveries, "m1")
    overs = over_scorecard(deliveries, "m1")

    assert innings["runs"].tolist() == [6, 3]
    assert innings["wickets"].tolist() == [1, 0]
    assert innings["legal_balls"].tolist() == [2, 1]
    assert overs["over_number"].tolist() == [1, 1]
    assert overs["runs"].tolist() == [6, 3]


def test_team_averages_average_match_innings_not_delivery_rows():
    averages = team_innings_averages(sample_deliveries()).set_index("team")

    assert averages.loc["A", "innings_batted"] == 2
    assert averages.loc["A", "average_runs_scored"] == pytest.approx(9)
    assert averages.loc["B", "innings_batted"] == 1
    assert averages.loc["B", "average_runs_scored"] == pytest.approx(3)


def test_venue_rates_use_only_resolved_results_and_report_samples():
    matches = pd.DataFrame(
        [
            {"match_id": "1", "venue": "X", "outcome_winner": "A", "outcome_by_runs": 10, "outcome_by_wickets": 0, "outcome_result": "", "is_tie": 0, "is_no_result": 0, "inn1_total_runs": 170},
            {"match_id": "2", "venue": "X", "outcome_winner": "B", "outcome_by_runs": 0, "outcome_by_wickets": 4, "outcome_result": "", "is_tie": 0, "is_no_result": 0, "inn1_total_runs": 160},
            {"match_id": "3", "venue": "X", "outcome_winner": None, "outcome_by_runs": 0, "outcome_by_wickets": 0, "outcome_result": "tie", "is_tie": 1, "is_no_result": 0, "inn1_total_runs": 150},
        ]
    )

    result = venue_result_summary(matches).iloc[0]

    assert result["matches"] == 3
    assert result["resolved_matches"] == 2
    assert result["batting_first_win_percent"] == pytest.approx(50)
    assert result["chase_win_percent"] == pytest.approx(50)
    assert result["ties"] == 1


def test_team_batting_order_uses_first_innings_batting_team():
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "team1": "A", "team2": "B", "outcome_winner": "A"},
            {"match_id": "m2", "team1": "A", "team2": "B", "outcome_winner": "A"},
        ]
    )
    deliveries = pd.DataFrame(
        [
            {"match_id": "m1", "innings": 1, "batting_team": "A", "is_super_over": 0},
            {"match_id": "m2", "innings": 1, "batting_team": "B", "is_super_over": 0},
        ]
    )

    summary = team_batting_order_summary(matches, deliveries).set_index("team")

    assert summary.loc["A", "batting_first_wins"] == 1
    assert summary.loc["A", "chase_wins"] == 1
    assert summary.loc["B", "batting_first_wins"] == 0
    assert summary.loc["B", "chase_wins"] == 0