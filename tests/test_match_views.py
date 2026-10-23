import pandas as pd

from src.match_views import filter_match_records


def sample_matches():
    return pd.DataFrame(
        [
            {"match_id": "1", "date": pd.Timestamp("2024-04-01"), "team1": "Mumbai Indians", "team2": "Chennai Super Kings", "venue": "Wankhede", "outcome_winner": "Mumbai Indians", "outcome_result": "", "is_tie": 0, "is_no_result": 0, "player_of_match": "Player A"},
            {"match_id": "2", "date": pd.Timestamp("2024-04-02"), "team1": "Rajasthan Royals", "team2": "Kolkata Knight Riders", "venue": "Eden Gardens", "outcome_winner": None, "outcome_result": "tie", "is_tie": 1, "is_no_result": 0, "player_of_match": None},
            {"match_id": "3", "date": pd.Timestamp("2024-04-03"), "team1": "Mumbai Indians", "team2": "Delhi Capitals", "venue": "Wankhede", "outcome_winner": None, "outcome_result": "no result", "is_tie": 0, "is_no_result": 1, "player_of_match": None},
        ]
    )


def test_search_and_result_filters_keep_match_grain():
    matches = sample_matches()

    searched = filter_match_records(matches, search="mumbai")
    ties = filter_match_records(matches, result_filter="Tie")
    resolved = filter_match_records(matches, result_filter="Winner recorded")
    no_results = filter_match_records(matches, result_filter="No result")

    assert searched["match_id"].tolist() == ["3", "1"]
    assert ties["match_id"].tolist() == ["2"]
    assert resolved["match_id"].tolist() == ["1"]
    assert no_results["match_id"].tolist() == ["3"]
    assert len(resolved) == resolved["match_id"].nunique()


def test_search_respects_requested_sort_order():
    matches = sample_matches()

    oldest = filter_match_records(matches, ascending=True)
    newest = filter_match_records(matches, ascending=False)

    assert oldest["match_id"].tolist() == ["1", "2", "3"]
    assert newest["match_id"].tolist() == ["3", "2", "1"]