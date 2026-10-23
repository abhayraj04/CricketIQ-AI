import pandas as pd
import pytest

from src.chase_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_chase_dataset,
    validate_feature_columns,
)


def delivery_row(**overrides):
    row = {
        "match_id": "match-a",
        "date": "2024-04-01",
        "innings": "2",
        "is_super_over": "False",
        "is_dls_affected": "False",
        "is_shortened_match": "False",
        "is_innings_ended": "False",
        "is_chasing_winner": "True",
        "target_runs": "160",
        "current_score": "45",
        "current_wickets": "2",
        "legal_balls_bowled": "24",
        "max_innings_balls": "120",
        "batting_team": "Team A",
        "bowling_team": "Team B",
        "venue": "Ground X",
        "ball_seq_idx": "30",
    }
    row.update(overrides)
    return row


def test_builds_only_post_delivery_features_and_derives_rates():
    result = build_chase_dataset(pd.DataFrame([delivery_row()]))
    state = result.iloc[0]

    assert state["current_score"] == 45
    assert state["runs_needed"] == 115
    assert state["wickets_remaining"] == 8
    assert state["balls_remaining"] == 96
    assert state["current_run_rate"] == pytest.approx(11.25)
    assert state["required_run_rate"] == pytest.approx(7.1875)
    assert state[TARGET_COLUMN] == 1
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert "is_innings_ended" not in result.columns
    assert "match_winner" not in result.columns


@pytest.mark.parametrize(
    "overrides",
    [
        {"innings": "1"},
        {"is_super_over": "True"},
        {"is_dls_affected": "True"},
        {"is_shortened_match": "True"},
        {"is_innings_ended": "True"},
        {"is_chasing_winner": ""},
    ],
)
def test_excludes_ineligible_delivery_states(overrides):
    result = build_chase_dataset(pd.DataFrame([delivery_row(**overrides)]))
    assert result.empty


def test_rejects_inconsistent_result_labels_within_a_match():
    rows = [
        delivery_row(ball_seq_idx="1", is_chasing_winner="True"),
        delivery_row(ball_seq_idx="2", is_chasing_winner="False"),
    ]
    with pytest.raises(ValueError, match="label changes within match"):
        build_chase_dataset(pd.DataFrame(rows))


def test_rejects_post_match_fields_in_model_features():
    with pytest.raises(ValueError, match="Target leakage"):
        validate_feature_columns(["current_score", "final_innings_score"])


def test_accepts_only_pre_prediction_state_fields():
    validate_feature_columns(FEATURE_COLUMNS)


def test_loader_rejects_labels_that_conflict_across_chunks(tmp_path):
    from src.chase_features import load_chase_dataset

    rows = [
        delivery_row(ball_seq_idx="1", is_chasing_winner="True"),
        delivery_row(ball_seq_idx="2", is_chasing_winner="False"),
    ]
    csv_path = tmp_path / "deliveries.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="across CSV chunks"):
        load_chase_dataset(str(csv_path), chunksize=1)