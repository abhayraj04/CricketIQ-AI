import joblib
import pandas as pd
import pytest
from pathlib import Path

from src.chase_features import FEATURE_COLUMNS
from src.chase_model import (
    MODEL_FORMAT_VERSION,
    build_candidate_models,
    fit_sigmoid_calibrator,
    match_balanced_weights,
    weighted_calibration_points,
)
from src.chase_predictor import (
    load_chase_model,
    make_chase_state,
    predict_chase_probability,
)


def valid_state(**overrides):
    state = {
        "batting_team": "Team A",
        "bowling_team": "Team B",
        "venue": "Ground X",
        "target_score": 160,
        "current_score": 45,
        "wickets_lost": 2,
        "legal_balls_bowled": 24,
    }
    state.update(overrides)
    return state


def trained_bundle():
    features = pd.DataFrame(
        [
            ["Team A", "Team B", "Ground X", 150, 10, 140, 0, 10, 6, 114, 10, 7],
            ["Team A", "Team B", "Ground X", 160, 60, 100, 2, 8, 48, 72, 7.5, 8.3],
            ["Team A", "Team B", "Ground X", 170, 140, 30, 4, 6, 96, 24, 8.75, 7.5],
            ["Team C", "Team D", "Ground Y", 140, 20, 120, 1, 9, 18, 102, 6.7, 7.1],
            ["Team C", "Team D", "Ground Y", 180, 80, 100, 5, 5, 60, 60, 8, 10],
            ["Team C", "Team D", "Ground Y", 120, 115, 5, 3, 7, 90, 30, 7.7, 1],
        ],
        columns=FEATURE_COLUMNS,
    )
    target = pd.Series([0, 0, 1, 0, 1, 1], dtype="int8")
    model = build_candidate_models()["Logistic regression"]
    model.fit(features, target)
    calibrator = fit_sigmoid_calibrator(
        model,
        features,
        target,
        match_balanced_weights(pd.Series(["a", "b", "c", "d", "e", "f"])),
    )
    return {
        "format_version": MODEL_FORMAT_VERSION,
        "model_name": "Logistic regression",
        "feature_columns": FEATURE_COLUMNS,
        "model": model,
        "calibrator": calibrator,
    }


def test_saved_model_loads_and_returns_normalized_probabilities(tmp_path):
    path = tmp_path / "test_chase_model.joblib"
    joblib.dump(trained_bundle(), path)

    bundle = load_chase_model(path)
    prediction = predict_chase_probability(valid_state(), bundle)

    assert 0 <= prediction["chasing_team_probability"] <= 1
    assert 0 <= prediction["bowling_team_probability"] <= 1
    assert prediction["chasing_team_probability"] + prediction["bowling_team_probability"] == pytest.approx(1)
    assert prediction["model_name"] == "Logistic regression"


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_score": 0},
        {"current_score": 160},
        {"current_score": -1},
        {"current_score": 45.5},
        {"wickets_lost": 10},
        {"legal_balls_bowled": 120},
        {"batting_team": ""},
        {"bowling_team": "Team A"},
    ],
)
def test_manual_state_validation_rejects_invalid_inputs(overrides):
    with pytest.raises(ValueError):
        make_chase_state(valid_state(**overrides))


def test_model_loader_rejects_missing_or_invalid_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chase_model(tmp_path / "missing.joblib")

    invalid_path = tmp_path / "invalid.joblib"
    joblib.dump({"format_version": 999}, invalid_path)
    with pytest.raises(ValueError, match="artifact format"):
        load_chase_model(invalid_path)


def test_weighted_calibration_bins_use_probability_order():
    predicted, observed = weighted_calibration_points(
        pd.Series([0, 1, 1, 0]),
        [0.05, 0.15, 0.75, 0.95],
        [1.0, 1.0, 1.0, 1.0],
        bin_count=2,
    )

    assert predicted.tolist() == pytest.approx([0.1, 0.85])
    assert observed.tolist() == pytest.approx([0.5, 0.5])


def test_match_weights_give_each_match_equal_total_weight():
    match_ids = pd.Series(["long", "long", "long", "short"])
    weights = match_balanced_weights(match_ids)
    weighted_by_match = pd.Series(weights).groupby(match_ids).sum()

    assert weighted_by_match["long"] == pytest.approx(weighted_by_match["short"])


def test_trained_artifact_runs_a_real_prediction():
    artifact_path = Path(__file__).resolve().parents[1] / "models" / "live_chase_probability.joblib"
    if not artifact_path.exists():
        pytest.skip("Train the model with `python -m src.train_chase_model` first.")

    bundle = load_chase_model(artifact_path)
    prediction = predict_chase_probability(
        valid_state(
            batting_team="Chennai Super Kings",
            bowling_team="Royal Challengers Bengaluru",
            venue="MA Chidambaram Stadium",
            target_score=174,
            current_score=140,
            wickets_lost=4,
            legal_balls_bowled=96,
        ),
        bundle,
    )

    assert prediction["model_name"] == "Logistic regression"
    assert 0 <= prediction["chasing_team_probability"] <= 1
    assert prediction["chasing_team_probability"] + prediction["bowling_team_probability"] == pytest.approx(1)