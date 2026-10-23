from collections.abc import Mapping
from pathlib import Path

import joblib
import math
import pandas as pd

from src.chase_features import FEATURE_COLUMNS, validate_feature_columns
from src.chase_model import (
    MODEL_FORMAT_VERSION,
    calibrated_positive_probability,
)


def _whole_number(value: object, field_name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a whole number.") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field_name} must be a whole number.")
    return int(number)


def load_chase_model(model_path: str | Path) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chase model not found: {path}. "
            "Train it with `python -m src.train_chase_model`."
        )

    try:
        bundle = joblib.load(path)
    except (AttributeError, ImportError, ModuleNotFoundError):
        from src.train_chase_model import main as train_chase
        train_chase()
        bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported or invalid chase model artifact format.")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Saved model feature schema does not match the current application.")
    validate_feature_columns(bundle["feature_columns"])
    return bundle


def make_chase_state(state: Mapping[str, object]) -> pd.DataFrame:
    """Validate a manual post-delivery state and derive the model feature row."""
    required = {
        "batting_team",
        "bowling_team",
        "venue",
        "target_score",
        "current_score",
        "wickets_lost",
        "legal_balls_bowled",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(f"Prediction state is missing fields: {sorted(missing)}")

    target = _whole_number(state["target_score"], "Target")
    score = _whole_number(state["current_score"], "Current score")
    wickets = _whole_number(state["wickets_lost"], "Wickets lost")
    legal_balls = _whole_number(state["legal_balls_bowled"], "Legal balls bowled")

    if target <= 0:
        raise ValueError("Target must be greater than zero.")
    if score < 0 or score >= target:
        raise ValueError("A live chase requires a score from zero up to target minus one.")
    if not 0 <= wickets <= 9:
        raise ValueError("Wickets lost must be between zero and nine for a live chase.")
    if not 0 <= legal_balls < 120:
        raise ValueError("Legal balls bowled must be between zero and 119.")
    if state["batting_team"].strip() == state["bowling_team"].strip():
        raise ValueError("Batting and bowling teams must be different.")
    if not all(isinstance(state[field], str) and state[field].strip() for field in ["batting_team", "bowling_team", "venue"]):
        raise ValueError("Team and venue values must be non-empty strings.")

    balls_remaining = 120 - legal_balls
    feature_row = {
        "batting_team": state["batting_team"].strip(),
        "bowling_team": state["bowling_team"].strip(),
        "venue": state["venue"].strip(),
        "target_score": target,
        "current_score": score,
        "runs_needed": target - score,
        "wickets_lost": wickets,
        "wickets_remaining": 10 - wickets,
        "legal_balls_bowled": legal_balls,
        "balls_remaining": balls_remaining,
        "current_run_rate": score * 6 / legal_balls if legal_balls else float("nan"),
        "required_run_rate": (target - score) * 6 / balls_remaining,
    }
    return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)


def predict_chase_probability(
    state: Mapping[str, object],
    model_bundle: dict,
) -> dict[str, float | str]:
    if model_bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported or invalid chase model artifact format.")
    if model_bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Saved model feature schema does not match the current application.")
    validate_feature_columns(model_bundle["feature_columns"])

    feature_row = make_chase_state(state)
    calibrator = model_bundle.get("calibrator")
    if calibrator is None:
        raise ValueError("Saved model artifact is missing its probability calibrator.")
    probability = float(
        calibrated_positive_probability(
            model_bundle["model"],
            calibrator,
            feature_row,
        )[0]
    )
    if not 0 <= probability <= 1:
        raise ValueError("Model returned a probability outside the valid range.")
    return {
        "chasing_team_probability": probability,
        "bowling_team_probability": 1.0 - probability,
        "model_name": str(model_bundle["model_name"]),
    }