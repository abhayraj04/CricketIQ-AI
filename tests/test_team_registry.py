from datetime import date

import pandas as pd
import pytest

from src.team_registry import (
    future_eligible_teams,
    load_team_registry,
    player_initials,
    team_short_name,
    validate_future_matchup,
)
from src.branding import resolve_asset_path, resolve_player_image, resolve_team_brand
from src.future_predictions import future_prediction_status


def test_only_verified_future_seasons_are_eligible():
    registry = load_team_registry()

    assert len(registry["seasons"]["2026"]["teams"]) == 10
    assert future_eligible_teams(registry, 2026, date(2026, 10, 21)) == []
    assert future_eligible_teams(registry, 2027, date(2026, 10, 21)) == []
    with pytest.raises(ValueError, match="participation is not verified"):
        validate_future_matchup(
            registry,
            2027,
            "Mumbai Indians",
            "Chennai Super Kings",
            date(2026, 10, 21),
        )


def test_team_short_names_and_historical_aliases_are_explicit():
    registry = load_team_registry()

    assert team_short_name("Royal Challengers Bengaluru", registry) == "RCB"
    assert team_short_name("Royal Challengers Bangalore", registry) == "RCB"
    assert team_short_name("Deccan Chargers", registry) is None


def test_player_initials_fallback_is_stable():
    assert player_initials("Virat Kohli") == "VK"
    assert player_initials("R Ashwin") == "RA"
    assert player_initials("") == "?"


def test_unlicensed_or_missing_brand_assets_use_fallbacks():
    team_registry = load_team_registry()
    current = resolve_team_brand("Mumbai Indians", team_registry)
    historical = resolve_team_brand("Deccan Chargers", team_registry)
    player = resolve_player_image("V Kohli")

    assert current["asset_path"] is None
    assert current["fallback_initials"] == "MI"
    assert historical["identity_status"] == "unmapped_historical_identity"
    assert player["image_path"] is None
    assert player["initials"] == "VK"
    assert resolve_asset_path("assets/../README.md") is None


def test_unverified_future_season_disables_pre_match_prediction(tmp_path):
    registry = load_team_registry()
    status = future_prediction_status(
        registry,
        2027,
        tmp_path / "metrics.csv",
        tmp_path / "model.joblib",
        date(2026, 10, 21),
    )

    assert status["enabled"] is False
    assert status["teams"] == []
    assert "verified" in status["reason"].lower()


def test_failed_pre_match_model_stays_disabled_even_for_verified_teams(tmp_path):
    registry = {
        "seasons": {
            "2027": {
                "status": "participation_verified_future",
                "teams": ["A", "B"],
                "fixture_status": "Hypothetical only",
            }
        }
    }
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {"split": "test", "model": "Majority baseline", "roc_auc": 0.5},
            {"split": "test", "model": "Logistic regression", "roc_auc": 0.4},
        ]
    ).to_csv(metrics_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"placeholder artifact")

    status = future_prediction_status(
        registry,
        2027,
        metrics_path,
        model_path,
        date(2026, 10, 21),
    )

    assert status["enabled"] is False
    assert status["teams"] == ["A", "B"]
    assert "did not beat" in status["reason"]


def test_backend_rejects_same_or_unverified_future_teams():
    registry = {
        "seasons": {
            "2027": {
                "status": "participation_verified_future",
                "teams": ["A", "B"],
            }
        }
    }
    current_day = date(2026, 10, 21)
    validate_future_matchup(registry, 2027, "A", "B", current_day)

    with pytest.raises(ValueError, match="different teams"):
        validate_future_matchup(registry, 2027, "A", "A", current_day)
    with pytest.raises(ValueError, match="not verified"):
        validate_future_matchup(registry, 2027, "A", "Historical XI", current_day)