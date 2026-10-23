from datetime import date
from pathlib import Path

import pandas as pd

from src.team_registry import future_eligible_teams


def future_prediction_status(
    registry: dict,
    season: int,
    evaluation_path: str | Path,
    model_path: str | Path,
    today: date | None = None,
) -> dict[str, object]:
    teams = future_eligible_teams(registry, season, today)
    if not teams:
        entry = registry.get("seasons", {}).get(str(season), {})
        return {
            "enabled": False,
            "teams": [],
            "reason": entry.get(
                "fixture_status",
                f"No official participation registry is verified for IPL {season}.",
            ),
        }

    artifact = Path(model_path)
    if not artifact.exists():
        return {
            "enabled": False,
            "teams": teams,
            "reason": "No validated pre-match inference artifact is available.",
        }

    report = Path(evaluation_path)
    if not report.exists():
        return {
            "enabled": False,
            "teams": teams,
            "reason": "Pre-match model test metrics are unavailable.",
        }
    metrics = pd.read_csv(report)
    test_rows = metrics.loc[metrics["split"].eq("test")].set_index("model")
    candidate_name = "Logistic regression"
    baseline_name = "Majority baseline"
    if candidate_name not in test_rows.index or baseline_name not in test_rows.index:
        return {
            "enabled": False,
            "teams": teams,
            "reason": "Pre-match model and baseline results could not be compared.",
        }

    model_auc = float(test_rows.loc[candidate_name, "roc_auc"])
    baseline_auc = float(test_rows.loc[baseline_name, "roc_auc"])
    if model_auc <= baseline_auc:
        return {
            "enabled": False,
            "teams": teams,
            "reason": (
                f"The current pre-match model did not beat its held-out baseline "
                f"(ROC-AUC {model_auc:.3f} vs {baseline_auc:.3f}); predictions are disabled."
            ),
        }

    return {
        "enabled": False,
        "teams": teams,
        "reason": "Future-match inference is not implemented for this validated model artifact.",
    }