import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from src.chase_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_chase_dataset,
    validate_feature_columns,
)
from src.chase_model import (
    build_candidate_models,
    calibrated_positive_probability,
    evaluate_probabilities,
    fit_sigmoid_calibrator,
    fit_weighted_model,
    match_balanced_weights,
    model_feature_importance,
    positive_class_probability,
    MODEL_FORMAT_VERSION,
    weighted_calibration_points,
)
from src.time_splits import chronological_match_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = PROJECT_ROOT / "csv" / "ipl_ball_by_ball_processed.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "live_chase_probability.joblib"
METRICS_PATH = REPORT_DIR / "live_chase_evaluation.csv"
IMPORTANCE_PATH = REPORT_DIR / "live_chase_feature_importance.csv"
CONFUSION_PATH = REPORT_DIR / "live_chase_confusion_matrix.png"
CALIBRATION_PATH = REPORT_DIR / "live_chase_calibration.png"

VALIDATION_START = "2020-01-01"
CALIBRATION_START = "2022-01-01"
TEST_START = "2024-01-01"


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source delivery CSV not found: {SOURCE_CSV}")
    validate_feature_columns(FEATURE_COLUMNS)

    states = load_chase_dataset(str(SOURCE_CSV))
    splits = chronological_match_splits(
        states,
        validation_start=VALIDATION_START,
        test_start=CALIBRATION_START,
    )
    training = splits["train"]
    validation = splits["validation"]
    later_validation = splits["test"]
    calibration = later_validation.loc[later_validation["date"] < TEST_START].copy()
    test = later_validation.loc[later_validation["date"] >= TEST_START].copy()

    all_splits = {
        "training": training,
        "model selection": validation,
        "calibration": calibration,
        "test": test,
    }
    for split_name, split in all_splits.items():
        if split.empty or split[TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"The {split_name} split must contain both chase outcomes.")

    x_train, y_train = training[FEATURE_COLUMNS], training[TARGET_COLUMN]
    x_validation, y_validation = validation[FEATURE_COLUMNS], validation[TARGET_COLUMN]
    train_weights = match_balanced_weights(training["match_id"])
    validation_weights = match_balanced_weights(validation["match_id"])

    candidates = build_candidate_models()
    validation_rows = []
    for model_name, model in candidates.items():
        fit_weighted_model(model, x_train, y_train, train_weights)
        metrics = evaluate_probabilities(
            model,
            x_validation,
            y_validation,
            validation_weights,
        )
        validation_rows.append({"model": model_name, **metrics})

    validation_metrics = pd.DataFrame(validation_rows).sort_values(
        ["brier_score", "roc_auc"], ascending=[True, False]
    )
    selected_name = str(validation_metrics.iloc[0]["model"])
    selected_model = candidates[selected_name]

    model_fit_data = pd.concat([training, validation], ignore_index=True)
    x_model_fit = model_fit_data[FEATURE_COLUMNS]
    y_model_fit = model_fit_data[TARGET_COLUMN]
    model_fit_weights = match_balanced_weights(model_fit_data["match_id"])
    fit_weighted_model(selected_model, x_model_fit, y_model_fit, model_fit_weights)

    calibration_weights = match_balanced_weights(calibration["match_id"])
    calibrator = fit_sigmoid_calibrator(
        selected_model,
        calibration[FEATURE_COLUMNS],
        calibration[TARGET_COLUMN],
        calibration_weights,
    )

    x_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]
    test_weights = match_balanced_weights(test["match_id"])
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(x_model_fit, y_model_fit, sample_weight=model_fit_weights)

    baseline_metrics = evaluate_probabilities(baseline, x_test, y_test, test_weights)
    uncalibrated_metrics = evaluate_probabilities(
        selected_model,
        x_test,
        y_test,
        test_weights,
    )
    selected_metrics = evaluate_probabilities(
        selected_model,
        x_test,
        y_test,
        test_weights,
        calibrator=calibrator,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_metadata = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_name": selected_name,
        "task": "post_delivery_live_chase_probability",
        "target": TARGET_COLUMN,
        "positive_class": "batting team eventually wins the chase",
        "feature_columns": FEATURE_COLUMNS,
        "validation_start": VALIDATION_START,
        "calibration_start": CALIBRATION_START,
        "test_start": TEST_START,
        "training_states": int(len(training)),
        "validation_states": int(len(validation)),
        "model_fit_states": int(len(model_fit_data)),
        "calibration_states": int(len(calibration)),
        "test_states": int(len(test)),
        "training_matches": int(training["match_id"].nunique()),
        "validation_matches": int(validation["match_id"].nunique()),
        "model_fit_matches": int(model_fit_data["match_id"].nunique()),
        "calibration_matches": int(calibration["match_id"].nunique()),
        "test_matches": int(test["match_id"].nunique()),
        "prediction_timing": "after a completed delivery; terminal states excluded",
        "excluded_data": ["super overs", "DLS/shortened matches", "terminal states", "unlabeled outcomes"],
        "evaluation_weighting": "each match has equal total weight within a split",
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "calibration_method": "match-weighted sigmoid calibration fitted on 2022-2023 only",
        "test_metrics": {
            "majority_baseline": baseline_metrics,
            "selected_model_uncalibrated": uncalibrated_metrics,
            "selected_model": selected_metrics,
        },
    }
    joblib.dump(
        {**model_metadata, "model": selected_model, "calibrator": calibrator},
        MODEL_PATH,
    )

    metrics_rows = [
        {"split": "validation", **row}
        for row in validation_rows
    ] + [
        {"split": "test", "model": "Prior baseline", **baseline_metrics},
        {"split": "test", "model": f"{selected_name} (uncalibrated)", **uncalibrated_metrics},
        {"split": "test", "model": f"{selected_name} (sigmoid calibrated)", **selected_metrics},
    ]
    pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)

    importance = model_feature_importance(selected_model)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    predictions = (
        calibrated_positive_probability(selected_model, calibrator, x_test) >= 0.5
    ).astype("int8")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Chase fails", "Chase succeeds"],
        yticklabels=["Chase fails", "Chase succeeds"],
        ax=axis,
    )
    axis.set(title="Untouched 2024-2026 test states", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=160)
    plt.close(figure)

    test_probabilities = calibrated_positive_probability(selected_model, calibrator, x_test)
    predicted_rate, observed_rate = weighted_calibration_points(
        y_test,
        test_probabilities,
        test_weights,
        bin_count=10,
    )
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Perfect calibration")
    axis.plot(predicted_rate, observed_rate, marker="o", color="#5b4bdb", label=selected_name)
    axis.set(
        title="Untouched test-state calibration",
        xlabel="Mean predicted chase probability",
        ylabel="Observed chase success rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(CALIBRATION_PATH, dpi=160)
    plt.close(figure)

    print("Chronological match-level partitions:")
    for split_name, split in all_splits.items():
        print(
            f"  {split_name}: {len(split):,} states, "
            f"{split['match_id'].nunique():,} matches, "
            f"{split[TARGET_COLUMN].value_counts().to_dict()}"
        )
    print("Validation candidate comparison (match-weighted):")
    print(validation_metrics.round(4).to_string(index=False))
    print("Untouched test comparison (match-weighted):")
    print(
        pd.DataFrame(
            [
                {"model": "Prior baseline", **baseline_metrics},
                {"model": f"{selected_name} (uncalibrated)", **uncalibrated_metrics},
                {"model": f"{selected_name} (sigmoid calibrated)", **selected_metrics},
            ]
        ).round(4).to_string(index=False)
    )
    print(f"Selected model: {selected_name}")
    print(f"Model artifact: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metrics: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Feature importance: {IMPORTANCE_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Confusion matrix: {CONFUSION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Calibration plot: {CALIBRATION_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()