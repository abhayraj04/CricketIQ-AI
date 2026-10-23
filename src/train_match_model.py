from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_matches_summary.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = MODEL_DIR / "team1_win_prediction.joblib"
METRICS_PATH = REPORT_DIR / "model_evaluation.csv"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix.png"

VALIDATION_START = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2024-01-01")
TARGET_COLUMN = "team1_won"

CATEGORICAL_FEATURES = ["team1", "team2", "venue"]
NUMERIC_FEATURES = [
    "team1_win_pct_prior",
    "team2_win_pct_prior",
    "team1_form_last5",
    "team2_form_last5",
    "h2h_matches_prior",
    "h2h_team1_wins_prior",
    "h2h_team1_win_pct",
    "venue_matches_prior",
    "venue_team1_win_pct",
    "venue_team2_win_pct",
    "venue_avg_first_innings_score",
    "venue_chase_win_pct",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "Logistic regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", LogisticRegression(max_iter=2_000, random_state=42)),
            ]
        ),
        "Random forest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline | DummyClassifier, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
        "brier_score": brier_score_loss(target, probabilities),
    }


def main() -> None:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"Processed match table not found: {MATCHES_PATH}. "
            "Run src/clean_data.py first."
        )

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"], low_memory=False)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, "date"])
    missing_columns = required_columns - set(matches.columns)
    if missing_columns:
        raise ValueError(f"Match table is missing columns: {sorted(missing_columns)}")

    # Ties and no-results have no team1_won label, so exclude them from training.
    labeled_matches = matches.dropna(subset=[TARGET_COLUMN, "date"]).copy()
    labeled_matches[TARGET_COLUMN] = labeled_matches[TARGET_COLUMN].astype(int)

    training = labeled_matches.loc[labeled_matches["date"] < VALIDATION_START]
    validation = labeled_matches.loc[
        (labeled_matches["date"] >= VALIDATION_START)
        & (labeled_matches["date"] < TEST_START)
    ]
    development = labeled_matches.loc[labeled_matches["date"] < TEST_START]
    test = labeled_matches.loc[labeled_matches["date"] >= TEST_START]

    for split_name, split in {
        "training": training,
        "validation": validation,
        "test": test,
    }.items():
        if split.empty or split[TARGET_COLUMN].nunique() < 2:
            raise ValueError(f"The {split_name} split is empty or has only one target class.")

    x_train = training[FEATURE_COLUMNS]
    y_train = training[TARGET_COLUMN]
    x_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    validation_results = []
    models = make_models()
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_validation, y_validation)
        validation_results.append({"model": model_name, **metrics})

    validation_table = pd.DataFrame(validation_results).sort_values(
        ["roc_auc", "brier_score"], ascending=[False, True]
    )
    selected_name = validation_table.iloc[0]["model"]
    selected_model = models[selected_name]

    # Keep the final years untouched until model selection is complete.
    x_development = development[FEATURE_COLUMNS]
    y_development = development[TARGET_COLUMN]
    x_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]
    selected_model.fit(x_development, y_development)

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(x_development, y_development)

    test_results = [
        {"model": "Majority baseline", **evaluate_model(baseline, x_test, y_test)},
        {"model": selected_name, **evaluate_model(selected_model, x_test, y_test)},
    ]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": selected_model,
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "validation_start": VALIDATION_START.isoformat(),
            "test_start": TEST_START.isoformat(),
        },
        MODEL_PATH,
    )

    metrics_table = pd.DataFrame(
        [
            {"split": "validation", **row}
            for row in validation_results
        ]
        + [{"split": "test", **row} for row in test_results]
    )
    metrics_table.to_csv(METRICS_PATH, index=False)

    test_predictions = selected_model.predict(x_test)
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Team 1 lost", "Team 1 won"],
        yticklabels=["Team 1 lost", "Team 1 won"],
        ax=axis,
    )
    axis.set(title=f"Final test set: {selected_name}", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close(figure)

    print(f"Training matches (< {VALIDATION_START.date()}): {len(training):,}")
    print(
        f"Validation matches ({VALIDATION_START.date()} to "
        f"{(TEST_START - pd.Timedelta(days=1)).date()}): {len(validation):,}"
    )
    print(f"Final test matches (>= {TEST_START.date()}): {len(test):,}")
    print("Validation comparison (model selection uses validation ROC-AUC):")
    print(validation_table.round(3).to_string(index=False))
    print("Final test comparison:")
    print(pd.DataFrame(test_results).round(3).to_string(index=False))
    print(f"Selected model saved to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Metrics saved to: {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()