import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.chase_features import FEATURE_COLUMNS


MODEL_FORMAT_VERSION = 2
CATEGORICAL_FEATURES = ["batting_team", "bowling_team", "venue"]
NUMERIC_FEATURES = [
    feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES
]


def match_balanced_weights(match_ids: pd.Series) -> np.ndarray:
    """Give each match equal total weight, regardless of its number of states."""
    counts = match_ids.groupby(match_ids).transform("size").to_numpy(dtype=float)
    if len(counts) == 0 or np.any(counts <= 0):
        raise ValueError("Sample weights require at least one non-empty match ID.")
    weights = 1.0 / counts
    return weights / weights.mean()


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_candidate_models() -> dict[str, Pipeline]:
    """Return modest candidates with preprocessing fitted inside each pipeline."""
    return {
        "Logistic regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", LogisticRegression(max_iter=2_000, random_state=42)),
            ]
        ),
        "Random forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=15,
                        max_features=0.8,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def fit_weighted_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: np.ndarray,
) -> Pipeline:
    model.fit(features, target, classifier__sample_weight=sample_weight)
    return model


def positive_class_probability(model, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Trained model does not contain the positive class 1.")
    return model.predict_proba(features)[:, classes.index(1)]


def _probability_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_sigmoid_calibrator(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: np.ndarray,
) -> LogisticRegression:
    """Fit a Platt-style sigmoid using only a later calibration partition."""
    raw_probabilities = positive_class_probability(model, features)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
    calibrator.fit(
        _probability_logit(raw_probabilities),
        target,
        sample_weight=sample_weight,
    )
    return calibrator


def calibrated_positive_probability(
    model,
    calibrator: LogisticRegression,
    features: pd.DataFrame,
) -> np.ndarray:
    raw_probabilities = positive_class_probability(model, features)
    classes = list(calibrator.classes_)
    if 1 not in classes:
        raise ValueError("Probability calibrator does not contain the positive class 1.")
    return calibrator.predict_proba(_probability_logit(raw_probabilities))[:, classes.index(1)]


def weighted_calibration_points(
    target: pd.Series,
    probabilities: np.ndarray,
    sample_weight: np.ndarray,
    bin_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-width reliability bins with match-weighted averages."""
    target_values = np.asarray(target, dtype=float)
    probability_values = np.asarray(probabilities, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if bin_count <= 0:
        raise ValueError("bin_count must be greater than zero.")
    if not (len(target_values) == len(probability_values) == len(weights)):
        raise ValueError("Targets, probabilities, and weights must have equal lengths.")
    if len(weights) == 0 or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Calibration weights must be finite positive values.")
    if not np.isin(target_values, [0, 1]).all():
        raise ValueError("Calibration targets must be binary 0/1 values.")
    if not np.isfinite(probability_values).all() or ((probability_values < 0) | (probability_values > 1)).any():
        raise ValueError("Calibration probabilities must be finite and within [0, 1].")

    bin_indexes = np.minimum((probability_values * bin_count).astype(int), bin_count - 1)
    mean_predicted = []
    observed_rate = []
    for bin_index in range(bin_count):
        in_bin = bin_indexes == bin_index
        if not in_bin.any():
            continue
        bin_weights = weights[in_bin]
        mean_predicted.append(np.average(probability_values[in_bin], weights=bin_weights))
        observed_rate.append(np.average(target_values[in_bin], weights=bin_weights))
    return np.asarray(mean_predicted), np.asarray(observed_rate)


def evaluate_probabilities(
    model,
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: np.ndarray,
    calibrator: LogisticRegression | None = None,
) -> dict[str, float]:
    probabilities = (
        positive_class_probability(model, features)
        if calibrator is None
        else calibrated_positive_probability(model, calibrator, features)
    )
    predictions = (probabilities >= 0.5).astype("int8")
    return {
        "accuracy": float(accuracy_score(target, predictions, sample_weight=sample_weight)),
        "precision": float(
            precision_score(target, predictions, sample_weight=sample_weight, zero_division=0)
        ),
        "recall": float(
            recall_score(target, predictions, sample_weight=sample_weight, zero_division=0)
        ),
        "f1": float(f1_score(target, predictions, sample_weight=sample_weight, zero_division=0)),
        "roc_auc": float(
            roc_auc_score(target, probabilities, sample_weight=sample_weight)
        ),
        "brier_score": float(
            brier_score_loss(target, probabilities, sample_weight=sample_weight)
        ),
        "log_loss": float(
            log_loss(
                target,
                np.column_stack([1 - probabilities, probabilities]),
                labels=[0, 1],
                sample_weight=sample_weight,
            )
        ),
    }


def model_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return model-based global importance; this is association, not causation."""
    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_).mean(axis=0)
    else:
        raise TypeError("Selected classifier does not expose supported feature importance.")

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def local_feature_contributions(
    model: Pipeline,
    calibrator: LogisticRegression,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Explain calibrated log-odds using signed transformed logistic contributions."""
    classifier = model.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Local signed contributions are currently available for logistic regression only.")
    if len(features) != 1:
        raise ValueError("Local explanation expects exactly one prediction state.")

    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(features)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    coefficients = classifier.coef_[0]
    calibration_slope = float(calibrator.coef_[0][0])
    contributions = transformed[0] * coefficients * calibration_slope
    feature_names = preprocessor.get_feature_names_out()
    if len(feature_names) != len(contributions):
        raise ValueError("Transformed model feature names do not match contribution values.")
    return (
        pd.DataFrame({"feature": feature_names, "log_odds_contribution": contributions})
        .assign(abs_contribution=lambda frame: frame["log_odds_contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .drop(columns="abs_contribution")
        .reset_index(drop=True)
    )