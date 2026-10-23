import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.chase_predictor import load_chase_model
from src.ui_theme import chart_colors, plotly_template


def render_model_performance(project_root: str | Path) -> None:
    root = Path(project_root)
    reports = root / "reports"
    metrics_path = reports / "live_chase_evaluation.csv"
    artifact_path = root / "models" / "live_chase_probability.joblib"
    colors = chart_colors(st.session_state.get("ui_theme", "Dark"))
    st.subheader("Model performance center")
    if not metrics_path.exists() or not artifact_path.exists():
        st.error("Live-chase evaluation artifacts are missing. Train the model first.")
        st.code(".\\.venv\\Scripts\\python.exe -m src.train_chase_model", language="powershell")
        return

    try:
        metrics = pd.read_csv(metrics_path)
        bundle = load_chase_model(artifact_path)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
        st.error(f"Could not load model performance artifacts: {error}")
        return

    test_metrics = metrics.loc[metrics["split"].eq("test")].copy()
    validation_metrics = metrics.loc[metrics["split"].eq("validation")].copy()
    test_model = test_metrics.loc[test_metrics["model"].str.contains("sigmoid calibrated", case=False)]
    baseline = test_metrics.loc[test_metrics["model"].eq("Prior baseline")]
    if test_model.empty or baseline.empty:
        st.error("Expected held-out calibrated model and baseline rows are missing from the report.")
        return

    model_row = test_model.iloc[0]
    baseline_row = baseline.iloc[0]
    metrics_row = st.columns(4)
    metrics_row[0].metric("Test ROC-AUC", f"{model_row['roc_auc']:.3f}")
    metrics_row[1].metric("Test Brier", f"{model_row['brier_score']:.3f}")
    metrics_row[2].metric("Test log loss", f"{model_row['log_loss']:.3f}")
    metrics_row[3].metric("Test matches", f"{bundle['test_matches']:,}")

    st.caption(
        f"Model: {bundle['model_name']} · Fit through 2021 · Calibrated 2022–2023 · "
        "Untouched test: 2024–2026"
    )
    st.markdown("#### Held-out comparison · match-weighted")
    st.dataframe(
        test_metrics.round(3),
        width="stretch",
        hide_index=True,
    )
    metric_plot = test_metrics.loc[
        test_metrics["model"].isin(
            ["Prior baseline", "Logistic regression (sigmoid calibrated)"]
        ),
        ["model", "roc_auc", "brier_score", "log_loss"],
    ].melt(id_vars="model", var_name="metric", value_name="score")
    st.plotly_chart(
        px.bar(
            metric_plot,
            x="metric",
            y="score",
            color="model",
            barmode="group",
            color_discrete_sequence=[colors["muted"], colors["gold"]],
            labels={"score": "Score", "metric": "Metric"},
        ).update_layout(template=plotly_template(st.session_state.get("ui_theme", "Dark"))),
        width="stretch",
    )
    st.caption(
        "Higher ROC-AUC is better; lower Brier score and log loss are better. "
        "The confusion matrix counts delivery states, while headline metrics weight each match equally."
    )

    artifact_col, plots_col = st.columns(2)
    with artifact_col:
        st.markdown("#### Training and leakage safeguards")
        split_data = pd.DataFrame(
            [
                {"Partition": "Fit", "Years": "2008–2021", "Matches": bundle["model_fit_matches"], "States": bundle["model_fit_states"]},
                {"Partition": "Calibration", "Years": "2022–2023", "Matches": bundle["calibration_matches"], "States": bundle["calibration_states"]},
                {"Partition": "Test", "Years": "2024–2026", "Matches": bundle["test_matches"], "States": bundle["test_states"]},
            ]
        )
        st.dataframe(split_data, width="stretch", hide_index=True)
        st.caption(
            "All states from a match remain in one date partition. Terminal states, super overs, "
            "shortened/DLS matches, and unlabeled outcomes are excluded. Inputs are post-delivery; "
            "final outcomes and scores are not model features."
        )
        st.download_button(
            "Download evaluation metrics",
            data=metrics.to_csv(index=False).encode("utf-8"),
            file_name="live_chase_evaluation.csv",
            mime="text/csv",
            width="stretch",
        )

    with plots_col:
        confusion_path = reports / "live_chase_confusion_matrix.png"
        calibration_path = reports / "live_chase_calibration.png"
        if confusion_path.exists():
            st.markdown("#### Confusion matrix")
            st.image(str(confusion_path), width="stretch")
        if calibration_path.exists():
            st.markdown("#### Probability calibration")
            st.image(str(calibration_path), width="stretch")

    importance_path = reports / "live_chase_feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(15)
        st.markdown("#### Global model factors")
        st.caption("Absolute coefficient magnitude after preprocessing; association is not causation.")
        st.dataframe(importance, width="stretch", hide_index=True)

    st.markdown("#### Candidate model selection")
    st.dataframe(
        validation_metrics.round(3),
        width="stretch",
        hide_index=True,
    )
    if model_row["roc_auc"] <= baseline_row["roc_auc"]:
        st.warning("The selected model did not beat the baseline on held-out ROC-AUC.")
    else:
        st.success(
            f"Held-out ROC-AUC improved over baseline by "
            f"{model_row['roc_auc'] - baseline_row['roc_auc']:.3f}. "
            "This is historical evidence, not a guarantee of future performance."
        )

    st.markdown("#### Live Model Governance and Retraining")
    retrain_col1, retrain_col2 = st.columns([2, 1])
    with retrain_col1:
        st.caption(
            "Re-fit the Sigmoid-calibrated In-Play Chase model directly from the verified historical dataset, "
            "re-calculating held-out ROC-AUC, Brier scores, calibration curves, and feature coefficients."
        )
    with retrain_col2:
        if st.button("Retrain Chase Model Now", type="primary", key="btn_retrain_report_view"):
            with st.spinner("Retraining in-play chase model across chronological partitions..."):
                try:
                    import subprocess
                    import sys
                    subprocess.run(
                        [sys.executable, "-m", "src.train_chase_model"],
                        cwd=str(root),
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    st.success("Chase model successfully retrained and verified!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Retraining error: {ex}")


def render_data_quality(
    project_root: str | Path,
    matches: pd.DataFrame,
    players: pd.DataFrame,
    deliveries: pd.DataFrame,
    database_path: str | Path,
) -> None:
    root = Path(project_root)
    report_path = root / "data" / "processed" / "data_quality_report.csv"
    registry_path = root / "config" / "team_branding.json"
    st.subheader("Data quality and methodology")

    source_summary = pd.DataFrame(
        [
            {"table": "Match summary", "rows": len(matches), "columns loaded": matches.shape[1], "key": "match_id"},
            {"table": "Player profiles", "rows": len(players), "columns loaded": players.shape[1], "key": "player_name (IDs need review)"},
            {"table": "Ball by ball", "rows": len(deliveries), "columns loaded": deliveries.shape[1], "key": "match_id + innings + ball sequence"},
        ]
    )
    st.markdown("#### Loaded sources")
    st.dataframe(source_summary, width="stretch", hide_index=True)
    st.caption(
        f"Match dates: {matches['date'].min().date()} to {matches['date'].max().date()} · "
        f"{matches['match_id'].nunique():,} unique matches · {matches['venue'].nunique()} venues"
    )

    duplicate_summary = pd.DataFrame(
        [
            {"check": "Exact duplicate match rows", "count": int(matches.duplicated().sum())},
            {"check": "Duplicate match IDs", "count": int(matches["match_id"].duplicated().sum())},
            {"check": "Exact duplicate profile rows", "count": int(players.duplicated().sum())},
            {"check": "Duplicate player names", "count": int(players["player_name"].duplicated().sum())},
            {
                "check": "Duplicate ball keys",
                "count": int(deliveries.duplicated(subset=["match_id", "innings", "ball_seq_idx"]).sum()),
            },
        ]
    )
    with sqlite3.connect(database_path) as connection:
        duplicate_profile_ids = connection.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT cricsheet_id FROM ipl_player_profiles "
            "WHERE cricsheet_id IS NOT NULL GROUP BY cricsheet_id "
            "HAVING COUNT(DISTINCT player_name) > 1)"
        ).fetchone()[0]
        season_mismatches = pd.read_sql_query(
            "SELECT season, SUBSTR(date, 1, 4) AS calendar_year, COUNT(*) AS matches "
            "FROM ipl_matches_summary "
            "WHERE CAST(season AS TEXT) <> SUBSTR(date, 1, 4) "
            "GROUP BY season, SUBSTR(date, 1, 4) ORDER BY season, calendar_year",
            connection,
        )
    duplicate_summary.loc[len(duplicate_summary)] = {
        "check": "Profile IDs shared by multiple names",
        "count": int(duplicate_profile_ids),
    }
    st.markdown("#### Duplicate and key checks")
    st.dataframe(duplicate_summary, width="stretch", hide_index=True)

    match_ids = set(matches["match_id"].astype("string"))
    delivery_match_ids = set(deliveries["match_id"].astype("string"))
    run_identity_errors = (
        pd.to_numeric(deliveries["runs_batter"], errors="coerce").fillna(0)
        + pd.to_numeric(deliveries["runs_extras"], errors="coerce").fillna(0)
        != pd.to_numeric(deliveries["total_runs"], errors="coerce").fillna(0)
    ).sum()
    integrity_summary = pd.DataFrame(
        [
            {"check": "Matches in summary", "value": len(match_ids)},
            {"check": "Distinct delivery match IDs", "value": len(delivery_match_ids)},
            {"check": "Delivery match IDs missing from summary", "value": len(delivery_match_ids - match_ids)},
            {"check": "Batter + extras != total run deliveries", "value": int(run_identity_errors)},
            {
                "check": "Profile and summary CSVs byte-identical",
                "value": (
                    "Yes"
                    if (root / "csv" / "ipl_player_profiles.csv").exists()
                    and (root / "csv" / "ipl_player_summary.csv").exists()
                    and (root / "csv" / "ipl_player_profiles.csv").read_bytes()
                    == (root / "csv" / "ipl_player_summary.csv").read_bytes()
                    else "No/unchecked"
                ),
            },
        ]
    )
    integrity_summary["value"] = integrity_summary["value"].astype("string")
    st.markdown("#### Relationship and scoring checks")
    st.dataframe(integrity_summary, width="stretch", hide_index=True)

    st.markdown("#### Source season labels versus calendar year")
    if season_mismatches.empty:
        st.success("All source season labels match calendar years.")
    else:
        st.caption("Year charts use dates rather than these inconsistent source labels.")
        st.dataframe(season_mismatches, width="stretch", hide_index=True)

    if report_path.exists():
        quality_report = pd.read_csv(report_path)
        selected_table = st.selectbox(
            "Missingness by table",
            sorted(quality_report["table"].unique()),
            key="quality_table",
        )
        table_quality = quality_report.loc[quality_report["table"].eq(selected_table)]
        missing = table_quality.loc[table_quality["missing_count"].gt(0)].sort_values(
            "missing_percent", ascending=False
        )
        st.markdown("#### Missing values")
        st.dataframe(missing.head(25), width="stretch", hide_index=True)
        st.download_button(
            "Download full quality report",
            data=quality_report.to_csv(index=False).encode("utf-8"),
            file_name="data_quality_report.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.warning("Per-column quality report is missing. Run `src/clean_data.py` to regenerate it.")

    with st.expander("Methodology and known limitations", expanded=True):
        st.markdown(
            "- Calendar year is derived from match date because source season labels are inconsistent.\n"
            "- Ties and no-results are not treated as losses in win-rate denominators.\n"
            "- Delivery totals are aggregated before joining to match-level data; super overs are excluded from regular-innings rates.\n"
            "- Player profiles and player summary CSVs are identical; only profiles are loaded.\n"
            "- Player career rates are not season-filtered; missing values are not converted to zero.\n"
            "- Three profile IDs map to multiple names; use exact profile names until reviewed.\n"
            "- Team logos/colors and player portraits are omitted unless reuse rights and identity are verified."
        )

    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        team_rows = []
        for team in registry.get("teams", []):
            team_rows.append(
                {
                    "team": team["name"],
                    "short name": team["short_name"],
                    "aliases": ", ".join(team.get("historical_aliases", [])) or "None recorded",
                    "logo": "Available" if team.get("logo_path") else "Initials fallback",
                    "verified colors": bool(team.get("primary_color") and team.get("secondary_color")),
                }
            )
        st.markdown("#### Team identity and assets")
        st.dataframe(pd.DataFrame(team_rows), width="stretch", hide_index=True)
        st.caption("Player portraits are not bundled; see docs/ASSET_SOURCES.md for sourcing and rights notes.")