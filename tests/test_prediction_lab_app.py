from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_prediction_lab_runs_manual_and_historical_paths():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    navigation = next(radio for radio in app.radio if radio.label == "Dashboard page")
    navigation.set_value("Prediction Lab").run()
    assert not app.exception

    estimate_button = next(
        button for button in app.button if button.label == "Estimate chase probability"
    )
    estimate_button.click().run()
    assert not app.exception
    metric_labels = {metric.label for metric in app.metric}
    assert "Chennai Super Kings chase success" in metric_labels

    input_mode = next(radio for radio in app.radio if radio.label == "Prediction input")
    input_mode.set_value("Historical chase state").run()
    assert not app.exception
    assert any(slider.label == "Select state after delivery" for slider in app.select_slider)

    estimate_button = next(
        button for button in app.button if button.label == "Estimate chase probability"
    )
    estimate_button.click().run()
    assert not app.exception


def test_future_prediction_mode_does_not_offer_historical_teams():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    next(radio for radio in app.radio if radio.label == "Dashboard page").set_value(
        "Prediction Lab"
    ).run()
    next(radio for radio in app.radio if radio.label == "Prediction mode").set_value(
        "Future match prediction"
    ).run()

    assert not app.exception
    assert any(selectbox.label == "Season" for selectbox in app.selectbox)
    assert not any(selectbox.label in {"Team A", "Team B"} for selectbox in app.selectbox)
    assert any("participation or fixture list verified" in warning.value.lower() for warning in app.warning)