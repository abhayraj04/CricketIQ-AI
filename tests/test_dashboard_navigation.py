from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_navigation_light_theme_filters_and_reset():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()

    next(control for control in app.selectbox if control.label == "Appearance").set_value("Light").run()
    assert not app.exception

    pages = [
        "Overview",
        "Team analytics",
        "Player analytics",
        "Match explorer",
        "Venue & toss",
        "Ball by ball",
        "Prediction Lab",
        "Model performance",
        "Data quality",
        "AI Intelligence Center",
    ]
    for page in pages:
        next(radio for radio in app.radio if radio.label == "Dashboard page").set_value(page).run()
        assert not app.exception

    next(radio for radio in app.radio if radio.label == "Dashboard page").set_value("Overview").run()
    app.slider[0].set_value((2024, 2026)).run()
    app.multiselect[0].set_value(["Mumbai Indians"]).run()
    assert not app.exception
    assert "2024–2026" in app.caption[1].value

    next(button for button in app.button if button.label == "Reset filters").click().run()
    assert app.slider[0].value == (2008, 2026)
    assert app.multiselect[0].value == []
    assert not app.exception
