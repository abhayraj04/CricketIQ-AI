from src.ui_theme import chart_colors, plotly_template, theme_variables


def test_dark_and_light_themes_use_distinct_plotly_templates():
    assert plotly_template("Dark") == "plotly_dark"
    assert plotly_template("Light") == "plotly_white"


def test_theme_variables_include_readable_surface_and_accent_tokens():
    for mode in ["Dark", "Light"]:
        variables = theme_variables(mode)
        colors = chart_colors(mode)
        assert "--app-bg:" in variables
        assert "--text:" in variables
        assert "--gold:" in variables
        assert set(colors) == {
            "gold",
            "blue",
            "orange",
            "teal",
            "text",
            "muted",
            "surface",
            "surface-raised",
            "line",
        }