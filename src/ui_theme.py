THEME_VARIABLES = {
    "Dark": {
        "app-bg": "#0b1220",
        "sidebar-bg": "#09101d",
        "surface": "#111b2b",
        "surface-raised": "#172438",
        "line": "#26364b",
        "text": "#edf3f8",
        "muted": "#9eacbd",
        "gold": "#f1bb4b",
        "blue": "#49bce8",
        "orange": "#f47c48",
        "teal": "#36c6a4",
    },
    "Light": {
        "app-bg": "#f3f6fa",
        "sidebar-bg": "#e7edf4",
        "surface": "#ffffff",
        "surface-raised": "#f9fbfd",
        "line": "#d5dee8",
        "text": "#152235",
        "muted": "#596a7f",
        "gold": "#a86b00",
        "blue": "#087ca8",
        "orange": "#c84d1c",
        "teal": "#087f68",
    },
}


def theme_variables(mode: str) -> str:
    values = THEME_VARIABLES.get(mode, THEME_VARIABLES["Dark"])
    return "\n".join(f"--{name}: {value};" for name, value in values.items())


def plotly_template(mode: str) -> str:
    return "plotly_white" if mode == "Light" else "plotly_dark"


def chart_colors(mode: str) -> dict[str, str]:
    values = THEME_VARIABLES.get(mode, THEME_VARIABLES["Dark"])
    return {
        "gold": values["gold"],
        "blue": values["blue"],
        "orange": values["orange"],
        "teal": values["teal"],
        "text": values["text"],
        "muted": values["muted"],
        "surface": values["surface"],
        "surface-raised": values["surface-raised"],
        "line": values["line"],
    }