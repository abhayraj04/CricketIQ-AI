import json
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "team_branding.json"


def load_team_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Team registry not found: {registry_path}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported team registry schema version.")

    known_names = {team["name"] for team in registry.get("teams", [])}
    for season, entry in registry.get("seasons", {}).items():
        if entry.get("status") == "participation_verified_completed":
            unknown_names = set(entry.get("teams", [])) - known_names
            if unknown_names:
                raise ValueError(f"Season {season} references unknown teams: {sorted(unknown_names)}")
    return registry


def future_eligible_teams(
    registry: dict[str, Any],
    season: int,
    today: date | None = None,
) -> list[str]:
    current_date = today or date.today()
    if season <= current_date.year:
        return []
    entry = registry.get("seasons", {}).get(str(season))
    if not entry or entry.get("status") != "participation_verified_future":
        return []
    return list(entry.get("teams", []))


def validate_future_matchup(
    registry: dict[str, Any],
    season: int,
    team_a: str,
    team_b: str,
    today: date | None = None,
) -> None:
    eligible = future_eligible_teams(registry, season, today)
    if not eligible:
        raise ValueError(f"Future predictions are disabled: IPL {season} participation is not verified.")
    if team_a == team_b:
        raise ValueError("Choose two different teams.")
    invalid = {team_a, team_b} - set(eligible)
    if invalid:
        raise ValueError(f"Teams are not verified for IPL {season}: {sorted(invalid)}")


def team_short_name(team_name: str, registry: dict[str, Any]) -> str | None:
    for team in registry.get("teams", []):
        if team_name == team["name"] or team_name in team.get("historical_aliases", []):
            return team["short_name"]
    return None


def player_initials(player_name: str) -> str:
    words = [word for word in player_name.strip().split() if word]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return f"{words[0][0]}{words[-1][0]}".upper()