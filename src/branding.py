from pathlib import Path
from typing import Any

from src.team_registry import load_team_registry, player_initials


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAYER_IMAGE_CONFIG = PROJECT_ROOT / "config" / "player_images.json"


def resolve_asset_path(asset_path: str | None) -> Path | None:
    """Return an existing in-project asset path; reject paths outside assets."""
    if not asset_path:
        return None
    root = (PROJECT_ROOT / "assets").resolve()
    candidate = (PROJECT_ROOT / asset_path).resolve()
    if root not in candidate.parents or not candidate.is_file():
        return None
    return candidate


def resolve_team_brand(
    team_name: str,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    team_registry = registry or load_team_registry()
    for team in team_registry.get("teams", []):
        if team_name == team["name"]:
            return {
                **team,
                "asset_path": resolve_asset_path(team.get("logo_path")),
                "identity_status": "verified_active_name",
                "fallback_initials": team["short_name"],
            }
        if team_name in team.get("historical_aliases", []):
            return {
                **team,
                "asset_path": resolve_asset_path(team.get("logo_path")),
                "identity_status": "documented_historical_alias",
                "fallback_initials": team["short_name"],
            }
    return {
        "name": team_name,
        "short_name": None,
        "historical_aliases": [],
        "logo_path": None,
        "primary_color": None,
        "secondary_color": None,
        "asset_path": None,
        "identity_status": "unmapped_historical_identity",
        "fallback_initials": player_initials(team_name),
    }


def resolve_player_image(
    player_name: str,
    player_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if player_mapping is None:
        import json

        player_mapping = json.loads(PLAYER_IMAGE_CONFIG.read_text(encoding="utf-8"))
    player = player_mapping.get("players", {}).get(player_name)
    if not player:
        return {"image_path": None, "initials": player_initials(player_name), "verified": False}

    image_path = resolve_asset_path(player.get("image_path"))
    if not image_path or not player.get("verified"):
        return {"image_path": None, "initials": player_initials(player_name), "verified": False}
    return {"image_path": image_path, "initials": player_initials(player_name), "verified": True}