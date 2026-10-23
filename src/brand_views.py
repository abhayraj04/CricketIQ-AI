from html import escape
from typing import Any

import streamlit as st

from src.branding import resolve_player_image, resolve_team_brand


def render_team_identity(team_name: str, registry: dict[str, Any]) -> None:
    identity = resolve_team_brand(team_name, registry)
    status = (
        "Name in verified 2026 team registry"
        if identity["identity_status"] == "verified_active_name"
        else "Historical identity; no active-team mapping"
    )
    image_column, name_column = st.columns([0.14, 0.86], vertical_alignment="center")
    if identity["asset_path"]:
        image_column.image(str(identity["asset_path"]), width=34)
    else:
        image_column.markdown(
            f"<span class='team-identity-fallback'>{escape(identity['fallback_initials'] or '?')}</span>",
            unsafe_allow_html=True,
        )
    name_column.markdown(
        f"<div class='team-identity-copy'><strong>{escape(team_name)}</strong>"
        f"<small>{escape(status)}</small></div>",
        unsafe_allow_html=True,
    )


def render_player_identity(player_name: str) -> None:
    identity = resolve_player_image(player_name)
    image_column, name_column = st.columns([0.14, 0.86], vertical_alignment="center")
    if identity["image_path"]:
        image_column.image(str(identity["image_path"]), width=34)
    else:
        name_column.markdown(
            f"<span class='player-identity-fallback'>{escape(identity['initials'])}</span> "
            f"<strong>{escape(player_name)}</strong>",
            unsafe_allow_html=True,
        )
        return
    name_column.markdown(f"<strong>{escape(player_name)}</strong>", unsafe_allow_html=True)