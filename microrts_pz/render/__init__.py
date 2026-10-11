# SPDX-License-Identifier: GPL-3.0-only
"""Renderers: browser canvas viewer (live and replay), RGB arrays and ANSI text."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np

from ..engine import TYPE_ATTACK_LOCATION, TYPE_NONE, TYPE_PRODUCE

if TYPE_CHECKING:
    from ..engine import GameState
    from ..game import MicroRTSGame

from .live import LiveViewer

VIEWER_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer.html")
DATA_MARKER = "/*__MICRORTS_DATA__*/"


# ---------------------------------------------------------------- encoding
def encode_meta(game: "MicroRTSGame", map_path: str, agents: list[str]) -> dict:
    gs = game.gs
    return {
        "map": os.path.basename(map_path),
        "width": game.width,
        "height": game.height,
        "terrain": "".join(str(int(v)) for v in (gs.terrain if gs is not None else np.zeros((game.height, game.width))).ravel()),
        "players": list(agents),
        "unitTypes": [
            {"name": ut.name, "hp": ut.hp, "attackRange": ut.attack_range, "cost": ut.cost} for ut in game.utt.unit_types
        ],
    }


def encode_frame(gs: "GameState") -> dict:
    """Compact per-tick snapshot.

    Units are ``[id, type, player, x, y, hp, carried, action, start, eta, p1, p2]``
    where ``action`` is -1 when idle; for attacks ``p1, p2`` is the target cell,
    for produce ``p1`` is the direction and ``p2`` the unit type, otherwise ``p1``
    is the direction.
    """
    units = []
    for u in gs.units:
        a = gs.assignments.get(u.id)
        if a is None:
            act = [-1, 0, 0, 0, 0]
        else:
            ua = a.action
            if ua.type == TYPE_ATTACK_LOCATION:
                p1, p2 = ua.x, ua.y
            elif ua.type == TYPE_PRODUCE:
                p1, p2 = ua.parameter, ua.unit_type.id
            elif ua.type == TYPE_NONE:
                p1, p2 = -1, 0
            else:
                p1, p2 = ua.parameter, 0
            act = [ua.type, a.time, ua.eta(u), p1, p2]
        units.append([u.id, u.type.id, u.player, u.x, u.y, u.hp, u.resources] + act)
    out = {"t": gs.time, "r": list(gs.player_resources), "u": units}
    if gs.gameover():
        out["winner"] = gs.winner()
    return out


def save_replay_html(meta: dict, frames: list[dict], path: str, title: str | None = None) -> str:
    """Write a self-contained HTML replay player for ``frames``."""
    with open(VIEWER_HTML, encoding="utf-8") as f:
        html = f.read()
    payload = json.dumps({"meta": meta, "frames": frames, "title": title}, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # keep the JSON from closing the script tag
    html = html.replace(DATA_MARKER, f"window.MICRORTS_REPLAY = {payload};", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# --------------------------------------------------------------- rgb array
PLAYER_COLORS = np.array([[63, 167, 255], [255, 93, 115]], dtype=np.uint8)
NEUTRAL = np.array([80, 220, 160], dtype=np.uint8)
BG_A = np.array([22, 28, 38], dtype=np.uint8)
BG_B = np.array([25, 32, 43], dtype=np.uint8)
WALL = np.array([60, 68, 84], dtype=np.uint8)


def _shape_mask(kind: str, n: int) -> np.ndarray:
    c = (n - 1) / 2.0
    ys, xs = np.mgrid[0:n, 0:n]
    dx, dy = (xs - c) / (n / 2.0), (ys - c) / (n / 2.0)
    if kind == "Base":
        return (np.abs(dx) <= 0.85) & (np.abs(dy) <= 0.85)
    if kind == "Barracks":
        return (np.abs(dx) <= 0.8) & (np.abs(dy) <= 0.65)
    if kind == "Worker":
        return dx**2 + dy**2 <= 0.45**2
    if kind == "Light":
        return dx**2 + dy**2 <= 0.6**2
    if kind == "Heavy":
        return (np.abs(dx) + np.abs(dy) <= 1.1) & (np.abs(dx) <= 0.78) & (np.abs(dy) <= 0.78)
    if kind == "Ranged":
        return np.abs(dx) + np.abs(dy) <= 0.75
    if kind == "Resource":
        return (np.abs(dx) * 1.6 + np.abs(dy) <= 0.9)
    return dx**2 + dy**2 <= 0.5**2


def render_rgb(gs: "GameState", cell: int = 32) -> np.ndarray:
    """Render the board as an ``(H*cell, W*cell, 3)`` uint8 image."""
    h, w = gs.height, gs.width
    img = np.empty((h * cell, w * cell, 3), dtype=np.uint8)
    checker = (np.add.outer(np.arange(h), np.arange(w)) % 2).astype(bool)
    tiles = np.where(checker[..., None], BG_B, BG_A)
    img[:] = np.repeat(np.repeat(tiles, cell, axis=0), cell, axis=1)
    walls = np.repeat(np.repeat(gs.terrain == 1, cell, axis=0), cell, axis=1)
    img[walls] = WALL
    masks: dict[str, np.ndarray] = {}
    for u in gs.units:
        m = masks.setdefault(u.type.name, _shape_mask(u.type.name, cell))
        color = NEUTRAL if u.player < 0 else PLAYER_COLORS[u.player]
        block = img[u.y * cell : (u.y + 1) * cell, u.x * cell : (u.x + 1) * cell]
        block[m] = color
        if u.player >= 0 and u.type.hp > 1:
            frac = max(0.0, min(1.0, u.hp / u.type.hp))
            bar = max(1, cell // 10)
            block[-bar:, :] = (40, 40, 40)
            block[-bar:, : int(round(frac * cell))] = (120, 230, 120)
        if u.resources > 0 and not u.type.is_resource:
            s = max(2, cell // 6)
            block[:s, -s:] = NEUTRAL
    return img


# ------------------------------------------------------------------- ansi
_GLYPHS = {"Base": "b", "Barracks": "k", "Worker": "w", "Light": "l", "Heavy": "h", "Ranged": "r"}


def render_ansi(gs: "GameState") -> str:
    """Text board: upper case = player 0, lower case = player 1, ``$`` resource, ``#`` wall."""
    grid = [["#" if gs.terrain[y, x] else "." for x in range(gs.width)] for y in range(gs.height)]
    for u in gs.units:
        if u.type.is_resource:
            ch = "$"
        else:
            ch = _GLYPHS.get(u.type.name, "?")
            ch = ch.upper() if u.player == 0 else ch
        grid[u.y][u.x] = ch
    header = f"t={gs.time}  resources: P0={gs.player_resources[0]} P1={gs.player_resources[1]}"
    return header + "\n" + "\n".join("".join(r) for r in grid)


__all__ = [
    "LiveViewer",
    "encode_frame",
    "encode_meta",
    "render_ansi",
    "render_rgb",
    "save_replay_html",
]
