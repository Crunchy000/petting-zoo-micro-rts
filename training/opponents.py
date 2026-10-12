# SPDX-License-Identifier: GPL-3.0-only
"""The scripted opponents the trainers and ``play.py`` know by name.

Each name maps to a ``microrts_pz.bots`` bot (``BOTS``, for the Python
engine) and to its settings (``BOT_SPECS``: kind, harvesters, army), which
the batch engines' ``bot_actions`` and the JAX trainer take.
"""

from __future__ import annotations

from microrts_pz.bots import RandomBot, WorkerRushBot

#: name -> (kind, harvesters, army) as ``BatchGame.bot_actions`` / ``JaxBatchGame.bot_actions`` take them.
BOT_SPECS = {
    "random": ("random", 1, None),
    "rush": ("rush", 1, None),
    "rush2": ("rush", 2, None),
    "light": ("rush", 2, "Light"),
    "heavy": ("rush", 2, "Heavy"),
    "ranged": ("rush", 2, "Ranged"),
    "mixed": ("rush", 2, ("Light", "Ranged", "Heavy")),
}


def make_bot(name: str, seed: int | None = None):
    """A fresh Python bot for ``name``."""
    kind, harvesters, army = BOT_SPECS[name]
    return RandomBot(seed) if kind == "random" else WorkerRushBot(harvesters=harvesters, army=army)


#: name -> factory taking a seed (kept for scripts written against ``train_selfplay.BOTS``).
BOTS = {name: (lambda seed, name=name: make_bot(name, seed)) for name in BOT_SPECS}
#: Bots the behaviour-cloning warm start imitates.
TEACHERS = ("rush", "rush2", "light", "heavy", "ranged", "mixed")


def parse_weights(spec: str) -> dict[str, float]:
    """``"mixed:2,random:1,self:3"`` -> ``{"mixed": 2.0, "random": 1.0, "self": 3.0}`` (a missing weight is 1)."""
    out = {}
    for part in spec.split(","):
        name, _, w = part.partition(":")
        name = name.strip()
        if name != "self" and name not in BOT_SPECS:
            raise ValueError(f"unknown opponent {name!r}; choose from {sorted(BOT_SPECS)} or 'self'")
        out[name] = float(w) if w else 1.0
    return out
