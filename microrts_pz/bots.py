# SPDX-License-Identifier: GPL-3.0-only
"""Small scripted opponents for testing and demos.

They read the engine state directly (not the agent observation), so they
are meant as baselines/sparring partners rather than fair agents under
partial observability. ``WorkerRushBot`` is a simplified take on MicroRTS's
``WorkerRush`` AI (GPL-3.0, (c) Santiago Ontañón).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .engine import (
    DIRECTION_OFFSET_X,
    DIRECTION_OFFSET_Y,
    TYPE_ATTACK_LOCATION,
    TYPE_HARVEST,
    TYPE_MOVE,
    TYPE_PRODUCE,
    TYPE_RETURN,
    GameState,
    Unit,
)
from .game import MicroRTSGame


class RandomBot:
    """Uniformly random valid action components (via the action mask)."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, game: MicroRTSGame, player: int) -> np.ndarray:
        mask = game.action_mask(player)
        out = np.zeros((game.num_cells, len(game.action_dims)), dtype=np.int64)
        off = game.mask_offsets
        for k in range(len(game.action_dims)):
            sub = mask[:, off[k] : off[k + 1]] > 0
            out[:, k] = np.argmax(np.where(sub, self.rng.random(sub.shape) + 1.0, 0.0), axis=1)
        return out


class WorkerRushBot:
    """Train workers non-stop, keep ``harvesters`` gathering, send the rest at the enemy.

    With ``army="Light"`` (or ``"Heavy"``/``"Ranged"``) a harvester builds one
    barracks once it can afford it, the base trains only enough workers to
    keep the economy going, and the barracks trains that unit type instead.
    """

    def __init__(self, harvesters: int = 1, army: str | None = None):
        self.harvesters = harvesters
        self.army = army

    # -- helpers
    @staticmethod
    def _first_step(gs: GameState, u: Unit, goals: set[tuple[int, int]]) -> int | None:
        """Direction of the first move on a shortest path to any cell in ``goals``."""
        if not goals:
            return None
        start = (u.x, u.y)
        seen = {start: None}
        q = deque([start])
        while q:
            cx, cy = q.popleft()
            if (cx, cy) in goals and (cx, cy) != start:
                # walk back to the first step
                node = (cx, cy)
                while seen[node] != start:
                    node = seen[node]
                for d in range(4):
                    if (start[0] + DIRECTION_OFFSET_X[d], start[1] + DIRECTION_OFFSET_Y[d]) == node:
                        return d
            for d in range(4):
                nx, ny = cx + DIRECTION_OFFSET_X[d], cy + DIRECTION_OFFSET_Y[d]
                if not (0 <= nx < gs.width and 0 <= ny < gs.height) or (nx, ny) in seen:
                    continue
                if gs.terrain[ny, nx] or gs.unit_at(nx, ny) is not None:
                    continue
                seen[(nx, ny)] = (cx, cy)
                q.append((nx, ny))
        return None

    @staticmethod
    def _adjacent_cells(gs: GameState, targets: list[Unit]) -> set[tuple[int, int]]:
        cells = set()
        for t in targets:
            for d in range(4):
                cells.add((t.x + DIRECTION_OFFSET_X[d], t.y + DIRECTION_OFFSET_Y[d]))
        return cells

    def act(self, game: MicroRTSGame, player: int) -> np.ndarray:
        gs = game.gs
        out = np.zeros((game.num_cells, len(game.action_dims)), dtype=np.int64)
        center = game.attack_grid // 2
        enemies = [u for u in gs.units if u.player == 1 - player]
        bases = [u for u in gs.units if u.player == player and u.type.is_stockpile]
        resources = [u for u in gs.units if u.type.is_resource]
        workers = sorted((u for u in gs.units if u.player == player and u.type.can_harvest), key=lambda u: u.id)
        harvester_ids = {w.id for w in workers[: self.harvesters]} if bases else set()
        has_barracks = any(u.player == player and u.type.name == "Barracks" for u in gs.units)
        building = any(
            a.unit.player == player and a.action.type == TYPE_PRODUCE and a.action.unit_type.name == "Barracks"
            for a in gs.assignments.values()
        )
        want_barracks = self.army is not None and not has_barracks and not building

        for u in gs.units:
            if u.player != player or not gs.is_idle(u):
                continue
            row = out[u.x + u.y * gs.width]
            legal = gs.unit_actions(u)
            attacks = [a for a in legal if a.type == TYPE_ATTACK_LOCATION]
            if attacks:
                a = attacks[0]
                row[0] = TYPE_ATTACK_LOCATION
                row[6] = (center + a.y - u.y) * game.attack_grid + (center + a.x - u.x)
                continue
            produces = [a for a in legal if a.type == TYPE_PRODUCE]
            if not u.type.can_move:  # buildings
                if u.type.is_stockpile:
                    wanted = "Worker" if self.army is None or len(workers) < self.harvesters + 1 else None
                else:
                    wanted = self.army or "Light"
                options = [a for a in produces if a.unit_type.name == wanted]
                if options:
                    a = options[0]
                    row[0], row[4], row[5] = TYPE_PRODUCE, a.parameter, a.unit_type.id
                continue
            if want_barracks and u.id in harvester_ids:
                options = [a for a in produces if a.unit_type.name == "Barracks"]
                if options:
                    a = options[0]
                    row[0], row[4], row[5] = TYPE_PRODUCE, a.parameter, a.unit_type.id
                    want_barracks = False
                    continue
            if u.id in harvester_ids:
                if u.resources > 0:
                    ret = [a for a in legal if a.type == TYPE_RETURN]
                    if ret:
                        row[0], row[3] = TYPE_RETURN, ret[0].parameter
                        continue
                    goals = self._adjacent_cells(gs, bases)
                else:
                    har = [a for a in legal if a.type == TYPE_HARVEST]
                    if har:
                        row[0], row[2] = TYPE_HARVEST, har[0].parameter
                        continue
                    goals = self._adjacent_cells(gs, resources)
            else:
                goals = self._adjacent_cells(gs, enemies)
            d = self._first_step(gs, u, goals)
            if d is not None:
                row[0], row[1] = TYPE_MOVE, d
        return out


__all__ = ["RandomBot", "WorkerRushBot"]
