# SPDX-License-Identifier: GPL-3.0-only
"""Gridnet-style two-player MicroRTS game: observations, action masks, rewards.

Mirrors ``tests.JNIGridnetClientSelfPlay`` and ``ai.reward.*`` from MicroRTS
(GPL-3.0, (c) Santiago Ontañón) and the feature-plane encoding from
MicroRTS-Py (MIT, (c) 2020 gym-microrts developers, (c) 2022 Farama Foundation).
"""

from __future__ import annotations

import numpy as np

from .engine import (
    NUMBER_OF_ACTION_TYPES,
    TYPE_ATTACK_LOCATION,
    TYPE_HARVEST,
    TYPE_MOVE,
    TYPE_NONE,
    TYPE_PRODUCE,
    TYPE_RETURN,
    GameState,
    MapSpec,
    load_map,
)
from .unit_types import VERSION_ORIGINAL, UnitTypeTable

REWARD_NAMES = (
    "win_loss",
    "resource_gather",
    "produce_worker",
    "produce_building",
    "attack",
    "produce_combat_unit",
)


class MicroRTSGame:
    """One MicroRTS match between player 0 and player 1, stepped one tick at a time.

    Action per player: integer array of shape ``(height * width, 7)`` with
    components ``[action_type(6), move_dir(4), harvest_dir(4), return_dir(4),
    produce_dir(4), produce_type(n_types), attack_target(attack_grid**2)]``.
    """

    def __init__(
        self,
        map_path: str = "maps/16x16/basesWorkers16x16.xml",
        utt: UnitTypeTable | None = None,
        partial_obs: bool = False,
        rng: np.random.Generator | None = None,
        map_spec: MapSpec | None = None,
    ):
        self.map_path = map_path
        self.map_spec = map_spec if map_spec is not None else load_map(map_path)
        self.utt = utt if utt is not None else UnitTypeTable(VERSION_ORIGINAL)
        self.partial_obs = partial_obs
        self.rng = rng if rng is not None else np.random.default_rng()
        self.height, self.width = self.map_spec.height, self.map_spec.width
        self.num_cells = self.height * self.width
        self.n_types = len(self.utt)
        self.attack_grid = 2 * self.utt.max_attack_range + 1
        self.action_dims = np.array([NUMBER_OF_ACTION_TYPES, 4, 4, 4, 4, self.n_types, self.attack_grid**2], dtype=np.int64)
        self.mask_offsets = np.concatenate([[0], np.cumsum(self.action_dims)])
        self.mask_size = int(self.mask_offsets[-1])
        # [hp(5), resources(5), owner(3), unit type(n+1), current action(6), terrain(2)] (+ visibility(2) if partial)
        self.num_planes = [5, 5, 3, self.n_types + 1, NUMBER_OF_ACTION_TYPES, 2]
        self.plane_offsets = np.concatenate([[0], np.cumsum(self.num_planes)])
        self.num_channels = int(self.plane_offsets[-1]) + (2 if partial_obs else 0)
        self.gs: GameState | None = None

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        self.gs = self.map_spec.new_game(self.utt, self.rng)

    def step(self, action0: np.ndarray, action1: np.ndarray) -> tuple[np.ndarray, bool]:
        """Issue both players' actions, run one tick, return ``(raw_rewards[2, 6], gameover)``."""
        gs = self.gs
        before = {u.x + u.y * gs.width: u.player for u in gs.units}
        trace: list[list] = []
        for player, action in ((0, action0), (1, action1)):
            grid = np.asarray(action, dtype=np.int64).reshape(self.num_cells, -1)
            visible = gs.visible_units(player) if self.partial_obs else None
            pairs = gs.player_action_from_grid(player, grid, self.attack_grid, visible)
            gs.issue_safe(pairs)
            trace.extend(pairs)
        gameover = gs.cycle()
        rewards = np.zeros((2, len(REWARD_NAMES)), dtype=np.float64)
        winner = gs.winner() if gameover else None
        for player in (0, 1):
            r = rewards[player]
            if gameover:
                r[0] = 1.0 if winner == player else -1.0
            for u, ua in trace:
                if u.player != player:
                    continue
                t = ua.type
                if t == TYPE_HARVEST or t == TYPE_RETURN:
                    r[1] += 1
                elif t == TYPE_PRODUCE and ua.unit_type is not None:
                    name = ua.unit_type.name
                    if name == "Worker":
                        r[2] += 1
                    elif name in ("Barracks", "Base"):
                        r[3] += 1
                    elif name in ("Light", "Heavy", "Ranged"):
                        r[5] += 1
                elif t == TYPE_ATTACK_LOCATION:
                    target = before.get(ua.x + ua.y * gs.width) if 0 <= ua.x < gs.width and 0 <= ua.y < gs.height else None
                    if target == 1 - player:
                        r[4] += 1
                    elif target == player:
                        r[4] -= 1
        return rewards, gameover

    # ---------------------------------------------------------- observation
    def raw_observation(self, player: int) -> np.ndarray:
        """Integer feature maps ``[6 or 8, H, W]`` (``GameState.getVectorObservation``)."""
        gs = self.gs
        n = 8 if self.partial_obs else 6
        obs = np.zeros((n, self.height, self.width), dtype=np.int64)
        units = gs.visible_units(player) if self.partial_obs else gs.units
        for u in units:
            a = gs.assignments.get(u.id)
            obs[0, u.y, u.x] = u.hp
            obs[1, u.y, u.x] = u.resources
            if u.player >= 0:
                obs[2, u.y, u.x] = ((u.player + player) % 2) + 1
            obs[3, u.y, u.x] = u.type.id + 1
            if a is not None:
                obs[4, u.y, u.x] = a.action.type
        obs[5] = gs.terrain
        if self.partial_obs:
            obs[6] = self._visibility([u for u in units if u.player == player])
            obs[7] = self._visibility([u for u in units if u.player >= 0 and u.player != player])
        return obs

    def _visibility(self, units) -> np.ndarray:
        vis = np.zeros((self.height, self.width), dtype=np.int64)
        if not units:
            return vis
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        for u in units:
            r = u.type.sight_radius
            vis |= ((xs - u.x) ** 2 + (ys - u.y) ** 2 <= r * r).astype(np.int64)
        return vis

    def observation(self, player: int) -> np.ndarray:
        """One-hot feature planes, shape ``(H, W, C)``, dtype int8."""
        raw = self.raw_observation(player)
        out = np.zeros((self.height, self.width, self.num_channels), dtype=np.int8)
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        for i, n in enumerate(self.num_planes):
            vals = np.clip(raw[i], 0, n - 1)
            out[ys, xs, self.plane_offsets[i] + vals] = 1
        if self.partial_obs:
            out[:, :, -2] = raw[6]
            out[:, :, -1] = raw[7]
        return out

    def action_mask(self, player: int) -> np.ndarray:
        """Valid action components per cell, shape ``(H*W, mask_size)``, dtype int8.

        Cells without an idle unit of ``player`` are all zero
        (``JNIGridnetClientSelfPlay.getMasks`` minus the source-unit column).
        """
        gs = self.gs
        mask = np.zeros((self.num_cells, self.mask_size), dtype=np.int8)
        off = self.mask_offsets
        center = self.attack_grid // 2
        for u in gs.units:
            if u.player != player or u.id in gs.assignments:
                continue
            row = mask[u.x + u.y * self.width]
            for ua in gs.unit_actions(u):
                t = ua.type
                row[t] = 1
                if t == TYPE_MOVE:
                    row[off[1] + ua.parameter] = 1
                elif t == TYPE_HARVEST:
                    row[off[2] + ua.parameter] = 1
                elif t == TYPE_RETURN:
                    row[off[3] + ua.parameter] = 1
                elif t == TYPE_PRODUCE:
                    row[off[4] + ua.parameter] = 1
                    row[off[5] + ua.unit_type.id] = 1
                elif t == TYPE_ATTACK_LOCATION:
                    rx, ry = ua.x - u.x, ua.y - u.y
                    row[off[6] + (center + ry) * self.attack_grid + (center + rx)] = 1
        return mask

    def source_unit_mask(self, player: int) -> np.ndarray:
        """1 for each cell holding an idle unit of ``player``."""
        m = np.zeros(self.num_cells, dtype=np.int8)
        for u in self.gs.units:
            if u.player == player and u.id not in self.gs.assignments:
                m[u.x + u.y * self.width] = 1
        return m

    @property
    def gameover(self) -> bool:
        return self.gs.gameover()

    @property
    def winner(self) -> int:
        return self.gs.winner()


__all__ = ["MicroRTSGame", "REWARD_NAMES", "TYPE_NONE"]
