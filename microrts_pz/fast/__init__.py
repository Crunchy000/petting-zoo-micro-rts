# SPDX-License-Identifier: GPL-3.0-only
"""Batched, compiled MicroRTS for fast training (needs ``numba``: ``pip install microrts-pz[fast]``).

``BatchGame`` steps many games of the same map size in one call, spread over
all CPU cores, and returns gridnet observations, action masks and rewards as
NumPy arrays. It runs the same rules as ``microrts_pz.engine`` (checked tick
for tick by ``tests/test_fast.py``) and has compiled versions of the scripted
bots, so a whole self-play rollout needs no Python per unit or per game.

    from microrts_pz.fast import BatchGame
    games = BatchGame(["maps/16x16/basesWorkers16x16.xml"], num_games=256, max_steps=2000)
    games.reset(seed=0)
    obs, mask = games.observe(player=0)              # (B, H, W, C), (B, H*W, 78)
    a1 = games.bot_actions(player=1, kind="rush", harvesters=2, army=("Light", "Ranged", "Heavy"))
    raw, over, trunc, winner = games.step(my_actions, a1)

Finished games restart on the next call to ``step`` (``auto_reset=True``);
the returned flags and rewards describe the step that ended them.

Partial observability and the non-deterministic unit table are not supported
here; use ``microrts_pz.microrts_v0`` for those. Random draws (damage ranges,
the random move-conflict rule, ``RandomBot``) come from Numba's generator, so
they do not reproduce the pure-Python engine's sequences.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..engine import load_map
from ..game import MicroRTSGame
from ..unit_types import MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH, VERSION_ORIGINAL, UnitTypeTable
from . import _kernels as K

KINDS = {"none": 0, "learner": 0, "random": 1, "rush": 2}
REWARD_CATEGORY = {"Worker": 2, "Barracks": 3, "Base": 3, "Light": 5, "Heavy": 5, "Ranged": 5}


def unit_type_arrays(utt: UnitTypeTable) -> tuple[np.ndarray, np.ndarray]:
    """``UTT[n_types, NFT]`` and ``PROD[n_types, max_produces]`` for the kernels."""
    n = len(utt)
    table = np.zeros((n, K.NFT), dtype=np.int64)
    width = max(1, max(len(ut.produces) for ut in utt.unit_types))
    prod = np.full((n, width), -1, dtype=np.int64)
    for ut in utt.unit_types:
        r = table[ut.id]
        r[K.T_COST], r[K.T_HP], r[K.T_MIND], r[K.T_MAXD] = ut.cost, ut.hp, ut.min_damage, ut.max_damage
        r[K.T_RANGE], r[K.T_PTIME], r[K.T_MTIME], r[K.T_ATIME] = ut.attack_range, ut.produce_time, ut.move_time, ut.attack_time
        r[K.T_HTIME], r[K.T_HAMT], r[K.T_SIGHT] = ut.harvest_time, ut.harvest_amount, ut.sight_radius
        r[K.T_ISRES], r[K.T_ISSTOCK], r[K.T_CANHARV] = ut.is_resource, ut.is_stockpile, ut.can_harvest
        r[K.T_CANMOVE], r[K.T_CANATT] = ut.can_move, ut.can_attack
        r[K.T_REWARD] = REWARD_CATEGORY.get(ut.name, 0)
        for k, p in enumerate(ut.produces):
            prod[ut.id, k] = p.id
    return table, prod


class BatchGame:
    """``num_games`` MicroRTS games stepped together.

    ``maps`` is a list of map paths of the same size; game ``i`` plays
    ``maps[i % len(maps)]`` unless ``map_index`` says otherwise.
    """

    def __init__(
        self,
        maps: Sequence[str] | str,
        num_games: int,
        max_steps: int = 2000,
        unit_type_table_version: int = VERSION_ORIGINAL,
        move_conflict_resolution: int = MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH,
        auto_reset: bool = True,
        map_index: Sequence[int] | None = None,
    ):
        maps = [maps] if isinstance(maps, str) else list(maps)
        self.utt = UnitTypeTable(unit_type_table_version, move_conflict_resolution)
        specs = [load_map(m) for m in maps]
        sizes = {(s.height, s.width) for s in specs}
        if len(sizes) != 1:
            raise ValueError(f"all maps must have the same size, got {sorted(sizes)}")
        self.height, self.width = sizes.pop()
        ref = MicroRTSGame(maps[0], self.utt, map_spec=specs[0])
        self.num_cells = ref.num_cells
        self.num_channels = ref.num_channels
        self.action_dims = ref.action_dims
        self.attack_grid = ref.attack_grid
        self.mask_size = ref.mask_size
        self.offsets = np.asarray(ref.mask_offsets, dtype=np.int64)
        self.n_types = len(self.utt)
        self.maps = maps
        self.num_games = num_games
        self.max_steps = max_steps
        self.auto_reset = auto_reset
        self.strategy = move_conflict_resolution

        self.UTT, self.PROD = unit_type_arrays(self.utt)
        cells = self.num_cells
        nmax = max(len(s.units) for s in specs)
        self.map_units = np.zeros((len(specs), max(nmax, 1), 7), dtype=np.int64)
        self.map_nunits = np.zeros(len(specs), dtype=np.int64)
        self.map_res = np.zeros((len(specs), 2), dtype=np.int64)
        self.map_ter = np.zeros((len(specs), cells), dtype=np.int8)
        for i, s in enumerate(specs):
            for j, (uid, tname, player, x, y, res, hp) in enumerate(s.units):
                self.map_units[i, j] = (uid, player, self.utt[tname].id, x, y, res, hp)
            self.map_nunits[i] = len(s.units)
            self.map_res[i] = s.resources
            self.map_ter[i] = np.asarray(s.terrain, dtype=np.int8)
        self.map_idx = np.asarray(map_index if map_index is not None else np.arange(num_games) % len(specs), dtype=np.int64)

        cap = 4 * cells + nmax
        self.U = np.zeros((num_games, cap, K.NF), dtype=np.int64)
        self.G = np.zeros((num_games, K.NG), dtype=np.int64)
        self.OCC = np.full((num_games, cells), -1, dtype=np.int64)
        self.TER = np.zeros((num_games, cells), dtype=np.int8)
        self._raw = np.zeros((num_games, 2, 6), dtype=np.float64)
        self._over = np.zeros(num_games, dtype=np.bool_)
        self._trunc = np.zeros(num_games, dtype=np.bool_)
        self._winner = np.zeros(num_games, dtype=np.int64)
        self._ids = {name: self.utt[name].id for name in ("Worker", "Barracks", "Light")}

    # ------------------------------------------------------------------ api
    def reset(self, seed: int | None = None, games: Sequence[int] | None = None) -> None:
        """Restart ``games`` (all by default). ``seed`` seeds Numba's generator."""
        if seed is not None:
            K.seed(int(seed))
        for g in range(self.num_games) if games is None else games:
            m = self.map_idx[g]
            K.reset_game(self.U, self.G, self.OCC, self.TER, int(g), self.width,
                         self.map_units[m, : self.map_nunits[m]], self.map_res[m], self.map_ter[m])

    def step(self, action0: np.ndarray, action1: np.ndarray):
        """Advance every game one tick.

        ``action0``/``action1``: ``(B, H*W, 7)`` integer gridnet actions.
        Returns ``(raw_rewards (B, 2, 6), gameover (B,), truncated (B,), winner (B,))``;
        ``winner`` is -1 for no winner.
        """
        a0 = np.ascontiguousarray(action0, dtype=np.int64).reshape(self.num_games, self.num_cells, -1)
        a1 = np.ascontiguousarray(action1, dtype=np.int64).reshape(self.num_games, self.num_cells, -1)
        K.step_batch(self.U, self.G, self.OCC, self.TER, self.UTT, self.PROD, self.width, self.height,
                     self.attack_grid, self.strategy, a0, a1, self._raw, self._over, self._winner,
                     self.max_steps, self._trunc, self.map_idx, self.map_units, self.map_nunits,
                     self.map_res, self.map_ter, self.auto_reset)
        return self._raw.copy(), self._over.copy(), self._trunc.copy(), self._winner.copy()

    def observe(self, player: int | np.ndarray, obs: np.ndarray | None = None, mask: np.ndarray | None = None):
        """One-hot observations ``(B, H, W, C)`` and masks ``(B, H*W, 78)`` (int8) for ``player`` (scalar or per game)."""
        players = np.broadcast_to(np.asarray(player, dtype=np.int64), (self.num_games,)).copy()
        if obs is None:
            obs = np.empty((self.num_games, self.num_cells, self.num_channels), dtype=np.int8)
        if mask is None:
            mask = np.empty((self.num_games, self.num_cells, self.mask_size), dtype=np.int8)
        K.observe_batch(self.U, self.G, self.OCC, self.TER, self.UTT, self.PROD, self.width, self.height,
                        self.attack_grid, self.n_types, self.offsets, players,
                        obs.reshape(self.num_games, self.num_cells, self.num_channels), mask)
        return obs.reshape(self.num_games, self.height, self.width, self.num_channels), mask

    def bot_actions(self, player, kind="rush", harvesters=1, army=None, masks: np.ndarray | None = None):
        """Scripted actions ``(B, H*W, 7)``.

        ``kind``: ``"rush"`` (``WorkerRushBot``), ``"random"`` (``RandomBot``,
        needs ``masks`` for ``player``) or ``"none"``, either one for all games
        or a sequence per game; likewise ``player`` and ``harvesters``. ``army``
        is a unit type name, a tuple of names or ``None`` for all games, or a
        list with one of those per game.
        """
        B = self.num_games
        players = np.broadcast_to(np.asarray(player, dtype=np.int64), (B,)).copy()
        kinds = np.array([KINDS[k] for k in ([kind] * B if isinstance(kind, str) else kind)], dtype=np.int64)
        harv = np.broadcast_to(np.asarray(harvesters, dtype=np.int64), (B,)).copy()
        armies = army if isinstance(army, list) and len(army) == B else [army] * B
        army_ids = np.zeros((B, 3), dtype=np.int64)
        n_army = np.zeros(B, dtype=np.int64)
        for g, a in enumerate(armies):
            names = () if a is None else ((a,) if isinstance(a, str) else tuple(a))
            n_army[g] = len(names)
            for k, name in enumerate(names):
                army_ids[g, k] = self.utt[name].id
        if masks is None:
            if (kinds == 1).any():
                raise ValueError("the random bot needs the action masks of its player")
            masks = np.zeros((B, 1, self.mask_size), dtype=np.int8)
        out = np.zeros((B, self.num_cells, len(self.action_dims)), dtype=np.int64)
        K.bots_batch(self.U, self.G, self.OCC, self.TER, self.UTT, self.PROD, self.width, self.height,
                     self.attack_grid, self.offsets, players, kinds, harv, army_ids, n_army,
                     self._ids["Worker"], self._ids["Barracks"], self._ids["Light"], masks, out)
        return out

    @property
    def time(self) -> np.ndarray:
        return self.G[:, K.G_TIME].copy()

    @property
    def episode_steps(self) -> np.ndarray:
        return self.G[:, K.G_STEPS].copy()

    def unit_types_seen(self, player: int) -> list[set[str]]:
        """Names of the unit types ``player`` currently owns, per game."""
        out = []
        for g in range(self.num_games):
            n = self.G[g, K.G_NSLOT]
            u = self.U[g, :n]
            alive = (u[:, K.ALIVE] == 1) & (u[:, K.PLAYER] == player)
            out.append({self.utt[int(t)].name for t in np.unique(u[alive, K.TYPE])})
        return out


__all__ = ["BatchGame"]
