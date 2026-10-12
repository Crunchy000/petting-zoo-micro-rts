# SPDX-License-Identifier: GPL-3.0-only
"""Batched MicroRTS in JAX, for training on a GPU or TPU (needs ``jax``: ``pip install microrts-pz[jax]``).

``JaxBatchGame`` has the same interface as ``microrts_pz.fast.BatchGame`` and
plays the same rules (checked tick for tick against ``microrts_pz.engine`` by
``tests/test_jax_engine.py``), but the whole batch is one array program, so it
runs on an accelerator next to the policy and can be traced into a jitted
rollout together with it:

    from microrts_pz.jax_engine import JaxBatchGame
    games = JaxBatchGame(["maps/16x16/basesWorkers16x16.xml"], num_games=4096, max_steps=2000)
    games.reset()
    obs, mask = games.observe(player=0)              # (B, H, W, C) int8, (B, H*W, 78) bool
    raw, over, trunc, winner = games.step(a0, a1)    # gridnet actions (B, H*W, 7)

For a fully jitted loop use the pure functions ``games.step_fn(state, a0, a1)``
and ``games.observe_fn(state, player)`` on ``games.state``.

Full observability and deterministic unit tables only (see ``core``).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..engine import MapSpec, load_map
from ..unit_types import MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH, VERSION_ORIGINAL, UnitTypeTable
from .core import Rules, State


class JaxBatchGame:
    """``num_games`` MicroRTS games stepped together as one JAX program.

    ``maps`` is a list of map paths (or ``MapSpec``s) of the same size; game ``i`` plays
    ``maps[i % len(maps)]`` unless ``map_index`` says otherwise. Finished games
    restart on the same ``step`` when ``auto_reset`` (the returned flags and
    rewards describe the step that ended them). ``move_conflict_resolution``
    is accepted for interface compatibility: with gridnet actions the three
    strategies never differ (see ``core``).
    """

    def __init__(
        self,
        maps: Sequence[str | MapSpec] | str,
        num_games: int,
        max_steps: int = 2000,
        unit_type_table_version: int = VERSION_ORIGINAL,
        move_conflict_resolution: int = MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH,
        auto_reset: bool = True,
        map_index: Sequence[int] | None = None,
        device: jax.Device | None = None,
    ):
        maps = [maps] if isinstance(maps, str) else list(maps)
        self.utt = UnitTypeTable(unit_type_table_version, move_conflict_resolution)
        specs = [m if isinstance(m, MapSpec) else load_map(m) for m in maps]
        sizes = {(s.height, s.width) for s in specs}
        if len(sizes) != 1:
            raise ValueError(f"all maps must have the same size, got {sorted(sizes)}")
        self.height, self.width = sizes.pop()
        self.rules = rules = Rules(self.utt, self.height, self.width)
        self.num_cells = rules.HW
        self.num_channels = rules.num_channels
        self.action_dims = rules.action_dims
        self.attack_grid = rules.AG
        self.mask_size = rules.mask_size
        self.n_types = rules.n_types
        self.maps = maps
        self.num_games = num_games
        self.max_steps = max_steps
        self.auto_reset = auto_reset
        self.device = device
        inits = [rules.initial_state(s, i) for i, s in enumerate(specs)]
        self.init_states = State(*[jnp.asarray(np.stack(f)) for f in zip(*inits)])
        self.map_idx = np.asarray(map_index if map_index is not None else np.arange(num_games) % len(specs), dtype=np.int32)
        self.state: State | None = None

        init_states = self.init_states

        def batch_step(state: State, a0, a1):
            s, raw, over, winner = jax.vmap(rules.step)(state, a0, a1)
            steps = s.steps + 1
            trunc = ~over & (steps >= max_steps)
            s = s._replace(steps=steps)
            if auto_reset:
                done = over | trunc
                fresh = jax.tree.map(lambda x: x[s.map_id], init_states)
                s = jax.tree.map(lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), s, fresh)
            return s, raw, over, trunc, winner

        def batch_observe(state: State, player):
            players = jnp.broadcast_to(jnp.asarray(player, jnp.int32), state.time.shape)
            obs = jax.vmap(rules.observation)(state, players)
            mask = jax.vmap(rules.action_mask)(state, players)
            return obs.reshape(-1, self.height, self.width, self.num_channels), mask

        def batch_reset(map_idx):
            return jax.tree.map(lambda x: x[map_idx], init_states)

        self.step_fn = jax.jit(batch_step)
        self.observe_fn = jax.jit(batch_observe)
        self.reset_fn = jax.jit(batch_reset)
        self.needs_decision_fn = jax.jit(jax.vmap(rules.needs_decision))

    # ------------------------------------------------------------------ api
    def reset(self, seed: int | None = None, games: Sequence[int] | None = None) -> None:
        """Restart ``games`` (all by default). ``seed`` is accepted for compatibility (the rules use no randomness)."""
        fresh = self.reset_fn(jnp.asarray(self.map_idx))
        if games is None or self.state is None:
            self.state = fresh
        else:
            sel = np.zeros(self.num_games, bool)
            sel[list(games)] = True
            sel = jnp.asarray(sel)
            self.state = jax.tree.map(lambda a, b: jnp.where(sel.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), self.state, fresh)
        if self.device is not None:
            self.state = jax.device_put(self.state, self.device)

    def step(self, action0, action1):
        """Advance every game one tick; returns NumPy ``(raw (B, 2, 6), gameover (B,), truncated (B,), winner (B,))``."""
        shape = (self.num_games, self.num_cells, len(self.action_dims))
        a0 = jnp.asarray(action0, jnp.int32).reshape(shape)
        a1 = jnp.asarray(action1, jnp.int32).reshape(shape)
        self.state, raw, over, trunc, winner = self.step_fn(self.state, a0, a1)
        return np.asarray(raw, np.float64), np.asarray(over), np.asarray(trunc), np.asarray(winner, np.int64)

    def observe(self, player):
        """One-hot observations ``(B, H, W, C)`` int8 and masks ``(B, H*W, mask_size)`` for ``player`` (scalar or per game)."""
        obs, mask = self.observe_fn(self.state, player)
        return np.asarray(obs), np.asarray(mask, np.int8)

    @property
    def time(self) -> np.ndarray:
        return np.asarray(self.state.time)

    @property
    def episode_steps(self) -> np.ndarray:
        return np.asarray(self.state.steps)


__all__ = ["JaxBatchGame", "Rules", "State"]
