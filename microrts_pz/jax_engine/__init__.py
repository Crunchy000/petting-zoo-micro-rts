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
    a1 = games.bot_actions(player=1, kind="rush", harvesters=2, army=("Light", "Ranged", "Heavy"))
    raw, over, trunc, winner = games.step(a0, a1)    # gridnet actions (B, H*W, 7)

For a fully jitted loop use the pure functions ``games.step_fn(state, a0, a1)``
(returning ``(state, raw, gameover, truncated, winner, ticks)``)
and ``games.observe_fn(state, player)`` on ``games.state``; the scripted
bots are in ``microrts_pz.jax_engine.bots``.

Full observability and deterministic unit tables only (see ``core``).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..engine import MapSpec, load_map
from ..unit_types import MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH, VERSION_ORIGINAL, UnitTypeTable
from .bots import random_actions, rush_actions
from .core import Rules, State

KINDS = {"none": 0, "learner": 0, "random": 1, "rush": 2}


class JaxBatchGame:
    """``num_games`` MicroRTS games stepped together as one JAX program.

    ``maps`` is a list of map paths (or ``MapSpec``s) of the same size; game ``i`` plays
    ``maps[i % len(maps)]`` unless ``map_index`` says otherwise. Finished games
    restart on the same ``step`` when ``auto_reset`` (the returned flags and
    rewards describe the step that ended them). ``move_conflict_resolution``
    is accepted for interface compatibility: with gridnet actions the three
    strategies never differ (see ``core``).

    With ``fast_forward`` a step keeps running ticks in a game until a unit
    of either player is idle again (as ``parallel_env(fast_forward=True)``);
    rewards are summed over those ticks, ``last_ticks`` says how many ran and
    ``max_steps`` counts ticks.
    """

    def __init__(
        self,
        maps: Sequence[str | MapSpec] | str,
        num_games: int,
        max_steps: int = 2000,
        unit_type_table_version: int = VERSION_ORIGINAL,
        move_conflict_resolution: int = MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH,
        auto_reset: bool = True,
        fast_forward: bool = False,
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
        self.fast_forward = fast_forward
        self.last_ticks = np.ones(num_games, dtype=np.int64)
        self.device = device
        inits = [rules.initial_state(s, i) for i, s in enumerate(specs)]
        self.init_states = State(*[jnp.asarray(np.stack(f)) for f in zip(*inits)])
        self.map_idx = np.asarray(map_index if map_index is not None else np.arange(num_games) % len(specs), dtype=np.int32)
        self.state: State | None = None
        self.bot_counter = np.zeros((num_games, 2), dtype=np.int32)  # each game's WorkerRushBot._next_unit per player
        self._ids = {name: self.utt[name].id for name in ("Worker", "Barracks", "Light")}

        init_states = self.init_states

        def where(pick, a, b):
            return jax.tree.map(lambda x, y: jnp.where(pick.reshape((-1,) + (1,) * (x.ndim - 1)), x, y), a, b)

        def skip_ticks(s, raw, over, winner):
            """``fast_forward``: run ticks in which no unit is idle (no action could change them)."""

            def pending(s, over):
                return ~over & (s.steps < max_steps) & ~jax.vmap(rules.needs_decision)(s)

            def body(c):
                s, raw, over, winner, ticks = c
                p = pending(s, over)
                s2 = jax.vmap(rules.cycle)(s)
                s2 = s2._replace(steps=s2.steps + 1)
                o2, w2 = jax.vmap(rules.outcome)(s2)
                raw = raw.at[:, :, 0].add(jnp.where(p[:, None], jax.vmap(rules.win_loss)(o2, w2), 0.0))
                return where(p, s2, s), raw, over | (p & o2), jnp.where(p, w2, winner), ticks + p

            ticks = jnp.ones_like(over, dtype=jnp.int32)
            return jax.lax.while_loop(lambda c: jnp.any(pending(c[0], c[2])), body, (s, raw, over, winner, ticks))

        def batch_step(state: State, a0, a1):
            s, raw, over, winner = jax.vmap(rules.step)(state, a0, a1)
            s = s._replace(steps=s.steps + 1)
            ticks = jnp.ones_like(over, dtype=jnp.int32)
            if fast_forward:
                s, raw, over, winner, ticks = skip_ticks(s, raw, over, winner)
            trunc = ~over & (s.steps >= max_steps)
            if auto_reset:
                fresh = jax.tree.map(lambda x: x[s.map_id], init_states)
                s = where(over | trunc, fresh, s)
            return s, raw, over, trunc, winner, ticks

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
        ids = self._ids

        def batch_rush(state, player, harvesters, army, n_army, counter):
            return jax.vmap(lambda s, p, h, a, n, c: rush_actions(rules, s, p, h, a, n, c, ids["Worker"], ids["Barracks"], ids["Light"]))(
                state, player, harvesters, army, n_army, counter)

        #: ``rush_fn(state, player (B,), harvesters (B,), army (B, 3), n_army (B,), counter (B,))`` -> ``(actions, counter)``
        self.rush_fn = jax.jit(batch_rush)
        self.random_fn = jax.jit(lambda key, mask: random_actions(rules, key, mask))
        self._key = jax.random.PRNGKey(0)

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
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self.bot_counter[slice(None) if games is None else list(games)] = 0

    def step(self, action0, action1):
        """Advance every game one tick; returns NumPy ``(raw (B, 2, 6), gameover (B,), truncated (B,), winner (B,))``."""
        shape = (self.num_games, self.num_cells, len(self.action_dims))
        a0 = jnp.asarray(action0, jnp.int32).reshape(shape)
        a1 = jnp.asarray(action1, jnp.int32).reshape(shape)
        self.state, raw, over, trunc, winner, ticks = self.step_fn(self.state, a0, a1)
        self.last_ticks = np.asarray(ticks, np.int64)
        if self.auto_reset:
            self.bot_counter[np.asarray(over) | np.asarray(trunc)] = 0
        return np.asarray(raw, np.float64), np.asarray(over), np.asarray(trunc), np.asarray(winner, np.int64)

    def observe(self, player):
        """One-hot observations ``(B, H, W, C)`` int8 and masks ``(B, H*W, mask_size)`` for ``player`` (scalar or per game)."""
        obs, mask = self.observe_fn(self.state, player)
        return np.asarray(obs), np.asarray(mask, np.int8)

    def bot_actions(self, player, kind="rush", harvesters=1, army=None, masks=None):
        """Scripted actions ``(B, H*W, 7)``, as ``microrts_pz.fast.BatchGame.bot_actions``.

        ``kind``: ``"rush"`` (``WorkerRushBot``), ``"random"`` (``RandomBot``)
        or ``"none"``, one for all games or a sequence per game; likewise
        ``player`` and ``harvesters``. ``army`` is a unit type name, a tuple of
        names or ``None`` for all games, or a list with one of those per game.
        ``masks`` is accepted for compatibility (the random bot computes its own).
        """
        B = self.num_games
        players = np.broadcast_to(np.asarray(player, dtype=np.int32), (B,)).copy()
        kinds = np.array([KINDS[k] for k in ([kind] * B if isinstance(kind, str) else kind)])
        harv = np.broadcast_to(np.asarray(harvesters, dtype=np.int32), (B,)).copy()
        armies = army if isinstance(army, list) and len(army) == B else [army] * B
        army_ids = np.zeros((B, 3), dtype=np.int32)
        n_army = np.zeros(B, dtype=np.int32)
        for g, a in enumerate(armies):
            names = () if a is None else ((a,) if isinstance(a, str) else tuple(a))
            n_army[g] = len(names)
            army_ids[g, : len(names)] = [self.utt[n].id for n in names]
        out = np.zeros((B, self.num_cells, len(self.action_dims)), dtype=np.int64)
        rush = kinds == 2
        if rush.any():
            idx = np.arange(B)
            acts, counter = self.rush_fn(self.state, jnp.asarray(players), jnp.asarray(harv), jnp.asarray(army_ids),
                                         jnp.asarray(n_army), jnp.asarray(self.bot_counter[idx, players]))
            out[rush] = np.asarray(acts)[rush]
            self.bot_counter[idx[rush], players[rush]] = np.asarray(counter)[rush]
        if (kinds == 1).any():
            self._key, sub = jax.random.split(self._key)
            _, mask = self.observe_fn(self.state, jnp.asarray(players))
            out[kinds == 1] = np.asarray(self.random_fn(sub, mask))[kinds == 1]
        return out

    @property
    def time(self) -> np.ndarray:
        return np.asarray(self.state.time)

    @property
    def episode_steps(self) -> np.ndarray:
        return np.asarray(self.state.steps)


__all__ = ["JaxBatchGame", "Rules", "State"]
