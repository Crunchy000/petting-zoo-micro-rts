# SPDX-License-Identifier: GPL-3.0-only
"""MicroRTS rules as pure JAX functions over per-cell arrays.

This is a re-derivation of ``microrts_pz.engine`` (a port of the Java MicroRTS
engine, GPL-3.0, (c) Santiago Ontañón) for accelerators: every rule is an
elementwise or shifted-grid operation on one game, and ``jax.vmap`` batches
games. ``tests/test_jax_engine.py`` checks it tick by tick against the Python
engine.

State is stored per map cell (a unit occupies exactly one cell), so
observations, action masks and gridnet actions need no scatter or gather.

How the order-dependent parts of the Java engine are reproduced without a
loop over units:

* ``player_action_from_grid`` accepts idle units one by one in cell order,
  keeping a running set of claimed cells and a running resource total. The
  accepted set is the unique fixed point of that triangular system, found by
  iterating the whole set in parallel until it stops changing (two or three
  rounds in practice).
* ``GameState.issue`` never has to cancel anything when actions come from
  the gridnet grid with full observability: ``player_action_from_grid``
  already rejects every action whose cell or cost clashes with a reservation,
  so every pair is assigned as decoded (after ``issue_safe``). The three
  move-conflict strategies therefore behave identically here.
* ``GameState.cycle`` executes finished actions in assignment order, which is
  (start tick, player, cell). Only a few rules depend on that order: whether
  an attack lands before its target walks away, which harvesters still find
  the resource, and whether a base is still standing when a worker returns.
  Each is decided by comparing order keys between neighbouring cells.
  Produce actions always succeed, because reservations never exceed the
  stockpile (``issue`` and ``player_action_from_grid`` keep that invariant).

Unit ids are not stored; ``birth`` keys (tick, order key) sort units in
creation order, which is what the scripted bots and the renderers use ids for.

Limitations (shared with ``microrts_pz.fast``): full observability only, and
deterministic damage (unit type table versions 1 and 2). Attacks are assumed
to finish before any unit can walk into or be produced on the attacked cell
(``attack_time < move_time`` and ``attack_time < produce_time``), which holds
for the bundled unit type tables and is checked at construction.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..unit_types import VERSION_NON_DETERMINISTIC, UnitTypeTable

NONE, MOVE, HARVEST, RETURN, PRODUCE, ATTACK = range(6)
DX = (0, 1, 0, -1)
DY = (-1, 0, 1, 0)
REWARD_CATEGORY = {"Worker": 2, "Barracks": 3, "Base": 3, "Light": 5, "Heavy": 5, "Ranged": 5}
I32 = jnp.int32
BIG = np.int32(2**30)
ATTRS = ("cost", "hp0", "damage", "range2", "ptime", "mtime", "atime", "htime", "hamt",
         "is_res", "is_stock", "can_harv", "can_move", "can_att", "reward_cat")


class State(NamedTuple):
    """One game (or a batch of games, with a leading batch axis on every field)."""

    utype: jax.Array  # (HW,) unit type id, -1 for an empty cell
    player: jax.Array  # (HW,) owner, -1 for resources
    hp: jax.Array  # (HW,)
    res: jax.Array  # (HW,) carried resources (workers) or remaining resources (resource fields)
    birth_t: jax.Array  # (HW,) tick the unit was created (-1 for map units)
    birth_s: jax.Array  # (HW,) order key of the produce action that made it (map index for map units)
    a_has: jax.Array  # (HW,) bool, the unit has an assignment
    a_type: jax.Array  # (HW,) action type
    a_param: jax.Array  # (HW,) direction, or the duration of a wait
    a_ut: jax.Array  # (HW,) unit type being produced, -1 otherwise
    a_tgt: jax.Array  # (HW,) attack target as an index into the attack grid
    a_time: jax.Array  # (HW,) tick the assignment was issued
    terrain: jax.Array  # (HW,) bool, wall
    resources: jax.Array  # (2,) player stockpiles
    time: jax.Array  # () game tick
    steps: jax.Array  # () steps in this episode
    map_id: jax.Array  # () which map the game restarts from


class Rules:
    """Static tables and shapes; methods are pure functions of one game's ``State``."""

    def __init__(self, utt: UnitTypeTable, height: int, width: int):
        if utt.version == VERSION_NON_DETERMINISTIC:
            raise NotImplementedError("the JAX engine supports deterministic damage only (unit type table versions 1 and 2)")
        self.utt = utt
        self.H, self.W = height, width
        self.HW = height * width
        self.n_types = n = len(utt)
        self.max_range = utt.max_attack_range
        self.AG = 2 * self.max_range + 1
        self.center = self.AG // 2
        self.action_dims = np.array([6, 4, 4, 4, 4, n, self.AG**2])
        self.mask_offsets = np.concatenate([[0], np.cumsum(self.action_dims)])
        self.mask_size = int(self.mask_offsets[-1])
        self.num_channels = 5 + 5 + 3 + (n + 1) + 6 + 2
        uts = utt.unit_types

        def col(f, dtype=np.int32):
            return np.array([f(u) for u in uts], dtype=dtype)

        self.cost = col(lambda u: u.cost)
        self.hp0 = col(lambda u: u.hp)
        self.damage = col(lambda u: u.min_damage)
        self.range2 = col(lambda u: u.attack_range * u.attack_range)
        self.ptime = col(lambda u: u.produce_time)
        self.mtime = col(lambda u: u.move_time)
        self.atime = col(lambda u: u.attack_time)
        self.htime = col(lambda u: u.harvest_time)
        self.hamt = col(lambda u: u.harvest_amount)
        self.is_res = col(lambda u: u.is_resource, bool)
        self.is_stock = col(lambda u: u.is_stockpile, bool)
        self.can_harv = col(lambda u: u.can_harvest, bool)
        self.can_move = col(lambda u: u.can_move, bool)
        self.can_att = col(lambda u: u.can_attack, bool)
        self.reward_cat = col(lambda u: REWARD_CATEGORY.get(u.name, 0))
        self.produces = np.zeros((n, n), dtype=bool)
        for u in uts:
            for p in u.produces:
                self.produces[u.id, p.id] = True
        self.produces_any = self.produces.any(1)
        self.table = np.stack([getattr(self, c).astype(np.int32) for c in ATTRS], 1)  # (n_types, len(ATTRS))
        if any(u.min_damage != u.max_damage for u in uts):
            raise NotImplementedError("the JAX engine supports deterministic damage only")
        attackers = [u for u in uts if u.can_attack]
        slowest_attack = max((u.attack_time for u in attackers), default=0)
        fastest_arrival = min([u.move_time for u in uts if u.can_move] + [u.produce_time for u in uts])
        if slowest_attack >= fastest_arrival:
            raise NotImplementedError("the JAX engine assumes attacks finish before a unit can enter the attacked cell")
        # Longest any action can take; bounds how far back a finishing action can have started.
        self.max_eta = int(max(self.ptime.max(), self.mtime.max(), self.atime.max(), self.htime.max(), 1)) + 2
        self.seq_span = 2 * self.HW
        # attack grid offsets
        k = np.arange(self.AG**2)
        self.att_ox = (k % self.AG - self.center).astype(np.int32)
        self.att_oy = (k // self.AG - self.center).astype(np.int32)
        self.cell_x = np.tile(np.arange(width, dtype=np.int32), height)
        self.cell_y = np.repeat(np.arange(height, dtype=np.int32), width)
        self.delta = np.array([-width, 1, width, -1], dtype=np.int32)
        # neighbour lookups as gathers: index of the cell at an offset, or HW (a padding slot) off the map
        self._index_cache: dict[tuple[int, int], np.ndarray] = {}
        self.nbr = np.stack([self.offset_index(DX[d], DY[d]) for d in range(4)], 1)  # (HW, 4)
        self.att_idx = np.stack([self.offset_index(int(ox), int(oy)) for ox, oy in zip(self.att_ox, self.att_oy)], 1)
        self.att_d2 = (self.att_ox**2 + self.att_oy**2).astype(np.int32)

    # ------------------------------------------------------------ helpers
    def offset_index(self, ox: int, oy: int) -> np.ndarray:
        """For every cell, the flat index of the cell at ``(+ox, +oy)``, or ``HW`` off the map."""
        key = (ox, oy)
        if key not in self._index_cache:
            x, y = self.cell_x + ox, self.cell_y + oy
            inside = (x >= 0) & (y >= 0) & (x < self.W) & (y < self.H)
            self._index_cache[key] = np.where(inside, x + y * self.W, self.HW).astype(np.int32)
        return self._index_cache[key]

    @staticmethod
    def ext(x, fill):
        """Append the padding slot that off-map indices point at."""
        return jnp.concatenate([x, jnp.full((1,) + x.shape[1:], fill, x.dtype)])

    def shifted(self, x, ox: int, oy: int, fill=0):
        """``out[(cx, cy)] = x[(cx + ox, cy + oy)]``, ``fill`` off the map; ``x`` is ``(HW,)`` or ``(HW, F)``."""
        return self.ext(x, fill)[self.offset_index(ox, oy)]

    def toward(self, d):
        """Index of the neighbouring cell in direction ``d[c]`` of every cell ``c`` (``HW`` off the map)."""
        return jnp.asarray(self.nbr)[jnp.arange(self.HW), jnp.clip(d, 0, 3)]

    def attrs(self, ut):
        """Unit type properties for an array of type ids (one gather): ``A["cost"]``, ``A["is_res"]``, ..."""
        t = jnp.asarray(self.table)[jnp.maximum(ut, 0)]
        return {c: (t[..., i] > 0 if c.startswith(("is_", "can_")) else t[..., i]) for i, c in enumerate(ATTRS)}

    @staticmethod
    def eta(A, A_made, atype, param):
        """``UnitAction.ETA``; ``A``: the unit's type properties, ``A_made``: those of the type it produces."""
        return jnp.select(
            [atype == NONE, atype == MOVE, atype == ATTACK, atype == HARVEST, atype == RETURN, atype == PRODUCE],
            [param, A["mtime"], A["atime"], A["htime"], A["mtime"], A_made["ptime"]],
            0,
        )

    # ------------------------------------------------------------- reset
    def initial_state(self, spec, map_id: int = 0) -> State:
        """``State`` of a fresh game from a ``MapSpec`` (NumPy arrays; stack several for a batch)."""
        HW = self.HW
        f = {name: np.full(HW, -1, np.int32) for name in ("utype", "player", "a_ut")}
        z = {name: np.zeros(HW, np.int32) for name in ("hp", "res", "birth_t", "birth_s", "a_type", "a_param", "a_tgt", "a_time")}
        for i, (uid, tname, player, x, y, res, hp) in enumerate(spec.units):
            c = x + y * self.W
            if f["utype"][c] >= 0:
                raise ValueError(f"two units at position ({x}, {y})")
            f["utype"][c] = self.utt[tname].id
            f["player"][c] = player
            z["hp"][c] = hp
            z["res"][c] = res
            z["birth_t"][c] = -1
            z["birth_s"][c] = i
        return State(
            utype=f["utype"], player=f["player"], hp=z["hp"], res=z["res"], birth_t=z["birth_t"], birth_s=z["birth_s"],
            a_has=np.zeros(HW, bool), a_type=z["a_type"], a_param=z["a_param"], a_ut=f["a_ut"], a_tgt=z["a_tgt"],
            a_time=z["a_time"], terrain=np.asarray(spec.terrain, dtype=bool), resources=np.asarray(spec.resources, np.int32),
            time=np.int32(0), steps=np.int32(0), map_id=np.int32(map_id),
        )

    # -------------------------------------------------------------- masks
    def action_mask(self, s: State, player) -> jax.Array:
        """``MicroRTSGame.action_mask``: ``(HW, mask_size)`` bool."""
        ut = jnp.maximum(s.utype, 0)
        A = self.attrs(s.utype)
        idle = (s.utype >= 0) & (s.player == player) & ~s.a_has
        nb = self.ext(jnp.stack([((~s.terrain) & (s.utype < 0)).astype(I32), s.utype, s.player], 1), jnp.array([0, -1, -2], I32))[self.nbr]
        free, nb_type, nb_player = nb[..., 0] > 0, nb[..., 1], nb[..., 2]  # (HW, 4) each
        NB = self.attrs(nb_type)
        res_dir = (nb_type >= 0) & NB["is_res"]
        stock_dir = (nb_type >= 0) & NB["is_stock"] & (nb_player == player)
        can_harv = A["can_harv"][:, None]
        move_dir = free & A["can_move"][:, None]
        harv_dir = res_dir & can_harv & (s.res == 0)[:, None]
        ret_dir = stock_dir & can_harv & (s.res > 0)[:, None]
        afford = jnp.asarray(self.produces)[ut] & (s.resources[player] >= jnp.asarray(self.cost))[None, :]  # (HW, n)
        prod_dir = free & afford.any(1)[:, None]
        prod_type = afford & free.any(1)[:, None]
        enemy = (s.utype >= 0) & (s.player >= 0) & (s.player != player)
        d2 = jnp.asarray(self.att_d2)[None, :]
        att = self.ext(enemy, False)[self.att_idx] & (d2 > 0) & (d2 <= A["range2"][:, None]) & A["can_att"][:, None]
        types = jnp.stack([
            jnp.ones(self.HW, bool), move_dir.any(1), harv_dir.any(1), ret_dir.any(1), prod_dir.any(1), att.any(1),
        ], 1)
        mask = jnp.concatenate([types, move_dir, harv_dir, ret_dir, prod_dir, prod_type, att], 1)
        return mask & idle[:, None]

    def observation(self, s: State, player) -> jax.Array:
        """``MicroRTSGame.observation`` flattened to ``(HW, C)`` int8."""
        occ = s.utype >= 0
        hp = jnp.clip(jnp.where(occ, s.hp, 0), 0, 4)
        res = jnp.clip(jnp.where(occ, s.res, 0), 0, 4)
        own = jnp.where(occ & (s.player >= 0), (s.player + player) % 2 + 1, 0)
        typ = jnp.where(occ, s.utype + 1, 0)
        act = jnp.where(occ & s.a_has, s.a_type, 0)
        n = self.n_types
        parts = [
            jax.nn.one_hot(hp, 5, dtype=jnp.int8), jax.nn.one_hot(res, 5, dtype=jnp.int8),
            jax.nn.one_hot(own, 3, dtype=jnp.int8), jax.nn.one_hot(typ, n + 1, dtype=jnp.int8),
            jax.nn.one_hot(act, 6, dtype=jnp.int8), jax.nn.one_hot(s.terrain.astype(I32), 2, dtype=jnp.int8),
        ]
        return jnp.concatenate(parts, 1)

    # ------------------------------------------------------------ actions
    def _issue(self, s: State, player, grid):
        """``player_action_from_grid`` + ``issue_safe`` + ``issue`` for one player.

        Returns the new state and what the reward trace needs: the issued
        action types, produced unit types and the owner of each attack target.
        """
        HW, W = self.HW, self.W
        cell = jnp.arange(HW, dtype=I32)
        idle = (s.utype >= 0) & (s.player == player) & ~s.a_has
        grid = grid.astype(I32)
        atype = grid[:, 0]
        param = jnp.select([atype == MOVE, atype == HARVEST, atype == RETURN, atype == PRODUCE],
                           [grid[:, 1], grid[:, 2], grid[:, 3], grid[:, 4]], -1)
        aut = jnp.where(atype == PRODUCE, grid[:, 5], -1)
        atgt = jnp.where(atype == ATTACK, grid[:, 6], 0)
        A = self.attrs(s.utype)
        MADE = self.attrs(aut)
        R = s.resources

        # --- player_action_from_grid: greedy acceptance in cell order
        has_pos = idle & ((atype == MOVE) | (atype == PRODUCE))
        delta = jnp.asarray(self.delta)
        target = cell + delta[jnp.clip(param, 0, 3)]  # flat, may wrap or leave the map (as in Java)
        my_cost = jnp.where(idle & (atype == PRODUCE), MADE["cost"], 0)
        # reservations of the current assignments (their positions are always on the map)
        a_pos = s.a_has & ((s.a_type == MOVE) | (s.a_type == PRODUCE))
        L = HW + 2 * W  # flat positions -W .. HW+W-1, stored at index pos + W
        reserved = jnp.zeros(L, bool).at[jnp.where(a_pos, cell + delta[jnp.clip(s.a_param, 0, 3)] + W, L)].set(True, mode="drop")
        a_cost = jnp.where(s.a_has & (s.a_type == PRODUCE), self.attrs(s.a_ut)["cost"], 0)
        stat = jnp.stack([jnp.sum(jnp.where(s.player == 0, a_cost, 0)), jnp.sum(jnp.where(s.player == 1, a_cost, 0))])
        opp = 1 - player
        opp_block = (stat[opp] != 0) & (stat[opp] > R[opp])
        base_ok = idle & ~opp_block & ~(has_pos & reserved[target + W])

        def accept(acc):
            claim = jnp.full(L, BIG, I32).at[jnp.where(acc & has_pos, target + W, L)].min(cell, mode="drop")
            clash = has_pos & (claim[target + W] < cell)
            mine = jnp.where(acc, my_cost, 0)
            spent = stat[player] + jnp.cumsum(mine) - mine
            over = (spent != 0) & (my_cost + spent > R[player])
            return base_ok & ~clash & ~over

        def body(c):
            acc, _ = c
            new = accept(acc)
            return new, jnp.any(new != acc)

        chosen, _ = jax.lax.while_loop(lambda c: c[1], body, (base_ok, jnp.array(True)))

        # --- issue_safe: illegal actions become waits of the same length
        ut = jnp.maximum(s.utype, 0)
        x, y = jnp.asarray(self.cell_x), jnp.asarray(self.cell_y)
        nb = self.ext(jnp.stack([((~s.terrain) & (s.utype < 0)).astype(I32), s.utype, s.player], 1), jnp.array([0, -1, -2], I32))[self.toward(param)]
        free_d, nbt, nbp = nb[:, 0] > 0, nb[:, 1], nb[:, 2]
        NB = self.attrs(nbt)
        tx = x + atgt % self.AG - self.center
        ty = y + atgt // self.AG - self.center
        tc = jnp.asarray(self.att_idx)[cell, atgt]
        tgt = self.ext(jnp.stack([s.utype, s.player], 1), jnp.array([-1, -9], I32))[tc]
        t_type, t_owner = tgt[:, 0], tgt[:, 1]
        t_enemy = (t_type >= 0) & (t_owner >= 0) & (t_owner != player)
        dist2 = (tx - x) ** 2 + (ty - y) ** 2
        safe_ut = jnp.maximum(aut, 0)
        valid = jnp.select(
            [atype == NONE, atype == MOVE, atype == HARVEST, atype == RETURN, atype == PRODUCE, atype == ATTACK],
            [
                jnp.ones(HW, bool),
                A["can_move"] & free_d,
                A["can_harv"] & (s.res == 0) & (nbt >= 0) & NB["is_res"],
                A["can_harv"] & (s.res > 0) & (nbt >= 0) & NB["is_stock"] & (nbp == player),
                jnp.asarray(self.produces)[ut, safe_ut] & (R[player] >= MADE["cost"]) & free_d,
                A["can_att"] & t_enemy & (dist2 <= A["range2"]),
            ],
            False,
        )
        keep = chosen & valid
        new_type = jnp.where(keep, atype, NONE)
        new_param = jnp.where(keep, param, jnp.where(chosen, self.eta(A, MADE, atype, param), 1))

        # --- issue: nothing can clash at this point (see the module docstring)
        s = s._replace(
            a_has=s.a_has | idle,
            a_type=jnp.where(idle, new_type, s.a_type),
            a_param=jnp.where(idle, new_param, s.a_param),
            a_ut=jnp.where(idle, jnp.where(keep, aut, -1), s.a_ut),
            a_tgt=jnp.where(idle, jnp.where(keep, atgt, 0), s.a_tgt),
            a_time=jnp.where(idle, s.time, s.a_time),
        )
        target_owner = jnp.where(t_type >= 0, t_owner, -9)
        return s, jnp.where(idle, new_type, NONE), jnp.where(keep, aut, -1), target_owner

    def _rewards(self, player, issued_type, issued_ut, target_owner):
        cat = self.attrs(issued_ut)["reward_cat"]
        prod = issued_type == PRODUCE
        att = issued_type == ATTACK
        return jnp.stack([
            jnp.zeros((), I32),
            jnp.sum((issued_type == HARVEST) | (issued_type == RETURN)),
            jnp.sum(prod & (cat == 2)),
            jnp.sum(prod & (cat == 3)),
            jnp.sum(att & (target_owner == 1 - player)) - jnp.sum(att & (target_owner == player)),
            jnp.sum(prod & (cat == 5)),
        ]).astype(jnp.float32)

    # -------------------------------------------------------------- cycle
    def cycle(self, s: State) -> State:
        """``GameState.cycle``: advance one tick and execute every finished action."""
        HW = self.HW
        t1 = s.time + 1
        occ = s.utype >= 0
        cell = jnp.arange(HW, dtype=I32)
        A = self.attrs(s.utype)
        MADE = self.attrs(s.a_ut)
        ready = occ & s.a_has & (self.eta(A, MADE, s.a_type, s.a_param) + s.a_time <= t1)
        # assignment order: (start tick, player, cell); finishing actions started at most max_eta ticks ago
        seq = (s.a_time - t1 + self.max_eta) * self.seq_span + jnp.maximum(s.player, 0) * HW + cell
        seq = jnp.where(ready, seq, BIG)
        kind = jnp.where(ready, s.a_type, -1)
        d = jnp.clip(s.a_param, 0, 3)
        tcell = self.toward(d)  # the cell a move/harvest/return/produce acts on
        mover = kind == MOVE
        returner = kind == RETURN

        # --- attacks land unless the target walked away first
        atk = kind == ATTACK
        tc = jnp.where(atk, jnp.asarray(self.att_idx)[cell, s.a_tgt], HW)
        leave = jnp.where(mover, seq, BIG)
        land = atk & self.ext(occ, False)[tc] & (seq < self.ext(leave, BIG)[tc])
        dmg = jnp.where(land, A["damage"], 0)
        taken = jnp.zeros(HW + 1, I32).at[tc].add(dmg)[:HW]
        killed = occ & (taken > 0) & (taken >= s.hp)

        # --- returns: the base must still stand when the worker arrives.
        # ret_seq[b, k]: order key of the worker returning into base b from direction k (it stands at b - D_k)
        ret_seq = jnp.stack([self.shifted(jnp.where(returner & (d == k), seq, BIG), -DX[k], -DY[k], BIG) for k in range(4)], 1)
        early = jnp.where(seq[:, None] < self.ext(ret_seq, BIG)[tc], dmg[:, None], 0)  # (HW, 4) per attacker
        early = jnp.zeros((HW + 1, 4), I32).at[tc].add(early)[:HW]
        dmg_before = jnp.zeros(HW, I32)
        for k in range(4):
            dmg_before = jnp.where(d == k, self.shifted(early[:, k], DX[k], DY[k], 0), dmg_before)
        b = self.ext(jnp.stack([s.utype, s.hp], 1), jnp.array([-1, 0], I32))[tcell]
        b_type, b_hp = b[:, 0], b[:, 1]
        BA = self.attrs(b_type)
        returned = returner & (b_type >= 0) & BA["is_stock"] & (s.res > 0) & (dmg_before < b_hp)
        gain = jnp.where(returned, s.res, 0)

        # --- harvests, in order, while the resource lasts
        hamt = A["hamt"]
        eligible = (kind == HARVEST) & A["can_harv"] & (s.res == 0) & (b_type >= 0) & BA["is_res"]
        # per resource cell r and direction k: the harvester at r - D_k (eligible flag, order key, amount)
        packed = jnp.stack([eligible.astype(I32), seq, hamt], 1)
        at_res = [self.shifted(jnp.where((d == k)[:, None], packed, jnp.array([0, BIG, 0], I32)), -DX[k], -DY[k], 0) for k in range(4)]
        succ = []
        for k in range(4):
            before = sum(jnp.where((at_res[j][:, 0] > 0) & (at_res[j][:, 1] < at_res[k][:, 1]), at_res[j][:, 2], 0)
                         for j in range(4) if j != k)
            succ.append((at_res[k][:, 0] > 0) & (before < s.res))
        drained = sum(jnp.where(succ[k], at_res[k][:, 2], 0) for k in range(4))
        harvested = jnp.zeros(HW, bool)
        for k in range(4):
            harvested = harvested | ((d == k) & eligible & self.shifted(succ[k], DX[k], DY[k], False))

        # --- produce (always affordable: reservations never exceed the stockpile)
        prod = kind == PRODUCE
        pcost = jnp.where(prod, MADE["cost"], 0)
        owner = jnp.maximum(s.player, 0)
        resources = s.resources + jnp.stack([jnp.sum(jnp.where(owner == p, gain - pcost, 0)) for p in (0, 1)])

        # --- the new board
        res_left = s.res - drained
        depleted = occ & A["is_res"] & (drained > 0) & (res_left <= 0)
        dead = killed | depleted
        stays = occ & ~dead & ~mover
        moves = mover & ~dead
        new_res = jnp.where(harvested, hamt, jnp.where(returned, 0, res_left))
        fields = jnp.stack([
            s.utype, s.player, s.hp - taken, new_res, s.birth_t, s.birth_s, (s.a_has & ~ready).astype(I32),
            s.a_type, s.a_param, s.a_ut, s.a_tgt, s.a_time,
        ], 1)
        empty = jnp.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], I32)
        born_ut = s.a_ut
        newborn = jnp.stack([
            born_ut, s.player, MADE["hp0"], jnp.zeros(HW, I32), jnp.full(HW, t1, I32), seq,
            jnp.zeros(HW, I32), jnp.zeros(HW, I32), jnp.zeros(HW, I32), jnp.full(HW, -1, I32), jnp.zeros(HW, I32), jnp.zeros(HW, I32),
        ], 1)
        out = jnp.where(stays[:, None], fields, empty)
        for k in range(4):
            src = jnp.where((moves & (d == k))[:, None], fields, jnp.where((prod & (d == k))[:, None], newborn, empty))
            incoming = self.shifted(src, -DX[k], -DY[k], -1)
            out = jnp.where((incoming[:, 0] >= 0)[:, None], incoming, out)
        return s._replace(
            utype=out[:, 0], player=out[:, 1], hp=out[:, 2], res=out[:, 3], birth_t=out[:, 4], birth_s=out[:, 5],
            a_has=out[:, 6] > 0, a_type=out[:, 7], a_param=out[:, 8], a_ut=out[:, 9], a_tgt=out[:, 10], a_time=out[:, 11],
            resources=resources, time=t1,
        )

    # --------------------------------------------------------------- step
    def step(self, s: State, a0, a1):
        """``MicroRTSGame.step``: returns ``(state, raw_rewards (2, 6), gameover, winner)``."""
        s, t0, u0, o0 = self._issue(s, 0, a0)
        s, t1, u1, o1 = self._issue(s, 1, a1)
        s = self.cycle(jax.lax.optimization_barrier(s))  # stops XLA re-computing the issue phase inside the cycle's fusions
        n0 = jnp.sum((s.utype >= 0) & (s.player == 0))
        n1 = jnp.sum((s.utype >= 0) & (s.player == 1))
        over = (n0 + n1 == 0) | ((n0 > 0) != (n1 > 0))
        winner = jnp.where((n0 > 0) & (n1 == 0), 0, jnp.where((n1 > 0) & (n0 == 0), 1, -1))
        wl = jnp.where(over, jnp.where(winner == jnp.arange(2), 1.0, -1.0), 0.0).astype(jnp.float32)
        raw = jnp.stack([self._rewards(0, t0, u0, o0), self._rewards(1, t1, u1, o1)])
        raw = raw.at[:, 0].set(wl)
        return s, raw, over, winner

    def needs_decision(self, s: State):
        """Some unit of either player is idle."""
        return jnp.any((s.utype >= 0) & (s.player >= 0) & ~s.a_has)
