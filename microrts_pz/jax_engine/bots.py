# SPDX-License-Identifier: GPL-3.0-only
"""The scripted bots of ``microrts_pz.bots`` as pure JAX functions of one game's ``State``.

``rush_actions`` is ``WorkerRushBot.act`` and ``random_actions`` is
``RandomBot.act`` (with JAX's generator). ``tests/test_jax_engine.py`` checks
that ``rush_actions`` picks exactly the Python bot's actions, tick by tick.

The Python bot walks the unit list and runs a breadth-first search per unit.
Here every unit decides at once:

* The first step of the Python BFS is the lowest direction whose neighbour
  lies on a shortest free path to a goal cell. (BFS queues each level in
  order of first step, so the first goal it dequeues is a nearest one in the
  lowest such direction.) A multi-source distance field from each goal set
  gives that for every unit in one go.
* "Lowest unit id" is the lowest birth key (units are created in id order).
* The two decisions that depend on earlier units in the list (only the first
  harvester builds a barracks, and each barracks advances the army counter)
  are resolved by an argmin and a short loop over the idle barracks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .core import ATTACK, DX, DY, HARVEST, I32, MOVE, PRODUCE, RETURN, Rules, State

INF = np.int32(1 << 20)


def birth_rank(s: State) -> jax.Array:
    """Position of each cell's unit in the unit list (empty cells last)."""
    occ = s.utype >= 0
    order = jnp.lexsort((s.birth_s, jnp.where(occ, s.birth_t, jnp.iinfo(jnp.int32).max)))
    return jnp.zeros_like(order).at[order].set(jnp.arange(order.shape[0], dtype=order.dtype))


def goal_distances(rules: Rules, free, goals):
    """Shortest free-path distance from every cell to the nearest free goal cell; ``goals`` is ``(G, HW)``."""
    nbr = jnp.asarray(rules.nbr)
    d0 = jnp.where(goals & free[None], 0, INF).astype(I32)

    def body(c):
        d, _ = c
        nd = jnp.concatenate([d, jnp.full((d.shape[0], 1), INF, I32)], 1)[:, nbr].min(-1) + 1
        new = jnp.where(free[None], jnp.minimum(d, nd), INF)
        return new, jnp.any(new != d)

    d, _ = jax.lax.while_loop(lambda c: c[1], body, (d0, jnp.array(True)))
    return d


def rush_actions(rules: Rules, s: State, player, harvesters, army, n_army, counter, worker_t: int, barracks_t: int, light_t: int):
    """``WorkerRushBot(harvesters, army).act`` for one game.

    ``army``: ``(3,)`` unit type ids of which the first ``n_army`` are used
    (``n_army == 0``: plain worker rush); ``counter``: the bot's
    ``_next_unit``. Returns gridnet actions ``(HW, 7)`` int32 and the new counter.
    """
    HW, AG = rules.HW, rules.AG
    occ = s.utype >= 0
    A = rules.attrs(s.utype)
    own = occ & (s.player == player)
    enemy = occ & (s.player == 1 - player)
    idle = own & ~s.a_has
    rank = birth_rank(s)
    nbr = jnp.asarray(rules.nbr)

    # who is who
    workers = own & A["can_harv"]
    n_workers = jnp.sum(workers)
    bases = own & A["is_stock"]
    n_bases = jnp.sum(bases)
    worker_rank = jnp.zeros(HW, I32).at[jnp.argsort(rank)].set(jnp.cumsum(workers[jnp.argsort(rank)]) - 1)
    is_harv = workers & (worker_rank < jnp.minimum(harvesters, n_workers)) & (n_bases > 0)
    has_barracks = jnp.any(own & (s.utype == barracks_t))
    building = jnp.any(own & s.a_has & (s.a_type == PRODUCE) & (s.a_ut == barracks_t))
    want_barracks = (n_army > 0) & ~has_barracks & ~building

    # neighbours
    free = ~s.terrain & ~occ
    nb = rules.ext(jnp.stack([free.astype(I32), s.utype, s.player], 1), jnp.array([0, -1, -2], I32))[nbr]
    nb_free, nb_type, nb_player = nb[..., 0] > 0, nb[..., 1], nb[..., 2]  # (HW, 4)
    NB = rules.attrs(nb_type)

    def first_true(m):
        """Lowest index along the last axis where ``m`` holds, and whether there is one."""
        return jnp.argmax(m, -1).astype(I32), m.any(-1)

    # attacks: the first legal one (neighbour order for range 1, unit-list order beyond)
    d2 = jnp.asarray(rules.att_d2)[None, :]
    in_range = (d2 > 0) & (d2 <= A["range2"][:, None]) & A["can_att"][:, None]
    att_ok = rules.ext(enemy, False)[jnp.asarray(rules.att_idx)] & in_range  # (HW, AG*AG)
    dir_order = np.full(AG * AG, 4, np.int32)
    for d in range(4):
        dir_order[(rules.center + DY[d]) * AG + rules.center + DX[d]] = d
    prio = jnp.where((A["range2"] <= 1)[:, None], jnp.asarray(dir_order)[None, :],
                     rules.ext(rank, HW)[jnp.asarray(rules.att_idx)])
    prio = jnp.where(att_ok, prio, jnp.iinfo(jnp.int32).max)
    att_k = jnp.argmin(prio, -1).astype(I32)
    attacks = idle & att_ok.any(-1)

    # produce: the first free direction, if the stockpile covers the cost
    stock = s.resources[player]
    produce_dir, any_free = first_true(nb_free)

    def can_make(t):
        tt = jnp.maximum(t, 0)
        return (t >= 0) & jnp.asarray(rules.produces)[jnp.maximum(s.utype, 0), tt] & (stock >= jnp.asarray(rules.cost)[tt]) & any_free

    mobile = A["can_move"]
    # buildings: the base trains workers; other buildings train the army in turn
    deciding = idle & ~attacks
    base_want = jnp.where((n_army == 0) | (n_workers < harvesters + 1), worker_t, -1)
    base_make = deciding & ~mobile & A["is_stock"] & can_make(jnp.full(HW, base_want, I32))
    barracks = deciding & ~mobile & ~A["is_stock"]
    slot_ok = jnp.stack([can_make(jnp.full(HW, army[j], I32)) for j in range(3)], 1)  # (HW, 3)
    light_ok = can_make(jnp.full(HW, light_t, I32))

    def barracks_body(c):
        k, last, made, out_t = c
        cand = barracks & (rank > last)
        nxt = jnp.argmin(jnp.where(cand, rank, HW + 1))
        j = k % jnp.maximum(n_army, 1)
        want = jnp.where(n_army > 0, army[j], light_t)
        ok = jnp.where(n_army > 0, slot_ok[nxt, j], light_ok[nxt])
        return (k + ok.astype(I32), rank[nxt], made.at[nxt].set(ok), out_t.at[nxt].set(want))

    def barracks_more(c):
        return jnp.any(barracks & (rank > c[1]))

    counter, _, barracks_make, barracks_t_out = jax.lax.while_loop(
        barracks_more, barracks_body, (jnp.asarray(counter, I32), jnp.array(-1, I32), jnp.zeros(HW, bool), jnp.full(HW, -1, I32)))

    # the first harvester able to builds a barracks
    unit = deciding & mobile
    builder_ok = unit & is_harv & can_make(jnp.full(HW, barracks_t, I32))
    builder = jnp.argmin(jnp.where(builder_ok, rank, HW + 1))
    builds = want_barracks & builder_ok & (jnp.arange(HW) == builder)

    # harvesters harvest or return when they can, else walk; everyone else walks to the enemy
    harv_unit = unit & is_harv & ~builds
    carrying = s.res > 0
    ret_dir, can_ret = first_true((nb_type >= 0) & NB["is_stock"] & (nb_player == player))
    har_dir, can_har = first_true((nb_type >= 0) & NB["is_res"])
    returns = harv_unit & carrying & can_ret
    harvests = harv_unit & ~carrying & can_har

    def adjacent(m):
        return jnp.zeros(HW + 1, bool).at[jnp.where(m[:, None], nbr, HW)].set(True)[:HW]

    goals = jnp.stack([adjacent(enemy), adjacent(bases), adjacent(occ & A["is_res"])])
    dist = goal_distances(rules, free, goals)  # (3, HW)
    which = jnp.where(harv_unit, jnp.where(carrying, 1, 2), 0)
    walkers = unit & ~builds & ~returns & ~harvests
    nd = jnp.concatenate([dist, jnp.full((3, 1), INF, I32)], 1)[which[:, None], nbr]  # (HW, 4)
    best = nd.min(-1)
    move_dir = jnp.argmax(nd == best[:, None], -1).astype(I32)
    moves = walkers & (best < INF)

    z = jnp.zeros(HW, I32)
    atype = jnp.select([attacks, base_make | barracks_make | builds, returns, harvests, moves],
                       [ATTACK, PRODUCE, RETURN, HARVEST, MOVE], 0).astype(I32)
    made = jnp.where(base_make, base_want, jnp.where(builds, barracks_t, barracks_t_out))
    out = jnp.stack([
        atype,
        jnp.where(moves, move_dir, z),
        jnp.where(harvests, har_dir, z),
        jnp.where(returns, ret_dir, z),
        jnp.where(atype == PRODUCE, produce_dir, z),
        jnp.where(atype == PRODUCE, made, z),
        jnp.where(attacks, att_k, z),
    ], 1)
    return out, counter


def random_actions(rules: Rules, key, mask):
    """``RandomBot.act``: a uniformly random valid value per action component (0 where none is valid)."""
    offs = rules.mask_offsets
    keys = jax.random.split(key, len(rules.action_dims))
    cols = []
    for k in range(len(rules.action_dims)):
        m = mask[..., offs[k] : offs[k + 1]]
        a = jax.random.categorical(keys[k], jnp.where(m, 0.0, -jnp.inf))
        cols.append(jnp.where(m.any(-1), a, 0))
    return jnp.stack(cols, -1).astype(I32)


__all__ = ["birth_rank", "goal_distances", "random_actions", "rush_actions"]
