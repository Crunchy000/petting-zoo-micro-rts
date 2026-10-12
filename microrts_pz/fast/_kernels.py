# SPDX-License-Identifier: GPL-3.0-only
"""Numba kernels for the batched MicroRTS engine.

A line-by-line port of ``microrts_pz.engine`` (itself a port of the Java
MicroRTS engine, GPL-3.0, (c) Santiago Ontañón) and ``microrts_pz.game`` to
flat integer arrays, so that many games can be stepped in one compiled call.
The rules, including the Java engine's quirks, follow ``engine.py`` exactly;
``tests/test_fast.py`` checks the two engines against each other tick by tick.

State layout (``B`` games, ``CAP`` unit slots per game):

* ``U[B, CAP, NF]`` units in insertion order (the order of ``GameState.units``),
  dead units keep their slot with ``ALIVE == 0``; the ``A_*`` fields hold the
  unit's current assignment, ``A_SEQ`` its position in ``GameState.assignments``.
* ``G[B, NG]`` per-game scalars (time, resources, id and slot counters).
* ``OCC[B, H*W]`` slot of the unit on each cell, or -1.
* ``TER[B, H*W]`` terrain.

Unit types come in as ``UTT[n_types, NFT]`` and ``PROD[n_types, max_produces]``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# unit fields
ALIVE, ID, PLAYER, TYPE, X, Y, RES, HP, A_HAS, A_TYPE, A_PARAM, A_X, A_Y, A_UT, A_TIME, A_SEQ = range(16)
NF = 16
# game scalars
G_TIME, G_RES0, G_RES1, G_NEXT_ID, G_NSLOT, G_CANCEL, G_SEQ, G_BOT0, G_BOT1, G_STEPS = range(10)
NG = 10
# unit type fields
(T_COST, T_HP, T_MIND, T_MAXD, T_RANGE, T_PTIME, T_MTIME, T_ATIME, T_HTIME, T_HAMT, T_SIGHT,
 T_ISRES, T_ISSTOCK, T_CANHARV, T_CANMOVE, T_CANATT, T_REWARD) = range(17)
NFT = 17

NONE, MOVE, HARVEST, RETURN, PRODUCE, ATTACK = range(6)
DX = np.array([0, 1, 0, -1], dtype=np.int64)
DY = np.array([-1, 0, 1, 0], dtype=np.int64)
MAX_ACTIONS = 128  # bound on UnitAction candidates for one unit


# ------------------------------------------------------------------ basics
@njit(cache=True)
def unit_at(OCC, g, W, H, x, y):
    if x < 0 or y < 0 or x >= W or y >= H:
        return -1
    return OCC[g, x + y * W]


@njit(cache=True)
def eta(UTT, ut, atype, param, aut):
    if atype == NONE:
        return param
    if atype == MOVE:
        return UTT[ut, T_MTIME]
    if atype == ATTACK:
        return UTT[ut, T_ATIME]
    if atype == HARVEST:
        return UTT[ut, T_HTIME]
    if atype == RETURN:
        return UTT[ut, T_MTIME]  # sic, as in the Java engine
    if atype == PRODUCE:
        return UTT[aut, T_PTIME]
    return 0


@njit(cache=True)
def usage(UTT, W, x, y, player, atype, param, aut, out):
    """ResourceUsage of an action: fills ``out = [has_pos, pos, r0, r1]``."""
    out[0] = 0
    out[1] = 0
    out[2] = 0
    out[3] = 0
    if atype == MOVE or atype == PRODUCE:
        pos = x + y * W
        if param == 0:
            pos -= W
        elif param == 1:
            pos += 1
        elif param == 2:
            pos += W
        elif param == 3:
            pos -= 1
        out[0] = 1
        out[1] = pos
        if atype == PRODUCE:
            out[2 + player] += UTT[aut, T_COST]


@njit(cache=True)
def add_unit(U, G, OCC, g, W, uid, player, ut, x, y, res, hp):
    s = G[g, G_NSLOT]
    G[g, G_NSLOT] = s + 1
    U[g, s, :] = 0
    U[g, s, ALIVE] = 1
    U[g, s, ID] = uid
    U[g, s, PLAYER] = player
    U[g, s, TYPE] = ut
    U[g, s, X] = x
    U[g, s, Y] = y
    U[g, s, RES] = res
    U[g, s, HP] = hp
    pos = x + y * W
    if OCC[g, pos] < 0:
        OCC[g, pos] = s
    if uid >= G[g, G_NEXT_ID]:
        G[g, G_NEXT_ID] = uid + 1


@njit(cache=True)
def remove_unit(U, OCC, g, W, s):
    if U[g, s, ALIVE] == 0:
        return
    U[g, s, ALIVE] = 0
    U[g, s, A_HAS] = 0
    pos = U[g, s, X] + U[g, s, Y] * W
    if OCC[g, pos] == s:
        OCC[g, pos] = -1


@njit(cache=True)
def rebuild_occupancy(U, G, OCC, g, W):
    OCC[g, :] = -1
    for s in range(G[g, G_NSLOT]):
        if U[g, s, ALIVE]:
            pos = U[g, s, X] + U[g, s, Y] * W
            if OCC[g, pos] < 0:
                OCC[g, pos] = s


@njit(cache=True)
def move_unit(U, G, OCC, g, W, s, nx, ny):
    alive = U[g, s, ALIVE]
    if alive:
        old = U[g, s, X] + U[g, s, Y] * W
        if OCC[g, old] == s:
            OCC[g, old] = -1
    U[g, s, X] = nx
    U[g, s, Y] = ny
    if alive:
        new = nx + ny * W
        if OCC[g, new] >= 0:
            rebuild_occupancy(U, G, OCC, g, W)
        else:
            OCC[g, new] = s


@njit(cache=True)
def compact(U, G, OCC, g, W):
    """Drop dead slots, keeping the order of the live ones."""
    n = 0
    for s in range(G[g, G_NSLOT]):
        if U[g, s, ALIVE]:
            if n != s:
                U[g, n, :] = U[g, s, :]
            n += 1
    G[g, G_NSLOT] = n
    rebuild_occupancy(U, G, OCC, g, W)


# ----------------------------------------------------------- unit actions
@njit(cache=True)
def unit_actions(U, G, OCC, TER, UTT, PROD, g, W, H, s, none_dur, out):
    """``GameState.unit_actions``: rows ``[type, param, x, y, unit_type]``; returns the count."""
    n = 0
    x = U[g, s, X]
    y = U[g, s, Y]
    ut = U[g, s, TYPE]
    player = U[g, s, PLAYER]
    neigh = np.empty(4, dtype=np.int64)
    neigh[0] = unit_at(OCC, g, W, H, x, y - 1)
    neigh[1] = unit_at(OCC, g, W, H, x + 1, y)
    neigh[2] = unit_at(OCC, g, W, H, x, y + 1)
    neigh[3] = unit_at(OCC, g, W, H, x - 1, y)

    if UTT[ut, T_CANATT]:
        rng = UTT[ut, T_RANGE]
        if rng == 1:
            for d in range(4):
                o = neigh[d]
                if o >= 0 and U[g, o, PLAYER] != player and U[g, o, PLAYER] >= 0:
                    out[n, 0] = ATTACK
                    out[n, 1] = -1
                    out[n, 2] = U[g, o, X]
                    out[n, 3] = U[g, o, Y]
                    out[n, 4] = -1
                    n += 1
        else:
            sq = rng * rng
            for o in range(G[g, G_NSLOT]):
                if U[g, o, ALIVE] == 0:
                    continue
                op = U[g, o, PLAYER]
                if op < 0 or op == player:
                    continue
                ddx = U[g, o, X] - x
                ddy = U[g, o, Y] - y
                if ddx * ddx + ddy * ddy <= sq:
                    out[n, 0] = ATTACK
                    out[n, 1] = -1
                    out[n, 2] = U[g, o, X]
                    out[n, 3] = U[g, o, Y]
                    out[n, 4] = -1
                    n += 1

    if UTT[ut, T_CANHARV]:
        if U[g, s, RES] == 0:
            for d in range(4):
                o = neigh[d]
                if o >= 0 and UTT[U[g, o, TYPE], T_ISRES]:
                    out[n, 0] = HARVEST
                    out[n, 1] = d
                    out[n, 2] = 0
                    out[n, 3] = 0
                    out[n, 4] = -1
                    n += 1
        if U[g, s, RES] > 0:
            for d in range(4):
                o = neigh[d]
                if o >= 0 and UTT[U[g, o, TYPE], T_ISSTOCK] and U[g, o, PLAYER] == player:
                    out[n, 0] = RETURN
                    out[n, 1] = d
                    out[n, 2] = 0
                    out[n, 3] = 0
                    out[n, 4] = -1
                    n += 1

    if PROD[ut, 0] >= 0 or UTT[ut, T_CANMOVE]:
        free = np.zeros(4, dtype=np.bool_)
        for d in range(4):
            nx = x + DX[d]
            ny = y + DY[d]
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            free[d] = TER[g, nx + ny * W] == 0 and neigh[d] < 0
        if player >= 0:
            for k in range(PROD.shape[1]):
                pt = PROD[ut, k]
                if pt < 0:
                    break
                if G[g, G_RES0 + player] >= UTT[pt, T_COST]:
                    for d in range(4):
                        if free[d]:
                            out[n, 0] = PRODUCE
                            out[n, 1] = d
                            out[n, 2] = 0
                            out[n, 3] = 0
                            out[n, 4] = pt
                            n += 1
        if UTT[ut, T_CANMOVE]:
            for d in range(4):
                if free[d]:
                    out[n, 0] = MOVE
                    out[n, 1] = d
                    out[n, 2] = 0
                    out[n, 3] = 0
                    out[n, 4] = -1
                    n += 1

    out[n, 0] = NONE
    out[n, 1] = none_dur
    out[n, 2] = 0
    out[n, 3] = 0
    out[n, 4] = -1
    return n + 1


@njit(cache=True)
def can_execute(U, G, OCC, TER, UTT, PROD, g, W, H, s, act, buf):
    atype = act[0]
    n = unit_actions(U, G, OCC, TER, UTT, PROD, g, W, H, s, eta(UTT, U[g, s, TYPE], act[0], act[1], act[4]), buf)
    for i in range(n):
        if buf[i, 0] != atype:
            continue
        if atype == ATTACK:
            if buf[i, 2] == act[2] and buf[i, 3] == act[3]:
                return True
        elif atype == PRODUCE:
            if buf[i, 1] == act[1] and buf[i, 4] == act[4]:
                return True
        elif buf[i, 1] == act[1]:
            return True
    return False


# --------------------------------------------------------------- one step
@njit(cache=True)
def consistent(a_pos, a_npos, a_r0, a_r1, b_pos, b_npos, b_r0, b_r1, pres0, pres1):
    """``a.consistent_with(b)`` for ResourceUsages ``a`` (self) and ``b`` (other)."""
    for i in range(b_npos):
        for j in range(a_npos):
            if a_pos[j] == b_pos[i]:
                return False
    if b_r0 != 0:
        t = a_r0 + b_r0
        if t > 0 and t > pres0:
            return False
    if b_r1 != 0:
        t = a_r1 + b_r1
        if t > 0 and t > pres1:
            return False
    return True


@njit(cache=True)
def assignment_order(U, G, g):
    """Slots holding an assignment, in ``GameState.assignments`` order."""
    n = 0
    slots = np.empty(G[g, G_NSLOT], dtype=np.int64)
    seqs = np.empty(G[g, G_NSLOT], dtype=np.int64)
    for s in range(G[g, G_NSLOT]):
        if U[g, s, ALIVE] and U[g, s, A_HAS]:
            slots[n] = s
            seqs[n] = U[g, s, A_SEQ]
            n += 1
    order = np.argsort(seqs[:n])
    return slots[:n][order]


@njit(cache=True)
def player_actions(U, G, OCC, TER, UTT, PROD, g, W, H, AG, player, grid, pslot, pact, np0):
    """``player_action_from_grid`` + ``issue_safe`` for one player. Appends to ``pslot/pact``; returns the new count."""
    nslot = G[g, G_NSLOT]
    pa_pos = np.empty(nslot + 1, dtype=np.int64)
    pa_n = 0
    pa_r0 = 0
    pa_r1 = 0
    ru = np.empty(4, dtype=np.int64)
    for s in range(nslot):
        if U[g, s, ALIVE] and U[g, s, A_HAS]:
            usage(UTT, W, U[g, s, X], U[g, s, Y], U[g, s, PLAYER], U[g, s, A_TYPE], U[g, s, A_PARAM], U[g, s, A_UT], ru)
            if ru[0]:
                pa_pos[pa_n] = ru[1]
                pa_n += 1
            pa_r0 += ru[2]
            pa_r1 += ru[3]

    # idle units of this player, by position
    idle = np.empty(nslot, dtype=np.int64)
    keys = np.empty(nslot, dtype=np.int64)
    ni = 0
    for s in range(nslot):
        if U[g, s, ALIVE] and U[g, s, PLAYER] == player and U[g, s, A_HAS] == 0:
            idle[ni] = s
            keys[ni] = U[g, s, X] + U[g, s, Y] * W
            ni += 1
    order = np.argsort(keys[:ni])
    chosen = np.zeros(nslot, dtype=np.bool_)
    n = np0
    center = AG // 2
    one = np.empty(1, dtype=np.int64)
    for k in range(ni):
        s = idle[order[k]]
        row = grid[U[g, s, X] + U[g, s, Y] * W]
        atype = row[0]
        param = -1
        ax = 0
        ay = 0
        aut = -1
        if atype == MOVE:
            param = row[1]
        elif atype == HARVEST:
            param = row[2]
        elif atype == RETURN:
            param = row[3]
        elif atype == PRODUCE:
            param = row[4]
            aut = row[5]
        elif atype == ATTACK:
            ax = U[g, s, X] + row[6] % AG - center
            ay = U[g, s, Y] + row[6] // AG - center
        usage(UTT, W, U[g, s, X], U[g, s, Y], player, atype, param, aut, ru)
        one[0] = ru[1]
        if consistent(one, ru[0], ru[2], ru[3], pa_pos, pa_n, pa_r0, pa_r1, G[g, G_RES0], G[g, G_RES1]):
            if ru[0]:
                pa_pos[pa_n] = ru[1]
                pa_n += 1
            pa_r0 += ru[2]
            pa_r1 += ru[3]
            pslot[n] = s
            pact[n, 0] = atype
            pact[n, 1] = param
            pact[n, 2] = ax
            pact[n, 3] = ay
            pact[n, 4] = aut
            chosen[s] = True
            n += 1
    for s in range(nslot):
        if U[g, s, ALIVE] and U[g, s, PLAYER] == player and U[g, s, A_HAS] == 0 and not chosen[s]:
            pslot[n] = s
            pact[n, 0] = NONE
            pact[n, 1] = 1
            pact[n, 2] = 0
            pact[n, 3] = 0
            pact[n, 4] = -1
            n += 1

    # issue_safe: illegal actions become waits of the same length
    buf = np.empty((MAX_ACTIONS, 5), dtype=np.int64)
    for i in range(np0, n):
        s = pslot[i]
        ut = U[g, s, TYPE]
        if not can_execute(U, G, OCC, TER, UTT, PROD, g, W, H, s, pact[i], buf):
            d = eta(UTT, ut, pact[i, 0], pact[i, 1], pact[i, 4])
            pact[i, 0] = NONE
            pact[i, 1] = d
            pact[i, 2] = 0
            pact[i, 3] = 0
            pact[i, 4] = -1
        usage(UTT, W, U[g, s, X], U[g, s, Y], player, pact[i, 0], pact[i, 1], pact[i, 4], ru)
        if ru[0]:
            pos = ru[1]
            if TER[g, pos] != 0 or OCC[g, pos] >= 0:
                d = eta(UTT, ut, pact[i, 0], pact[i, 1], pact[i, 4])
                pact[i, 0] = NONE
                pact[i, 1] = d
                pact[i, 2] = 0
                pact[i, 3] = 0
                pact[i, 4] = -1
    return n


@njit(cache=True)
def issue(U, G, UTT, g, W, strategy, pslot, pact, lo, hi, trace):
    """``GameState.issue`` for pairs ``lo..hi``; ``trace`` gets the action each pair reports for rewards."""
    ru = np.empty(4, dtype=np.int64)
    aru = np.empty(4, dtype=np.int64)
    one = np.empty(1, dtype=np.int64)
    aone = np.empty(1, dtype=np.int64)
    t = G[g, G_TIME]
    for i in range(lo, hi):
        s = pslot[i]
        player = U[g, s, PLAYER]
        ut = U[g, s, TYPE]
        trace[i, :] = pact[i, :]
        a0, a1, a2, a3, a4 = pact[i, 0], pact[i, 1], pact[i, 2], pact[i, 3], pact[i, 4]
        usage(UTT, W, U[g, s, X], U[g, s, Y], player, a0, a1, a4, ru)
        one[0] = ru[1]
        replaced = False
        for o in assignment_order(U, G, g):
            usage(UTT, W, U[g, o, X], U[g, o, Y], U[g, o, PLAYER], U[g, o, A_TYPE], U[g, o, A_PARAM], U[g, o, A_UT], aru)
            aone[0] = aru[1]
            if consistent(aone, aru[0], aru[2], aru[3], one, ru[0], ru[2], ru[3], G[g, G_RES0], G[g, G_RES1]):
                continue
            if U[g, o, A_TIME] == t:
                cancel_old = False
                cancel_new = False
                if strategy == 2:
                    if np.random.randint(2) == 0:
                        cancel_new = True
                    else:
                        cancel_old = True
                elif strategy == 3:
                    if G[g, G_CANCEL] % 2 == 0:
                        cancel_new = True
                    else:
                        cancel_old = True
                    G[g, G_CANCEL] += 1
                else:
                    cancel_old = True
                    cancel_new = True
                d = min(eta(UTT, U[g, o, TYPE], U[g, o, A_TYPE], U[g, o, A_PARAM], U[g, o, A_UT]), eta(UTT, ut, a0, a1, a4))
                if cancel_old:
                    U[g, o, A_TYPE] = NONE
                    U[g, o, A_PARAM] = d
                    U[g, o, A_X] = 0
                    U[g, o, A_Y] = 0
                    U[g, o, A_UT] = -1
                if cancel_new:
                    a0, a1, a2, a3, a4 = NONE, d, 0, 0, -1
                    replaced = True
            else:
                a0, a1, a2, a3, a4 = NONE, -1, 0, 0, -1
                if not replaced:
                    trace[i, 0] = a0
                    trace[i, 1] = a1
                    trace[i, 2] = a2
                    trace[i, 3] = a3
                    trace[i, 4] = a4
        U[g, s, A_HAS] = 1
        U[g, s, A_TYPE] = a0
        U[g, s, A_PARAM] = a1
        U[g, s, A_X] = a2
        U[g, s, A_Y] = a3
        U[g, s, A_UT] = a4
        U[g, s, A_TIME] = t
        U[g, s, A_SEQ] = G[g, G_SEQ]
        G[g, G_SEQ] += 1


@njit(cache=True)
def execute(U, G, OCC, UTT, g, W, H, s, a0, a1, a2, a3, a4):
    if a0 == MOVE:
        move_unit(U, G, OCC, g, W, s, U[g, s, X] + DX[a1], U[g, s, Y] + DY[a1])
    elif a0 == ATTACK:
        o = unit_at(OCC, g, W, H, a2, a3)
        if o >= 0:
            ut = U[g, s, TYPE]
            lo = UTT[ut, T_MIND]
            hi = UTT[ut, T_MAXD]
            dmg = lo if lo == hi else lo + np.random.randint(1 + hi - lo)
            U[g, o, HP] -= dmg
            if U[g, o, HP] <= 0:
                remove_unit(U, OCC, g, W, o)
    elif a0 == HARVEST:
        r = unit_at(OCC, g, W, H, U[g, s, X] + DX[a1], U[g, s, Y] + DY[a1])
        ut = U[g, s, TYPE]
        if r >= 0 and UTT[U[g, r, TYPE], T_ISRES] and UTT[ut, T_CANHARV] and U[g, s, RES] == 0:
            U[g, r, RES] -= UTT[ut, T_HAMT]
            if U[g, r, RES] <= 0:
                remove_unit(U, OCC, g, W, r)
            U[g, s, RES] = UTT[ut, T_HAMT]
    elif a0 == RETURN:
        b = unit_at(OCC, g, W, H, U[g, s, X] + DX[a1], U[g, s, Y] + DY[a1])
        if b >= 0 and UTT[U[g, b, TYPE], T_ISSTOCK] and U[g, s, RES] > 0:
            G[g, G_RES0 + U[g, s, PLAYER]] += U[g, s, RES]
            U[g, s, RES] = 0
    elif a0 == PRODUCE:
        cost = UTT[a4, T_COST]
        p = U[g, s, PLAYER]
        if G[g, G_RES0 + p] - cost >= 0:
            add_unit(U, G, OCC, g, W, G[g, G_NEXT_ID], p, a4, U[g, s, X] + DX[a1], U[g, s, Y] + DY[a1], 0, UTT[a4, T_HP])
            G[g, G_RES0 + p] -= cost


@njit(cache=True)
def cycle(U, G, OCC, UTT, g, W, H):
    G[g, G_TIME] += 1
    t = G[g, G_TIME]
    order = assignment_order(U, G, g)
    n = 0
    ready = np.empty((order.shape[0], 6), dtype=np.int64)
    for s in order:
        if eta(UTT, U[g, s, TYPE], U[g, s, A_TYPE], U[g, s, A_PARAM], U[g, s, A_UT]) + U[g, s, A_TIME] <= t:
            ready[n, 0] = s
            ready[n, 1] = U[g, s, A_TYPE]
            ready[n, 2] = U[g, s, A_PARAM]
            ready[n, 3] = U[g, s, A_X]
            ready[n, 4] = U[g, s, A_Y]
            ready[n, 5] = U[g, s, A_UT]
            n += 1
    for i in range(n):
        s = ready[i, 0]
        U[g, s, A_HAS] = 0
        if ready[i, 1] != NONE:
            execute(U, G, OCC, UTT, g, W, H, s, ready[i, 1], ready[i, 2], ready[i, 3], ready[i, 4], ready[i, 5])


@njit(cache=True)
def unit_counts(U, G, g):
    c0 = 0
    c1 = 0
    for s in range(G[g, G_NSLOT]):
        if U[g, s, ALIVE]:
            p = U[g, s, PLAYER]
            if p == 0:
                c0 += 1
            elif p == 1:
                c1 += 1
    return c0, c1


@njit(cache=True)
def step_game(U, G, OCC, TER, UTT, PROD, g, W, H, AG, strategy, a0, a1, raw):
    """``MicroRTSGame.step`` for game ``g``: fills ``raw[2, 6]``; returns ``(gameover, winner)``."""
    cap = U.shape[1]
    if G[g, G_NSLOT] + 2 * W * H > cap:
        compact(U, G, OCC, g, W)
    nslot = G[g, G_NSLOT]
    before = np.full(W * H, -1, dtype=np.int64)
    for s in range(nslot):
        if U[g, s, ALIVE]:
            before[U[g, s, X] + U[g, s, Y] * W] = U[g, s, PLAYER]
    pslot = np.empty(2 * nslot + 2, dtype=np.int64)
    pact = np.empty((2 * nslot + 2, 5), dtype=np.int64)
    trace = np.empty((2 * nslot + 2, 5), dtype=np.int64)
    n0 = player_actions(U, G, OCC, TER, UTT, PROD, g, W, H, AG, 0, a0, pslot, pact, 0)
    issue(U, G, UTT, g, W, strategy, pslot, pact, 0, n0, trace)
    n1 = player_actions(U, G, OCC, TER, UTT, PROD, g, W, H, AG, 1, a1, pslot, pact, n0)
    issue(U, G, UTT, g, W, strategy, pslot, pact, n0, n1, trace)
    cycle(U, G, OCC, UTT, g, W, H)
    c0, c1 = unit_counts(U, G, g)
    gameover = (c0 + c1 == 0) or ((c0 > 0) != (c1 > 0))
    winner = -1
    if c0 > 0 and c1 == 0:
        winner = 0
    elif c1 > 0 and c0 == 0:
        winner = 1
    raw[:, :] = 0.0
    for p in range(2):
        if gameover:
            raw[p, 0] = 1.0 if winner == p else -1.0
        for i in range(n1):
            s = pslot[i]
            if U[g, s, PLAYER] != p:
                continue
            t = trace[i, 0]
            if t == HARVEST or t == RETURN:
                raw[p, 1] += 1
            elif t == PRODUCE and trace[i, 4] >= 0:
                cat = UTT[trace[i, 4], T_REWARD]
                if cat > 0:
                    raw[p, cat] += 1
            elif t == ATTACK:
                x = trace[i, 2]
                y = trace[i, 3]
                if 0 <= x < W and 0 <= y < H:
                    tgt = before[x + y * W]
                    if tgt == 1 - p:
                        raw[p, 4] += 1
                    elif tgt == p:
                        raw[p, 4] -= 1
    return gameover, winner


# ------------------------------------------------------- observation, mask
@njit(cache=True)
def observe(U, G, TER, g, W, H, player, n_types, out):
    """One-hot planes ``(H*W, C)`` for ``player`` (full observability)."""
    cells = W * H
    hp = np.zeros(cells, dtype=np.int64)
    res = np.zeros(cells, dtype=np.int64)
    own = np.zeros(cells, dtype=np.int64)
    typ = np.zeros(cells, dtype=np.int64)
    act = np.zeros(cells, dtype=np.int64)
    for s in range(G[g, G_NSLOT]):
        if U[g, s, ALIVE] == 0:
            continue
        pos = U[g, s, X] + U[g, s, Y] * W
        hp[pos] = U[g, s, HP]
        res[pos] = U[g, s, RES]
        p = U[g, s, PLAYER]
        own[pos] = ((p + player) % 2) + 1 if p >= 0 else 0
        typ[pos] = U[g, s, TYPE] + 1
        act[pos] = U[g, s, A_TYPE] if U[g, s, A_HAS] else 0
    out[:, :] = 0
    o_res = 5
    o_own = 10
    o_typ = 13
    o_act = o_typ + n_types + 1
    o_ter = o_act + 6
    for c in range(cells):
        out[c, min(max(hp[c], 0), 4)] = 1
        out[c, o_res + min(max(res[c], 0), 4)] = 1
        out[c, o_own + own[c]] = 1
        out[c, o_typ + min(typ[c], n_types)] = 1
        out[c, o_act + act[c]] = 1
        out[c, o_ter + min(TER[g, c], 1)] = 1


@njit(cache=True)
def action_mask(U, G, OCC, TER, UTT, PROD, g, W, H, AG, player, offs, out):
    out[:, :] = 0
    buf = np.empty((MAX_ACTIONS, 5), dtype=np.int64)
    center = AG // 2
    for s in range(G[g, G_NSLOT]):
        if U[g, s, ALIVE] == 0 or U[g, s, PLAYER] != player or U[g, s, A_HAS]:
            continue
        x = U[g, s, X]
        y = U[g, s, Y]
        row = x + y * W
        n = unit_actions(U, G, OCC, TER, UTT, PROD, g, W, H, s, 10, buf)
        for i in range(n):
            t = buf[i, 0]
            out[row, t] = 1
            if t == MOVE:
                out[row, offs[1] + buf[i, 1]] = 1
            elif t == HARVEST:
                out[row, offs[2] + buf[i, 1]] = 1
            elif t == RETURN:
                out[row, offs[3] + buf[i, 1]] = 1
            elif t == PRODUCE:
                out[row, offs[4] + buf[i, 1]] = 1
                out[row, offs[5] + buf[i, 4]] = 1
            elif t == ATTACK:
                out[row, offs[6] + (center + buf[i, 3] - y) * AG + (center + buf[i, 2] - x)] = 1


# --------------------------------------------------------------- the bots
@njit(cache=True)
def first_step(U, OCC, TER, g, W, H, sx, sy, goal):
    """Direction of the first move on a shortest path from ``(sx, sy)`` to a goal cell, or -1."""
    cells = W * H
    parent = np.full(cells, -2, dtype=np.int64)
    q = np.empty(cells, dtype=np.int64)
    start = sx + sy * W
    parent[start] = -1
    q[0] = start
    head = 0
    tail = 1
    while head < tail:
        c = q[head]
        head += 1
        if goal[c] and c != start:
            node = c
            while parent[node] != start:
                node = parent[node]
            nx = node % W
            ny = node // W
            for d in range(4):
                if sx + DX[d] == nx and sy + DY[d] == ny:
                    return d
        cx = c % W
        cy = c // W
        for d in range(4):
            nx = cx + DX[d]
            ny = cy + DY[d]
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            nc = nx + ny * W
            if parent[nc] != -2:
                continue
            if TER[g, nc] or OCC[g, nc] >= 0:
                continue
            parent[nc] = c
            q[tail] = nc
            tail += 1
    return -1


@njit(cache=True)
def mark_adjacent(goal, W, H, x, y):
    for d in range(4):
        nx = x + DX[d]
        ny = y + DY[d]
        if 0 <= nx < W and 0 <= ny < H:
            goal[nx + ny * W] = True


@njit(cache=True)
def worker_rush(U, G, OCC, TER, UTT, PROD, g, W, H, AG, player, harvesters, army, n_army,
                worker_t, barracks_t, light_t, out):
    """``bots.WorkerRushBot.act``; ``army`` holds unit type ids (``n_army == 0``: plain worker rush)."""
    out[:, :] = 0
    nslot = G[g, G_NSLOT]
    center = AG // 2
    cells = W * H
    enemy_goal = np.zeros(cells, dtype=np.bool_)
    base_goal = np.zeros(cells, dtype=np.bool_)
    res_goal = np.zeros(cells, dtype=np.bool_)
    n_bases = 0
    has_barracks = False
    building = False
    wids = np.empty(nslot, dtype=np.int64)
    n_workers = 0
    for s in range(nslot):
        if U[g, s, ALIVE] == 0:
            continue
        p = U[g, s, PLAYER]
        ut = U[g, s, TYPE]
        if p == 1 - player:
            mark_adjacent(enemy_goal, W, H, U[g, s, X], U[g, s, Y])
        if p == player and UTT[ut, T_ISSTOCK]:
            mark_adjacent(base_goal, W, H, U[g, s, X], U[g, s, Y])
            n_bases += 1
        if UTT[ut, T_ISRES]:
            mark_adjacent(res_goal, W, H, U[g, s, X], U[g, s, Y])
        if p == player and UTT[ut, T_CANHARV]:
            wids[n_workers] = U[g, s, ID]
            n_workers += 1
        if p == player and ut == barracks_t:
            has_barracks = True
        if p == player and U[g, s, A_HAS] and U[g, s, A_TYPE] == PRODUCE and U[g, s, A_UT] == barracks_t:
            building = True
    wsorted = np.sort(wids[:n_workers])
    n_harv = min(harvesters, n_workers) if n_bases > 0 else 0
    want_barracks = n_army > 0 and not has_barracks and not building
    buf = np.empty((MAX_ACTIONS, 5), dtype=np.int64)

    for s in range(nslot):
        if U[g, s, ALIVE] == 0 or U[g, s, PLAYER] != player or U[g, s, A_HAS]:
            continue
        x = U[g, s, X]
        y = U[g, s, Y]
        ut = U[g, s, TYPE]
        row = x + y * W
        n = unit_actions(U, G, OCC, TER, UTT, PROD, g, W, H, s, 10, buf)
        done = False
        for i in range(n):
            if buf[i, 0] == ATTACK:
                out[row, 0] = ATTACK
                out[row, 6] = (center + buf[i, 3] - y) * AG + (center + buf[i, 2] - x)
                done = True
                break
        if done:
            continue
        if not UTT[ut, T_CANMOVE]:  # buildings
            wanted = -1
            if UTT[ut, T_ISSTOCK]:
                if n_army == 0 or n_workers < harvesters + 1:
                    wanted = worker_t
            else:
                k = G[g, G_BOT0 + player]
                wanted = army[k % n_army] if n_army > 0 else light_t
            for i in range(n):
                if buf[i, 0] == PRODUCE and buf[i, 4] == wanted:
                    out[row, 0] = PRODUCE
                    out[row, 4] = buf[i, 1]
                    out[row, 5] = wanted
                    if not UTT[ut, T_ISSTOCK]:
                        G[g, G_BOT0 + player] += 1
                    break
            continue
        is_harv = False
        for k in range(n_harv):
            if wsorted[k] == U[g, s, ID]:
                is_harv = True
        if want_barracks and is_harv:
            for i in range(n):
                if buf[i, 0] == PRODUCE and buf[i, 4] == barracks_t:
                    out[row, 0] = PRODUCE
                    out[row, 4] = buf[i, 1]
                    out[row, 5] = barracks_t
                    want_barracks = False
                    done = True
                    break
            if done:
                continue
        if is_harv:
            want = RETURN if U[g, s, RES] > 0 else HARVEST
            for i in range(n):
                if buf[i, 0] == want:
                    out[row, 0] = want
                    out[row, 3 if want == RETURN else 2] = buf[i, 1]
                    done = True
                    break
            if done:
                continue
            d = first_step(U, OCC, TER, g, W, H, x, y, base_goal if want == RETURN else res_goal)
        else:
            d = first_step(U, OCC, TER, g, W, H, x, y, enemy_goal)
        if d >= 0:
            out[row, 0] = MOVE
            out[row, 1] = d


@njit(cache=True)
def random_bot(mask, offs, out):
    """``bots.RandomBot.act``: a uniformly random valid value for each action component."""
    cells = mask.shape[0]
    for k in range(offs.shape[0] - 1):
        for c in range(cells):
            best = 0
            bestv = -1.0
            for j in range(offs[k + 1] - offs[k]):
                if mask[c, offs[k] + j]:
                    v = np.random.random()
                    if v > bestv:
                        bestv = v
                        best = j
            out[c, k] = best


# ----------------------------------------------------------- batched calls
@njit(cache=True)
def reset_game(U, G, OCC, TER, g, W, map_units, map_res, map_ter):
    U[g, :, :] = 0
    G[g, :] = 0
    OCC[g, :] = -1
    TER[g, :] = map_ter
    G[g, G_RES0] = map_res[0]
    G[g, G_RES1] = map_res[1]
    for i in range(map_units.shape[0]):
        r = map_units[i]
        add_unit(U, G, OCC, g, W, r[0], r[1], r[2], r[3], r[4], r[5], r[6])


@njit(cache=True, parallel=True)
def step_batch(U, G, OCC, TER, UTT, PROD, W, H, AG, strategy, a0, a1, raw, over, winner, max_steps,
               trunc, map_idx, map_units, map_nunits, map_res, map_ter, auto_reset):
    """Step every game; finished games are reset when ``auto_reset``."""
    for g in prange(U.shape[0]):
        ov, w = step_game(U, G, OCC, TER, UTT, PROD, g, W, H, AG, strategy, a0[g], a1[g], raw[g])
        G[g, G_STEPS] += 1
        over[g] = ov
        winner[g] = w
        trunc[g] = (not ov) and G[g, G_STEPS] >= max_steps
        if auto_reset and (ov or trunc[g]):
            m = map_idx[g]
            reset_game(U, G, OCC, TER, g, W, map_units[m, : map_nunits[m]], map_res[m], map_ter[m])


@njit(cache=True, parallel=True)
def observe_batch(U, G, OCC, TER, UTT, PROD, W, H, AG, n_types, offs, players, obs, mask):
    """Observation and action mask of ``players[g]`` in every game."""
    for g in prange(U.shape[0]):
        observe(U, G, TER, g, W, H, players[g], n_types, obs[g])
        action_mask(U, G, OCC, TER, UTT, PROD, g, W, H, AG, players[g], offs, mask[g])


@njit(cache=True, parallel=True)
def bots_batch(U, G, OCC, TER, UTT, PROD, W, H, AG, offs, players, kinds, harvesters, army, n_army,
               worker_t, barracks_t, light_t, masks, out):
    """Scripted actions for ``players[g]``: ``kinds[g]`` 0 = none, 1 = random, 2 = worker rush."""
    for g in prange(U.shape[0]):
        k = kinds[g]
        if k == 1:
            random_bot(masks[g], offs, out[g])
        elif k == 2:
            worker_rush(U, G, OCC, TER, UTT, PROD, g, W, H, AG, players[g], harvesters[g], army[g], n_army[g],
                        worker_t, barracks_t, light_t, out[g])


@njit(cache=True)
def seed(s):
    np.random.seed(s)
