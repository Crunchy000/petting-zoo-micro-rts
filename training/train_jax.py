# SPDX-License-Identifier: GPL-3.0-only
"""Train a MicroRTS gridnet agent with PPO and invalid-action masking, entirely in JAX.

This is ``train_selfplay.py`` rebuilt around ``microrts_pz.jax_engine``: the
games, the scripted bots, the self-play opponents and the network all run in
one jitted rollout on the accelerator, and the PPO update reads the rollout
straight from device memory. Nothing goes through Python per step, per game
or per unit, so a GPU is kept busy end to end.

    pip install -e ".[jax]"           # add "jax[cuda12]" on a GPU machine
    python training/train_jax.py --map maps/8x8/basesWorkers8x8A.xml --total-steps 5000000 --bc-steps 200000 --out runs/8x8
    python training/train_jax.py --map maps/16x16/basesWorkers16x16.xml --init runs/8x8/latest.pkl --out runs/16x16

How it trains (the same recipe as ``train_selfplay.py``):

* The learner plays a random side in every episode against an opponent from
  a weighted pool (``--opponents``): the scripted bots of ``microrts_pz.bots``
  and frozen snapshots of itself (``self``).
* ``--bc-steps`` starts with behaviour cloning: the bots play the learner's
  side and the network learns to imitate them, which gets a working economy
  and army going much sooner than PPO from scratch.
* Then PPO with GAE, clipped objective, entropy bonus and a linearly decayed
  learning rate. Every ``--snapshot-every`` updates the current weights join
  the ring of ``--keep-snapshots`` self-play opponents.

What is different from the PyTorch trainer, for speed:

* Opponents are fixed per game slot instead of drawn per episode: the
  ``--num-envs`` games are split in proportion to the pool weights, so each
  opponent type is computed for its own block of games only. Self-play games
  are split evenly over the snapshots and all snapshots run in one batched
  call (``vmap`` over stacked weights).
* Action masks are stored bit-packed (8x smaller), so long rollouts of many
  games fit in GPU memory.
* ``--dtype bfloat16`` (the default on a GPU) runs the network in bfloat16
  with float32 weights.

Outputs in ``--out``: ``latest.pkl`` (weights, optimiser, snapshots; what
``--resume`` reads), ``latest.pt`` (the same weights as a PyTorch checkpoint
for ``play.py`` and the website, written when PyTorch is installed) and
``metrics.csv``. ``--init`` accepts either format, so a PyTorch agent can be
fine-tuned here.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opponents import BOT_SPECS, TEACHERS, parse_weights  # noqa: E402


#: Training statistics written to ``metrics.csv`` (averaged over an update's minibatches).
STAT_COLUMNS = ("pg_loss", "v_loss", "entropy", "bc_loss", "approx_kl", "clipfrac")


def slot_layout(num_envs: int, weights: dict[str, float], keep_snapshots: int):
    """Opponent name for every game slot: self-play first (a multiple of ``keep_snapshots``), then each bot in a block."""
    total = sum(weights.values())
    n_self = 0
    if weights.get("self", 0) > 0:
        n_self = keep_snapshots * max(1, round(num_envs * weights["self"] / total / keep_snapshots))
        n_self = min(n_self, num_envs - num_envs % keep_snapshots)
    bots = [n for n in weights if n != "self" and weights[n] > 0]
    rest = num_envs - n_self
    if rest and not bots:
        if num_envs % keep_snapshots:
            raise SystemExit("with only self-play opponents, --num-envs must be a multiple of --keep-snapshots")
        n_self, rest = num_envs, 0
    if not bots and not n_self:
        raise SystemExit("--opponents has no positive weight")
    # largest-remainder split of the remaining slots between the bots, random last so the rush bots form one block
    bots.sort(key=lambda n: BOT_SPECS[n][0] == "random")
    w = np.array([weights[n] for n in bots], dtype=float)
    share = rest * w / w.sum() if bots else w
    counts = np.floor(share).astype(int)
    for i in np.argsort(-(share - counts))[: rest - counts.sum()]:
        counts[i] += 1
    names = ["self"] * n_self
    for n, c in zip(bots, counts):
        names += [n] * int(c)
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", help="map to train on (repeat for several maps of the same size)")
    ap.add_argument("--out", default="runs/microrts")
    ap.add_argument("--total-steps", type=int, default=10_000_000, help="environment steps (summed over games)")
    ap.add_argument("--time-limit", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--num-envs", type=int, default=1024, help="games played in parallel")
    ap.add_argument("--num-steps", type=int, default=64, help="rollout length per game per update")
    ap.add_argument("--max-steps", type=int, default=2000, help="episode length cap in ticks")
    ap.add_argument("--opponents", default="mixed:1,light:1,heavy:1,ranged:1,rush:1,random:0.5,self:3",
                    help="weighted opponent pool, e.g. 'mixed:2,random:1,self:3' (self = past snapshots)")
    ap.add_argument("--bc-steps", type=int, default=0, help="behaviour-cloning warm start: env steps imitating the bots")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between self-play snapshots")
    ap.add_argument("--keep-snapshots", type=int, default=4, help="snapshots used as self-play opponents")
    ap.add_argument("--init", help="start from this checkpoint (.pkl from this script or a PyTorch .pt)")
    ap.add_argument("--resume", action="store_true", help="continue from --out/latest.pkl")
    ap.add_argument("--width", type=int, default=64, help="network channels")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--minibatch-size", type=int, default=4096, help="samples per gradient step")
    ap.add_argument("--epochs", type=int, default=2, help="passes over each rollout")
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default=None,
                    help="network compute precision (default: bfloat16 on a GPU or TPU, float32 on a CPU)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp

    import jax_model as P
    from microrts_pz.env import DEFAULT_REWARD_WEIGHT
    from microrts_pz.jax_engine import JaxBatchGame
    from microrts_pz.jax_engine.bots import random_actions, rush_actions

    maps = args.map or ["maps/16x16/basesWorkers16x16.xml"]
    os.makedirs(args.out, exist_ok=True)
    backend = jax.default_backend()
    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[args.dtype or ("float32" if backend == "cpu" else "bfloat16")]
    weights = parse_weights(args.opponents)
    K = args.keep_snapshots
    N, T = args.num_envs, args.num_steps
    names = slot_layout(N, weights, K)
    pool = sorted(set(names), key=names.index)
    n_self = names.count("self")
    rush_lo = n_self
    rush_hi = n_self + sum(BOT_SPECS[n][0] == "rush" for n in names[n_self:])
    rng = np.random.default_rng(args.seed)

    games = JaxBatchGame(maps, num_games=N, max_steps=args.max_steps, map_index=rng.integers(len(maps), size=N))
    games.reset()
    rules, ids = games.rules, games._ids
    H, W, C = games.height, games.width, games.num_channels
    dims = tuple(int(d) for d in games.action_dims)
    mask_size = games.mask_size
    reward_w = jnp.asarray(DEFAULT_REWARD_WEIGHT, jnp.float32)

    def bot_arrays(specs):
        harv = np.array([s[1] for s in specs], np.int32)
        army = np.zeros((len(specs), 3), np.int32)
        n_army = np.zeros(len(specs), np.int32)
        for i, s in enumerate(specs):
            a = () if s[2] is None else ((s[2],) if isinstance(s[2], str) else tuple(s[2]))
            army[i, : len(a)] = [games.utt[x].id for x in a]
            n_army[i] = len(a)
        return jnp.asarray(harv), jnp.asarray(army), jnp.asarray(n_army)

    opp_rush = bot_arrays([BOT_SPECS[n] for n in names[rush_lo:rush_hi]])
    teacher = bot_arrays([BOT_SPECS[TEACHERS[i % len(TEACHERS)]] for i in range(N)])
    opp_id = jnp.asarray([pool.index(n) for n in names], jnp.int32)

    # ---------------------------------------------------------------- weights
    latest = os.path.join(args.out, "latest.pkl")
    start_step, n_snapped, update0 = 0, 0, 0
    if args.resume and os.path.exists(latest):
        ckpt = P.load_checkpoint(latest)
        params, opt = ckpt["params"], jax.tree.map(jnp.asarray, ckpt["optimizer"])
        snapshots = jax.tree.map(jnp.asarray, ckpt["snapshots"])
        start_step, n_snapped, update0 = ckpt["global_step"], ckpt["n_snapshots"], ckpt["update"]
        if jax.tree.leaves(snapshots)[0].shape[0] != K:
            raise SystemExit(f"{latest} has {jax.tree.leaves(snapshots)[0].shape[0]} snapshots; pass that --keep-snapshots")
        print(f"resumed from {latest} at step {start_step:,}")
    else:
        if args.init:
            params, cfg = P.load_params(args.init)
            if tuple(cfg["dilations"]) != P.DEFAULT_DILATIONS:
                raise SystemExit("--init checkpoint has a different network shape")
            print(f"initialised from {args.init}")
        else:
            params = P.init_params(jax.random.PRNGKey(args.seed), C, dims, width=args.width)
        opt = P.adam_init(params)
        snapshots = jax.tree.map(lambda p: jnp.broadcast_to(p, (K,) + p.shape), params)
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
    print(f"map {H}x{W}, {n_params:,} parameters, {backend} ({jax.devices()[0].device_kind}), {jnp.dtype(dtype).name}, "
          f"{N} games: " + ", ".join(f"{n} {names.count(n)}" for n in pool))

    # ---------------------------------------------------------------- rollout
    def forward(p, obs):
        return P.apply(p, obs, dtype=dtype)

    def rush_block(state, player, harv, army, n_army, counter):
        return jax.vmap(lambda s, p, h, a, n, c: rush_actions(rules, s, p, h, a, n, c, ids["Worker"], ids["Barracks"], ids["Light"]))(
            state, player, harv, army, n_army, counter)

    def block(tree, lo, hi):
        return jax.tree.map(lambda x: x[lo:hi], tree)

    def rollout(params, snapshots, carry, teaching: bool):
        """``T`` steps of every game; returns the new carry, the buffer and episode statistics."""

        def step(c, _):
            gs, side, counter, key, ep_ret = c
            key, k_learn, k_opp, k_rand, k_side = jax.random.split(key, 5)
            idx = jnp.arange(N)
            obs, mask = games.observe_fn(gs, side)
            opp_obs, opp_mask = games.observe_fn(gs, 1 - side)
            logits, value = forward(params, obs)
            if teaching:
                act, counter_l = rush_block(gs, side, *teacher, counter[idx, side])
                counter = counter.at[idx, side].set(counter_l)
                logp = jnp.zeros(N)
            else:
                act, logp = P.sample(k_learn, logits, mask, dims)
            # opponents, block by block
            parts = []
            if n_self:
                o = opp_obs[:n_self].reshape((K, n_self // K) + opp_obs.shape[1:])
                lg, _ = jax.vmap(forward)(snapshots, o)
                a, _ = P.sample(k_opp, lg.reshape((n_self,) + lg.shape[2:]), opp_mask[:n_self], dims)
                parts.append(a)
            if rush_hi > rush_lo:
                opp = 1 - side[rush_lo:rush_hi]
                rows = idx[rush_lo:rush_hi]
                a, cnt = rush_block(block(gs, rush_lo, rush_hi), opp, *opp_rush, counter[rows, opp])
                counter = counter.at[rows, opp].set(cnt)
                parts.append(a)
            if N > rush_hi:
                parts.append(random_actions(rules, k_rand, opp_mask[rush_hi:]))
            opp_act = jnp.concatenate(parts) if len(parts) > 1 else parts[0]
            first = (side == 0)[:, None, None]
            gs, raw, over, trunc, winner, _ = games.step_fn(gs, jnp.where(first, act, opp_act), jnp.where(first, opp_act, act))
            r = (raw @ reward_w)[idx, side]
            done = over | trunc
            result = jnp.where(winner < 0, 0, jnp.where(winner == side, 1, -1))
            ep_ret = ep_ret + r
            out = {"obs": obs, "mask": P.pack_mask(mask), "act": act.astype(jnp.int8), "logp": logp, "val": value,
                   "rew": r, "done": done, "result": jnp.where(done, result, 0), "ep_ret": jnp.where(done, ep_ret, 0.0)}
            ep_ret = jnp.where(done, 0.0, ep_ret)
            counter = jnp.where(done[:, None], 0, counter)
            side = jnp.where(done, jax.random.bernoulli(k_side, 0.5, (N,)).astype(jnp.int32), side)
            return (gs, side, counter, key, ep_ret), out

        carry, buf = jax.lax.scan(step, carry, None, length=T)
        obs, _ = games.observe_fn(carry[0], carry[1])
        next_value = forward(params, obs)[1]
        # GAE
        def gae(last, x):
            rew, val, nval, done = x
            delta = rew + args.gamma * nval * (1 - done) - val
            last = delta + args.gamma * args.gae_lambda * (1 - done) * last
            return last, last

        done = buf["done"].astype(jnp.float32)
        nvals = jnp.concatenate([buf["val"][1:], next_value[None]])
        _, adv = jax.lax.scan(gae, jnp.zeros(N), (buf["rew"], buf["val"], nvals, done), reverse=True)
        buf["adv"], buf["ret"] = adv, adv + buf["val"]
        # episode statistics per opponent: wins, draws, losses, return sum
        d = buf["done"]
        per = lambda x: jax.ops.segment_sum(x.sum(0), opp_id, num_segments=len(pool))  # noqa: E731
        stats = {"episodes": per(d.astype(jnp.int32)), "wins": per(d & (buf["result"] == 1)),
                 "losses": per(d & (buf["result"] == -1)), "return": per(buf["ep_ret"])}
        data = {k: buf[k].reshape((T * N,) + buf[k].shape[2:]) for k in ("obs", "mask", "act", "logp", "adv", "ret")}
        return carry, data, stats

    rollout_ppo = jax.jit(lambda p, s, c: rollout(p, s, c, False))
    rollout_bc = jax.jit(lambda p, s, c: rollout(p, s, c, True))

    # ----------------------------------------------------------------- update
    def minibatch(params, opt, data, idx, lr, teaching: bool):
        batch = jax.tree.map(lambda x: x[idx], data)
        batch["mask"] = P.unpack_mask(batch["mask"], mask_size)
        if teaching:
            fn = lambda p: P.bc_loss_and_stats(p, batch, dims, dtype, args.vf_coef)  # noqa: E731
        else:
            fn = lambda p: P.ppo_loss_and_stats(p, batch, dims, dtype, args.clip, args.ent_coef, args.vf_coef)  # noqa: E731
        (_, stats), grads = jax.value_and_grad(fn, has_aux=True)(params)
        params, opt = P.adam_step(params, grads, opt, lr=lr, max_norm=args.max_grad_norm)
        return params, opt, stats

    mb_ppo = jax.jit(lambda *a: minibatch(*a, False), donate_argnums=(0, 1))
    mb_bc = jax.jit(lambda *a: minibatch(*a, True), donate_argnums=(0, 1))
    snap_fn = jax.jit(lambda s, p, slot: jax.tree.map(lambda a, b: a.at[slot].set(b), s, p))

    batch = T * N
    mb = min(args.minibatch_size, batch)
    n_mb = batch // mb

    # ---------------------------------------------------------------- logging
    metrics_path = os.path.join(args.out, "metrics.csv")
    new_file = not (args.resume and os.path.exists(metrics_path))
    metrics = open(metrics_path, "w" if new_file else "a", newline="")
    writer = csv.writer(metrics)
    if new_file:
        writer.writerow(["step", "phase", "sps", "episodes", "mean_return", "win_rate", "per_opponent", *STAT_COLUMNS,
                         "rollout_s", "update_s", "peak_mem_gb"])
    has_torch = importlib.util.find_spec("torch") is not None  # only needed to export latest.pt

    def save(step, update):
        extra = {"optimizer": jax.tree.map(np.asarray, opt), "snapshots": jax.tree.map(np.asarray, snapshots),
                 "n_snapshots": n_snapped, "global_step": step, "update": update, "maps": maps, "args": vars(args)}
        P.save_checkpoint(latest, params, extra, action_dims=dims)
        if has_torch:
            P.export_torch(os.path.join(args.out, "latest.pt"), params, {"global_step": step, "maps": maps}, action_dims=dims)

    # ------------------------------------------------------------------- loop
    key = jax.random.PRNGKey(args.seed + 7919 * update0)
    side0 = jnp.asarray(rng.integers(2, size=N), jnp.int32)
    carry = (games.state, side0, jnp.zeros((N, 2), jnp.int32), key, jnp.zeros(N))
    num_updates = max(1, math.ceil((args.total_steps - start_step) / batch))
    global_step = start_step
    t0 = time.time()
    perm_key = jax.random.PRNGKey(args.seed + 1)
    recent = {"episodes": np.zeros(len(pool)), "wins": np.zeros(len(pool)), "losses": np.zeros(len(pool)), "return": np.zeros(len(pool))}
    for update in range(update0 + 1, update0 + num_updates + 1):
        teaching = global_step < args.bc_steps
        t_roll = time.time()
        carry, data, ep = (rollout_bc if teaching else rollout_ppo)(params, snapshots, carry)
        jax.block_until_ready(data)
        t_roll = time.time() - t_roll
        t_upd = time.time()
        frac = 1.0 - (global_step - start_step) / max(1, args.total_steps - start_step)
        lr = jnp.float32(args.lr * max(frac, 0.0))
        step_fn = mb_bc if teaching else mb_ppo
        stats = []
        for _ in range(args.epochs):
            perm_key, k = jax.random.split(perm_key)
            perm = jax.random.permutation(k, batch)[: n_mb * mb].reshape(n_mb, mb)
            for i in range(n_mb):
                params, opt, s = step_fn(params, opt, data, perm[i], lr)
                stats.append(s)
        stats = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        t_upd = time.time() - t_upd
        global_step += batch

        # episode results, smoothed over recent updates
        ep = {k: np.asarray(v, float) for k, v in ep.items()}
        for k in recent:
            recent[k] = 0.9 * recent[k] + ep[k]
        n_ep = recent["episodes"].sum()
        score = (recent["wins"].sum() - recent["losses"].sum()) / n_ep if n_ep else float("nan")
        mean_ret = recent["return"].sum() / n_ep if n_ep else float("nan")
        per_opp = " ".join(f"{n}={recent['wins'][i] / recent['episodes'][i]:.0%}" for i, n in enumerate(pool) if recent["episodes"][i] > 0)
        sps = int((global_step - start_step) / (time.time() - t0))
        phase = "bc" if teaching else "ppo"
        print(f"[{phase}] step {global_step:>11,}  sps {sps:>6}  return {mean_ret:7.2f}  score {score:+.2f}  "
              f"rollout {t_roll:.1f}s update {t_upd:.1f}s  wins {per_opp}", flush=True)
        mem = (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", float("nan")) / 2**30
        writer.writerow([global_step, phase, sps, int(ep["episodes"].sum()), mean_ret, score, per_opp,
                         *(stats.get(k, float("nan")) for k in STAT_COLUMNS), round(t_roll, 4), round(t_upd, 4), round(mem, 3)])
        metrics.flush()
        if not teaching and update % args.snapshot_every == 0:
            snapshots = snap_fn(snapshots, params, n_snapped % K)
            n_snapped += 1
        save(global_step, update)
        if args.time_limit and time.time() - t0 > args.time_limit * 60:
            print(f"time limit reached at step {global_step:,}")
            break
        if global_step >= args.total_steps:
            break
    metrics.close()
    print(f"done; latest checkpoint at {latest}" + (" and latest.pt" if has_torch else ""))


if __name__ == "__main__":
    main()
