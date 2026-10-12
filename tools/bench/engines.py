# SPDX-License-Identifier: GPL-3.0-only
"""Time the batched MicroRTS engines, alone and in a self-play rollout with the gridnet policy.

    python tools/bench/engines.py --map maps/16x16/basesWorkers16x16.xml --games 256 1024 4096

Engines:

* ``numba``: ``microrts_pz.fast.BatchGame`` on all CPU cores (needs ``numba``).
* ``jax``:   ``microrts_pz.jax_engine`` on the default JAX device (GPU when there is one).

Workloads, each reported as environment steps (game ticks) per second:

* ``engine``: observations + masks for both players, uniformly random legal
  actions, one tick. No network.
* ``rollout``: the same with the gridnet actor-critic choosing both players'
  actions (self-play), storing what PPO keeps (obs, masks, actions, log-probs,
  values). PyTorch on the GPU for the Numba engine, as in
  ``training/train_selfplay.py``; the JAX port of the same network for the JAX
  engine, traced into one ``lax.scan`` with the engine.
* ``update``: one PPO epoch (forward, backward, Adam) of the same network over
  ``--update-batch`` mid-game samples, PyTorch with AMP vs JAX in bfloat16
  (reported as samples per second; the ``numba`` row is PyTorch).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))


def timed(fn, repeats):
    fn()  # warm up / compile
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


# ---------------------------------------------------------------- numba
def numba_engine(map_path, B, T, repeats):
    from microrts_pz.fast import BatchGame

    g = BatchGame([map_path], num_games=B, max_steps=2000)
    g.reset(seed=0)

    def run():
        for _ in range(T):
            acts = []
            for p in (0, 1):
                _, mask = g.observe(p)
                acts.append(g.bot_actions(p, "random", masks=mask))
            g.step(acts[0], acts[1])

    return B * T / timed(run, repeats)


def numba_rollout(map_path, B, T, repeats, model_path, device, amp):
    import torch

    sys.path.insert(0, os.path.dirname(model_path))
    from model import GridnetPolicy, masked_policy

    from microrts_pz.fast import BatchGame

    g = BatchGame([map_path], num_games=B, max_steps=2000)
    g.reset(seed=0)
    dev = torch.device(device)
    net = GridnetPolicy(in_channels=g.num_channels).to(dev).eval()
    H, W, C, cells = g.height, g.width, g.num_channels, g.num_cells
    buf = {
        "obs": torch.zeros((T, 2 * B, H, W, C), dtype=torch.int8, device=dev),
        "mask": torch.zeros((T, 2 * B, cells, 78), dtype=torch.bool, device=dev),
        "act": torch.zeros((T, 2 * B, cells, 7), dtype=torch.long, device=dev),
        "logp": torch.zeros((T, 2 * B), device=dev),
        "val": torch.zeros((T, 2 * B), device=dev),
    }

    def run():
        for t in range(T):
            o0, m0 = g.observe(0)
            o1, m1 = g.observe(1)
            obs = torch.as_tensor(np.concatenate([o0, o1])).to(dev, non_blocking=True)
            mask = torch.as_tensor(np.concatenate([m0, m1])).to(dev, non_blocking=True).bool()
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits, value = net(obs)
            act, logp, _, _ = masked_policy(logits.float(), mask)
            buf["obs"][t], buf["mask"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = obs, mask, act, logp, value.float()
            a = act.cpu().numpy()
            g.step(a[:B], a[B:])
        if dev.type == "cuda":
            torch.cuda.synchronize()

    return B * T / timed(run, repeats)


# ------------------------------------------------------------------ jax
def jax_engine(map_path, B, T, repeats):
    import jax
    import jax.numpy as jnp

    from microrts_pz.jax_engine import JaxBatchGame

    g = JaxBatchGame([map_path], num_games=B, max_steps=2000)
    g.reset()
    offs = np.asarray(g.rules.mask_offsets)

    def random_actions(key, mask):
        keys = jax.random.split(key, 7)
        cols = []
        for k in range(7):
            m = mask[..., offs[k] : offs[k + 1]]
            cols.append(jnp.argmax(jnp.where(m, jax.random.uniform(keys[k], m.shape) + 1.0, 0.0), -1))
        return jnp.stack(cols, -1).astype(jnp.int32)

    def body(carry, _):
        s, key = carry
        key, k0, k1 = jax.random.split(key, 3)
        _, m0 = g.observe_fn(s, 0)
        _, m1 = g.observe_fn(s, 1)
        s, *_ = g.step_fn(s, random_actions(k0, m0), random_actions(k1, m1))
        return (s, key), None

    roll = jax.jit(lambda s, key: jax.lax.scan(body, (s, key), None, length=T)[0])
    state = {"c": (g.state, jax.random.PRNGKey(0))}

    def run():
        state["c"] = roll(*state["c"])
        jax.block_until_ready(state["c"])

    return B * T / timed(run, repeats)


def jax_rollout(map_path, B, T, repeats, dtype):
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, HERE)
    import jax_policy as P

    from microrts_pz.jax_engine import JaxBatchGame

    g = JaxBatchGame([map_path], num_games=B, max_steps=2000)
    g.reset()
    dims = tuple(int(d) for d in g.action_dims)
    params = P.init_params(jax.random.PRNGKey(0), g.num_channels, dims)
    dt = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[dtype]

    def body(carry, _):
        s, key = carry
        key, k = jax.random.split(key)
        o0, m0 = g.observe_fn(s, 0)
        o1, m1 = g.observe_fn(s, 1)
        obs, mask = jnp.concatenate([o0, o1]), jnp.concatenate([m0, m1])
        logits, value = P.apply(params, obs, dtype=dt)
        act, logp = P.sample(k, logits, mask, dims)
        s, raw, over, trunc, *_ = g.step_fn(s, act[:B], act[B:])
        return (s, key), (obs, mask, act, logp, value, raw, over | trunc)

    roll = jax.jit(lambda s, key: jax.lax.scan(body, (s, key), None, length=T))
    state = {"c": (g.state, jax.random.PRNGKey(1))}

    def run():
        c, out = roll(*state["c"])
        jax.block_until_ready(out)
        state["c"] = c

    return B * T / timed(run, repeats)


# --------------------------------------------------------------- update
def sample_batch(map_path, M, seed=0):
    """``M`` realistic training samples (mid-game observations, masks, legal actions) as NumPy arrays."""
    import jax
    import jax.numpy as jnp

    from microrts_pz.jax_engine import JaxBatchGame

    g = JaxBatchGame([map_path], num_games=M, max_steps=10**6)
    g.reset()
    offs = np.asarray(g.rules.mask_offsets)
    rng = np.random.default_rng(seed)

    def rand(mask):
        cols = []
        for k in range(7):
            m = mask[..., offs[k] : offs[k + 1]] > 0
            cols.append(np.argmax(np.where(m, rng.random(m.shape) + 1.0, 0.0), -1))
        return np.stack(cols, -1)

    for _ in range(int(rng.integers(60, 120))):
        acts = [rand(g.observe(p)[1]) for p in (0, 1)]
        g.step(*acts)
    obs, mask = g.observe(0)
    del jax, jnp
    return {"obs": obs, "mask": mask.astype(bool), "act": rand(mask).astype(np.int32),
            "logp": rng.normal(size=M).astype(np.float32) - 5, "adv": rng.normal(size=M).astype(np.float32),
            "ret": rng.normal(size=M).astype(np.float32)}


def torch_update(data, minibatches, repeats, model_path, device, amp):
    """One PPO epoch (forward, backward, Adam) over ``data`` in ``minibatches``; returns samples/s."""
    import torch

    sys.path.insert(0, os.path.dirname(model_path))
    from model import GridnetPolicy, masked_policy

    dev = torch.device(device)
    t = {k: torch.as_tensor(v).to(dev) for k, v in data.items()}
    t["act"] = t["act"].long()
    net = GridnetPolicy(in_channels=t["obs"].shape[-1]).to(dev).train()
    opt = torch.optim.Adam(net.parameters(), lr=2.5e-4, eps=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    N = t["obs"].shape[0]
    mb = N // minibatches

    def run():
        perm = torch.randperm(N, device=dev)
        for s0 in range(0, N, mb):
            i = perm[s0 : s0 + mb]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits, value = net(t["obs"][i])
            _, newlogp, ent, _ = masked_policy(logits.float(), t["mask"][i], t["act"][i])
            ratio = (newlogp - t["logp"][i]).exp()
            ad = t["adv"][i]
            ad = (ad - ad.mean()) / (ad.std() + 1e-8)
            pg = torch.max(-ad * ratio, -ad * ratio.clamp(0.9, 1.1)).mean()
            loss = pg - 0.01 * ent.mean() + 0.5 * 0.5 * ((value.float() - t["ret"][i]) ** 2).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
        if dev.type == "cuda":
            torch.cuda.synchronize()

    return N / timed(run, repeats)


def jax_update(data, minibatches, repeats, dtype):
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, HERE)
    import jax_policy as P

    dims = (6, 4, 4, 4, 4, 7, 49)
    dt = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}[dtype]
    d = {k: jnp.asarray(v) for k, v in data.items()}
    N = d["obs"].shape[0]
    B = d["obs"].shape
    d["mask"] = d["mask"].reshape(N, -1, 78)
    params = P.init_params(jax.random.PRNGKey(0), B[-1], dims)
    opt = P.adam_init(params)
    mb = N // minibatches

    @jax.jit
    def minibatch(params, opt, d, idx):
        batch = jax.tree.map(lambda x: x[idx], d)
        grads = jax.grad(P.ppo_loss)(params, batch, dims, dt)
        return P.adam_step(params, grads, opt)

    st = {"c": (params, opt), "k": jax.random.PRNGKey(0)}

    def run():
        st["k"], k = jax.random.split(st["k"])
        perm = jax.random.permutation(k, N)[: mb * minibatches].reshape(minibatches, mb)
        for i in range(minibatches):  # a Python loop: XLA:CPU runs convolutions inside lax.scan very slowly
            st["c"] = minibatch(*st["c"], d, perm[i])
        jax.block_until_ready(st["c"])

    return N / timed(run, repeats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="maps/16x16/basesWorkers16x16.xml")
    ap.add_argument("--games", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--ticks", type=int, default=64, help="ticks per timed run (the rollout length)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workloads", nargs="+", default=["engine", "rollout"])
    ap.add_argument("--engines", nargs="+", default=["numba", "jax"])
    ap.add_argument("--model", default=os.path.join(HERE, "..", "..", "training", "model.py"),
                    help="path of training/model.py (the PyTorch policy)")
    ap.add_argument("--device", default=None, help="PyTorch device (default: cuda when available)")
    ap.add_argument("--jax-dtype", default=None, help="float32, bfloat16 or float16 (default: bfloat16 on GPU)")
    ap.add_argument("--json", help="append results to this JSON-lines file")
    ap.add_argument("--update-batch", type=int, default=4096, help="samples per PPO epoch in the update workload")
    ap.add_argument("--minibatches", type=int, default=4)
    args = ap.parse_args(argv)

    import jax

    on_gpu = jax.default_backend() != "cpu"
    jax_dtype = args.jax_dtype or ("bfloat16" if on_gpu else "float32")
    device = args.device
    if device is None and importlib.util.find_spec("torch"):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"jax backend {jax.default_backend()} ({jax.devices()[0].device_kind}), torch device {device}, "
          f"{os.cpu_count()} CPU cores, map {args.map}, {args.ticks} ticks per run", flush=True)
    if "update" in args.workloads:
        data = sample_batch(args.map, args.update_batch)
        for eng in args.engines:
            try:
                sps = (torch_update(data, args.minibatches, args.repeats, os.path.abspath(args.model), device, device == "cuda")
                       if eng == "numba" else jax_update(data, args.minibatches, args.repeats, jax_dtype))
            except Exception as e:
                print(f"update   {'torch' if eng == 'numba' else 'jax':6s} failed: {type(e).__name__}: {e}", flush=True)
                continue
            name = "torch" if eng == "numba" else "jax"
            print(f"update   {name:6s} N={args.update_batch:6d}  {sps:12,.0f} samples/s per PPO epoch", flush=True)
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps({"workload": "update", "engine": name, "samples": args.update_batch,
                                        "samples_per_s": sps, "backend": jax.default_backend()}) + "\n")
    for B in args.games:
        for work in [w for w in args.workloads if w != "update"]:
            for eng in args.engines:
                try:
                    if eng == "numba" and work == "engine":
                        sps = numba_engine(args.map, B, args.ticks, args.repeats)
                    elif eng == "numba":
                        sps = numba_rollout(args.map, B, args.ticks, args.repeats, os.path.abspath(args.model),
                                            device, amp=device == "cuda")
                    elif work == "engine":
                        sps = jax_engine(args.map, B, args.ticks, args.repeats)
                    else:
                        sps = jax_rollout(args.map, B, args.ticks, args.repeats, jax_dtype)
                except Exception as e:  # keep going so one missing dependency does not lose the other numbers
                    print(f"{work:8s} {eng:6s} B={B:6d}  failed: {type(e).__name__}: {e}", flush=True)
                    continue
                print(f"{work:8s} {eng:6s} B={B:6d}  {sps:12,.0f} steps/s", flush=True)
                if args.json:
                    with open(args.json, "a") as f:
                        f.write(json.dumps({"workload": work, "engine": eng, "games": B, "steps_per_s": sps,
                                            "map": args.map, "backend": jax.default_backend()}) + "\n")


if __name__ == "__main__":
    main()
