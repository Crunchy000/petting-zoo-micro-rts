# SPDX-License-Identifier: GPL-3.0-only
"""Train a MicroRTS gridnet agent with PPO and invalid-action masking.

The learner plays a random side in each episode against an opponent drawn
from a weighted pool: the scripted bots in ``microrts_pz.bots`` and frozen
snapshots of the learner itself (self-play). An optional behaviour-cloning
warm start imitates the scripted bots first, which gets a competent economy
and army going in minutes instead of hours.

Environments run in worker processes (one per CPU core by default); the
network runs in the main process, on the GPU when there is one.

    pip install -e ".[train]"
    python training/train_selfplay.py --map maps/8x8/basesWorkers8x8A.xml --total-steps 2000000 --out runs/8x8
    python training/train_selfplay.py --map maps/16x16/basesWorkers16x16.xml --init runs/8x8/latest.pt --out runs/16x16

Checkpoints (``latest.pt``, ``snapshot_<n>.pt``) and a ``metrics.csv`` are
written to ``--out``. Use ``training/play.py`` to evaluate a checkpoint and
record HTML replays.
"""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ACTION_DIMS, GridnetPolicy, load_policy, masked_policy, save_checkpoint  # noqa: E402

from microrts_pz import microrts_v0  # noqa: E402
from microrts_pz.bots import RandomBot, WorkerRushBot  # noqa: E402

BOTS = {
    "random": lambda seed: RandomBot(seed),
    "rush": lambda seed: WorkerRushBot(),
    "rush2": lambda seed: WorkerRushBot(harvesters=2),
    "light": lambda seed: WorkerRushBot(harvesters=2, army="Light"),
    "heavy": lambda seed: WorkerRushBot(harvesters=2, army="Heavy"),
    "ranged": lambda seed: WorkerRushBot(harvesters=2, army="Ranged"),
    "mixed": lambda seed: WorkerRushBot(harvesters=2, army=("Light", "Ranged", "Heavy")),
}
#: Bots the behaviour-cloning warm start imitates.
TEACHERS = ("rush", "rush2", "light", "heavy", "ranged", "mixed")
AGENTS = ("player_0", "player_1")


def parse_weights(spec: str) -> dict[str, float]:
    out = {}
    for part in spec.split(","):
        name, _, w = part.partition(":")
        name = name.strip()
        if name != "self" and name not in BOTS:
            raise ValueError(f"unknown opponent {name!r}; choose from {sorted(BOTS)} or 'self'")
        out[name] = float(w) if w else 1.0
    return out


# ------------------------------------------------------------------ workers
class Slot:
    """One environment plus the bookkeeping for whoever is playing in it."""

    def __init__(self, maps, env_kwargs, seed, opp_weights, teach):
        self.rng = np.random.default_rng(seed)
        self.envs = {m: microrts_v0.parallel_env(map_path=m, **env_kwargs) for m in maps}
        self.opp_weights = opp_weights
        self.teach = teach
        self.seed = seed

    def new_episode(self, n_snapshots):
        self.env = self.envs[list(self.envs)[self.rng.integers(len(self.envs))]]
        names = [n for n in self.opp_weights if n != "self" or n_snapshots > 0]
        w = np.array([self.opp_weights[n] for n in names])
        self.opp = names[self.rng.choice(len(names), p=w / w.sum())] if names else "mixed"
        self.snapshot = int(self.rng.integers(n_snapshots)) if self.opp == "self" else -1
        self.bot = BOTS[self.opp](int(self.rng.integers(1 << 31))) if self.opp != "self" else None
        self.teacher = BOTS[TEACHERS[self.rng.integers(len(TEACHERS))]](0) if self.teach else None
        self.side = int(self.rng.integers(2))
        self.ret = 0.0
        self.length = 0
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 31)))
        return obs

    def pack(self, obs):
        me, opp = AGENTS[self.side], AGENTS[1 - self.side]
        return (
            obs[me]["observation"],
            obs[me]["action_mask"],
            obs[opp]["observation"] if self.opp == "self" else None,
            obs[opp]["action_mask"] if self.opp == "self" else None,
            self.snapshot,
        )


def worker_main(remote, maps, env_kwargs, seeds, opp_weights, teach):
    slots = [Slot(maps, env_kwargs, s, opp_weights, teach) for s in seeds]
    n_snapshots = 0
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                n_snapshots = data
                remote.send([s.pack(s.new_episode(n_snapshots)) for s in slots])
            elif cmd == "snapshots":
                n_snapshots = data
            elif cmd == "teach":
                for s in slots:
                    s.teach = bool(data)
                    s.teacher = BOTS[TEACHERS[s.rng.integers(len(TEACHERS))]](0) if s.teach else None
            elif cmd == "step":
                learner_actions, opp_actions = data
                out = []
                for i, s in enumerate(slots):
                    game = s.env.unwrapped.game
                    me, opp = AGENTS[s.side], AGENTS[1 - s.side]
                    if s.teacher is not None:
                        a = s.teacher.act(game, s.side).reshape(-1, len(ACTION_DIMS))
                    else:
                        a = learner_actions[i]
                    b = opp_actions[i] if s.opp == "self" else s.bot.act(game, 1 - s.side)
                    obs, rew, term, trunc, info = s.env.step({me: a.reshape(-1), opp: np.asarray(b).reshape(-1)})
                    r = rew[me]
                    s.ret += r
                    s.length += 1
                    done = term[me] or trunc[me]
                    episode = None
                    if done:
                        w = info[me].get("winner")
                        result = 0 if w is None else (1 if w == me else -1)
                        episode = (s.opp, result, s.ret, s.length)
                        obs = s.new_episode(n_snapshots)
                    out.append((*s.pack(obs), r, done, episode, a if s.teacher is not None else None))
                remote.send(out)
            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        remote.close()


class VecGames:
    def __init__(self, num_envs, num_workers, maps, env_kwargs, seed, opp_weights, teach):
        ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
        num_workers = max(1, min(num_workers, num_envs))
        split = np.array_split(np.arange(num_envs), num_workers)
        self.split = [list(s) for s in split]
        self.remotes, self.procs = [], []
        for idx in self.split:
            parent, child = ctx.Pipe()
            p = ctx.Process(
                target=worker_main,
                args=(child, maps, env_kwargs, [seed * 100003 + int(i) for i in idx], opp_weights, teach),
                daemon=True,
            )
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)

    def broadcast(self, cmd, data=None):
        for r in self.remotes:
            r.send((cmd, data))

    def reset(self, n_snapshots):
        self.broadcast("reset", n_snapshots)
        return [x for r in self.remotes for x in r.recv()]

    def step(self, learner_actions, opp_actions):
        for r, idx in zip(self.remotes, self.split):
            r.send(("step", ([learner_actions[i] for i in idx], [opp_actions[i] for i in idx])))
        return [x for r in self.remotes for x in r.recv()]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)


# ------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", action="append", help="map to train on (repeat for several maps of the same size)")
    ap.add_argument("--out", default="runs/microrts")
    ap.add_argument("--total-steps", type=int, default=2_000_000, help="environment steps (summed over envs)")
    ap.add_argument("--time-limit", type=float, default=0, help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--num-envs", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--num-steps", type=int, default=256, help="rollout length per env per update")
    ap.add_argument("--max-steps", type=int, default=2000, help="episode length cap in ticks")
    ap.add_argument("--opponents", default="mixed:1,light:1,heavy:1,ranged:1,rush:1,random:0.5,self:3",
                    help="weighted opponent pool, e.g. 'mixed:2,random:1,self:3' (self = past snapshots)")
    ap.add_argument("--bc-steps", type=int, default=0, help="behaviour-cloning warm start: env steps imitating the bots")
    ap.add_argument("--snapshot-every", type=int, default=20, help="updates between self-play snapshots")
    ap.add_argument("--keep-snapshots", type=int, default=10, help="most recent snapshots used as opponents")
    ap.add_argument("--init", help="start from this checkpoint (e.g. one trained on a smaller map)")
    ap.add_argument("--resume", action="store_true", help="continue from --out/latest.pt")
    ap.add_argument("--width", type=int, default=64, help="network channels")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--async-update", action=argparse.BooleanOptionalAction, default=None,
                    help="learn from one rollout while playing the next (default: on with a GPU)")
    ap.add_argument("--no-amp", action="store_true", help="train in float32 on the GPU (mixed precision is ~10x faster on a T4)")
    args = ap.parse_args(argv)

    maps = args.map or ["maps/16x16/basesWorkers16x16.xml"]
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    opp_weights = parse_weights(args.opponents)
    env_kwargs = {"max_steps": args.max_steps}

    probe = [microrts_v0.parallel_env(map_path=m) for m in maps]
    shapes = {(e.height, e.width) for e in probe}
    if len(shapes) != 1:
        raise SystemExit(f"all --map entries must have the same size, got {sorted(shapes)}")
    (H, W), C = shapes.pop(), probe[0].game.num_channels
    cells, mask_w = H * W, int(sum(ACTION_DIMS))

    # Workers fork before CUDA is initialised.
    vec = VecGames(args.num_envs, args.num_workers, maps, env_kwargs, args.seed, opp_weights, teach=args.bc_steps > 0)
    device = torch.device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def forward(net, x):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits, value = net(x)
        return logits.float(), value.float()

    latest = os.path.join(args.out, "latest.pt")
    start_step, snapshots = 0, []
    if args.resume and os.path.exists(latest):
        model = load_policy(latest, device)
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        start_step = ckpt.get("global_step", 0)
        snapshots = [p for p in ckpt.get("snapshots", []) if os.path.exists(p)]
        print(f"resumed from {latest} at step {start_step}")
    elif args.init:
        model = load_policy(args.init, device)
        print(f"initialised from {args.init}")
    else:
        model = GridnetPolicy(in_channels=C, width=args.width).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.resume and os.path.exists(latest) and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])
    print(f"map {H}x{W}, {sum(p.numel() for p in model.parameters()):,} parameters, device {device}, "
          f"{args.num_envs} envs on {len(vec.procs)} workers")

    snapshot_models = [load_policy(p, device) for p in snapshots[-args.keep_snapshots:]]

    def take_snapshot(n_update):
        path = os.path.join(args.out, f"snapshot_{n_update:05d}.pt")
        save_checkpoint(path, model)
        snapshots.append(path)
        snapshot_models.append(load_policy(path, device))
        del snapshot_models[: -args.keep_snapshots]
        vec.broadcast("snapshots", len(snapshot_models))

    if not snapshot_models:
        take_snapshot(0)

    N, T = args.num_envs, args.num_steps
    batch = N * T

    def new_buffers():
        # Rollout buffers live on the training device, so minibatches never cross the PCIe bus.
        return {
            "obs": torch.zeros((T, N, H, W, C), dtype=torch.int8, device=device),
            "mask": torch.zeros((T, N, cells, mask_w), dtype=torch.bool, device=device),
            "act": torch.zeros((T, N, cells, len(ACTION_DIMS)), dtype=torch.long, device=device),
            "logp": torch.zeros((T, N), device=device),
            "rew": torch.zeros((T, N), device=device),
            "done": torch.zeros((T, N), device=device),
            "val": torch.zeros((T, N), device=device),
        }

    # With --async-update the GPU learns from one rollout while the workers play the next
    # one with the previous weights (a one-update policy lag that PPO's clipping absorbs).
    async_update = device.type == "cuda" if args.async_update is None else args.async_update
    buffers = [new_buffers() for _ in range(2 if async_update else 1)]
    actor = copy.deepcopy(model).eval() if async_update else model

    state = vec.reset(len(snapshot_models))
    next_done = torch.zeros(N, device=device)
    recent = defaultdict(lambda: deque(maxlen=100))
    returns = deque(maxlen=100)
    metrics_path = os.path.join(args.out, "metrics.csv")
    new_file = not os.path.exists(metrics_path) or not args.resume
    metrics = open(metrics_path, "w" if new_file else "a", newline="")
    writer = csv.writer(metrics)
    if new_file:
        writer.writerow(["step", "phase", "sps", "episodes", "mean_return", "win_rate", "per_opponent", "pg_loss", "v_loss", "entropy", "bc_loss"])

    global_step = start_step
    num_updates = max(1, (args.total_steps - start_step) // batch)
    t0 = time.time()
    teaching = args.bc_steps > 0 and start_step < args.bc_steps
    if not teaching:
        vec.broadcast("teach", False)

    def collect(buf):
        """Play T steps in every env with ``actor``; fill ``buf``."""
        nonlocal state, next_done, global_step
        actor.eval()
        for t in range(T):
            obs = torch.as_tensor(np.stack([s[0] for s in state])).to(device, non_blocking=True)
            mask = torch.as_tensor(np.stack([s[1] for s in state]).reshape(N, cells, mask_w)).to(device, non_blocking=True).bool()
            buf["obs"][t], buf["mask"][t], buf["done"][t] = obs, mask, next_done
            with torch.no_grad():
                if teaching:  # the bots act; only the value head is trained
                    act, logp, value = None, None, forward(actor, obs)[1]
                else:
                    logits, value = forward(actor, obs)
                    act, logp, _, _ = masked_policy(logits, mask)
            buf["val"][t] = value
            learner = act.cpu().numpy() if act is not None else [None] * N

            opp_actions = [None] * N
            by_snap = defaultdict(list)
            for i, s in enumerate(state):
                if s[2] is not None:
                    by_snap[s[4]].append(i)
            for k, idx in by_snap.items():
                net = snapshot_models[min(k, len(snapshot_models) - 1)]
                o = torch.as_tensor(np.stack([state[i][2] for i in idx])).to(device)
                m = torch.as_tensor(np.stack([state[i][3] for i in idx]).reshape(len(idx), cells, mask_w)).to(device).bool()
                with torch.no_grad():
                    a, *_ = masked_policy(forward(net, o)[0], m)
                a = a.cpu().numpy()
                for j, i in enumerate(idx):
                    opp_actions[i] = a[j]

            results = vec.step(list(learner), opp_actions)
            global_step += N
            if teaching:
                buf["act"][t] = torch.as_tensor(np.stack([r[8] for r in results])).to(device)
            else:
                buf["act"][t] = act
                buf["logp"][t] = logp
            buf["rew"][t] = torch.tensor([r[5] for r in results], dtype=torch.float32).to(device)
            next_done = torch.tensor([float(r[6]) for r in results]).to(device)
            for r in results:
                if r[7] is not None:
                    opp, result, ret, _ = r[7]
                    recent[opp].append(result)
                    recent["all"].append(result)
                    returns.append(ret)
            state = [r[:5] for r in results]
        buf["next_obs"] = torch.as_tensor(np.stack([s[0] for s in state])).to(device)
        buf["next_done"] = next_done
        buf["teaching"] = teaching

    def learn(buf, lr, out):
        """PPO (or behaviour-cloning) epochs on ``buf``; losses go to ``out``."""
        t_upd = time.time()
        opt.param_groups[0]["lr"] = lr
        val_buf, done_buf, rew_buf = buf["val"], buf["done"], buf["rew"]
        with torch.no_grad():
            next_value = forward(model, buf["next_obs"])[1]
        adv = torch.zeros((T, N), device=device)
        last = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nonterminal = 1.0 - (buf["next_done"] if t == T - 1 else done_buf[t + 1])
            nv = next_value if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * nv * nonterminal - val_buf[t]
            last = delta + args.gamma * args.gae_lambda * nonterminal * last
            adv[t] = last
        ret_buf = adv + val_buf

        b_obs, b_mask, b_act = buf["obs"].reshape(batch, H, W, C), buf["mask"].reshape(batch, cells, mask_w), buf["act"].reshape(batch, cells, -1)
        b_logp, b_adv, b_ret = buf["logp"].reshape(-1), adv.reshape(-1), ret_buf.reshape(-1)
        mb = batch // args.minibatches
        model.train()
        stats = defaultdict(list)
        for _ in range(args.epochs):
            perm = torch.randperm(batch, device=device)
            for s in range(0, batch, mb):
                i = perm[s : s + mb]
                o, m, a = b_obs[i], b_mask[i], b_act[i]
                logits, value = forward(model, o)
                _, newlogp, ent, _ = masked_policy(logits, m, a)
                v_loss = 0.5 * ((value - b_ret[i]) ** 2).mean()
                if buf["teaching"]:
                    bc_loss = -newlogp.mean()
                    loss = bc_loss + args.vf_coef * v_loss
                    stats["bc_loss"].append(bc_loss.detach())
                else:
                    ratio = (newlogp - b_logp[i]).exp()
                    ad = b_adv[i]
                    ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                    pg_loss = torch.max(-ad * ratio, -ad * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    loss = pg_loss - args.ent_coef * ent.mean() + args.vf_coef * v_loss
                    stats["pg_loss"].append(pg_loss.detach())
                    stats["entropy"].append(ent.mean().detach())
                stats["v_loss"].append(v_loss.detach())
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(opt)
                scaler.update()
        out["stats"] = {k: float(torch.stack(v).mean()) for k, v in stats.items()}
        out["time"] = time.time() - t_upd

    def finish(update, phase, t_roll, step, out):
        """Log one update, take snapshots and save the checkpoint."""
        stats = out["stats"]
        sps = int((step - start_step) / (time.time() - t0))
        win = np.mean(recent["all"]) if recent["all"] else float("nan")
        per_opp = " ".join(
            f"{k}={np.mean([x == 1 for x in v]):.0%}" for k, v in sorted(recent.items()) if k != "all" and v
        )
        print(f"[{phase}] step {step:>9,}  sps {sps:>5}  return {np.mean(returns) if returns else float('nan'):7.2f}  "
              f"score {win:+.2f}  rollout {t_roll:.1f}s update {out['time']:.1f}s  wins {per_opp}", flush=True)
        writer.writerow([step, phase, sps, len(recent["all"]), np.mean(returns) if returns else "", win, per_opp,
                         *(stats.get(k, float("nan")) for k in ("pg_loss", "v_loss", "entropy", "bc_loss"))])
        metrics.flush()
        if phase == "ppo" and update % args.snapshot_every == 0:
            take_snapshot(update)
        save_checkpoint(latest, model, {"global_step": step, "optimizer": opt.state_dict(),
                                        "snapshots": snapshots, "maps": maps, "args": vars(args)})
        if actor is not model:
            actor.load_state_dict(model.state_dict())

    pending = None  # (thread, update, phase, t_roll, step, out) of an update still training
    for update in range(1, num_updates + 1):
        if teaching and global_step >= args.bc_steps:
            teaching = False
            vec.broadcast("teach", False)
            print("behaviour cloning done, switching to PPO")
        buf = buffers[update % len(buffers)]
        t_roll = time.time()
        collect(buf)
        t_roll = time.time() - t_roll
        if pending is not None:
            pending[0].join()
            finish(*pending[1:])
        out = {}
        lr = (1.0 - (update - 1.0) / num_updates) * args.lr
        phase = "bc" if buf["teaching"] else "ppo"
        if async_update:
            th = threading.Thread(target=learn, args=(buf, lr, out))
            th.start()
            pending = (th, update, phase, t_roll, global_step, out)
        else:
            learn(buf, lr, out)
            finish(update, phase, t_roll, global_step, out)
        if args.time_limit and time.time() - t0 > args.time_limit * 60:
            print(f"time limit reached at step {global_step:,}")
            break
    if pending is not None:
        pending[0].join()
        finish(*pending[1:])

    metrics.close()
    vec.close()
    print(f"done; latest checkpoint at {latest}")


if __name__ == "__main__":
    main()
