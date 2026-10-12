# SPDX-License-Identifier: GPL-3.0-only
"""Throughput of the trainers, one option at a time.

Runs ``training/train_jax.py`` for a few updates per configuration, each in
a fresh process, and reports environment steps per second measured between
log lines after the first (compiling) update. That is real training speed:
rollout, update and checkpointing together. Each configuration changes one
option from the baseline (16×16 map, 1,024 games, 64 steps, minibatches of
4,096, two epochs, bfloat16, width 64, the default opponent pool), so the
table shows what each option costs or buys.

    python tools/bench/trainer.py                       # the whole sweep
    python tools/bench/trainer.py --only base,envs=4096 # some of it
    python tools/bench/trainer.py --pytorch             # also time train_selfplay.py (Numba + PyTorch)

Results go to stdout as a Markdown table and to ``--json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = {"--map": "maps/16x16/basesWorkers16x16.xml", "--num-envs": "1024", "--num-steps": "64", "--minibatch-size": "4096",
        "--epochs": "2", "--width": "64", "--keep-snapshots": "4"}
BOTS_ONLY = "mixed:1,light:1,heavy:1,ranged:1,rush:1,random:0.5"

#: name -> (option changes from BASE, what it shows)
CONFIGS = {
    "base": ({}, "baseline"),
    "envs=256": ({"--num-envs": "256"}, "--num-envs 256"),
    "envs=512": ({"--num-envs": "512"}, "--num-envs 512"),
    "envs=2048": ({"--num-envs": "2048"}, "--num-envs 2048"),
    "envs=4096": ({"--num-envs": "4096"}, "--num-envs 4096"),
    "steps=32": ({"--num-steps": "32"}, "--num-steps 32"),
    "steps=128": ({"--num-steps": "128"}, "--num-steps 128"),
    "mb=2048": ({"--minibatch-size": "2048"}, "--minibatch-size 2048"),
    "mb=8192": ({"--minibatch-size": "8192"}, "--minibatch-size 8192"),
    "mb=16384": ({"--minibatch-size": "16384"}, "--minibatch-size 16384"),
    "epochs=1": ({"--epochs": "1"}, "--epochs 1"),
    "epochs=4": ({"--epochs": "4"}, "--epochs 4"),
    "float32": ({"--dtype": "float32"}, "--dtype float32"),
    "width=32": ({"--width": "32"}, "--width 32"),
    "width=128": ({"--width": "128"}, "--width 128"),
    "self-only": ({"--opponents": "self:1"}, "--opponents self:1 (all self-play)"),
    "bots-only": ({"--opponents": BOTS_ONLY}, "--opponents without self"),
    "snapshots=8": ({"--keep-snapshots": "8"}, "--keep-snapshots 8"),
    "bc": ({"--bc-steps": "1000000000"}, "behaviour cloning phase (--bc-steps)"),
    "map=8x8": ({"--map": "maps/8x8/basesWorkers8x8A.xml"}, "8×8 map"),
    "map=24x24": ({"--map": "maps/24x24/basesWorkers24x24.xml"}, "24×24 map"),
}


def run_logged(cmd, out_lines: list, timeout: float):
    """Run ``cmd``; record ``(arrival time, line)`` for every stdout line."""
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t_end = time.time() + timeout
    for line in p.stdout:
        out_lines.append((time.time(), line.rstrip()))
        if time.time() > t_end:
            p.kill()
            break
    return p.wait()


def steady_sps(lines, batch, skip=1):
    """Steps per second from the gaps between update lines, after ``skip`` compiling updates."""
    t = [ts for ts, line in lines if line.startswith(("[ppo]", "[bc]"))]
    gaps = [b - a for a, b in zip(t[skip:], t[skip + 1 :])]
    return (batch / statistics.median(gaps), len(gaps)) if gaps else (float("nan"), 0)


def bench_jax(name, changes, updates, timeout):
    opts = {**BASE, **changes}
    batch = int(opts["--num-envs"]) * int(opts["--num-steps"])
    with tempfile.TemporaryDirectory() as out:
        cmd = [sys.executable, "training/train_jax.py", *sum(([k, v] for k, v in opts.items()), []),
               "--total-steps", str(batch * (updates + 2)), "--out", out]
        lines: list = []
        t0 = time.time()
        code = run_logged(cmd, lines, timeout)
        rows = list(csv.DictReader(open(os.path.join(out, "metrics.csv")))) if os.path.exists(os.path.join(out, "metrics.csv")) else []
    sps, n = steady_sps(lines, batch, skip=2)
    steady = rows[2:] or rows[1:]
    med = lambda k: statistics.median(float(r[k]) for r in steady) if steady else float("nan")  # noqa: E731
    result = {"name": name, "ok": code == 0 and n > 0, "sps": sps, "updates_timed": n, "batch": batch,
              "rollout_s": med("rollout_s"), "update_s": med("update_s"),
              "peak_mem_gb": max((float(r["peak_mem_gb"]) for r in rows), default=float("nan")),
              "first_update_s": float(rows[0]["rollout_s"]) + float(rows[0]["update_s"]) if rows else float("nan"),
              "wall_s": time.time() - t0}
    if not result["ok"]:
        result["tail"] = [line for _, line in lines[-15:]]
    return result


def bench_pytorch(updates, timeout, num_envs=512, num_steps=128):
    """``train_selfplay.py`` on the Numba engine, with the trainer's own defaults otherwise."""
    with tempfile.TemporaryDirectory() as out:
        batch = num_envs * num_steps
        cmd = [sys.executable, "training/train_selfplay.py", "--map", BASE["--map"], "--engine", "fast",
               "--num-envs", str(num_envs), "--num-steps", str(num_steps), "--total-steps", str(batch * (updates + 2)), "--out", out]
        lines: list = []
        t0 = time.time()
        code = run_logged(cmd, lines, timeout)
    sps, n = steady_sps(lines, batch, skip=2)
    return {"name": f"pytorch (train_selfplay.py, {num_envs} games x {num_steps} steps)", "ok": code == 0 and n > 0,
            "sps": sps, "updates_timed": n, "batch": batch, "wall_s": time.time() - t0,
            **({} if code == 0 and n else {"tail": [line for _, line in lines[-15:]]})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated configuration names (default: all): " + ",".join(CONFIGS))
    ap.add_argument("--updates", type=int, default=5, help="updates per configuration after the compiling ones (the first gap is dropped too)")
    ap.add_argument("--base", default="", help="override baseline options, e.g. 'num-envs=64,width=16' for a quick CPU check")
    ap.add_argument("--timeout", type=float, default=900, help="seconds per configuration")
    ap.add_argument("--pytorch", action="store_true", help="also time the PyTorch trainer on the Numba engine")
    ap.add_argument("--json", default="trainer_bench.json")
    args = ap.parse_args(argv)

    import jax

    for kv in filter(None, args.base.split(",")):
        k, _, v = kv.partition("=")
        BASE["--" + k.lstrip("-")] = v

    dev = jax.devices()[0]
    print(f"device: {dev.platform} {dev.device_kind}, {os.cpu_count()} CPU cores", flush=True)
    names = args.only.split(",") if args.only else list(CONFIGS)
    results = []
    for name in names:
        changes, label = CONFIGS[name]
        r = bench_jax(name, changes, args.updates, args.timeout)
        r["label"] = label
        results.append(r)
        print(json.dumps(r), flush=True)
        json.dump({"device": dev.device_kind, "results": results}, open(args.json, "w"), indent=1)
    if args.pytorch:
        r = bench_pytorch(args.updates, args.timeout)
        r["label"] = r["name"]
        results.append(r)
        print(json.dumps(r), flush=True)
        json.dump({"device": dev.device_kind, "results": results}, open(args.json, "w"), indent=1)

    base = next((r["sps"] for r in results if r["name"] == "base" and r["ok"]), None)
    print(f"\n{dev.device_kind}, {BASE['--map']} unless noted\n")
    print("| option | steps/s | vs baseline | rollout s | update s | peak GPU memory GB |")
    print("|---|---|---|---|---|---|")
    for r in results:
        if not r["ok"]:
            print(f"| {r['label']} | failed | | | | |")
            continue
        rel = f"{r['sps'] / base:.2f}×" if base else ""
        print(f"| {r['label']} | {r['sps']:,.0f} | {rel} | {r.get('rollout_s', float('nan')):.2f} | "
              f"{r.get('update_s', float('nan')):.2f} | {r.get('peak_mem_gb', float('nan')):.1f} |")


if __name__ == "__main__":
    main()
