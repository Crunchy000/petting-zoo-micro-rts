#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Step-by-step parity check of microrts_pz against the reference Java MicroRTS engine.

Usage::

    git clone https://github.com/Farama-Foundation/MicroRTS /path/to/MicroRTS
    python tools/parity/check_parity.py --microrts /path/to/MicroRTS --episodes 5

The script compiles MicroRTS and ``ParityHarness.java`` (needs a JDK), then
plays random games where both engines receive identical actions, comparing
observations, action masks, raw rewards, player resources and termination
after every tick. Actions are sampled from the action mask most of the time
and uniformly at random otherwise, so illegal-action handling is exercised too.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from microrts_pz.engine import resolve_map_path  # noqa: E402
from microrts_pz.game import MicroRTSGame  # noqa: E402
from microrts_pz.unit_types import UnitTypeTable  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def build(microrts: str, out: str) -> str:
    classes = os.path.join(out, "classes")
    os.makedirs(classes, exist_ok=True)
    srcs = glob.glob(os.path.join(microrts, "src", "**", "*.java"), recursive=True)
    cp = os.pathsep.join([os.path.join(microrts, "lib", "*"), os.path.join(microrts, "lib", "bots", "*")])
    listfile = os.path.join(out, "srcs.txt")
    with open(listfile, "w") as f:
        f.write("\n".join(srcs))
    subprocess.run(["javac", "-nowarn", "-d", classes, "-cp", cp, "@" + listfile], check=True, stderr=subprocess.DEVNULL)
    subprocess.run(["javac", "-d", classes, "-cp", classes + os.pathsep + cp, os.path.join(HERE, "ParityHarness.java")], check=True)
    return classes + os.pathsep + cp


def sample(mask: np.ndarray, dims: np.ndarray, rng: np.random.Generator, p_random: float) -> np.ndarray:
    offsets = np.concatenate([[0], np.cumsum(dims)])
    out = np.zeros((mask.shape[0], len(dims)), dtype=np.int64)
    for c in range(mask.shape[0]):
        for k, n in enumerate(dims):
            valid = np.flatnonzero(mask[c, offsets[k] : offsets[k + 1]])
            if len(valid) == 0 or rng.random() < p_random:
                out[c, k] = rng.integers(n)
            else:
                out[c, k] = rng.choice(valid)
    return out


class Java:
    def __init__(self, cp: str, map_file: str, version: int, partial: bool):
        self.p = subprocess.Popen(
            ["java", "-cp", cp, "ParityHarness", map_file, str(version), "1" if partial else "0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def read(self):
        players = []
        for _ in range(2):
            line = self.p.stdout.readline().split("|")
            head = line[0].split()
            players.append(
                (
                    np.array(head[2:], dtype=np.int64),
                    np.array(line[1].split(), dtype=np.int64),
                    np.array(line[2].split(), dtype=np.int64),
                )
            )
        done = self.p.stdout.readline().split()[1] == "1"
        res = [int(v) for v in self.p.stdout.readline().split()[1:]]
        return players, done, res

    def step(self, acts):
        msg = ["STEP"]
        for a in acts:
            msg.append(str(len(a)))
            msg.extend(str(int(v)) for v in a.ravel())
        self.p.stdin.write(" ".join(msg) + "\n")
        self.p.stdin.flush()
        return self.read()

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
        except BrokenPipeError:
            pass
        self.p.wait()


def compare(game: MicroRTSGame, java_players, rewards, step: int, label: str) -> None:
    for p in range(2):
        jr, jobs, jmask = java_players[p]
        pobs = game.raw_observation(p).ravel()
        if not np.array_equal(pobs, jobs):
            diff = np.argwhere(pobs.reshape(-1, game.height, game.width) != jobs.reshape(-1, game.height, game.width))
            raise AssertionError(f"{label} step {step}: observation mismatch for player {p} at [plane,y,x] {diff[:5].tolist()}")
        jmask = jmask.reshape(game.num_cells, -1)
        if not np.array_equal(game.source_unit_mask(p), jmask[:, 0]):
            raise AssertionError(f"{label} step {step}: source unit mask mismatch for player {p}")
        if not np.array_equal(game.action_mask(p), jmask[:, 1:]):
            cells = np.flatnonzero((game.action_mask(p) != jmask[:, 1:]).any(1))
            raise AssertionError(f"{label} step {step}: action mask mismatch for player {p} at cells {cells.tolist()}")
        if rewards is not None and not np.array_equal(rewards[p].astype(np.int64), jr):
            raise AssertionError(f"{label} step {step}: reward mismatch for player {p}: {rewards[p]} vs {jr}")


def run(cp: str, map_path: str, version: int, partial: bool, steps: int, seed: int, p_random: float) -> int:
    rng = np.random.default_rng(seed)
    game = MicroRTSGame(map_path, UnitTypeTable(version), partial_obs=partial)
    game.reset()
    java = Java(cp, resolve_map_path(map_path), version, partial)
    label = f"{map_path} v{version}{' partial' if partial else ''} seed {seed}"
    try:
        players, done, res = java.read()
        compare(game, players, None, 0, label)
        for t in range(1, steps + 1):
            acts, rows = [], []
            for p in range(2):
                a = sample(game.action_mask(p), game.action_dims, rng, p_random)
                acts.append(a)
                src = np.flatnonzero(game.source_unit_mask(p))
                rows.append(np.concatenate([src[:, None], a[src]], axis=1))
            rewards, gameover = game.step(acts[0], acts[1])
            players, done, res = java.step(rows)
            compare(game, players, rewards, t, label)
            if res != game.gs.player_resources:
                raise AssertionError(f"{label} step {t}: resources {game.gs.player_resources} vs {res}")
            if gameover != done:
                raise AssertionError(f"{label} step {t}: gameover {gameover} vs {done}")
            if done:
                return t
        return steps
    finally:
        java.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--microrts", required=True, help="path to a MicroRTS checkout")
    ap.add_argument("--episodes", type=int, default=3, help="seeds per map")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--p-random", type=float, default=0.1)
    ap.add_argument(
        "--maps",
        nargs="*",
        default=[
            "maps/4x4/base4x4.xml",
            "maps/8x8/basesWorkers8x8A.xml",
            "maps/8x8/melee8x8Mixed6.xml",
            "maps/10x10/basesWorkers10x10.xml",
            "maps/16x16/basesWorkers16x16.xml",
            "maps/16x16/melee16x16Mixed12.xml",
            "maps/16x16/TwoBasesBarracks16x16.xml",
        ],
    )
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        cp = build(args.microrts, tmp)
        for map_path in args.maps:
            for version in (1, 2):
                for partial in (False, True):
                    for seed in range(args.episodes):
                        n = run(cp, map_path, version, partial, args.steps, seed, args.p_random)
                        print(f"ok  {map_path} v{version} partial={partial} seed={seed}: {n} ticks identical")


if __name__ == "__main__":
    main()
