# Training MicroRTS agents

This folder trains gridnet agents for MicroRTS with PPO and invalid-action
masking, evaluates them against the scripted bots and records HTML replays.

| file | what it is |
|---|---|
| [`train_jax.py`](train_jax.py) | **The trainer.** Games, bots, self-play opponents and the network all run in JAX on the GPU. |
| [`jax_model.py`](jax_model.py) | The network, action sampling, PPO/BC losses, Adam, checkpoints and PyTorch conversion, in JAX. |
| [`play.py`](play.py) | Plays a checkpoint against a bot or another checkpoint and saves HTML replays. |
| [`opponents.py`](opponents.py) | The named scripted opponents (`rush`, `light`, `mixed`, ...) shared by all scripts. |
| [`train_colab.ipynb`](train_colab.ipynb) | Runs the two-stage recipe below on a Colab GPU and packs up the results. |
| [`train_selfplay.py`](train_selfplay.py), [`model.py`](model.py) | The earlier PyTorch trainer and network, on the Numba or Python engine. Still works; slower. |
| [`agents/agent_16x16.pt`](agents/) | A trained 16×16 agent (the one on the website). |

The agents only ever see what a PettingZoo agent sees: the one-hot
observation and the action mask from `microrts_pz.microrts_v0`. A trained
checkpoint therefore plays in the PettingZoo environment, and the batch
engines are just faster ways to produce the same games.

## Quick start

```bash
pip install -e ".[jax]"                  # CPU
pip install -e ".[jax]" "jax[cuda12]"    # NVIDIA GPU (Colab already has this)

# stage 1: 8x8, starting by imitating the bots
python training/train_jax.py --map maps/8x8/basesWorkers8x8A.xml --out runs/8x8 \
    --total-steps 20000000 --bc-steps 1000000 --max-steps 1000
# stage 2: 16x16, starting from the 8x8 agent (the network is fully convolutional)
python training/train_jax.py --map maps/16x16/basesWorkers16x16.xml --out runs/16x16 \
    --init runs/8x8/latest.pkl --total-steps 100000000

# how good is it?
python training/play.py runs/16x16/latest.pkl --opponent mixed --games 20
python training/play.py runs/16x16/latest.pkl --opponent light --games 4 --record replays/
```

Training prints one line per update (this one is from a small CPU run on 8×8):

```
[ppo] step     176,128  sps    653  return   34.03  score +0.18  rollout 2.0s update 4.1s  wins self=90% mixed=55% light=59% heavy=32% ranged=54% rush=12% random=100%
```

`score` is wins minus losses over finished episodes (+1 means it wins every
game), `wins` is the win rate per opponent, both smoothed over recent updates.
During behaviour cloning (`[bc]` lines) the bots play the learner's side, so
those numbers describe the teachers, not the network.
The same numbers, plus the training losses, go to `metrics.csv`.

`--time-limit MINUTES` stops cleanly, and `--resume` carries on from
`--out/latest.pkl` (weights, optimiser, self-play snapshots and step count),
so a disconnected Colab session loses at most one update.

## How it trains

1. **Opponents.** The learner plays a random side in every episode against
   an opponent from a weighted pool, `--opponents` (default
   `mixed:1,light:1,heavy:1,ranged:1,rush:1,random:0.5,self:3`). The names
   are the scripted bots in [`opponents.py`](opponents.py); `self` means
   earlier copies of the learner. The `--num-envs` games are divided between
   the opponents in proportion to those weights (the start-up line prints
   the split), and each game keeps its opponent for the whole run.
2. **Behaviour cloning** (`--bc-steps`). For the first steps the scripted
   bots (all six rush variants, spread over the games) play the learner's
   side, and the network is trained to imitate their actions while its
   value head learns from the rewards. This gets an economy and an army going
   long before PPO alone would find them.
3. **PPO** with GAE (`--gamma`, `--gae-lambda`), the clipped objective
   (`--clip`), an entropy bonus (`--ent-coef`), gradient-norm clipping and a
   learning rate that decays linearly to zero at `--total-steps`. Each update
   plays `--num-steps` steps in every game and then makes `--epochs` passes
   over the `num-envs × num-steps` samples in minibatches of
   `--minibatch-size`.
4. **Self-play snapshots.** Every `--snapshot-every` updates the current
   weights replace the oldest of `--keep-snapshots` frozen copies. The
   self-play games are split evenly between those copies.
5. **Rewards** are the environment's default shaped reward
   (`microrts_pz.env.DEFAULT_REWARD_WEIGHT`: win/loss 10, harvesting 1,
   training a worker 1, building 0.2, attacking 1, training a combat unit 4).

The network ([`jax_model.py`](jax_model.py)) is a 3×3 stem convolution, four
residual blocks with dilations 1, 2, 4, 1, a pooled summary of the board
added back to every cell, a 1×1 actor head with 78 logits per cell (one
distribution per action component) and an MLP critic on the pooled summary,
about 346k parameters at the default `--width 64`. Only cells holding one of
the player's idle units are sampled and scored, and an action's
log-probability only counts the components its type uses (the direction of a
move, the direction and unit type of a produce, and so on).

### What makes it fast

Everything an update needs happens in two compiled programs on the GPU.
Nothing is copied to the host, and Python does nothing per step, per game or
per unit.

* **The rollout** is one `jax.lax.scan` over `--num-steps` steps. Each step
  observes both players, runs the learner network, computes the opponents'
  actions, steps every game, and stores the sample. Each opponent type only
  computes its own block of games: the scripted bots run on their slots, and
  all self-play snapshots run in one batched call (`jax.vmap` over the stacked
  snapshot weights).
* **The update** reads minibatches straight from the rollout buffer in
  device memory. Action masks are stored bit-packed, 10 bytes instead of 78
  per cell, so a rollout of 1,024 games × 64 steps on 16×16 takes under 1 GB.
* **bfloat16.** On a GPU or TPU the network runs in bfloat16 with float32
  weights by default (`--dtype`).
* **Two epochs** per rollout by default instead of the PyTorch trainer's four.
  The batches are much larger, so each epoch already makes many gradient
  steps.

### Choosing sizes

`--num-envs` is the main lever: more games make larger batches and use the GPU
better. 1,024 (the default) suits a 16 GB or 24 GB GPU on 16×16 maps; on 8×8
maps you can go higher. Keep `num-envs × num-steps` a multiple of
`--minibatch-size`. If you run out of GPU memory, lower `--minibatch-size`
first (it sets the activation memory of the update), then `--num-envs`.

## Checkpoints

`train_jax.py` writes to `--out`:

* `latest.pkl`: a pickle of NumPy arrays with the weights (`params`),
  network shape (`config`), optimiser state, self-play snapshots, step count
  and command-line arguments. You don't need JAX to read it.
* `latest.pt`: the same weights as a PyTorch `GridnetPolicy` checkpoint, when
  PyTorch is installed. This is what the website's exporter and older scripts
  read.
* `metrics.csv`: one row per update.

`--init` and `play.py` accept either format, and `jax_model` converts between
them exactly (the two networks produce the same logits):

```python
import jax_model as P
params, config = P.load_params("training/agents/agent_16x16.pt")   # PyTorch -> JAX
P.export_torch("agent.pt", params)                                  # JAX -> PyTorch
bot = P.JaxPolicyBot("runs/16x16/latest.pkl")                       # act(game, player) like the scripted bots
```

## Evaluating with `play.py`

```bash
python training/play.py CHECKPOINT --opponent mixed --games 20             # win/draw/loss summary
python training/play.py CHECKPOINT --opponent light --record replays/       # also save HTML replays
python training/play.py CHECKPOINT --opponent other_agent.pkl              # agent against agent
python training/play.py CHECKPOINT --map maps/letMeOut.xml --deterministic # another map, greedy actions
```

`play.py` plays through the PettingZoo environment with the reference Python
engine, alternating sides. It keeps the most interesting games as replays:
wins first, then the games where the agent built the most unit types.

## Using the pieces in your own training code

The trainer is built from public pieces that you can use on their own.

**The batch engine.** `microrts_pz.jax_engine.JaxBatchGame` steps any number
of games of one map size as a JAX program. It plays exactly the rules of
`microrts_pz.engine`, checked tick for tick by `tests/test_jax_engine.py`.
It covers full observability and the deterministic unit tables (versions
1 and 2).

```python
from microrts_pz.jax_engine import JaxBatchGame
games = JaxBatchGame(["maps/16x16/basesWorkers16x16.xml"], num_games=1024, max_steps=2000)
games.reset()
obs, mask = games.observe(player=0)        # (B, 16, 16, 29) int8, (B, 256, 78) int8
a1 = games.bot_actions(player=1, kind="rush", harvesters=2, army="Light")
raw, over, trunc, winner = games.step(a0, a1)   # raw rewards (B, 2, 6); finished games restart
```

These methods take and return NumPy arrays, which is convenient but copies
data on every call. For speed, trace the pure functions into your own jitted
loop. They work on `games.state`, a pytree of per-game arrays:

```python
import jax
from microrts_pz.jax_engine.bots import random_actions

@jax.jit
def play(state, key):
    def step(state, key):
        _, m0 = games.observe_fn(state, 0)
        _, m1 = games.observe_fn(state, 1)
        k0, k1 = jax.random.split(key)
        state, raw, over, trunc, winner, _ = games.step_fn(
            state, random_actions(games.rules, k0, m0), random_actions(games.rules, k1, m1))
        return state, (over, winner)
    return jax.lax.scan(step, state, jax.random.split(key, 1000))

state, (over, winner) = play(games.state, jax.random.PRNGKey(0))
```

* `games.step_fn(state, a0, a1) -> (state, raw (B,2,6), gameover, truncated, winner, ticks)`:
  one tick, with automatic restart of finished games.
* `games.observe_fn(state, player) -> (obs, mask)`: `player` is a scalar or
  one per game.
* `games.rush_fn(state, player, harvesters, army, n_army, counter) -> (actions, counter)`:
  `WorkerRushBot` for every game. `army` is `(B, 3)` unit type ids, of which
  the first `n_army` are used. `counter` is the bot's army counter; reset it
  to 0 when a game restarts. For a block of games, call
  `microrts_pz.jax_engine.bots.rush_actions` under `jax.vmap`, as
  `train_jax.py` does.
* `random_actions(rules, key, mask)`: `RandomBot`.
* `jax_model.apply / sample / evaluate / ppo_loss_and_stats / adam_step`:
  the network and the PPO pieces.

The scripted bots in `microrts_pz.jax_engine.bots` choose exactly the same
actions as the Python `WorkerRushBot`. The tests check this every tick;
the `bots.py` docstring explains how the per-unit search becomes a few
whole-board operations. `RandomBot` uses JAX's random generator, so its
draws differ from the Python bot's.

## The PyTorch trainer

`train_selfplay.py` is the original trainer: PyTorch for the network, with
games on the Numba engine (`microrts_pz.fast`, all CPU cores) or the Python
environment in worker processes. It trains the same network with the same
recipe and writes `.pt` checkpoints. Use it if you can't install JAX. On a
GPU it is several times slower, because the games run on the CPU and every
step crosses to the GPU and back.

```bash
pip install -e ".[train]"        # torch + numba
python training/train_selfplay.py --map maps/16x16/basesWorkers16x16.xml --bc-steps 2000000 --out runs/16x16
```

## Speed

`tools/bench/engines.py` times the engines on their own, the self-play
rollout with the network, and the PPO update. On an NVIDIA L4 (16×16 map, 1,024
games):

| | Numba engine + PyTorch | JAX |
|---|---|---|
| engine only, ticks/s | 52,626 | 408,783 |
| rollout with the policy, steps/s | 13,389 | 55,403 |
| PPO update, samples/s per epoch | 28,331 | 36,843 |

```bash
python tools/bench/engines.py --games 1024 --workloads engine,rollout,update --engines numba,jax
```
