# SPDX-License-Identifier: GPL-3.0-only
"""The gridnet actor-critic of ``model.py`` in plain JAX, for the jitted trainer (``train_jax.py``).

Parameters are a nested dict of arrays with the same layers as the PyTorch
``GridnetPolicy``: a 3x3 stem convolution, residual blocks with dilations,
pooled global features added back to every cell, a 1x1 actor head with
``sum(action_dims)`` logits per cell and an MLP critic. ``to_torch`` and
``from_torch`` convert between the two exactly, so a checkpoint trained here
plays in ``play.py`` and on the website, and a PyTorch checkpoint can seed
JAX training.

Sampling and scoring follow ``model.masked_policy``: only cells holding an
idle unit of the player (source cells) count, other cells get action 0, and
an action's log-probability sums the components its type uses.

Checkpoints written by ``save_checkpoint`` are pickles of NumPy arrays
(``.pkl``, no JAX or PyTorch needed to read them); ``load_params`` reads those
and PyTorch ``.pt`` checkpoints alike.
"""

from __future__ import annotations

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from orient import mask_permutation, rotate_action, rotate_mask, rotate_obs

NEG = -1e8
#: For components 1..6, the action type they parameterise (move, harvest, return, produce, produce, attack).
COMPONENT_TYPE = (0, 1, 2, 3, 4, 4, 5)
DEFAULT_DILATIONS = (1, 2, 4, 1)


# ------------------------------------------------------------------ network
def init_params(key, in_channels: int, action_dims, width: int = 64, dilations=DEFAULT_DILATIONS):
    """Random parameters (He-normal convolutions, a near-zero actor output layer as in ``model.py``)."""
    keys = iter(jax.random.split(key, 64))

    def conv(cin, cout, k, std=None):
        std = np.sqrt(2.0 / (cin * k * k)) if std is None else std
        return {"w": jax.random.normal(next(keys), (k, k, cin, cout)) * std, "b": jnp.zeros(cout)}

    def dense(cin, cout):
        return {"w": jax.random.normal(next(keys), (cin, cout)) * np.sqrt(2.0 / cin), "b": jnp.zeros(cout)}

    a = int(sum(action_dims))
    return {
        "stem": conv(in_channels, width, 3),
        "blocks": [{"c1": conv(width, width, 3), "c2": conv(width, width, 3)} for _ in dilations],
        "glob": dense(2 * width, width),
        "actor1": conv(width, width, 1),
        "actor2": conv(width, a, 1, std=0.01),
        "critic1": dense(2 * width, 128),
        "critic2": dense(128, 1),
    }


def _conv(x, p, dilation=1, dtype=jnp.float32):
    k = p["w"].shape[0]
    pad = dilation * (k // 2)
    y = jax.lax.conv_general_dilated(
        x, p["w"].astype(dtype), (1, 1), [(pad, pad), (pad, pad)], rhs_dilation=(dilation, dilation),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return y + p["b"].astype(dtype)


def apply(params, obs, dilations=DEFAULT_DILATIONS, dtype=jnp.float32):
    """``obs`` ``(B, H, W, C)`` -> logits ``(B, H*W, A)`` float32, values ``(B,)``.

    ``dtype`` is the compute precision (``jnp.bfloat16`` roughly doubles
    throughput on recent GPUs); parameters stay float32.
    """
    x = obs.astype(dtype)
    h = jax.nn.relu(_conv(x, params["stem"], dtype=dtype))
    for blk, d in zip(params["blocks"], dilations):
        h = jax.nn.relu(h + _conv(jax.nn.relu(_conv(h, blk["c1"], d, dtype)), blk["c2"], 1, dtype))
    pooled = jnp.concatenate([h.mean((1, 2)), h.max((1, 2))], -1)
    g = jax.nn.relu(pooled @ params["glob"]["w"].astype(dtype) + params["glob"]["b"].astype(dtype))
    h = h + g[:, None, None, :]
    logits = _conv(jax.nn.relu(_conv(h, params["actor1"], dtype=dtype)), params["actor2"], dtype=dtype)
    v = jax.nn.relu(pooled @ params["critic1"]["w"].astype(dtype) + params["critic1"]["b"].astype(dtype))
    v = v @ params["critic2"]["w"].astype(dtype) + params["critic2"]["b"].astype(dtype)
    B = obs.shape[0]
    return logits.reshape(B, -1, logits.shape[-1]).astype(jnp.float32), v[:, 0].astype(jnp.float32)


# ------------------------------------------------------------- distribution
def _components(logits, mask, action_dims):
    offs = np.concatenate([[0], np.cumsum(action_dims)])
    for k in range(len(action_dims)):
        m = mask[..., offs[k] : offs[k + 1]]
        yield m, jax.nn.log_softmax(jnp.where(m, logits[..., offs[k] : offs[k + 1]], NEG))


def _relevant(act, mask, action_dims):
    """Which components count: the type everywhere, the others for their action type; source cells only."""
    source = mask[..., : action_dims[0]].any(-1)
    rel = (jnp.arange(len(action_dims)) == 0) | (act[..., :1] == jnp.asarray(COMPONENT_TYPE))
    return rel & source[..., None]


def sample(key, logits, mask, action_dims, deterministic: bool = False):
    """Masked per-component sampling; returns ``(actions (B, cells, 7) int32, logprob (B,))``.

    Non-source cells get action 0. With ``deterministic`` each component takes its most likely value.
    """
    keys = jax.random.split(key, len(action_dims))
    acts, logps = [], []
    for k, (m, lp) in enumerate(_components(logits, mask, action_dims)):
        a = jnp.argmax(lp, -1) if deterministic else jax.random.categorical(keys[k], lp)
        acts.append(a)
        logps.append(jnp.take_along_axis(lp, a[..., None], -1)[..., 0])
    act = jnp.stack(acts, -1).astype(jnp.int32)
    rel = _relevant(act, mask, action_dims)
    act = jnp.where(rel[..., :1], act, 0)
    return act, (jnp.stack(logps, -1) * rel).sum((1, 2))


def evaluate(logits, mask, act, action_dims):
    """Log-probability and entropy ``(B,)`` of given actions, counting only the components each action type uses."""
    logps, ents = [], []
    for k, (m, lp) in enumerate(_components(logits, mask, action_dims)):
        logps.append(jnp.take_along_axis(lp, act[..., k : k + 1].astype(jnp.int32), -1)[..., 0])
        ents.append(-jnp.where(m, jnp.exp(lp) * lp, 0.0).sum(-1))
    rel = _relevant(act, mask, action_dims)
    return (jnp.stack(logps, -1) * rel).sum((1, 2)), (jnp.stack(ents, -1) * rel).sum((1, 2))


def pack_mask(mask):
    """``(..., A)`` bool -> ``(..., ceil(A/8))`` uint8, an 8x smaller rollout buffer."""
    return jnp.packbits(mask, axis=-1)


def unpack_mask(packed, size: int):
    return jnp.unpackbits(packed, axis=-1, count=size).astype(bool)


# ------------------------------------------------------------------- losses
def ppo_loss(params, batch, action_dims, dtype=jnp.float32, clip=0.1, ent_coef=0.01, vf_coef=0.5, dilations=DEFAULT_DILATIONS):
    """Clipped PPO loss on a minibatch dict (``obs``, ``mask``, ``act``, ``logp``, ``adv``, ``ret``)."""
    loss, _ = ppo_loss_and_stats(params, batch, action_dims, dtype, clip, ent_coef, vf_coef, dilations)
    return loss


def ppo_loss_and_stats(params, batch, action_dims, dtype=jnp.float32, clip=0.1, ent_coef=0.01, vf_coef=0.5,
                       dilations=DEFAULT_DILATIONS):
    logits, value = apply(params, batch["obs"], dilations, dtype)
    logp, ent = evaluate(logits, batch["mask"], batch["act"], action_dims)
    ratio = jnp.exp(logp - batch["logp"])
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - clip, 1 + clip)).mean()
    v = 0.5 * ((value - batch["ret"]) ** 2).mean()
    stats = {"pg_loss": pg, "v_loss": v, "entropy": ent.mean(),
             "approx_kl": ((ratio - 1) - jnp.log(ratio)).mean(), "clipfrac": (jnp.abs(ratio - 1) > clip).mean()}
    return pg - ent_coef * ent.mean() + vf_coef * v, stats


def bc_loss_and_stats(params, batch, action_dims, dtype=jnp.float32, vf_coef=0.5, dilations=DEFAULT_DILATIONS):
    """Behaviour cloning: negative log-likelihood of the teacher's actions, plus the value loss."""
    logits, value = apply(params, batch["obs"], dilations, dtype)
    logp, _ = evaluate(logits, batch["mask"], batch["act"], action_dims)
    bc = -logp.mean()
    v = 0.5 * ((value - batch["ret"]) ** 2).mean()
    return bc + vf_coef * v, {"bc_loss": bc, "v_loss": v}


# ---------------------------------------------------------------- optimiser
def adam_init(params):
    return {"m": jax.tree.map(jnp.zeros_like, params), "v": jax.tree.map(jnp.zeros_like, params), "t": jnp.zeros((), jnp.int32)}


def adam_step(params, grads, opt, lr=2.5e-4, b1=0.9, b2=0.999, eps=1e-5, max_norm=0.5):
    """Adam (as ``torch.optim.Adam``) after clipping the global gradient norm to ``max_norm``."""
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, max_norm / (norm + 1e-6)), grads)
    t = opt["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
    c1, c2 = 1 - b1**t, 1 - b2**t
    params = jax.tree.map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}


# ---------------------------------------------------------- PyTorch interop
def to_torch(params) -> dict[str, np.ndarray]:
    """``GridnetPolicy.state_dict()`` contents (as NumPy arrays) for these parameters."""
    conv = lambda p: (np.asarray(p["w"], np.float32).transpose(3, 2, 0, 1), np.asarray(p["b"], np.float32))  # noqa: E731
    dense = lambda p: (np.asarray(p["w"], np.float32).T, np.asarray(p["b"], np.float32))  # noqa: E731
    out = {}

    def put(name, wb):
        out[f"{name}.weight"], out[f"{name}.bias"] = (np.ascontiguousarray(a) for a in wb)

    put("stem", conv(params["stem"]))
    for i, blk in enumerate(params["blocks"]):
        put(f"blocks.{i}.c1", conv(blk["c1"]))
        put(f"blocks.{i}.c2", conv(blk["c2"]))
    put("glob.0", dense(params["glob"]))
    put("actor.0", conv(params["actor1"]))
    put("actor.2", conv(params["actor2"]))
    put("critic.0", dense(params["critic1"]))
    put("critic.2", dense(params["critic2"]))
    return out


def from_torch(state_dict) -> dict:
    """Parameters from a ``GridnetPolicy.state_dict()`` (tensors or arrays)."""
    sd = {k: np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, np.float32) for k, v in state_dict.items()}
    conv = lambda n: {"w": jnp.asarray(sd[f"{n}.weight"].transpose(2, 3, 1, 0)), "b": jnp.asarray(sd[f"{n}.bias"])}  # noqa: E731
    dense = lambda n: {"w": jnp.asarray(sd[f"{n}.weight"].T), "b": jnp.asarray(sd[f"{n}.bias"])}  # noqa: E731
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    return {
        "stem": conv("stem"),
        "blocks": [{"c1": conv(f"blocks.{i}.c1"), "c2": conv(f"blocks.{i}.c2")} for i in range(n_blocks)],
        "glob": dense("glob.0"),
        "actor1": conv("actor.0"),
        "actor2": conv("actor.2"),
        "critic1": dense("critic.0"),
        "critic2": dense("critic.2"),
    }


def _default_dims(total: int) -> list[int]:
    if total != 78:
        raise ValueError("pass action_dims for a unit type table other than the bundled ones")
    return [6, 4, 4, 4, 4, 7, 49]


def config_of(params, dilations=DEFAULT_DILATIONS, action_dims=None) -> dict:
    """``GridnetPolicy`` constructor arguments for these parameters."""
    a = params["actor2"]["w"].shape[-1]
    return {
        "in_channels": int(params["stem"]["w"].shape[2]),
        "width": int(params["stem"]["w"].shape[3]),
        "dilations": list(dilations),
        "action_dims": list(action_dims) if action_dims is not None else _default_dims(a),
    }


# -------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, params, extra: dict | None = None, dilations=DEFAULT_DILATIONS, action_dims=None) -> None:
    """Write ``{"config", "params", **extra}`` as a pickle of NumPy arrays (atomically)."""
    blob = {"format": "microrts-jax-1", "config": config_of(params, dilations, action_dims),
            "params": jax.tree.map(np.asarray, params), **(extra or {})}
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def export_torch(path: str, params, extra: dict | None = None, dilations=DEFAULT_DILATIONS, action_dims=None) -> None:
    """Write a PyTorch ``.pt`` checkpoint that ``model.load_policy`` (``play.py``, the website exporter) reads."""
    import torch

    sd = {k: torch.from_numpy(v.copy()) for k, v in to_torch(params).items()}
    torch.save({"config": config_of(params, dilations, action_dims), "state_dict": sd, **(extra or {})}, path)


def load_checkpoint(path: str) -> dict:
    """The checkpoint dict, with ``params`` as JAX arrays, from a ``.pkl`` (this module) or ``.pt`` (PyTorch) file."""
    if path.endswith(".pt"):
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return {"config": ckpt["config"], "params": from_torch(ckpt["state_dict"]), "rotate_side": ckpt.get("rotate_side")}
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    ckpt["params"] = jax.tree.map(jnp.asarray, ckpt["params"])
    return ckpt


def load_params(path: str):
    """``(params, config)`` from a ``.pkl`` or ``.pt`` checkpoint."""
    ckpt = load_checkpoint(path)
    return ckpt["params"], ckpt["config"]


class JaxPolicyBot:
    """A trained policy with the ``act(game, player)`` interface of ``microrts_pz.bots`` (no PyTorch needed)."""

    def __init__(self, path_or_params, deterministic: bool = False, seed: int = 0, config: dict | None = None,
                 rotate_side: int | None | str = "auto"):
        #: the side whose board is turned 180 degrees before the policy sees it (``orient.py``);
        #: ``"auto"`` takes the checkpoint's ``rotate_side``
        self.rotate_side = None if rotate_side == "auto" else rotate_side
        if isinstance(path_or_params, str):
            ckpt = load_checkpoint(path_or_params)
            self.params, config = ckpt["params"], ckpt["config"]
            if rotate_side == "auto":
                self.rotate_side = ckpt.get("rotate_side")
        else:
            self.params = path_or_params
        config = config or config_of(self.params)
        dims = tuple(config["action_dims"])
        dil = tuple(config["dilations"])
        self.key = jax.random.PRNGKey(seed)

        @jax.jit
        def act(params, key, obs, mask):
            logits, _ = apply(params, obs, dil)
            return sample(key, logits, mask, dims, deterministic)[0]

        self._act = act

    def act(self, game, player: int) -> np.ndarray:
        self.key, k = jax.random.split(self.key)
        obs, mask = game.observation(player), game.action_mask(player)
        rot = player == self.rotate_side
        if rot:
            obs, mask = rotate_obs(obs), rotate_mask(mask, mask_permutation(game.action_dims))
        act = np.asarray(self._act(self.params, k, jnp.asarray(obs[None]), jnp.asarray(mask[None]).astype(bool)))[0]
        return rotate_action(act.reshape(game.num_cells, -1), game.action_dims) if rot else act
