# SPDX-License-Identifier: GPL-3.0-only
"""The gridnet actor-critic of ``training/model.py`` in plain JAX, for jitted rollouts.

Same layers and shapes as the PyTorch ``GridnetPolicy`` (stem conv, residual
blocks with dilations, pooled global features added back to every cell, 1x1
actor head with 78 logits per cell, MLP critic), so the two can be timed like
for like. Sampling is dense over all cells with a per-component mask.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

NEG = -1e8


def init_params(key, in_channels: int, action_dims, width: int = 64, dilations=(1, 2, 4, 1)):
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


def apply(params, obs, dilations=(1, 2, 4, 1), dtype=jnp.float32):
    """``obs`` ``(B, H, W, C)`` -> logits ``(B, H*W, A)`` float32, values ``(B,)``."""
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


def sample(key, logits, mask, action_dims):
    """Masked per-component sampling; returns ``(actions (B, cells, 7) int32, logprob (B,))``."""
    offs = np.concatenate([[0], np.cumsum(action_dims)])
    keys = jax.random.split(key, len(action_dims))
    acts, logps = [], []
    for k in range(len(action_dims)):
        lg = jnp.where(mask[..., offs[k] : offs[k + 1]], logits[..., offs[k] : offs[k + 1]], NEG)
        a = jax.random.categorical(keys[k], lg)
        lp = jax.nn.log_softmax(lg)
        acts.append(a)
        logps.append(jnp.take_along_axis(lp, a[..., None], -1)[..., 0])
    act = jnp.stack(acts, -1).astype(jnp.int32)
    comp_type = jnp.array([0, 1, 2, 3, 4, 4, 5])
    relevant = (jnp.arange(7) == 0) | (act[..., :1] == comp_type)
    source = mask[..., : action_dims[0]].any(-1)
    logp = (jnp.stack(logps, -1) * relevant * source[..., None]).sum((1, 2))
    return act, logp


def evaluate(logits, mask, act, action_dims):
    """Log-probability and entropy ``(B,)`` of given actions, counting only the components each action type uses."""
    offs = np.concatenate([[0], np.cumsum(action_dims)])
    logps, ents = [], []
    for k in range(len(action_dims)):
        m = mask[..., offs[k] : offs[k + 1]]
        lp = jax.nn.log_softmax(jnp.where(m, logits[..., offs[k] : offs[k + 1]], NEG))
        logps.append(jnp.take_along_axis(lp, act[..., k : k + 1], -1)[..., 0])
        ents.append(-jnp.where(m, jnp.exp(lp) * lp, 0.0).sum(-1))
    comp_type = jnp.array([0, 1, 2, 3, 4, 4, 5])
    relevant = ((jnp.arange(7) == 0) | (act[..., :1] == comp_type)) & mask[..., : action_dims[0]].any(-1)[..., None]
    return (jnp.stack(logps, -1) * relevant).sum((1, 2)), (jnp.stack(ents, -1) * relevant).sum((1, 2))


def ppo_loss(params, batch, action_dims, dtype=jnp.float32, clip=0.1, ent_coef=0.01, vf_coef=0.5):
    logits, value = apply(params, batch["obs"], dtype=dtype)
    logp, ent = evaluate(logits, batch["mask"], batch["act"], action_dims)
    ratio = jnp.exp(logp - batch["logp"])
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - clip, 1 + clip)).mean()
    v = 0.5 * ((value - batch["ret"]) ** 2).mean()
    return pg - ent_coef * ent.mean() + vf_coef * v


def adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.zeros((), jnp.int32)}


def adam_step(params, grads, opt, lr=2.5e-4, b1=0.9, b2=0.999, eps=1e-5, max_norm=0.5):
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, max_norm / (norm + 1e-6)), grads)
    t = opt["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
    c1, c2 = 1 - b1**t, 1 - b2**t
    params = jax.tree.map(lambda p, m, v: p - lr * (m / c1) / (jnp.sqrt(v / c2) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}
