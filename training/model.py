# SPDX-License-Identifier: GPL-3.0-only
"""Gridnet policy for MicroRTS: a fully convolutional actor-critic with action masking.

The actor scores all 7 action components for every map cell (78 logits per
cell on the default unit table) and the critic reads a pooled summary of the
board. The network is fully convolutional, so a checkpoint plays on any map
size. The masked, per-cell factorised action distribution follows the
invalid-action-masking PPO of Huang & Ontañón (2020) and CleanRL's
``ppo_gridnet`` (MIT).

Only the action components that matter for the chosen action type count
towards the log-probability and entropy: the move direction for a move, the
produce direction and unit type for a produce, and so on.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Component dims ``[type, move_dir, harvest_dir, return_dir, produce_dir, produce_type, attack_target]``
#: for the bundled unit type tables (7 unit types, max attack range 3).
ACTION_DIMS = (6, 4, 4, 4, 4, 7, 49)
#: For components 1..6, the action type they parameterise (move, harvest, return, produce, produce, attack).
COMPONENT_TYPE = (None, 1, 2, 3, 4, 4, 5)
NEG = -1e8


class ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class GridnetPolicy(nn.Module):
    """Actor-critic over ``(B, H, W, C)`` one-hot observations.

    ``forward`` returns per-cell logits ``(B, H*W, sum(ACTION_DIMS))`` and
    values ``(B,)``. A pooled board summary is added back to every cell so
    that each unit's choice can depend on the whole map, not only on the
    local receptive field.
    """

    def __init__(self, in_channels: int = 29, width: int = 64, dilations=(1, 2, 4, 1), action_dims=ACTION_DIMS):
        super().__init__()
        self.config = {"in_channels": in_channels, "width": width, "dilations": list(dilations), "action_dims": list(action_dims)}
        self.action_dims = tuple(action_dims)
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(width, d) for d in dilations])
        self.glob = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())
        self.actor = nn.Sequential(nn.Conv2d(width, width, 1), nn.ReLU(), nn.Conv2d(width, sum(action_dims), 1))
        self.critic = nn.Sequential(nn.Linear(2 * width, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.normal_(self.actor[-1].weight, std=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, obs: torch.Tensor):
        x = obs.permute(0, 3, 1, 2).float()
        h = self.blocks(F.relu(self.stem(x)))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        h = h + self.glob(pooled)[:, :, None, None]
        logits = self.actor(h).permute(0, 2, 3, 1).reshape(obs.shape[0], -1, sum(self.action_dims))
        return logits, self.critic(pooled).squeeze(-1)


def masked_policy(
    logits: torch.Tensor,
    mask: torch.Tensor,
    action: torch.Tensor | None = None,
    deterministic: bool = False,
):
    """Sample (or score) gridnet actions under the action mask.

    ``logits``: ``(B, cells, 78)``; ``mask``: same shape, bool; ``action``:
    ``(B, cells, 7)`` long to score, or ``None`` to sample.

    Returns ``(action, logprob (B,), entropy (B,), relevant (B, cells, 7) bool)``.

    Only cells holding an idle unit of the player (a "source" cell) are
    evaluated; usually that is a few percent of the board, so this is much
    cheaper than scoring every cell. Other cells get action 0, which the
    environment ignores.
    """
    B, cells, _ = logits.shape
    source = mask[..., : ACTION_DIMS[0]].any(-1)
    b_idx, c_idx = source.nonzero(as_tuple=True)
    lg_src = logits[b_idx, c_idx]
    m_src = mask[b_idx, c_idx]
    given = action[b_idx, c_idx] if action is not None else None
    dims = list(F.pad(torch.tensor(ACTION_DIMS), (1, 0)).cumsum(0).tolist())
    picks, logps, ents = [], [], []
    for k in range(len(ACTION_DIMS)):
        lg = lg_src[:, dims[k] : dims[k + 1]]
        m = m_src[:, dims[k] : dims[k + 1]]
        lg = torch.where(m, lg, torch.full_like(lg, NEG))
        logp = torch.log_softmax(lg, dim=-1)
        if given is None:
            a = lg.argmax(-1) if deterministic else torch.multinomial(logp.exp(), 1).squeeze(-1) if lg.shape[0] else lg.argmax(-1)
        else:
            a = given[:, k]
        picks.append(a)
        logps.append(logp.gather(-1, a.unsqueeze(-1)).squeeze(-1))
        p = logp.exp()
        ents.append(-(torch.where(m, p * logp, torch.zeros_like(p))).sum(-1))
    act_src = torch.stack(picks, -1)
    rel_src = torch.ones_like(act_src, dtype=torch.bool)
    for k in range(1, len(ACTION_DIMS)):
        rel_src[:, k] = act_src[:, 0] == COMPONENT_TYPE[k]
    relf = rel_src.to(logits.dtype)
    logprob = logits.new_zeros(B).index_add(0, b_idx, (torch.stack(logps, -1) * relf).sum(-1))
    entropy = logits.new_zeros(B).index_add(0, b_idx, (torch.stack(ents, -1) * relf).sum(-1))
    if action is None:
        act = torch.zeros((B, cells, len(ACTION_DIMS)), dtype=torch.long, device=logits.device)
        act[b_idx, c_idx] = act_src
    else:
        act = action
    relevant = torch.zeros((B, cells, len(ACTION_DIMS)), dtype=torch.bool, device=logits.device)
    relevant[b_idx, c_idx] = rel_src
    return act, logprob, entropy, relevant


def save_checkpoint(path: str, model: GridnetPolicy, extra: dict | None = None) -> None:
    torch.save({"config": model.config, "state_dict": model.state_dict(), **(extra or {})}, path)


def load_policy(path: str, device: str | torch.device = "cpu") -> GridnetPolicy:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GridnetPolicy(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


class PolicyBot:
    """Wraps a trained policy with the same ``act(game, player)`` interface as ``microrts_pz.bots``."""

    def __init__(self, model: GridnetPolicy | str, device: str = "cpu", deterministic: bool = False, seed: int | None = None):
        self.model = load_policy(model, device) if isinstance(model, str) else model
        self.device = device
        self.deterministic = deterministic
        if seed is not None:
            torch.manual_seed(seed)

    @torch.no_grad()
    def act(self, game, player: int) -> np.ndarray:
        obs = torch.as_tensor(game.observation(player)[None], device=self.device)
        mask = torch.as_tensor(game.action_mask(player)[None], device=self.device).bool()
        logits, _ = self.model(obs)
        act, *_ = masked_policy(logits, mask, deterministic=self.deterministic)
        return act[0].cpu().numpy()
