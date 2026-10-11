# SPDX-License-Identifier: GPL-3.0-only
"""Versioned entry point, following PettingZoo's ``<name>_v<N>`` convention."""

from .env import MicroRTSParallelEnv, env, parallel_env, raw_env

__all__ = ["MicroRTSParallelEnv", "env", "parallel_env", "raw_env"]
