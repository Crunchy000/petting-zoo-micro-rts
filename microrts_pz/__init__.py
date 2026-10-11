# SPDX-License-Identifier: GPL-3.0-only
"""MicroRTS as a PettingZoo environment.

A pure-Python port of the MicroRTS engine (Santiago Ontañón, GPL-3.0) with the
Gym-MicroRTS gridnet interface (MIT), exposed through PettingZoo's Parallel
and AEC APIs.

    from microrts_pz import microrts_v0
    env = microrts_v0.parallel_env(map_path="maps/16x16/basesWorkers16x16.xml")
"""

from . import microrts_v0
from .engine import list_maps
from .env import MicroRTSParallelEnv
from .game import REWARD_NAMES, MicroRTSGame
from .unit_types import UnitTypeTable

__version__ = "0.1.0"

__all__ = ["MicroRTSGame", "MicroRTSParallelEnv", "REWARD_NAMES", "UnitTypeTable", "list_maps", "microrts_v0", "__version__"]
