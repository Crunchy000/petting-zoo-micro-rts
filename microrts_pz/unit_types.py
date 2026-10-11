# SPDX-License-Identifier: GPL-3.0-only
"""Unit type tables.

A Python port of ``rts.units.UnitTypeTable`` / ``rts.units.UnitType`` from
MicroRTS (https://github.com/Farama-Foundation/MicroRTS, GPL-3.0,
(c) Santiago Ontañón). Stats and version switches are copied verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VERSION_ORIGINAL = 1
VERSION_ORIGINAL_FINETUNED = 2
VERSION_NON_DETERMINISTIC = 3

MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH = 1
MOVE_CONFLICT_RESOLUTION_CANCEL_RANDOM = 2
MOVE_CONFLICT_RESOLUTION_CANCEL_ALTERNATING = 3


@dataclass(eq=False)
class UnitType:
    """Static properties of a kind of unit (defaults match the Java class)."""

    id: int = 0
    name: str = ""
    cost: int = 1
    hp: int = 1
    min_damage: int = 1
    max_damage: int = 1
    attack_range: int = 1
    produce_time: int = 10
    move_time: int = 10
    attack_time: int = 10
    harvest_time: int = 10
    return_time: int = 10
    harvest_amount: int = 1
    sight_radius: int = 4
    is_resource: bool = False
    is_stockpile: bool = False
    can_harvest: bool = False
    can_move: bool = True
    can_attack: bool = True
    produces: list["UnitType"] = field(default_factory=list)
    produced_by: list["UnitType"] = field(default_factory=list)

    def add_produces(self, other: "UnitType") -> None:
        self.produces.append(other)
        other.produced_by.append(self)

    def __repr__(self) -> str:  # dataclass repr would recurse through produces
        return f"UnitType({self.name!r})"

    def to_dict(self) -> dict:
        return {
            "ID": self.id,
            "name": self.name,
            "cost": self.cost,
            "hp": self.hp,
            "minDamage": self.min_damage,
            "maxDamage": self.max_damage,
            "attackRange": self.attack_range,
            "produceTime": self.produce_time,
            "moveTime": self.move_time,
            "attackTime": self.attack_time,
            "harvestTime": self.harvest_time,
            "returnTime": self.return_time,
            "harvestAmount": self.harvest_amount,
            "sightRadius": self.sight_radius,
            "isResource": self.is_resource,
            "isStockpile": self.is_stockpile,
            "canHarvest": self.can_harvest,
            "canMove": self.can_move,
            "canAttack": self.can_attack,
            "produces": [u.name for u in self.produces],
            "producedBy": [u.name for u in self.produced_by],
        }


class UnitTypeTable:
    """The set of unit types used by a game, plus the move-conflict strategy."""

    def __init__(
        self,
        version: int = VERSION_ORIGINAL,
        move_conflict_resolution: int = MOVE_CONFLICT_RESOLUTION_CANCEL_BOTH,
    ):
        if version not in (VERSION_ORIGINAL, VERSION_ORIGINAL_FINETUNED, VERSION_NON_DETERMINISTIC):
            raise ValueError(f"unknown unit type table version {version}")
        if move_conflict_resolution not in (1, 2, 3):
            raise ValueError(f"unknown move conflict resolution strategy {move_conflict_resolution}")
        self.version = version
        self.move_conflict_resolution = move_conflict_resolution
        self.unit_types: list[UnitType] = []
        self._build(version)
        self._by_name = {ut.name: ut for ut in self.unit_types}

    def _add(self, ut: UnitType) -> UnitType:
        ut.id = len(self.unit_types)
        self.unit_types.append(ut)
        return ut

    def _build(self, version: int) -> None:
        original = version == VERSION_ORIGINAL
        finetuned = version == VERSION_ORIGINAL_FINETUNED
        nondet = version == VERSION_NON_DETERMINISTIC

        self._add(
            UnitType(
                name="Resource",
                is_resource=True,
                can_harvest=False,
                can_move=False,
                can_attack=False,
                sight_radius=0,
            )
        )

        base = UnitType(name="Base", cost=10, hp=10, is_stockpile=True, can_move=False, can_attack=False, sight_radius=5)
        if original:
            base.produce_time = 250
        elif finetuned:
            base.produce_time = 200
        self._add(base)

        barracks = UnitType(name="Barracks", cost=5, hp=4, can_move=False, can_attack=False, sight_radius=3)
        barracks.produce_time = 200 if original else 100
        self._add(barracks)

        worker = UnitType(
            name="Worker",
            cost=1,
            hp=1,
            attack_range=1,
            produce_time=50,
            move_time=10,
            attack_time=5,
            harvest_time=20,
            return_time=10,
            can_harvest=True,
            sight_radius=3,
        )
        worker.min_damage, worker.max_damage = (0, 2) if nondet else (1, 1)
        self._add(worker)

        light = UnitType(name="Light", cost=2, hp=4, attack_range=1, produce_time=80, move_time=8, attack_time=5, sight_radius=2)
        light.min_damage, light.max_damage = (1, 3) if nondet else (2, 2)
        self._add(light)

        heavy = UnitType(name="Heavy", attack_range=1, produce_time=120, attack_time=5, sight_radius=2)
        heavy.min_damage, heavy.max_damage = (0, 6) if nondet else (4, 4)
        if original:
            heavy.move_time, heavy.hp, heavy.cost = 12, 4, 2
        else:
            heavy.move_time, heavy.hp, heavy.cost = 10, 8, 3
        self._add(heavy)

        ranged = UnitType(name="Ranged", cost=2, hp=1, attack_range=3, produce_time=100, move_time=10, attack_time=5, sight_radius=3)
        ranged.min_damage, ranged.max_damage = (1, 2) if nondet else (1, 1)
        self._add(ranged)

        base.add_produces(worker)
        barracks.add_produces(light)
        barracks.add_produces(heavy)
        barracks.add_produces(ranged)
        worker.add_produces(base)
        worker.add_produces(barracks)

    def __getitem__(self, key: int | str) -> UnitType:
        if isinstance(key, str):
            return self._by_name[key]
        return self.unit_types[key]

    def __len__(self) -> int:
        return len(self.unit_types)

    @property
    def max_attack_range(self) -> int:
        return max(ut.attack_range for ut in self.unit_types)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "moveConflictResolutionStrategy": self.move_conflict_resolution,
            "unitTypes": [ut.to_dict() for ut in self.unit_types],
        }
