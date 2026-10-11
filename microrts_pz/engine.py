# SPDX-License-Identifier: GPL-3.0-only
"""Pure-Python MicroRTS game engine.

This is a port of the core rules of MicroRTS
(https://github.com/Farama-Foundation/MicroRTS, GPL-3.0, (c) Santiago Ontañón):
``rts.GameState``, ``rts.PhysicalGameState``, ``rts.UnitAction``,
``rts.ResourceUsage``, ``rts.PlayerAction`` and ``rts.units.Unit``, plus the
gridnet action/observation conversion from ``ai.jni.JNIAI`` and
``tests.JNIGridnetClientSelfPlay``.

The port deliberately keeps the Java engine's quirks (action ordering,
resource reservation, cancelled actions turning into waits of the same
duration, units killed in a cycle still finishing that cycle's action) so
that trajectories match the reference implementation step for step.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .unit_types import (
    MOVE_CONFLICT_RESOLUTION_CANCEL_ALTERNATING,
    MOVE_CONFLICT_RESOLUTION_CANCEL_RANDOM,
    UnitType,
    UnitTypeTable,
)

TERRAIN_NONE = 0
TERRAIN_WALL = 1

TYPE_NONE = 0
TYPE_MOVE = 1
TYPE_HARVEST = 2
TYPE_RETURN = 3
TYPE_PRODUCE = 4
TYPE_ATTACK_LOCATION = 5
NUMBER_OF_ACTION_TYPES = 6
ACTION_NAMES = ("wait", "move", "harvest", "return", "produce", "attack_location")

DIRECTION_NONE = -1
DIRECTION_UP = 0
DIRECTION_RIGHT = 1
DIRECTION_DOWN = 2
DIRECTION_LEFT = 3
DIRECTION_OFFSET_X = (0, 1, 0, -1)
DIRECTION_OFFSET_Y = (-1, 0, 1, 0)

MAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")


class Unit:
    __slots__ = ("id", "player", "type", "x", "y", "resources", "hp", "alive")

    def __init__(self, uid: int, player: int, utype: UnitType, x: int, y: int, resources: int = 0, hp: int | None = None):
        self.id = uid
        self.player = player
        self.type = utype
        self.x = x
        self.y = y
        self.resources = resources
        self.hp = utype.hp if hp is None else hp
        self.alive = True

    def __repr__(self) -> str:
        return f"{self.type.name}(id={self.id}, player={self.player}, pos=({self.x},{self.y}), hp={self.hp}, res={self.resources})"


class ResourceUsage:
    __slots__ = ("positions", "resources")

    def __init__(self, positions: list[int] | None = None, resources: list[int] | None = None):
        self.positions = positions if positions is not None else []
        self.resources = resources if resources is not None else [0, 0]

    def merge(self, other: "ResourceUsage") -> None:
        self.positions.extend(other.positions)
        self.resources[0] += other.resources[0]
        self.resources[1] += other.resources[1]

    def clone(self) -> "ResourceUsage":
        return ResourceUsage(list(self.positions), list(self.resources))

    def consistent_with(self, other: "ResourceUsage", gs: "GameState") -> bool:
        for pos in other.positions:
            if pos in self.positions:
                return False
        for i in (0, 1):
            if other.resources[i] == 0:
                continue
            total = self.resources[i] + other.resources[i]
            if total > 0 and total > gs.player_resources[i]:
                return False
        return True


class UnitAction:
    __slots__ = ("type", "parameter", "x", "y", "unit_type")

    def __init__(self, atype: int, parameter: int = DIRECTION_NONE, x: int = 0, y: int = 0, unit_type: UnitType | None = None):
        self.type = atype
        self.parameter = parameter
        self.x = x
        self.y = y
        self.unit_type = unit_type

    @staticmethod
    def none(duration: int = DIRECTION_NONE) -> "UnitAction":
        return UnitAction(TYPE_NONE, duration)

    @staticmethod
    def attack(x: int, y: int) -> "UnitAction":
        return UnitAction(TYPE_ATTACK_LOCATION, DIRECTION_NONE, x, y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitAction) or other.type != self.type:
            return False
        if self.type in (TYPE_NONE, TYPE_MOVE, TYPE_HARVEST, TYPE_RETURN):
            return other.parameter == self.parameter
        if self.type == TYPE_ATTACK_LOCATION:
            return other.x == self.x and other.y == self.y
        return other.parameter == self.parameter and other.unit_type is self.unit_type

    __hash__ = None  # mutable, compared structurally

    def __repr__(self) -> str:
        name = ACTION_NAMES[self.type]
        if self.type == TYPE_ATTACK_LOCATION:
            return f"{name}({self.x},{self.y})"
        if self.type == TYPE_NONE:
            return f"{name}({self.parameter})"
        extra = f",{self.unit_type.name}" if self.unit_type is not None else ""
        return f"{name}({self.parameter}{extra})"

    def target_pos(self, u: Unit) -> tuple[int, int]:
        return u.x + DIRECTION_OFFSET_X[self.parameter], u.y + DIRECTION_OFFSET_Y[self.parameter]

    def resource_usage(self, u: Unit, width: int) -> ResourceUsage:
        ru = ResourceUsage()
        if self.type == TYPE_MOVE or self.type == TYPE_PRODUCE:
            pos = u.x + u.y * width
            p = self.parameter
            if p == DIRECTION_UP:
                pos -= width
            elif p == DIRECTION_RIGHT:
                pos += 1
            elif p == DIRECTION_DOWN:
                pos += width
            elif p == DIRECTION_LEFT:
                pos -= 1
            ru.positions.append(pos)
            if self.type == TYPE_PRODUCE:
                ru.resources[u.player] += self.unit_type.cost
        return ru

    def eta(self, u: Unit) -> int:
        t = self.type
        if t == TYPE_NONE:
            return self.parameter
        if t == TYPE_MOVE:
            return u.type.move_time
        if t == TYPE_ATTACK_LOCATION:
            return u.type.attack_time
        if t == TYPE_HARVEST:
            return u.type.harvest_time
        if t == TYPE_RETURN:
            return u.type.move_time  # sic: the Java engine uses moveTime for returns
        if t == TYPE_PRODUCE:
            return self.unit_type.produce_time
        return 0

    def copy(self) -> "UnitAction":
        return UnitAction(self.type, self.parameter, self.x, self.y, self.unit_type)


@dataclass
class Assignment:
    unit: Unit
    action: UnitAction
    time: int


class GameState:
    """Full MicroRTS game state for two players."""

    def __init__(
        self,
        width: int,
        height: int,
        terrain: Sequence[int],
        player_resources: Sequence[int],
        units: Iterable[Unit],
        utt: UnitTypeTable,
        rng: np.random.Generator | None = None,
    ):
        self.width = width
        self.height = height
        self.terrain = np.asarray(terrain, dtype=np.int8).reshape(height, width)
        self.player_resources = list(player_resources)
        self.utt = utt
        self.rng = rng if rng is not None else np.random.default_rng()
        self.time = 0
        self.unit_cancelation_counter = 0
        self.units: list[Unit] = []
        self.assignments: dict[int, Assignment] = {}
        self._occ: dict[int, Unit] = {}
        self.next_id = 0
        for u in units:
            self.add_unit(u)

    # ------------------------------------------------------------------ units
    def add_unit(self, u: Unit) -> None:
        pos = u.x + u.y * self.width
        if pos in self._occ:
            raise ValueError(f"two units at position ({u.x}, {u.y})")
        self.units.append(u)
        self._occ[pos] = u
        if u.id >= self.next_id:
            self.next_id = u.id + 1

    def remove_unit(self, u: Unit) -> None:
        if not u.alive:
            return
        u.alive = False
        self.units.remove(u)
        self.assignments.pop(u.id, None)
        pos = u.x + u.y * self.width
        if self._occ.get(pos) is u:
            del self._occ[pos]

    def unit_at(self, x: int, y: int) -> Unit | None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        return self._occ.get(x + y * self.width)

    def _move_unit(self, u: Unit, nx: int, ny: int) -> None:
        if u.alive:
            old = u.x + u.y * self.width
            if self._occ.get(old) is u:
                del self._occ[old]
        u.x, u.y = nx, ny
        if u.alive:
            new = nx + ny * self.width
            if new in self._occ:
                # Cannot happen when actions go through issue_safe; keep
                # first-in-list semantics of PhysicalGameState.getUnitAt.
                self._rebuild_occupancy()
            else:
                self._occ[new] = u

    def _rebuild_occupancy(self) -> None:
        self._occ = {}
        for u in self.units:
            self._occ.setdefault(u.x + u.y * self.width, u)

    def terrain_at(self, x: int, y: int) -> int:
        return int(self.terrain[y, x])

    # --------------------------------------------------------------- queries
    def gameover(self) -> bool:
        counts = [0, 0]
        for u in self.units:
            if u.player >= 0:
                counts[u.player] += 1
        if counts[0] + counts[1] == 0:
            return True
        return (counts[0] > 0) != (counts[1] > 0)

    def winner(self) -> int:
        counts = [0, 0]
        for u in self.units:
            if u.player >= 0:
                counts[u.player] += 1
        if counts[0] > 0 and counts[1] > 0:
            return -1
        if counts[0] > 0:
            return 0
        if counts[1] > 0:
            return 1
        return -1

    def is_idle(self, u: Unit) -> bool:
        return u.id not in self.assignments

    def unit_actions(self, u: Unit, none_duration: int = 10) -> list[UnitAction]:
        """All actions ``u`` may start now (``Unit.getUnitActions``)."""
        actions: list[UnitAction] = []
        x, y, t = u.x, u.y, u.type
        w, h = self.width, self.height
        uup = self.unit_at(x, y - 1)
        uright = self.unit_at(x + 1, y)
        udown = self.unit_at(x, y + 1)
        uleft = self.unit_at(x - 1, y)
        player = u.player

        if t.can_attack:
            if t.attack_range == 1:
                for other in (uup, uright, udown, uleft):
                    if other is not None and other.player != player and other.player >= 0:
                        actions.append(UnitAction.attack(other.x, other.y))
            else:
                sq = t.attack_range * t.attack_range
                for other in self.units:
                    if other.player < 0 or other.player == player:
                        continue
                    if (other.x - x) ** 2 + (other.y - y) ** 2 <= sq:
                        actions.append(UnitAction.attack(other.x, other.y))

        if t.can_harvest:
            neigh = (uup, uright, udown, uleft)
            if u.resources == 0:
                for d, other in enumerate(neigh):
                    if other is not None and other.type.is_resource:
                        actions.append(UnitAction(TYPE_HARVEST, d))
            if u.resources > 0:
                for d, other in enumerate(neigh):
                    if other is not None and other.type.is_stockpile and other.player == player:
                        actions.append(UnitAction(TYPE_RETURN, d))

        if t.produces or t.can_move:
            tup = self.terrain_at(x, y - 1) if y > 0 else TERRAIN_WALL
            tright = self.terrain_at(x + 1, y) if x < w - 1 else TERRAIN_WALL
            tdown = self.terrain_at(x, y + 1) if y < h - 1 else TERRAIN_WALL
            tleft = self.terrain_at(x - 1, y) if x > 0 else TERRAIN_WALL
            terr = (tup, tright, tdown, tleft)
            neigh = (uup, uright, udown, uleft)
            if player >= 0:
                for ut in t.produces:
                    if self.player_resources[player] >= ut.cost:
                        for d in range(4):
                            if terr[d] == TERRAIN_NONE and neigh[d] is None:
                                actions.append(UnitAction(TYPE_PRODUCE, d, unit_type=ut))
            if t.can_move:
                for d in range(4):
                    if terr[d] == TERRAIN_NONE and neigh[d] is None:
                        actions.append(UnitAction(TYPE_MOVE, d))

        actions.append(UnitAction.none(none_duration))
        return actions

    def can_execute(self, u: Unit, ua: UnitAction) -> bool:
        return ua in self.unit_actions(u, ua.eta(u))

    def reserved_resources(self, units: Iterable[Unit] | None = None) -> ResourceUsage:
        ru = ResourceUsage()
        for u in self.units if units is None else units:
            a = self.assignments.get(u.id)
            if a is not None:
                ru.merge(a.action.resource_usage(a.unit, self.width))
        return ru

    # -------------------------------------------------------------- dynamics
    def issue(self, pairs: list[list]) -> bool:
        """``GameState.issue``: register actions, resolving same-cycle conflicts."""
        issued_something = False
        strategy = self.utt.move_conflict_resolution
        for pair in pairs:
            u, ua = pair
            ru = ua.resource_usage(u, self.width)
            replaced = False
            for asg in list(self.assignments.values()):
                if asg.action.resource_usage(asg.unit, self.width).consistent_with(ru, self):
                    continue
                if asg.time == self.time:
                    cancel_old = cancel_new = False
                    if strategy == MOVE_CONFLICT_RESOLUTION_CANCEL_RANDOM:
                        if self.rng.integers(2) == 0:
                            cancel_new = True
                        else:
                            cancel_old = True
                    elif strategy == MOVE_CONFLICT_RESOLUTION_CANCEL_ALTERNATING:
                        if self.unit_cancelation_counter % 2 == 0:
                            cancel_new = True
                        else:
                            cancel_old = True
                        self.unit_cancelation_counter += 1
                    else:
                        cancel_old = cancel_new = True
                    d = min(asg.action.eta(asg.unit), ua.eta(u))
                    if cancel_old:
                        asg.action = UnitAction.none(d)
                    if cancel_new:
                        ua = UnitAction.none(d)
                        replaced = True
                else:
                    ua = UnitAction.none()
                    if not replaced:
                        pair[1] = ua
            self.assignments[u.id] = Assignment(u, ua, self.time)
            if ua.type != TYPE_NONE:
                issued_something = True
        return issued_something

    def issue_safe(self, pairs: list[list]) -> bool:
        """``GameState.issueSafe``: illegal actions become waits of the same length."""
        for pair in pairs:
            u, ua = pair
            if not self.can_execute(u, ua):
                ua = UnitAction.none(ua.eta(u))
                pair[1] = ua
            for pos in ua.resource_usage(u, self.width).positions:
                px, py = pos % self.width, pos // self.width
                if self.terrain[py, px] != TERRAIN_NONE or self.unit_at(px, py) is not None:
                    ua = UnitAction.none(ua.eta(u))
                    pair[1] = ua
        return self.issue(pairs)

    def cycle(self) -> bool:
        """Advance one game tick, executing every finished action. Returns gameover."""
        self.time += 1
        ready = [a for a in self.assignments.values() if a.action.eta(a.unit) + a.time <= self.time]
        for a in ready:
            self.assignments.pop(a.unit.id, None)
            self._execute(a.unit, a.action)
        return self.gameover()

    def _execute(self, u: Unit, ua: UnitAction) -> None:
        t = ua.type
        if t == TYPE_NONE:
            return
        if t == TYPE_MOVE:
            nx, ny = ua.target_pos(u)
            self._move_unit(u, nx, ny)
        elif t == TYPE_ATTACK_LOCATION:
            other = self.unit_at(ua.x, ua.y)
            if other is not None:
                lo, hi = u.type.min_damage, u.type.max_damage
                damage = lo if lo == hi else lo + int(self.rng.integers(1 + hi - lo))
                other.hp -= damage
                if other.hp <= 0:
                    self.remove_unit(other)
        elif t == TYPE_HARVEST:
            tx, ty = ua.target_pos(u)
            res = self.unit_at(tx, ty)
            if res is not None and res.type.is_resource and u.type.can_harvest and u.resources == 0:
                res.resources -= u.type.harvest_amount
                if res.resources <= 0:
                    self.remove_unit(res)
                u.resources = u.type.harvest_amount
        elif t == TYPE_RETURN:
            tx, ty = ua.target_pos(u)
            base = self.unit_at(tx, ty)
            if base is not None and base.type.is_stockpile and u.resources > 0:
                self.player_resources[u.player] += u.resources
                u.resources = 0
        elif t == TYPE_PRODUCE:
            tx, ty = ua.target_pos(u)
            cost = ua.unit_type.cost
            if self.player_resources[u.player] - cost >= 0:
                self.add_unit(Unit(self.next_id, u.player, ua.unit_type, tx, ty, 0))
                self.player_resources[u.player] -= cost

    # ---------------------------------------------------------- vector API
    def observable(self, player: int, x: int, y: int) -> bool:
        for u in self.units:
            if u.player == player:
                r = u.type.sight_radius
                if (u.x - x) ** 2 + (u.y - y) ** 2 <= r * r:
                    return True
        return False

    def visible_units(self, player: int) -> list[Unit]:
        """Units ``player`` can see (``PartiallyObservableGameState``)."""
        own = [u for u in self.units if u.player == player]
        out = []
        for u in self.units:
            if u.player == player:
                out.append(u)
                continue
            for o in own:
                r = o.type.sight_radius
                if (o.x - u.x) ** 2 + (o.y - u.y) ** 2 <= r * r:
                    out.append(u)
                    break
        return out

    def decode_action(self, u: Unit, row: Sequence[int], attack_grid: int) -> UnitAction:
        """``UnitAction.fromVectorAction`` for one grid cell (``row`` excludes the cell index)."""
        atype = int(row[0])
        ua = UnitAction(atype)
        if atype == TYPE_MOVE:
            ua.parameter = int(row[1])
        elif atype == TYPE_HARVEST:
            ua.parameter = int(row[2])
        elif atype == TYPE_RETURN:
            ua.parameter = int(row[3])
        elif atype == TYPE_PRODUCE:
            ua.parameter = int(row[4])
            ua.unit_type = self.utt[int(row[5])]
        elif atype == TYPE_ATTACK_LOCATION:
            center = attack_grid // 2
            ua.x = u.x + int(row[6]) % attack_grid - center
            ua.y = u.y + int(row[6]) // attack_grid - center
        return ua

    def player_action_from_grid(
        self,
        player: int,
        grid_actions: np.ndarray,
        attack_grid: int,
        visible: list[Unit] | None = None,
    ) -> list[list]:
        """``PlayerAction.fromVectorAction`` + ``fillWithNones(duration=1)``.

        ``grid_actions`` has shape ``(height * width, 7)``. Only cells holding an
        idle unit of ``player`` are read. ``visible`` restricts which units'
        ongoing actions count as reserved (partial observability).
        """
        pa_ru = self.reserved_resources(visible)
        pairs: list[list] = []
        chosen: set[int] = set()
        w = self.width
        for u in sorted((u for u in self.units if u.player == player and u.id not in self.assignments), key=lambda u: u.x + u.y * w):
            ua = self.decode_action(u, grid_actions[u.x + u.y * w], attack_grid)
            ru = ua.resource_usage(u, w)
            if ru.consistent_with(pa_ru, self):
                pa_ru.merge(ru)
                pairs.append([u, ua])
                chosen.add(u.id)
        for u in self.units:
            if u.player == player and u.id not in self.assignments and u.id not in chosen:
                pairs.append([u, UnitAction.none(1)])
        return pairs

    def to_dict(self) -> dict:
        """JSON-friendly snapshot used by the renderers."""
        units = []
        for u in self.units:
            a = self.assignments.get(u.id)
            entry = {
                "id": u.id,
                "type": u.type.name,
                "player": u.player,
                "x": u.x,
                "y": u.y,
                "hp": u.hp,
                "maxHp": u.type.hp,
                "resources": u.resources,
            }
            if a is not None:
                act = {"type": ACTION_NAMES[a.action.type], "start": a.time, "eta": a.action.eta(u)}
                if a.action.type == TYPE_ATTACK_LOCATION:
                    act["x"], act["y"] = a.action.x, a.action.y
                elif a.action.type != TYPE_NONE:
                    act["dir"] = a.action.parameter
                    if a.action.unit_type is not None:
                        act["unitType"] = a.action.unit_type.name
                entry["action"] = act
            units.append(entry)
        return {
            "time": self.time,
            "width": self.width,
            "height": self.height,
            "resources": list(self.player_resources),
            "units": units,
        }


# ---------------------------------------------------------------------- maps
def resolve_map_path(map_path: str) -> str:
    """Accept an absolute path, a path relative to cwd, or one inside the bundled maps."""
    if os.path.isfile(map_path):
        return map_path
    rel = map_path[len("maps/"):] if map_path.startswith("maps/") else map_path
    bundled = os.path.join(MAPS_DIR, rel)
    if os.path.isfile(bundled):
        return bundled
    raise FileNotFoundError(f"MicroRTS map not found: {map_path!r}")


def _decode_terrain(text: str, size: int) -> list[int]:
    text = text.strip()
    if "A" in text or "B" in text:  # run-length compressed format
        out: list[int] = []
        counter = ""
        for ch in text:
            if ch in "AB":
                if counter:
                    out.extend([out[-1]] * (int(counter) - 1))
                    counter = ""
                out.append(0 if ch == "A" else 1)
            else:
                counter += ch
        if counter:
            out.extend([out[-1]] * (int(counter) - 1))
        return out
    return [int(c) for c in text[:size]]


@dataclass
class MapSpec:
    width: int
    height: int
    terrain: list[int]
    resources: list[int]
    units: list[tuple[int, str, int, int, int, int, int]]  # id, type, player, x, y, resources, hp

    def new_game(self, utt: UnitTypeTable, rng: np.random.Generator | None = None) -> GameState:
        units = [Unit(uid, player, utt[tname], x, y, res, hp) for uid, tname, player, x, y, res, hp in self.units]
        return GameState(self.width, self.height, self.terrain, self.resources, units, utt, rng)


def load_map(map_path: str) -> MapSpec:
    root = ET.parse(resolve_map_path(map_path)).getroot()
    width, height = int(root.get("width")), int(root.get("height"))
    terrain = _decode_terrain(root.find("terrain").text or "", width * height)
    players = sorted(root.find("players"), key=lambda p: int(p.get("ID")))
    if len(players) != 2:
        raise ValueError("only two-player maps are supported")
    resources = [int(p.get("resources")) for p in players]
    units = []
    seen = set()
    for e in root.find("units"):
        uid = int(e.get("ID"))
        if uid in seen:
            raise ValueError(f"repeated unit ID {uid} in map")
        seen.add(uid)
        units.append(
            (uid, e.get("type"), int(e.get("player")), int(e.get("x")), int(e.get("y")), int(e.get("resources")), int(e.get("hitpoints")))
        )
    return MapSpec(width, height, terrain, resources, units)


def list_maps() -> list[str]:
    """Bundled map paths, relative to the maps directory (e.g. ``16x16/basesWorkers16x16.xml``)."""
    out = []
    for dirpath, _, files in os.walk(MAPS_DIR):
        for f in files:
            if f.endswith(".xml"):
                out.append(os.path.relpath(os.path.join(dirpath, f), MAPS_DIR))
    return sorted(out)
