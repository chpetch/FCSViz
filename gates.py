"""
Gate data model for FCSViz.

No Streamlit dependency — pure data model safe to import anywhere.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gate type constants
# ---------------------------------------------------------------------------
POLYGON = "polygon"
RECTANGLE = "rectangle"
QUADRANT = "quadrant"
THRESHOLD_V = "threshold_v"
THRESHOLD_H = "threshold_h"

GATE_TYPES = [POLYGON, RECTANGLE, QUADRANT, THRESHOLD_V, THRESHOLD_H]


# ---------------------------------------------------------------------------
# Point-in-polygon helper (numpy ray-casting, no extra dependencies)
# ---------------------------------------------------------------------------
def _pip(vertices: list[tuple[float, float]], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return bool mask: True where (x[i], y[i]) is inside the polygon defined by vertices."""
    vx = np.array([v[0] for v in vertices], dtype=float)
    vy = np.array([v[1] for v in vertices], dtype=float)
    n = len(vx)
    inside = np.zeros(len(x), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = vx[i], vy[i]
        xj, yj = vx[j], vy[j]
        # Edge crosses the horizontal ray from (x, y) rightward
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi)
        inside ^= cross
        j = i
    return inside


# ---------------------------------------------------------------------------
# Gate dataclass
# ---------------------------------------------------------------------------
@dataclass
class Gate:
    """
    A single gate node in a GateTree.

    params schema by gate_type
    --------------------------
    polygon     : {"vertices": [(x1,y1), (x2,y2), ...]}
    rectangle   : {"x_min": float, "x_max": float, "y_min": float, "y_max": float}
    quadrant    : {"x0": float, "y0": float, "quadrant": "Q1"|"Q2"|"Q3"|"Q4"}
                  Q1=top-right, Q2=top-left, Q3=bottom-left, Q4=bottom-right
    threshold_v : {"x0": float, "side": "left"|"right"}
    threshold_h : {"y0": float, "side": "top"|"bottom"}
    """
    name: str
    gate_type: str
    x_channel: str
    y_channel: Optional[str]
    params: dict
    color: str = "#e74c3c"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    children_ids: list = field(default_factory=list)

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Bool mask of events passing THIS gate (ancestors not applied)."""
        if self.gate_type == RECTANGLE:
            p = self.params
            xv = data[self.x_channel].values
            yv = data[self.y_channel].values
            return (xv >= p["x_min"]) & (xv <= p["x_max"]) & (yv >= p["y_min"]) & (yv <= p["y_max"])

        if self.gate_type == POLYGON:
            verts = self.params["vertices"]
            if len(verts) < 3:
                return np.ones(len(data), dtype=bool)
            return _pip(verts, data[self.x_channel].values, data[self.y_channel].values)

        if self.gate_type == QUADRANT:
            x0, y0 = self.params["x0"], self.params["y0"]
            q = self.params["quadrant"]
            xv = data[self.x_channel].values
            yv = data[self.y_channel].values
            if q == "Q1":
                return (xv > x0) & (yv > y0)
            if q == "Q2":
                return (xv <= x0) & (yv > y0)
            if q == "Q3":
                return (xv <= x0) & (yv <= y0)
            if q == "Q4":
                return (xv > x0) & (yv <= y0)

        if self.gate_type == THRESHOLD_V:
            xv = data[self.x_channel].values
            if self.params["side"] == "left":
                return xv <= self.params["x0"]
            return xv > self.params["x0"]

        if self.gate_type == THRESHOLD_H:
            yv = data[self.y_channel].values
            if self.params["side"] == "bottom":
                return yv <= self.params["y0"]
            return yv > self.params["y0"]

        return np.ones(len(data), dtype=bool)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "gate_type": self.gate_type,
            "x_channel": self.x_channel,
            "y_channel": self.y_channel,
            "params": self.params,
            "color": self.color,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Gate":
        g = cls(
            name=d["name"],
            gate_type=d["gate_type"],
            x_channel=d["x_channel"],
            y_channel=d["y_channel"],
            params=d["params"],
            color=d.get("color", "#e74c3c"),
            id=d["id"],
            parent_id=d["parent_id"],
            children_ids=list(d["children_ids"]),
        )
        return g


# ---------------------------------------------------------------------------
# GateTree
# ---------------------------------------------------------------------------
class GateTree:
    """
    Tree of Gate objects representing a hierarchical gating strategy for one FCS file.

    The virtual root (ROOT_ID) is not a real Gate — it represents "All Events".
    All top-level gates have parent_id == ROOT_ID.
    """

    ROOT_ID = "__root__"

    def __init__(self) -> None:
        self._gates: dict[str, Gate] = {}
        # Virtual root children list — top-level gate ids
        self._root_children: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _parent_children(self, parent_id: str) -> list[str]:
        """Return the children_ids list for a parent (or root list)."""
        if parent_id == self.ROOT_ID:
            return self._root_children
        return self._gates[parent_id].children_ids

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add_gate(self, gate: Gate, parent_id: str = ROOT_ID) -> str:
        """
        Insert a gate into the tree under parent_id.
        Returns the gate's id.
        """
        gate.parent_id = parent_id
        self._gates[gate.id] = gate
        self._parent_children(parent_id).append(gate.id)
        return gate.id

    def remove_gate(self, gate_id: str) -> None:
        """
        Remove a gate and all its descendants.
        Also unlinks from its parent's children_ids.
        """
        if gate_id not in self._gates:
            return
        gate = self._gates[gate_id]
        # Recurse into children first (copy list — we're mutating during iteration)
        for child_id in list(gate.children_ids):
            self.remove_gate(child_id)
        # Unlink from parent
        siblings = self._parent_children(gate.parent_id)
        if gate_id in siblings:
            siblings.remove(gate_id)
        del self._gates[gate_id]

    def rename_gate(self, gate_id: str, name: str) -> None:
        self._gates[gate_id].name = name

    def update_params(self, gate_id: str, params: dict) -> None:
        """Replace a gate's params (e.g. after user adjusts a drawn gate)."""
        self._gates[gate_id].params = params

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_gate(self, gate_id: str) -> Gate:
        return self._gates[gate_id]

    def get_children(self, parent_id: str) -> list[Gate]:
        return [self._gates[gid] for gid in self._parent_children(parent_id)]

    def get_ancestors(self, gate_id: str) -> list[Gate]:
        """
        Return ancestors from root down to (but not including) gate_id.
        Returns [] if gate_id is a direct child of ROOT.
        """
        ancestors: list[Gate] = []
        current_id = self._gates[gate_id].parent_id
        while current_id and current_id != self.ROOT_ID:
            ancestors.append(self._gates[current_id])
            current_id = self._gates[current_id].parent_id
        ancestors.reverse()
        return ancestors

    def get_descendants(self, gate_id: str) -> list[str]:
        """Return all descendant ids (excludes gate_id itself), breadth-first."""
        result: list[str] = []
        queue = list(self._gates[gate_id].children_ids)
        while queue:
            gid = queue.pop(0)
            result.append(gid)
            queue.extend(self._gates[gid].children_ids)
        return result

    def flat_list(self, parent_id: str = ROOT_ID, depth: int = 0) -> list[tuple[int, Gate]]:
        """
        Pre-order traversal returning (depth, gate) pairs.
        Used to render an indented gate manager UI.
        """
        result: list[tuple[int, Gate]] = []
        for gid in self._parent_children(parent_id):
            gate = self._gates[gid]
            result.append((depth, gate))
            result.extend(self.flat_list(gid, depth + 1))
        return result

    def __len__(self) -> int:
        return len(self._gates)

    # ------------------------------------------------------------------
    # Mask / population
    # ------------------------------------------------------------------
    def get_parent_mask(self, gate_id: str, data: pd.DataFrame) -> np.ndarray:
        """
        Boolean mask of events in the PARENT population (ancestors only, not self).
        All-True when gate's parent is ROOT.
        """
        ancestors = self.get_ancestors(gate_id)
        mask = np.ones(len(data), dtype=bool)
        for ancestor in ancestors:
            mask &= ancestor.evaluate(data)
        # Also apply direct parent if not root
        parent_id = self._gates[gate_id].parent_id
        if parent_id and parent_id != self.ROOT_ID:
            mask &= self._gates[parent_id].evaluate(data)
        return mask

    def get_mask(self, gate_id: str, data: pd.DataFrame) -> np.ndarray:
        """Boolean mask for events passing this gate AND all ancestors."""
        mask = self.get_parent_mask(gate_id, data)
        mask &= self._gates[gate_id].evaluate(data)
        return mask

    def event_count(self, gate_id: str, data: pd.DataFrame) -> int:
        return int(self.get_mask(gate_id, data).sum())

    def parent_count(self, gate_id: str, data: pd.DataFrame) -> int:
        return int(self.get_parent_mask(gate_id, data).sum())

    def percent_of_parent(self, gate_id: str, data: pd.DataFrame) -> float:
        denom = self.parent_count(gate_id, data)
        if denom == 0:
            return 0.0
        return self.event_count(gate_id, data) / denom * 100.0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    def add_quadrant_gates(
        self,
        parent_id: str,
        x_ch: str,
        y_ch: str,
        x0: float,
        y0: float,
        name_prefix: str = "Q",
        color: str = "#e74c3c",
    ) -> list[str]:
        """
        Create 4 quadrant Gate siblings (Q1–Q4) under parent_id.
        Returns list of 4 gate ids in Q1, Q2, Q3, Q4 order.
        """
        ids = []
        for q in ("Q1", "Q2", "Q3", "Q4"):
            gate = Gate(
                name=f"{name_prefix}{q[-1]}",
                gate_type=QUADRANT,
                x_channel=x_ch,
                y_channel=y_ch,
                params={"x0": x0, "y0": y0, "quadrant": q},
                color=color,
            )
            ids.append(self.add_gate(gate, parent_id))
        return ids

    def add_threshold_pair(
        self,
        parent_id: str,
        axis: str,
        channel: str,
        value: float,
        name_prefix: str = "T",
        color: str = "#e74c3c",
    ) -> list[str]:
        """
        Create 2 threshold Gate siblings under parent_id.
        axis: "v" (vertical threshold on x) or "h" (horizontal threshold on y).
        Returns [left_id, right_id] for axis="v" or [bottom_id, top_id] for axis="h".
        """
        if axis == "v":
            pairs = [
                (f"{name_prefix}-L", THRESHOLD_V, channel, None, {"x0": value, "side": "left"}),
                (f"{name_prefix}-R", THRESHOLD_V, channel, None, {"x0": value, "side": "right"}),
            ]
        else:
            pairs = [
                (f"{name_prefix}-B", THRESHOLD_H, None, channel, {"y0": value, "side": "bottom"}),
                (f"{name_prefix}-T", THRESHOLD_H, None, channel, {"y0": value, "side": "top"}),
            ]
        ids = []
        for name, gtype, x_ch, y_ch, params in pairs:
            gate = Gate(
                name=name,
                gate_type=gtype,
                x_channel=x_ch or channel,
                y_channel=y_ch,
                params=params,
                color=color,
            )
            ids.append(self.add_gate(gate, parent_id))
        return ids

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "gates": {gid: g.to_dict() for gid, g in self._gates.items()},
            "root_children": list(self._root_children),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GateTree":
        tree = cls()
        for gid, gdict in d["gates"].items():
            gate = Gate.from_dict(gdict)
            tree._gates[gid] = gate
        tree._root_children = list(d["root_children"])
        return tree
