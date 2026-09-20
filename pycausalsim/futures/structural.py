"""
Structural interventions.

Pearl's do() sets a variable to a value. It does not delete an edge. The most
consequential futures questions are structural: automation does not lower the
value of apprenticeship, it severs the mechanism by which apprenticeship used
to happen as a free byproduct of junior production.

A StructuralRule makes a graph change conditional on a value, which is what
lets the model represent structure change as itself caused.

Rules fire per draw. If the condition holds in 40 percent of draws at a given
step, the edge is severed in exactly those draws, so the population splits
into worlds with different graphs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_COND = re.compile(r"^\s*([A-Za-z_]\w*)\s*(>=|<=|>|<|==)\s*(-?[\d.eE+]+)\s*$")


class StructuralIntervention:
    """Factory for edge-level interventions."""

    def __init__(self, kind, u, v, on_removed=0.0, new_parent=None):
        self.kind = kind
        self.u = u
        self.v = v
        self.on_removed = on_removed
        self.new_parent = new_parent

    @classmethod
    def remove_edge(cls, u, v, on_removed=0.0):
        """Sever u -> v. Inside v's equation, u reads as `on_removed`."""
        return cls("remove", u, v, on_removed=on_removed)

    @classmethod
    def add_edge(cls, u, v):
        """Restore a previously severed u -> v."""
        return cls("add", u, v)

    @classmethod
    def rewire(cls, u, v, to):
        """Sever u -> v and route `to` into v's slot for u instead."""
        return cls("rewire", u, v, new_parent=to)

    def __repr__(self):
        arrow = f"{self.u} -> {self.v}"
        if self.kind == "rewire":
            return f"<rewire {arrow} to {self.new_parent}>"
        return f"<{self.kind}_edge {arrow}>"


@dataclass
class StructuralRule:
    when: str | Callable
    apply: StructuralIntervention
    note: str = ""
    armed: bool = False
    _tested: bool = field(default=False, repr=False)

    def __post_init__(self):
        if isinstance(self.when, str):
            m = _COND.match(self.when)
            if not m:
                raise ValueError(
                    f"cannot parse condition {self.when!r}. Use "
                    f"'node > 0.55' or pass a callable taking the context.")
            self._var, self._op, self._thr = m.group(1), m.group(2), float(m.group(3))
        else:
            self._var = None

    def depends_on(self, just_computed, cur):
        """Fire only once the tested node exists for this step."""
        if self._tested:
            return False
        if self._var is None:
            return True
        return just_computed == self._var

    def evaluate(self, ctx, cur, levers, t):
        """Return a boolean array: True where the edge stays intact."""
        self._tested = True
        if self._var is None:
            cond = np.asarray(self.when(ctx), dtype=bool)
        else:
            val = cur.get(self._var)
            if val is None:
                val = levers.get(self._var, [None] * (t + 1))[t]
            if val is None:
                return None
            ops = {">": np.greater, "<": np.less, ">=": np.greater_equal,
                   "<=": np.less_equal, "==": np.equal}
            cond = ops[self._op](np.asarray(val, dtype=float), self._thr)
        cond = np.atleast_1d(cond)
        if self.apply.kind == "add":
            return ~cond          # condition true means the edge is present
        return ~cond              # condition true means the edge is severed

    def install(self, masks, active):
        si = self.apply
        if si.kind in ("remove", "rewire"):
            masks[(si.u, si.v)] = (active, si.on_removed)
        elif si.kind == "add":
            masks[(si.u, si.v)] = (active, si.on_removed)

    def reset_step(self):
        self._tested = False
