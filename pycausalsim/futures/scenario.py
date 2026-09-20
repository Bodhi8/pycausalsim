"""
The scenario language.

A scenario is a named bundle of timed interventions on policy levers. It is
declarative so that the people who should be arguing with a forecast, who are
mostly not engineers, can read and edit one.

    kc = Scenario("Ownership and institutions")
    kc.at(2027).set("apprenticeship", ramp_to=0.72, over=4)
    kc.at(2028).set("institutions",   ramp_to=0.78, over=6)
    kc.at(2031).set("income_floor",   to=0.60)

Scenarios are comparable, composable and diffable. `kc | other` merges, with
the right-hand side winning on conflicts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Intervention:
    lever: str
    year: int
    to: float | None = None
    ramp_to: float | None = None
    over: int = 1
    note: str = ""

    @property
    def target(self):
        return self.ramp_to if self.ramp_to is not None else self.to

    def describe(self):
        if self.ramp_to is not None:
            return (f"{self.year}: {self.lever} ramps to {self.ramp_to:g} "
                    f"over {self.over} {'step' if self.over == 1 else 'steps'}")
        return f"{self.year}: {self.lever} set to {self.to:g}"


class _At:
    def __init__(self, scenario, year):
        self._s = scenario
        self._y = year

    def set(self, lever, to=None, ramp_to=None, over=1, note=""):
        if to is None and ramp_to is None:
            raise ValueError("pass to= or ramp_to=")
        self._s.interventions.append(
            Intervention(lever, self._y, to, ramp_to, over, note))
        return self._s


@dataclass
class Scenario:
    name: str
    interventions: list = field(default_factory=list)

    def at(self, year):
        return _At(self, int(year))

    def set(self, lever, value, note=""):
        """Hold a lever at a constant value for the whole horizon."""
        self.interventions.append(
            Intervention(lever, -10**9, to=float(value), note=note))
        return self

    def __or__(self, other):
        merged = Scenario(f"{self.name} + {other.name}")
        merged.interventions = list(self.interventions) + list(other.interventions)
        return merged

    def copy(self, name=None):
        s = Scenario(name or self.name)
        s.interventions = list(self.interventions)
        return s

    def without(self, lever, name=None):
        """This scenario with one lever dropped back to its default."""
        s = Scenario(name or f"{self.name} without {lever}")
        s.interventions = [i for i in self.interventions if i.lever != lever]
        return s

    def levers(self):
        return sorted({i.lever for i in self.interventions})

    def describe(self):
        lines = [f"Scenario: {self.name}"]
        for i in sorted(self.interventions, key=lambda x: (x.year, x.lever)):
            lines.append("  " + i.describe() + (f"   # {i.note}" if i.note else ""))
        return "\n".join(lines)

    def to_dict(self):
        return dict(name=self.name, interventions=[
            dict(lever=i.lever, year=i.year, to=i.to, ramp_to=i.ramp_to,
                 over=i.over, note=i.note) for i in self.interventions])

    def to_json(self, path=None, indent=2):
        s = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d):
        s = cls(d["name"])
        s.interventions = [Intervention(**i) for i in d["interventions"]]
        return s


def resolve_lever(scenario, lever, default, years):
    """Build the value path for one lever across the horizon."""
    path = np.full(len(years), float(default))
    if scenario is None:
        return path
    acts = [i for i in scenario.interventions if i.lever == lever]
    if not acts:
        return path
    acts.sort(key=lambda i: i.year)
    for iv in acts:
        if iv.year <= years[0]:
            start = 0
        elif iv.year > years[-1]:
            continue
        else:
            start = years.index(iv.year)
        base = path[start - 1] if start > 0 else path[0]
        if iv.ramp_to is not None:
            for k in range(start, len(years)):
                frac = min(1.0, (k - start + 1) / max(1, iv.over))
                path[k] = base + (iv.ramp_to - base) * frac
        else:
            path[start:] = iv.to
    return path


def ramp(year, to, over=3):
    """Convenience for inline use: ramp(2028, 0.78, over=6)."""
    return dict(year=int(year), ramp_to=float(to), over=int(over))


def step(year, to):
    return dict(year=int(year), to=float(to))
