"""
Mechanism priors.

Forecasting pushes edges far outside the range anything was observed in. A
fitted response surface has nothing to say there, and it says it silently. A
MechanismPrior makes the extrapolation explicit: a declared functional form, a
validity range, and stated behaviour at the boundary.

The engine records every step where a parent leaves its declared range, so any
result can report which conclusions rest on observed data and which rest on
assumed mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FORMS = {"logistic", "linear", "log_linear", "power", "saturating",
         "declared", "identity"}


class OutOfSupport(RuntimeError):
    pass


@dataclass
class MechanismPrior:
    parent: str
    child: str
    form: str = "declared"
    valid_range: tuple | None = None
    saturates_at: float | None = None
    outside: str = "flag"          # flag | clip | error
    note: str = ""

    def __post_init__(self):
        if self.form not in FORMS:
            raise ValueError(f"unknown form {self.form!r}; one of {sorted(FORMS)}")
        if self.outside not in {"flag", "clip", "error"}:
            raise ValueError("outside must be flag, clip or error")

    @property
    def edge(self):
        return f"{self.child} ~ {self.parent}"

    def enforce(self, value, year, report):
        if self.valid_range is None:
            return value
        lo, hi = self.valid_range
        below = value < lo
        above = value > hi
        frac = float(np.mean(below | above))
        if frac > 0:
            report.record(self.edge, year, frac, self.outside)
            if self.outside == "error":
                raise OutOfSupport(
                    f"{self.edge} left its declared range [{lo}, {hi}] at "
                    f"{year} in {frac:.1%} of draws")
            if self.outside == "clip":
                value = np.clip(value, lo, hi)
        if self.saturates_at is not None:
            value = np.minimum(value, self.saturates_at)
        return value


@dataclass
class SupportReport:
    """Where the model extrapolated, and how far."""
    entries: list = field(default_factory=list)

    def record(self, edge, year, frac, mode):
        self.entries.append(dict(edge=edge, year=int(year),
                                 fraction=float(frac), mode=mode))

    @property
    def clean(self):
        return not self.entries

    def by_edge(self):
        out = {}
        for e in self.entries:
            d = out.setdefault(e["edge"], dict(
                first_year=e["year"], last_year=e["year"],
                max_fraction=0.0, mode=e["mode"], steps=0))
            d["first_year"] = min(d["first_year"], e["year"])
            d["last_year"] = max(d["last_year"], e["year"])
            d["max_fraction"] = max(d["max_fraction"], e["fraction"])
            d["steps"] += 1
        return out

    def summary(self):
        if self.clean:
            return "Support: every declared edge stayed inside its valid range."
        lines = ["Support: edges extrapolated outside their declared range"]
        for edge, d in sorted(self.by_edge().items()):
            lines.append(
                f"  {edge:34s} {d['first_year']}-{d['last_year']}  "
                f"up to {d['max_fraction']:.0%} of draws  ({d['mode']})")
        lines.append("  Conclusions downstream of these edges rest on declared "
                     "mechanism, not on observed data.")
        return "\n".join(lines)

    def to_dict(self):
        return self.by_edge()
