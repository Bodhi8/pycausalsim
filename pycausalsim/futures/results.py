"""
Comparing scenarios, ranking drivers, and emitting falsifiable claims.

rank_drivers here is the forecasting analogue of the same call in the
observational API: raise each lever to matched intensity, one at a time, hold
everything else at the baseline, and report the effect on the target. Effects
are not additive, because combined scenarios interact, and the comparison says
so rather than letting a reader assume otherwise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from .scenario import Scenario


@dataclass
class DriverEffect:
    lever: str
    effect: float
    ci: tuple
    p_positive: float
    intensity: float

    def __repr__(self):
        return (f"{self.lever:28s} {self.effect:+.4f}  "
                f"[{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]  "
                f"p(+)={self.p_positive:.2f}")


@dataclass
class FalsifiableClaim:
    key: str
    statement: str
    resolve_by: int
    p: float
    scenario: str
    source: str = ""
    observed: bool | None = None

    def __repr__(self):
        seen = "" if self.observed is None else f"  observed={self.observed}"
        return (f"[{self.resolve_by}] p={self.p:.2f}  {self.statement}"
                f"  ({self.scenario}){seen}")


class ClaimSet:
    """Dated, checkable claims with probabilities and a named source."""

    def __init__(self, claims, model_version="", generated=""):
        self.claims = list(claims)
        self.model_version = model_version
        self.generated = generated

    def __iter__(self):
        return iter(self.claims)

    def __len__(self):
        return len(self.claims)

    def __repr__(self):
        return "\n".join(repr(c) for c in
                         sorted(self.claims, key=lambda c: c.resolve_by))

    def export(self, path, indent=2):
        payload = dict(model_version=self.model_version,
                       generated=self.generated,
                       claims=[asdict(c) for c in self.claims])
        with open(path, "w") as f:
            json.dump(payload, f, indent=indent)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls([FalsifiableClaim(**c) for c in d["claims"]],
                   d.get("model_version", ""), d.get("generated", ""))

    def score(self, observed: dict):
        """
        Brier score against resolved outcomes.

        observed maps claim key -> True/False. Unresolved keys are skipped, so
        a claim set can be scored progressively as the years arrive.
        """
        rows, sq = [], []
        for c in self.claims:
            if c.key not in observed:
                continue
            o = bool(observed[c.key])
            c.observed = o
            err = (c.p - (1.0 if o else 0.0)) ** 2
            sq.append(err)
            rows.append((c.key, c.p, o, err))
        if not sq:
            return dict(n=0, brier=None, rows=[],
                        note="no claims resolved yet")
        brier = float(np.mean(sq))
        hits = sum(1 for _, p, o, _ in rows if (p >= 0.5) == o)
        return dict(n=len(rows), brier=brier, accuracy=hits / len(rows),
                    rows=rows,
                    note="Brier score: 0 is perfect, 0.25 is a coin flip.")


class Comparison:
    """The result of running several scenarios on the same parameter draws."""

    def __init__(self, scm, runs, target, baseline=None):
        self.scm = scm
        self.runs = {r.name: r for r in runs}
        self.target = target
        self.baseline = baseline or list(self.runs)[0]
        self.years = scm.years

    def __getitem__(self, name):
        return self.runs[name]

    # -------------------------------------------------------------- reading
    def table(self, year=None, node=None):
        year = year or self.years[-1]
        node = node or self.target
        base = self.runs[self.baseline].at(node, year)
        lines = [f"{node} at {year}",
                 f"  {'scenario':32s} {'mean':>8s} {'90% interval':>20s} "
                 f"{'P(>base)':>9s}"]
        for name, r in self.runs.items():
            v = r.at(node, year)
            lines.append(
                f"  {name:32s} {v.mean():8.3f} "
                f"[{np.percentile(v, 5):8.3f}, {np.percentile(v, 95):7.3f}] "
                f"{float((v > base.mean()).mean()):9.2f}")
        return "\n".join(lines)

    def diff(self, a, b, node=None, year=None):
        """Paired difference between two scenarios on the same draws."""
        node = node or self.target
        year = year or self.years[-1]
        d = self.runs[a].at(node, year) - self.runs[b].at(node, year)
        return dict(
            a=a, b=b, node=node, year=year,
            mean=float(d.mean()),
            ci=(float(np.percentile(d, 5)), float(np.percentile(d, 95))),
            p_positive=float((d > 0).mean()))

    def share_of_gap(self, partial, full, node=None, year=None):
        """
        How much of the distance from baseline to `full` does `partial` cover?

        This is the number that separates a policy that helps from one that
        substitutes for the thing that would actually work.
        """
        node = node or self.target
        year = year or self.years[-1]
        base = self.runs[self.baseline].at(node, year).mean()
        p = self.runs[partial].at(node, year).mean()
        f = self.runs[full].at(node, year).mean()
        if np.isclose(f, base):
            return float("nan")
        return float((p - base) / (f - base))

    # ------------------------------------------------------------- claims
    def falsifiable_claims(self, scenario=None, threshold=0.0,
                           model_version=""):
        from datetime import date
        name = scenario or self.baseline
        run = self.runs[name]
        out = []
        for spec in self.scm._claims:
            year = spec["resolve_by"]
            if year not in self.years:
                continue
            p = float(np.mean(np.asarray(spec["test"](_At(run, year)),
                                         dtype=bool)))
            if p < threshold:
                continue
            out.append(FalsifiableClaim(
                key=spec["key"], statement=spec["statement"],
                resolve_by=year, p=p, scenario=name, source=spec["source"]))
        return ClaimSet(out, model_version, str(date.today()))


class _At:
    """Read-only view of one run pinned to a year, passed to claim tests."""

    def __init__(self, run, year):
        self._r = run
        self.year = year

    def __getitem__(self, node):
        return self._r.at(node, self.year)

    def lever(self, name):
        return self._r.levers[name][self._r.years.index(self.year)]


def rank_drivers(scm, baseline: Scenario, levers=None, target="wellbeing",
                 at_year=None, intensity=0.75, ramp_from=None, over=5,
                 draws=None, seed_params=None):
    """
    Single-lever do() from a baseline, at matched intensity.

    The same parameter draws are reused across levers, so the comparison is
    paired and the intervals are tighter than independent sampling would give.
    """
    at_year = at_year or scm.years[-1]
    ramp_from = ramp_from or scm.years[min(2, len(scm.years) - 1)]
    p = seed_params if seed_params is not None else scm.sample_params(draws or scm.draws)
    levers = levers or sorted(scm._levers)

    base_run = scm.run(baseline, params=p)
    base = base_run.at(target, at_year)

    effects = []
    for lv in levers:
        s = baseline.copy(f"{baseline.name} + {lv}")
        s.interventions = [i for i in s.interventions if i.lever != lv]
        s.at(ramp_from).set(lv, ramp_to=intensity, over=over)
        d = scm.run(s, params=p).at(target, at_year) - base
        effects.append(DriverEffect(
            lever=lv, effect=float(d.mean()),
            ci=(float(np.percentile(d, 5)), float(np.percentile(d, 95))),
            p_positive=float((d > 0).mean()), intensity=intensity))
    effects.sort(key=lambda e: -e.effect)
    return effects
