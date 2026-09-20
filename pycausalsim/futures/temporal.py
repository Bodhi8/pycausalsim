"""
Temporal structural causal models.

A TemporalSCM unrolls a causal graph over a horizon. The unrolled graph stays
acyclic, so identification and simulation results for static SCMs carry over.
Everything is vectorized across Monte Carlo draws: each node value at each step
is an array of shape (draws,).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np

from .mechanism import MechanismPrior, SupportReport
from .structural import StructuralRule
from .scenario import Scenario, resolve_lever

_LAG = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*lag\(\s*(\d+)\s*\)\s*$")


def _parse_parent(spec: str):
    """'junior:lag(10)' -> ('junior', 10); 'automation' -> ('automation', 0)."""
    m = _LAG.match(spec)
    if m:
        return m.group(1), int(m.group(2))
    if ":" in spec:
        raise ValueError(
            f"bad parent spec {spec!r}; use 'name' or 'name:lag(k)'")
    return spec.strip(), 0


class Context:
    """Passed to every equation. Gives current values, lags and parameters."""

    def __init__(self, scm, t, cur, hist, params, levers, masks, seed=None):
        self._scm = scm
        self.t = t
        self.year = scm.years[t]
        self._cur = cur
        self._hist = hist
        self._masks = masks
        self.p = params
        self._levers = levers
        self._seed = seed or {}
        self._node = None

    def __getitem__(self, name):
        if name in self._levers:
            return self._levers[name][self.t]
        if name not in self._cur:
            raise KeyError(
                f"{name!r} is not available at step {self.year}. Either it is "
                f"not a declared node, or it is computed later in the "
                f"topological order and needs a lag.")
        val = self._cur[name]
        mask = self._masks.get((name, self._node))
        if mask is not None:
            active, on_removed = mask
            val = np.where(active, val, on_removed)
        return val

    def lag(self, name, k=1):
        """Value of `name` k steps back. Before the horizon, uses init history."""
        if k < 1:
            return self[name]
        idx = self.t - k
        series = self._hist[name]
        if idx >= 0:
            val = series[idx]
        else:
            pre = self._scm._pre_history.get(name)
            if pre is not None:
                # pre[-1] is the step immediately before t=0
                j = max(0, len(pre) + idx)
                val = pre[j]
            elif name in self._seed:
                val = self._seed[name]
            else:
                raise KeyError(
                    f"{name!r} needs {k} steps of history before the horizon. "
                    f"Declare pre_history=... or init=... on it.")
        mask = self._masks.get((name, self._node))
        if mask is not None:
            active, on_removed = mask
            val = np.where(active, val, on_removed)
        return val

    def lever(self, name):
        return self._levers[name][self.t]


class TemporalSCM:
    """
    A causal graph unrolled over a horizon.

    scm = TemporalSCM(horizon=range(2026, 2042), draws=4000, seed=7)
    scm.parameter("attrition", "normal", loc=0.045, scale=0.008)
    scm.lever("apprenticeship", default=0.05)
    scm.add_equation("junior", f, parents=["automation", "apprenticeship"])
    scm.state("senior", f_senior, parents=["junior:lag(10)", "senior:lag(1)"],
              init=lambda p: p.conversion / p.attrition * 100,
              pre_history=[100.0] * 11)
    """

    def __init__(self, horizon: Iterable[int], step: str = "year",
                 draws: int = 1000, seed: int | None = None):
        self.years = list(horizon)
        self.step = step
        self.draws = draws
        self.rng = np.random.default_rng(seed)

        self._equations: dict[str, Callable] = {}
        self._parents: dict[str, list[tuple[str, int]]] = {}
        self._states: set[str] = set()
        self._inits: dict[str, Callable] = {}
        self._pre_history: dict[str, list] = {}
        self._param_specs: dict[str, dict] = {}
        self._levers: dict[str, float] = {}
        self._mechanisms: dict[tuple[str, str], MechanismPrior] = {}
        self._rules: list[StructuralRule] = []
        self._claims: list[dict] = []
        self._order: list[str] | None = None

    # ---------------------------------------------------------------- setup
    def parameter(self, name, dist="normal", **kw):
        """Declare a prior. dist in {normal, lognormal, uniform, fixed}."""
        self._param_specs[name] = dict(dist=dist, **kw)
        return self

    def lever(self, name, default=0.0):
        """A policy variable that scenarios intervene on."""
        self._levers[name] = float(default)
        return self

    def add_equation(self, name, fn, parents: Sequence[str] = (),
                     pre_history=None):
        self._equations[name] = fn
        self._parents[name] = [_parse_parent(s) for s in parents]
        if pre_history is not None:
            self._pre_history[name] = list(pre_history)
        self._order = None
        return self

    def state(self, name, fn, parents: Sequence[str] = (), init=None,
              pre_history=None):
        """A node carrying memory. `init` sets its value before the horizon."""
        self.add_equation(name, fn, parents, pre_history)
        self._states.add(name)
        if init is not None:
            self._inits[name] = init
        return self

    def exogenous(self, name, fn):
        """A node with no parents inside the graph."""
        return self.add_equation(name, fn, parents=())

    def mechanism(self, edge: str, form: str = "declared",
                  valid_range=None, saturates_at=None, outside="flag",
                  note=""):
        """
        Declare the mechanism on an edge, e.g. "automation ~ token_price".

        Required before the engine will extrapolate that edge outside
        `valid_range`. outside in {flag, clip, error}.
        """
        child, _, parent = [s.strip() for s in edge.partition("~")]
        if not parent:
            raise ValueError("edge must look like 'child ~ parent'")
        self._mechanisms[(parent, child)] = MechanismPrior(
            parent=parent, child=child, form=form, valid_range=valid_range,
            saturates_at=saturates_at, outside=outside, note=note)
        return self

    def structural_rule(self, when, apply, note=""):
        """Change the graph itself when a condition fires. See StructuralRule."""
        self._rules.append(StructuralRule(when=when, apply=apply, note=note))
        return self

    def claim(self, key, statement, test, resolve_by, source=""):
        """Register a dated, checkable claim evaluated on a run."""
        self._claims.append(dict(key=key, statement=statement, test=test,
                                 resolve_by=int(resolve_by), source=source))
        return self

    # ------------------------------------------------------------ internals
    def _topo(self):
        if self._order is not None:
            return self._order
        deps = {n: {p for p, k in ps if k == 0 and p in self._equations}
                for n, ps in self._parents.items()}
        order, seen, temp = [], set(), set()

        def visit(n):
            if n in seen:
                return
            if n in temp:
                raise ValueError(
                    f"cycle through {n!r} at lag 0. Break it with a lag, which "
                    f"is what makes the unrolled graph acyclic.")
            temp.add(n)
            for d in sorted(deps.get(n, ())):
                visit(d)
            temp.discard(n)
            seen.add(n)
            order.append(n)

        for n in sorted(self._equations):
            visit(n)
        self._order = order
        return order

    def sample_params(self, draws=None):
        n = draws or self.draws
        out = {}
        for name, spec in self._param_specs.items():
            d = spec["dist"]
            if d == "normal":
                out[name] = self.rng.normal(spec["loc"], spec["scale"], n)
            elif d == "lognormal":
                out[name] = self.rng.lognormal(spec["mean"], spec["sigma"], n)
            elif d == "uniform":
                out[name] = self.rng.uniform(spec["low"], spec["high"], n)
            elif d == "fixed":
                out[name] = np.full(n, float(spec["value"]))
            else:
                raise ValueError(f"unknown dist {d!r}")
        return _Params(out)

    def _lever_paths(self, scenario, n):
        paths = {}
        for name, default in self._levers.items():
            paths[name] = resolve_lever(scenario, name, default, self.years)
        return paths

    # ----------------------------------------------------------------- run
    def run(self, scenario: Scenario | None = None, draws=None, params=None):
        n = draws or self.draws
        p = params if params is not None else self.sample_params(n)
        levers = self._lever_paths(scenario, n)
        order = self._topo()

        hist = {name: [] for name in self._equations}
        support = SupportReport()
        fired = defaultdict(list)

        # seed state variables; the init value also serves as pre-horizon lag
        prev, seed = {}, {}
        for name in self._states:
            init = self._inits.get(name)
            v = (np.asarray(init(p), dtype=float) * np.ones(n)
                 if init is not None else np.zeros(n))
            prev[name] = v
            seed[name] = v

        for t in range(len(self.years)):
            cur = dict(prev)
            masks = {}
            ctx = Context(self, t, cur, hist, p, levers, masks, seed=seed)

            for name in order:
                ctx._node = name
                # check declared mechanisms on incoming edges
                for parent, lag in self._parents[name]:
                    mech = self._mechanisms.get((parent, name))
                    if mech is None:
                        continue
                    try:
                        val = ctx[parent] if lag == 0 else ctx.lag(parent, lag)
                    except KeyError:
                        continue
                    val = mech.enforce(val, self.years[t], support)
                    if lag == 0:
                        cur[parent] = val

                value = np.asarray(self._equations[name](ctx), dtype=float)
                if value.ndim == 0:
                    value = np.full(n, float(value))
                cur[name] = value

                # structural rules fire after the node they test is available
                for rule in self._rules:
                    if rule.armed or not rule.depends_on(name, cur):
                        continue
                    active = rule.evaluate(ctx, cur, levers, t)
                    if active is None:
                        continue
                    frac = float(np.mean(~active))
                    if frac > 0:
                        rule.install(masks, active)
                        fired[rule.note or repr(rule.apply)].append(
                            (self.years[t], frac))

            for name in order:
                hist[name].append(cur[name])
            prev = {k: cur[k] for k in self._states}
            for rule in self._rules:
                rule.reset_step()

        series = {k: np.array(v) for k, v in hist.items()}  # (T, draws)
        return RunResult(self, series, levers, p, support, dict(fired),
                         scenario_name=(scenario.name if scenario else "baseline"))


class _Params:
    """Attribute access over sampled parameter arrays."""

    def __init__(self, d):
        self.__dict__.update(d)
        self._d = d

    def __getitem__(self, k):
        return self._d[k]

    def keys(self):
        return self._d.keys()


class RunResult:
    def __init__(self, scm, series, levers, params, support, fired,
                 scenario_name="baseline"):
        self.scm = scm
        self.series = series
        self.levers = levers
        self.params = params
        self.support = support
        self.fired = fired
        self.name = scenario_name
        self.years = scm.years

    def __getitem__(self, name):
        return self.series[name]

    def at(self, name, year):
        return self.series[name][self.years.index(int(year))]

    def mean(self, name):
        return self.series[name].mean(axis=1)

    def interval(self, name, lo=5, hi=95):
        s = self.series[name]
        return np.percentile(s, lo, axis=1), np.percentile(s, hi, axis=1)

    def summary(self, nodes=None, year=None):
        year = year or self.years[-1]
        nodes = nodes or sorted(self.series)
        rows = []
        for nd in nodes:
            v = self.at(nd, year)
            rows.append(f"  {nd:22s} {v.mean():9.3f}   "
                        f"[{np.percentile(v, 5):8.3f}, {np.percentile(v, 95):8.3f}]")
        head = (f"Scenario: {self.name}\n"
                f"Year: {year}   draws: {self.series[nodes[0]].shape[1]}\n"
                f"  {'node':22s} {'mean':>9s}   {'90% interval':>20s}")
        tail = self.support.summary()
        return "\n".join([head, *rows, "", tail])
