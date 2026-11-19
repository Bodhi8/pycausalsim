#   Copyright 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Grounded access to a causal simulator for language models and agents.

Nothing here calls a language model or needs a network. It gives one what it
needs to talk about a causal model without inventing numbers:

``brief(sim)``
    the data, the graph and every reason for caution as plain data, with the
    warnings decided by fixed rules rather than by prose.

``ToolBox(sim)``
    the simulator's methods as tools with JSON schemas, in the layouts chat
    model APIs accept, and a dispatcher that returns JSON-ready results. Each
    result carries the caveats that apply to it.

``SYSTEM_PROMPT``
    instructions that keep an assistant to the tool results.

:mod:`pycausalsim.chat` runs a conversation over these tools, with a built-in
offline agent or a language model of your choice.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

__all__ = ["brief", "ToolBox", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = (
    "You are helping someone read a PyCausalSim causal model. Every number you state must come from a "
    "tool result in this conversation; if you have not called the tool, call it or say you do not have the "
    "figure. Give effects with their 95% interval and say plainly when the interval includes zero. When "
    "someone asks whether something works or causes the outcome, compare the causal estimate with the "
    "naive correlation. Repeat any item in a result's `caveats` that bears on the question. The causal "
    "graph comes from the data and the declared roles; edges listed as assumed are not settled by the data, "
    "and every estimate holds only if the graph has no hidden common causes. Never call an estimate proven. "
    "Changing the graph changes every later estimate, so propose it and let the user approve it."
)


def _clean(x):
    """Make pandas and numpy values JSON-ready."""
    if isinstance(x, pd.DataFrame):
        return [_clean(r) for r in x.to_dict("records")]
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [_clean(v) for v in list(x)]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def _edges(graph):
    return [[p, c] for c, ps in (graph or {}).items() for p in ps]


def brief(sim, graph_source: str = "discovered") -> dict:
    """The data, the causal graph and the cautions that apply, as plain data.

    ``graph_source`` says where the graph came from ("discovered" from the
    data, or "set" by a person); it decides one of the cautions.
    """
    caveats = []
    g = sim.graph
    numeric = [c for c in sim.data.columns if pd.api.types.is_numeric_dtype(sim.data[c])]
    out = {"target": sim.target, "rows": len(sim.data), "variables": list(sim.data.columns),
           "treatments": list(sim.treatment_vars), "confounders": list(sim.confounders),
           "target_mean": float(sim.data[sim.target].mean()) if sim.target in numeric else None,
           "graph_source": graph_source if g else None}
    if not g:
        caveats.append("No causal graph yet: call discover_graph() or set_graph() before estimating effects.")
        out["caveats"] = caveats
        return _clean(out)
    edges = _edges(g)
    assumed = list(getattr(sim, "_oriented_edges", []) or []) if graph_source == "discovered" else []
    causes = sorted(sim._ancestors(sim.target))
    out.update({"edges": edges, "n_edges": len(edges), "direct_causes": list(g.get(sim.target, [])),
                "causes_of_target": causes, "assumed_edges": assumed})
    if assumed:
        caveats.append("These edge directions were not settled by the data and were set from the declared "
                       f"roles and column order: {', '.join(assumed)}.")
    if graph_source == "discovered":
        caveats.append("The graph was discovered from observational data. Check each edge against what you "
                       "know; a hidden common cause the data cannot see would change every estimate.")
    for t in sim.treatment_vars:
        if t not in causes:
            caveats.append(f"{t} has no path to {sim.target} in the graph, so its effect is zero by assumption.")
    out["caveats"] = caveats
    return _clean(out)


_VAR = {"type": "string", "description": "A column of the data, exactly as named."}

_TOOLS = [
    ("brief", "The data, the causal graph (edges as [cause, effect]), what causes the target, and every "
              "caution that applies.",
     {"type": "object", "properties": {}}),
    ("data_summary", "Mean, standard deviation, minimum and maximum of each numeric column.",
     {"type": "object", "properties": {}}),
    ("get_graph", "The causal graph: each variable's direct causes, the edge list, and which variables "
                  "cause the target directly or through others.",
     {"type": "object", "properties": {}}),
    ("simulate_intervention", "Causal effect on the target of setting a variable to a value for everyone, "
                              "E[target | do(variable = value)] minus today's average, with a 95% bootstrap "
                              "interval. Give `value`, or `change` to add to today's average.",
     {"type": "object", "properties": {
         "variable": _VAR, "value": {"type": "number"},
         "change": {"type": "number", "description": "Amount added to the variable's current average."},
         "method": {"type": "string", "enum": ["backdoor", "scm"],
                    "description": "backdoor: linear model with the adjustment set from the graph (fast). "
                                   "scm: tree-based mechanisms propagated through the graph."},
         "n_bootstrap": {"type": "integer", "minimum": 2, "maximum": 200}},
      "required": ["variable"]}),
    ("naive_effect", "What the raw correlation says: a straight line of the target on the variable alone, "
                     "with no adjustment. Use it to show how far correlation is from the causal effect.",
     {"type": "object", "properties": {"variable": _VAR, "value": {"type": "number"}, "change": {"type": "number"}},
      "required": ["variable"]}),
    ("rank_drivers", "Rank variables by the causal effect on the target of a one standard deviation increase.",
     {"type": "object", "properties": {"n_bootstrap": {"type": "integer", "minimum": 2, "maximum": 100}}}),
    ("validate", "Stress-test one variable's effect: robustness value against hidden confounding, effect "
                 "bounds under confounders of given strength, a placebo test and refutation checks.",
     {"type": "object", "properties": {"variable": _VAR,
                                       "n_bootstrap": {"type": "integer", "minimum": 2, "maximum": 100}},
      "required": ["variable"]}),
    ("optimize_policy", "Settings of the treatment variables that maximize or minimize the target, searched "
                        "on a grid within each variable's observed range unless constraints are given.",
     {"type": "object", "properties": {
         "objective": {"type": "string", "enum": ["maximize", "minimize"]},
         "constraints": {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "number"},
                                                                     "minItems": 2, "maxItems": 2},
                         "description": "{variable: [min, max]}"},
         "grid_points": {"type": "integer", "minimum": 2, "maximum": 9}}}),
    ("set_graph", "Replace the causal graph. Changes every later estimate, so the user must approve it.",
     {"type": "object", "properties": {"graph": {
         "type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}},
         "description": "{variable: [its direct causes]} for every variable."}}, "required": ["graph"]}),
]


class ToolBox:
    """A :class:`pycausalsim.CausalSimulator` as callable tools.

    >>> sim = CausalSimulator(df, target="conversion_rate", treatment_vars=["page_load_time"])
    >>> sim.discover_graph()
    >>> box = ToolBox(sim)
    >>> box.specs("tool_use")                      # pass as the tool list of a chat model API
    >>> box.call("simulate_intervention", {"variable": "page_load_time", "value": 2.0})

    ``set_graph`` is the one tool that changes the model; it is listed in
    ``mutating`` so :class:`pycausalsim.chat.Chat` asks before running it.
    """

    system_prompt = SYSTEM_PROMPT
    mutating = frozenset({"set_graph"})

    def __init__(self, sim, graph_source: str = "discovered") -> None:
        self.sim = sim
        self.graph_source = graph_source

    def specs(self, format: str = "tool_use") -> list:
        """Tool definitions: "tool_use" (``input_schema``), "function" or "mcp" (``inputSchema``)."""
        if format == "tool_use":
            return [{"name": n, "description": d, "input_schema": s} for n, d, s in _TOOLS]
        if format == "function":
            return [{"type": "function", "function": {"name": n, "description": d, "parameters": s}}
                    for n, d, s in _TOOLS]
        if format == "mcp":
            return [{"name": n, "description": d, "inputSchema": s} for n, d, s in _TOOLS]
        raise ValueError('format must be "tool_use", "function" or "mcp"')

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Run one tool. Always returns a dict; errors come back as ``{"error": ...}``."""
        a = dict(arguments or {})
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name!r}", "tools": [t[0] for t in _TOOLS]}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result, caveats = fn(a)
        except (KeyError, ValueError, TypeError, RuntimeError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        for w in caught:
            msg = str(w.message).split("\n")[0]
            if issubclass(w.category, UserWarning) and msg not in caveats:
                caveats.append(msg)
        return _clean({"tool": name, "result": result, "caveats": caveats,
                       "units": f"effects are in the units of {self.sim.target}"})

    def call_json(self, name: str, arguments: dict | None = None) -> str:
        return json.dumps(self.call(name, arguments))

    # helpers --------------------------------------------------------------
    def _var(self, v):
        if v not in self.sim.data.columns:
            raise ValueError(f"unknown variable {v!r}; variables are {list(self.sim.data.columns)}")
        if v == self.sim.target:
            raise ValueError("that is the target; pick a variable that might cause it")
        return v

    def _value(self, var, a):
        mean = float(self.sim.data[var].mean())
        if a.get("value") is not None:
            return float(a["value"]), mean
        if a.get("change") is not None:
            return mean + float(a["change"]), mean
        raise ValueError("give `value` or `change`")

    def _need_graph(self):
        if self.sim.graph is None:
            raise RuntimeError("no causal graph yet; discover or set one first")

    def _assumed_on(self, var):
        assumed = getattr(self.sim, "_oriented_edges", []) or []
        hits = [e for e in assumed if var in e.replace(" ", "").split("->")]
        return ([f"The direction of {', '.join(hits)} was assumed, not settled by the data; the estimate "
                 "depends on it."] if hits and self.graph_source == "discovered" else [])

    def describe_call(self, name: str, arguments: dict) -> str | None:
        """A plain-language summary of what a call that changes the model would do, for approval prompts."""
        if name != "set_graph" or not isinstance((arguments or {}).get("graph"), dict):
            return None
        old = {tuple(e) for e in _edges(self.sim.graph)}
        new = {(p, c) for c, ps in arguments["graph"].items() for p in (ps or [])}
        fmt = lambda es: ", ".join(f"{a} -> {b}" for a, b in sorted(es))
        parts = ([f"add {fmt(new - old)}"] if new - old else []) + ([f"remove {fmt(old - new)}"] if old - new else [])
        return "; ".join(parts) or "no change to the edges"

    # tools ----------------------------------------------------------------
    def _t_brief(self, a):
        b = brief(self.sim, self.graph_source)
        return b, list(b["caveats"])

    def _t_data_summary(self, a):
        num = self.sim.data.select_dtypes("number")
        rows = [{"variable": c, "mean": num[c].mean(), "std": num[c].std(), "min": num[c].min(),
                 "max": num[c].max()} for c in num.columns]
        return rows, []

    def _t_get_graph(self, a):
        self._need_graph()
        sim = self.sim
        g = sim.graph
        assumed = list(getattr(sim, "_oriented_edges", []) or []) if self.graph_source == "discovered" else []
        return ({"parents": g, "edges": _edges(g), "direct_causes": list(g.get(sim.target, [])),
                 "causes_of_target": sorted(sim._ancestors(sim.target)), "assumed_edges": assumed,
                 "source": self.graph_source},
                (["Edges listed in assumed_edges were oriented by declared role and column order."]
                 if assumed else []))

    def _t_simulate_intervention(self, a):
        self._need_graph()
        sim = self.sim
        var = self._var(a["variable"])
        value, mean = self._value(var, a)
        method = a.get("method", "backdoor")
        n_boot = int(min(200, max(2, a.get("n_bootstrap", 50))))
        e = sim.simulate_intervention(var, value, method=method, n_bootstrap=n_boot)
        base = e.baseline_outcome
        res = {"variable": var, "value": value, "current_mean": mean, "target": sim.target,
               "effect": e.point_estimate, "ci_lower": e.ci_lower, "ci_upper": e.ci_upper, "p_value": e.p_value,
               "baseline_outcome": base, "outcome_after": base + e.point_estimate if base is not None else None,
               "relative_effect": (e.point_estimate / base) if base else None, "method": method,
               "n_bootstrap": n_boot}
        cav = []
        if e.ci_lower <= 0 <= e.ci_upper:
            cav.append("The 95% interval includes zero: the data do not establish an effect of this size.")
        if sim.target not in sim._descendants(var):
            cav.append(f"{sim.target} is not downstream of {var} in the graph, so the effect is zero by assumption.")
        if method == "backdoor":
            adj = sim._get_adjustment_set(var)
            cav.append("Backdoor estimate from a linear model adjusting for "
                       + (", ".join(adj) if adj else "nothing (no adjustment set in the graph)") + ".")
        return res, cav + self._assumed_on(var)

    def _t_naive_effect(self, a):
        sim = self.sim
        var = self._var(a["variable"])
        value, mean = self._value(var, a)
        d = sim.data[[var, sim.target]].dropna()
        slope = float(np.polyfit(d[var].astype(float), d[sim.target].astype(float), 1)[0])
        return ({"variable": var, "value": value, "current_mean": mean, "slope": slope,
                 "effect": slope * (value - mean)},
                ["This is the raw correlation with no adjustment, not a causal estimate."])

    def _t_rank_drivers(self, a):
        self._need_graph()
        n_boot = int(min(100, max(2, a.get("n_bootstrap", 20))))
        r = self.sim.rank_drivers(n_bootstrap=n_boot)
        rows = []
        for var, eff in r.drivers:
            e = r.effects.get(var)
            rows.append({"variable": var, "effect": eff, "ci_lower": e.ci_lower if e else None,
                         "ci_upper": e.ci_upper if e else None, "shift": float(self.sim.data[var].std())})
        cav = ["Each effect is for a one standard deviation increase in that variable (the `shift`), estimated "
               "with tree-based mechanisms propagated through the graph."]
        zero = [x["variable"] for x in rows if x["ci_lower"] is not None and x["ci_lower"] <= 0 <= x["ci_upper"]
                and x["effect"] != 0]
        if zero:
            cav.append(f"Intervals that include zero: {', '.join(zero)}.")
        return rows, cav

    def _t_validate(self, a):
        self._need_graph()
        var = self._var(a["variable"])
        n_boot = int(min(100, max(2, a.get("n_bootstrap", 20))))
        s = self.sim.validate(var, n_bootstrap=n_boot)
        ref = s.refutation_results or {}
        bounds = {k: {"lower": lo, "upper": hi} for k, (lo, hi) in (s.confounding_bounds or {}).items()}
        e = s.original_effect
        res = {"variable": var, "shift": float(self.sim.data[var].std()),
               "effect": e.point_estimate, "ci_lower": e.ci_lower, "ci_upper": e.ci_upper,
               "robustness_value": s.robustness_value, "confounding_bounds": bounds,
               "placebo_p_value": (s.placebo_results or {}).get("placebo_treatment"),
               "subset_stability": (s.placebo_results or {}).get("subset_stability"),
               "refutations": ref,
               "passed": {k: bool(v.get("passed")) for k, v in ref.items() if isinstance(v, dict)}}
        cav = ["The robustness value, bounds and refutations use a linear outcome model with the backdoor "
               "adjustment set. The effect reported here is for a one standard deviation increase, from the "
               "tree-based structural model, so it can differ from a backdoor estimate of the same change."]
        rv = s.robustness_value
        if rv is not None and rv < 0.1:
            cav.append(f"Fragile: a hidden confounder explaining {rv:.0%} of the leftover variation in both "
                       f"{var} and the target would erase the effect.")
        crossing = [k for k, b in bounds.items() if b["lower"] <= 0 <= b["upper"]]
        if crossing:
            cav.append(f"A hidden confounder of strength {crossing[0]} or more could bring the effect to zero.")
        failed = [k for k, ok in res["passed"].items() if not ok]
        if failed:
            cav.append(f"Failed checks: {', '.join(failed)}.")
        return res, cav + self._assumed_on(var)

    def _t_optimize_policy(self, a):
        self._need_graph()
        sim = self.sim
        cons = {self._var(k): tuple(float(x) for x in v) for k, v in (a.get("constraints") or {}).items()}
        gp = int(min(9, max(2, a.get("grid_points", 5))))
        p = sim.optimize_policy(objective=a.get("objective", "maximize"), constraints=cons or None, grid_points=gp)
        res = {"objective": p.objective, "settings": p.settings, "expected_outcome": p.expected_outcome,
               "baseline_outcome": p.baseline_outcome, "improvement": p.improvement,
               "relative_improvement": p.improvement / p.baseline_outcome if p.baseline_outcome else None,
               "search_ranges": {k: list(v) for k, v in p.constraints.items()},
               "current_means": {k: float(sim.data[k].mean()) for k in p.settings}, "grid_points": gp}
        edge = [k for k, v in p.settings.items()
                if np.isclose(v, p.constraints[k][0]) or np.isclose(v, p.constraints[k][1])]
        cav = [f"Searched {gp} values per variable, by default within the range seen in the data."]
        if edge:
            cav.append(f"{', '.join(edge)} landed at the edge of the range searched; the best setting may lie "
                       "beyond it, where the data cannot say.")
        cav.append("Setting several variables at once assumes they can be changed independently of each other.")
        cav.append(f"The search weighs {sim.target} alone; it ignores costs, such as revenue lost to a lower price.")
        return res, cav

    def _t_set_graph(self, a):
        g = a["graph"]
        if not isinstance(g, dict):
            raise TypeError("graph must be {variable: [direct causes]}")
        full = {c: list(g.get(c, [])) for c in self.sim.data.columns}
        self.sim.set_graph(full)
        self.graph_source = "set"
        return ({"parents": self.sim.graph, "edges": _edges(self.sim.graph)},
                ["The graph is now the one you set; later estimates rest on it, not on discovery."])
