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
"""PyCausalSim's offline agent: question to plan of tool calls, results to answer.

Deterministic and keyword-driven. Every number in an answer is formatted from a
tool result or from the question, so the grounding check in
:mod:`pycausalsim.chat` can trace it. Used by :class:`pycausalsim.chat.OfflineAgent`.
"""

from __future__ import annotations

import re

TASKS = {
    "overview": "Give me an overview: what is in the data, what causes what, and what is assumed?",
    "effect_audit": ("Is the effect of the main treatment real? Estimate it, compare it with the naive "
                     "correlation, and stress-test it against hidden confounding."),
    "act_first": ("What drives the outcome most, and what should we act on first? Rank the drivers, "
                  "stress-test the top one and find the best settings."),
}

HELP = ("I am the built-in offline agent, so I answer a fixed range of questions. Try: what causes the "
        "outcome; what if we set a variable to a value (or cut it, or raise it); is that effect real; what "
        "drives the outcome most; what are the best settings; assume one variable causes another. Connect a "
        "language model for open-ended questions.")

_GENERIC = {"the", "rate", "type", "time", "value", "total", "num", "count", "level", "score", "index", "var"}
_DOWN = ("cut", "reduce", "lower", "decrease", "less", "drop", "faster", "speed up", "shorten", "down", "fewer")
_UP = ("raise", "increase", "more", "higher", "slower", "add", "grow", "up")


class Plan:
    def __init__(self, steps, write):
        self.steps, self.write = steps, write


def _nice(c) -> str:
    return str(c).replace("_", " ")


def _has(q: str, *words) -> bool:
    return any(w in q for w in words)


def _aliases(c: str):
    low = c.lower()
    return {low, _nice(c).lower()} | {p for p in re.split(r"[_\s]+", low) if len(p) > 2 and p not in _GENERIC}


def _vars_in(q: str, columns):
    """Columns mentioned in the question, in order of first mention."""
    found = []
    for c in columns:
        hits = [m.start() for a in _aliases(c) for m in re.finditer(r"\b" + re.escape(a) + r"\b", q)]
        if hits:
            found.append((min(hits), c))
    return [c for _, c in sorted(found)]


def _nums(q: str):
    return [(float(m.group(1)), bool(m.group(2)))
            for m in re.finditer(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)\s*(%|percent)?", q)]


def _caveats(done):
    seen, out = set(), []
    for d in done:
        for c in d["result"].get("caveats", []) if isinstance(d["result"], dict) else []:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def _cav_block(done):
    cav = _caveats(done)
    return ("\n\n**Cautions from the model**\n" + "\n".join(f"- {c}" for c in cav)) if cav else ""


def _err(done):
    bad = [d for d in done if isinstance(d["result"], dict) and "error" in d["result"]]
    return "\n".join(f"The `{d['name']}` call failed: {d['result']['error']}" for d in bad)


def _result(done, name):
    for d in reversed(done):
        if d["name"] == name and isinstance(d["result"], dict) and "result" in d["result"]:
            return d["result"]["result"]
    return None


def _s(x, d=2):
    return f"{x:+.{d}f}"


# ------------------------------------------------------------------ writers

def _w_effect(e, naive=None):
    lines = [f"Setting **{_nice(e['variable'])}** to {e['value']:.2f} for everyone (it averages "
             f"{e['current_mean']:.2f} today) changes **{_nice(e['target'])}** by **{_s(e['effect'])}** "
             f"(95% interval {_s(e['ci_lower'])} to {_s(e['ci_upper'])})."]
    if e.get("relative_effect") is not None and e.get("baseline_outcome"):
        lines[0] += (f" That is {e['relative_effect']:+.1%} on today's average of {e['baseline_outcome']:.2f}, "
                     f"to about {e['outcome_after']:.2f}.")
    if e["ci_lower"] <= 0 <= e["ci_upper"]:
        lines.append("The interval includes zero, so the data do not establish this effect.")
    if naive is not None:
        n, c = naive["effect"], e["effect"]
        if abs(c) < 1e-12:
            rel = "while the causal estimate is zero"
        elif n * c < 0:
            rel = "which points the opposite way to the causal estimate"
        elif abs(n) > abs(c):
            rel = "which overstates the causal effect"
        else:
            rel = "which understates the causal effect"
        lines.append(f"\nThe naive correlation (a straight line through the raw data, no adjustment) says "
                     f"{_s(n)}, {rel}. The gap is what confounding does to a straight comparison.")
    return "\n".join(lines)


def _w_validate(v, show_effect=True):
    var = _nice(v["variable"])
    rv, p = v.get("robustness_value"), v.get("placebo_p_value")
    passed = v.get("passed", {})
    ok = (rv is None or rv >= 0.1) and all(passed.values())
    lines = [("**Verdict: the effect holds up to these checks.**" if ok else
              "**Verdict: treat this effect with caution.**") +
             f" They test a one standard deviation increase in {var} (a shift of {v['shift']:.2f})" +
             (f", whose effect is {_s(v['effect'])} (95% interval {_s(v['ci_lower'])} to {_s(v['ci_upper'])})."
              if show_effect else ".")]
    if rv is not None:
        lines.append(f"- **Hidden confounding.** Robustness value {rv:.2f}: an unmeasured common cause would "
                     f"have to explain {rv:.0%} of the leftover variation in both {var} and the outcome to erase "
                     "the effect.")
    b = v.get("confounding_bounds") or {}
    if b:
        k = sorted(b, key=float)[min(2, len(b) - 1)]
        lines.append(f"- **If such a confounder existed** with strength {k}, the effect could lie anywhere from "
                     f"{_s(b[k]['lower'])} to {_s(b[k]['upper'])}.")
    if p is not None:
        lines.append(f"- **Placebo.** Shuffling {var} destroys any real effect; the estimate on the shuffled data "
                     f"has p = {p:.2f}" + (", so no false signal." if p > 0.05 else ", a warning sign."))
    ref = v.get("refutations") or {}
    if "random_common_cause" in ref:
        r = ref["random_common_cause"]
        lines.append(f"- **Adding an unrelated variable** " + (
            f"leaves the estimate where it was ({r['original_coef']:.2f}, then {r['new_coef']:.2f}), as it should."
            if r.get("passed") else
            f"moves the estimate from {r['original_coef']:.2f} to {r['new_coef']:.2f}, a warning sign."))
    if "data_subset" in ref:
        lines.append(f"- **Half-samples** agree on the sign {ref['data_subset']['same_sign_share']:.0%} of the time.")
    return "\n".join(lines)


def _w_rank(rows, target):
    lines = [f"What moves **{_nice(target)}** most, for a one standard deviation increase in each variable "
             "(95% intervals):"]
    for i, r in enumerate(rows, 1):
        ci = (f" ({_s(r['ci_lower'])} to {_s(r['ci_upper'])})" if r.get("ci_lower") is not None and r["effect"]
              else "")
        note = " (no path to the outcome in the graph)" if r["effect"] == 0 else ""
        lines.append(f"{i}. **{_nice(r['variable'])}**: {_s(r['effect'])}{ci}, per {r['shift']:.2f} increase{note}")
    return "\n".join(lines)


def _w_policy(p, target):
    goal = "highest" if "max" in p["objective"] else "lowest"
    lines = [f"The settings with the {goal} **{_nice(target)}** on the grid searched:"]
    for k, v in p["settings"].items():
        lines.append(f"- **{_nice(k)}**: {v:.2f} (today it averages {p['current_means'][k]:.2f})")
    lines.append(f"\nExpected {_nice(target)}: {p['expected_outcome']:.2f}, against {p['baseline_outcome']:.2f} "
                 f"today ({_s(p['improvement'])}, {p['relative_improvement']:+.1%}).")
    return "\n".join(lines)


def _w_graph(g, target):
    lines = [f"Direct causes of **{_nice(target)}**: " + (", ".join(_nice(c) for c in g["direct_causes"]) or "none")
             + ".", "\nEvery edge in the graph (cause → effect):"]
    lines += [f"- {_nice(a)} → {_nice(b)}" for a, b in g["edges"]] or ["- none"]
    if g.get("assumed_edges"):
        lines.append("\nNot settled by the data, oriented by declared role and column order: "
                     + ", ".join(g["assumed_edges"]) + ".")
    return "\n".join(lines)


def _w_brief(b, summary=None):
    lines = [f"The data has {b['rows']} rows and {len(b['variables'])} variables. The outcome is "
             f"**{_nice(b['target'])}**" + (f", averaging {b['target_mean']:.2f}" if b.get("target_mean") is not None
                                            else "") + "."]
    if b.get("treatments"):
        lines.append("Variables you can act on: " + ", ".join(_nice(t) for t in b["treatments"]) + ".")
    if b.get("confounders"):
        lines.append("Declared confounders (causes of both the treatments and the outcome): "
                     + ", ".join(_nice(t) for t in b["confounders"]) + ".")
    if b.get("edges") is not None:
        how = "discovered from the data" if b.get("graph_source") == "discovered" else "set by hand"
        lines.append(f"The causal graph was {how}. Direct causes of the outcome: "
                     + (", ".join(_nice(c) for c in b["direct_causes"]) or "none") + ".")
        lines += ["", "Edges (cause → effect):"] + [f"- {_nice(a)} → {_nice(c)}" for a, c in b["edges"]]
    if summary:
        lines += ["", "Ranges in the data:"]
        lines += [f"- {_nice(r['variable'])}: mean {r['mean']:.2f}, from {r['min']:.2f} to {r['max']:.2f}"
                  for r in summary]
    return "\n".join(lines)


# --------------------------------------------------------------------- plans

def _intervention(q, var, sim):
    """{"value": v} or {"change": c} for the variable from the question."""
    nums = _nums(q)
    col = sim.data[var]
    down = _has(q, *_DOWN)
    for v, pct in nums:
        if pct:
            c = abs(v) / 100 * float(col.mean())
            return {"change": round(-c if down else c, 6)}
    if nums:
        v = nums[0][0]
        if re.search(r"\bto\s+[-+]?\d", q) and not re.search(r"\bby\s+[-+]?\d", q):
            return {"value": v}
        if re.search(r"\bby\s+[-+]?\d", q) or _has(q, "cut", "reduce", "shorten", "speed up", "faster"):
            return {"change": -abs(v) if down else abs(v)}
        return {"value": v}
    sd = float(col.std())
    return {"change": round(-sd if down else sd, 6)}


def make_plan(question: str, box) -> Plan:
    sim = box.sim
    q = question.lower()
    target = sim.target
    cols = [c for c in sim.data.columns if c != target]
    mentioned = [c for c in _vars_in(q, cols)]
    var = mentioned[0] if mentioned else None
    main = (list(sim.treatment_vars) or cols)[0]

    def fail_or(done, text):
        return _err(done) or text

    # overview
    if _has(q, "overview", "summary", "summar", "what is in the data", "describe the data", "what's in the data"):
        return Plan([("brief", {}), ("data_summary", {})],
                    lambda done: fail_or(done, _w_brief(_result(done, "brief"), _result(done, "data_summary"))
                                         + _cav_block(done)))

    # change the graph: "assume X causes Y", "X does not cause Y"
    m_add = re.search(r"(assume|suppose|add an edge|add edge|what if)", q) and re.search(r"\bcauses?\b|->|→", q)
    m_del = re.search(r"(does not|doesn't|do not|don't) (cause|affect)|remove the edge|remove edge|drop the edge", q)
    if (m_add or m_del) and len(_vars_in(q, list(sim.data.columns))) >= 2 and sim.graph is not None:
        a, b = _vars_in(q, list(sim.data.columns))[:2]
        g = {c: list(p) for c, p in sim.graph.items()}
        g.setdefault(b, [])
        if m_del:
            g[b] = [p for p in g[b] if p != a]
            g[a] = [p for p in g.get(a, []) if p != b]
        else:
            g[a] = [p for p in g.get(a, []) if p != b]
            if a not in g[b]:
                g[b].append(a)
        verb = "no longer causes" if m_del else "causes"

        def write(done):
            if done and "declined" in str(done[0]["result"].get("error", "")):
                return ("Understood. I left the graph as it was, so every estimate still rests on the "
                        "current graph.")
            if _err(done):
                return _err(done)
            return (f"Done: the graph now says **{_nice(a)}** {verb} **{_nice(b)}**. Every later estimate uses "
                    "this graph.\n\n" + _w_graph(_result(done, "get_graph"), target) + _cav_block(done))
        return Plan([("set_graph", {"graph": g}), ("get_graph", {})], write)

    # agent task: drivers, then act
    if _has(q, "drive", "driver", "matter", "important", "rank", "biggest", "act on", "lever"):
        def top(done):
            rows = _result(done, "rank_drivers") or []
            nonzero = [r for r in rows if r["effect"]]
            return {"variable": (nonzero or rows or [{"variable": main}])[0]["variable"]}
        full = _has(q, "act", "first", "stress", "best setting", "what should")
        steps = [("rank_drivers", {})] + ([("validate", top), ("optimize_policy", {"objective": "maximize"})]
                                          if full else [])

        def write(done):
            if _err(done):
                return _err(done)
            text = _w_rank(_result(done, "rank_drivers"), target)
            if full:
                v = _result(done, "validate")
                text += (f"\n\n**Stress test of the top driver, {_nice(v['variable'])}**\n" + _w_validate(v) +
                         "\n\n**Best settings of the variables you can act on**\n" +
                         _w_policy(_result(done, "optimize_policy"), target))
            return text + _cav_block(done)
        return Plan(steps, write)

    # agent task: is the effect real
    if _has(q, "real", "trust", "robust", "confound", "validat", "believe", "really", "stress", "spurious",
            "sure about"):
        v = var or main
        iv = _intervention(q, v, sim)
        args = {"variable": v, **iv}

        def write(done):
            if _err(done):
                return _err(done)
            return (_w_effect(_result(done, "simulate_intervention"), _result(done, "naive_effect")) +
                    "\n\n" + _w_validate(_result(done, "validate"), show_effect=False) + _cav_block(done))
        return Plan([("simulate_intervention", args), ("naive_effect", dict(args)),
                     ("validate", {"variable": v})], write)

    # best settings
    if _has(q, "optimi", "best setting", "maximi", "minimi", "best value", "best price", "ideal", "optimal"):
        obj = "minimize" if _has(q, "minimi", "lowest", "reduce the outcome") else "maximize"
        return Plan([("optimize_policy", {"objective": obj})],
                    lambda done: fail_or(done, _w_policy(_result(done, "optimize_policy"), target)
                                         + _cav_block(done)))

    # what if
    if var and (_nums(q) or _has(q, "what if", "if we", "if i", "set ", "effect of", "impact of", *_DOWN, *_UP)):
        args = {"variable": var, **_intervention(q, var, sim)}
        return Plan([("simulate_intervention", args), ("naive_effect", dict(args))],
                    lambda done: fail_or(done, _w_effect(_result(done, "simulate_intervention"),
                                                         _result(done, "naive_effect")) + _cav_block(done)))

    # graph
    if _has(q, "graph", "cause", "structure", "depend", "affect", "edge", "influence", "what leads"):
        return Plan([("get_graph", {})],
                    lambda done: fail_or(done, _w_graph(_result(done, "get_graph"), target) + _cav_block(done)))

    def write(done):
        return HELP + "\n\nHere is the overview in the meantime.\n\n" + fail_or(
            done, _w_brief(_result(done, "brief")) + _cav_block(done))
    return Plan([("brief", {})], write)


def cli_toolbox(argv):
    """Build a :class:`pycausalsim.ai.ToolBox` from command line options (``python -m pycausalsim.chat``)."""
    import argparse

    import pandas as pd

    from .ai import ToolBox
    from .simulator import CausalSimulator

    ap = argparse.ArgumentParser(prog="python -m pycausalsim.chat",
                                 description="Chat with a PyCausalSim causal model built from a CSV file.")
    ap.add_argument("--data", required=True, help="CSV with one row per unit")
    ap.add_argument("--target", required=True)
    ap.add_argument("--treatments", default="", help="comma-separated variables you can act on")
    ap.add_argument("--confounders", default="", help="comma-separated known common causes")
    ap.add_argument("--method", default="pc", help="discovery method (default pc)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    split = lambda s: [x for x in s.split(",") if x]
    df = pd.read_csv(a.data)
    sim = CausalSimulator(df, a.target, treatment_vars=split(a.treatments), confounders=split(a.confounders),
                          random_state=a.seed)
    sim.discover_graph(method=a.method)
    return ToolBox(sim)
