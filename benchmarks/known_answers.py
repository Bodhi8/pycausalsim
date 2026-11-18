#   Copyright 2025 - 2026 Brian Curry / Vector1 Research
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
"""Grade PyCausalSim against problems whose answers are known.

Every dataset here is simulated from a stated model, so the right answer is
known before any method runs. The script grades causal discovery on graphs with
a known structure and marketing attribution on channels with known effects.

It uses only the public API, so it runs against any installed version:

    python benchmarks/known_answers.py run --out results-0.7.0.json
    python benchmarks/known_answers.py report results-0.6.0.json results-0.7.0.json

Run each version in its own environment (pip install pycausalsim==0.6.0 in one,
the current release in another) and from a directory where the installed package,
not a source checkout, is the one imported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Graph helpers (independent of the package under test)
# ---------------------------------------------------------------------------


def edges_of(graph):
    """{child: [parents]} -> set of (parent, child)."""
    return {(p, c) for c, ps in graph.items() for p in ps if p != c}


def self_loops(graph):
    return sum(1 for c, ps in graph.items() if c in ps)


def cpdag(names, parents):
    """Edges of the completed partially directed graph of a DAG: compelled edges
    once, reversible edges in both directions. V-structures, then Meek's rules."""
    dag = {(p, c) for c, ps in parents.items() for p in ps}
    adj = {frozenset(e) for e in dag}
    directed = set()
    for c in names:
        for a, b in combinations(sorted(parents[c]), 2):
            if frozenset((a, b)) not in adj:
                directed |= {(a, c), (b, c)}
    undirected = {e for e in adj if not any(tuple(x) in directed for x in [tuple(e), tuple(e)[::-1]])}

    def is_adj(x, y):
        return frozenset((x, y)) in adj

    changed = True
    while changed:
        changed = False
        for e in list(undirected):
            x, y = tuple(e)
            for u, v in [(x, y), (y, x)]:
                # R1: w -> u - v, w not adjacent to v  => u -> v
                r1 = any((w, u) in directed and not is_adj(w, v) for w in names if w not in (u, v))
                # R2: u -> w -> v and u - v  => u -> v
                r2 = any((u, w) in directed and (w, v) in directed for w in names if w not in (u, v))
                # R3: u - w1 -> v, u - w2 -> v, w1, w2 not adjacent  => u -> v
                ws = [w for w in names if w not in (u, v) and frozenset((u, w)) in undirected
                      and (w, v) in directed]
                r3 = any(not is_adj(a, b) for a, b in combinations(ws, 2))
                if r1 or r2 or r3:
                    directed.add((u, v))
                    undirected.discard(e)
                    changed = True
                    break
            if changed:
                break
    out = set(directed)
    for e in undirected:
        x, y = tuple(e)
        out |= {(x, y), (y, x)}
    return out


def grade(got, truth_dag, truth_class):
    """Compare a learned graph with the truth."""
    skel_got = {frozenset(e) for e in got}
    skel_true = {frozenset(e) for e in truth_dag}
    tp = len(skel_got & skel_true)
    arrows = {e for e in got if (e[1], e[0]) not in got}
    return {
        "exact": got == truth_class,
        "missing_edges": len(skel_true - skel_got),
        "extra_edges": len(skel_got - skel_true),
        "adjacency_f1": (2 * tp / (len(skel_got) + len(skel_true))) if (skel_got or skel_true) else 1.0,
        "arrows": len(arrows),
        "wrong_arrows": sum(1 for a in arrows if (a[1], a[0]) in truth_dag),
        "unsupported_arrows": sum(1 for a in arrows if a in truth_dag and (a[1], a[0]) in truth_class),
        "two_way_edges": sum(1 for e in got if (e[1], e[0]) in got) // 2,
    }


# ---------------------------------------------------------------------------
# Data with known answers
# ---------------------------------------------------------------------------


def named_graphs(n=3000, seed=0):
    rng = np.random.default_rng(seed)

    def e():
        return rng.normal(0, 1, n)

    out = {}
    a = e(); b = 0.8 * a + e(); c = 0.8 * b + e()                               # noqa: E702
    out["chain a->b->c"] = (pd.DataFrame({"a": a, "b": b, "c": c}), {"a": [], "b": ["a"], "c": ["b"]})
    a = e(); b = 0.8 * a + e(); c = 0.8 * a + e()                               # noqa: E702
    out["fork b<-a->c"] = (pd.DataFrame({"a": a, "b": b, "c": c}), {"a": [], "b": ["a"], "c": ["a"]})
    a = e(); b = e(); c = 0.8 * a + 0.8 * b + e()                               # noqa: E702
    out["collider a->c<-b"] = (pd.DataFrame({"a": a, "b": b, "c": c}), {"a": [], "b": [], "c": ["a", "b"]})
    a = e(); b = 0.8 * a + e(); c = 0.8 * a + e(); d = 0.7 * b + 0.7 * c + e()  # noqa: E702
    out["diamond"] = (pd.DataFrame({"a": a, "b": b, "c": c, "d": d}),
                      {"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]})
    return out


def random_graphs(count=12, d=5, p=0.4, n=5000, noise="gauss"):
    for s in range(count):
        rng = np.random.default_rng(1000 + s)
        order = rng.permutation(d)
        names = [f"v{i}" for i in range(d)]
        parents = {v: [] for v in names}
        for i in range(d):
            for j in range(i + 1, d):
                if rng.random() < p:
                    parents[names[order[j]]].append(names[order[i]])
        X = {}
        for k in order:
            v = names[k]
            eps = rng.normal(0, 1, n) if noise == "gauss" else rng.uniform(-1.73, 1.73, n)
            X[v] = sum(rng.uniform(0.7, 1.2) * X[q] for q in sorted(parents[v])) + eps
        yield names, parents, pd.DataFrame(X)[names]


CHANNELS = ["email", "search", "display", "social"]
TRUE_EFFECT = {"email": 0.10, "search": 0.04, "display": 0.0, "social": -0.03}


def attribution_data(n=40000, seed=0):
    """Conversion is additive in exposure. Display does nothing but is shown mostly
    to people who also saw search. Social lowers conversion."""
    rng = np.random.default_rng(seed)
    search = (rng.random(n) < 0.35).astype(int)
    display = (rng.random(n) < 0.15 + 0.5 * search).astype(int)
    email = (rng.random(n) < 0.4).astype(int)
    social = (rng.random(n) < 0.3).astype(int)
    p = 0.05 + 0.10 * email + 0.04 * search + 0.0 * display - 0.03 * social
    df = pd.DataFrame({"email": email, "search": search, "display": display, "social": social,
                       "converted": (rng.random(n) < p).astype(int)})
    reach = {c: float(df[c].mean()) for c in CHANNELS}
    contrib = {c: TRUE_EFFECT[c] * reach[c] for c in CHANNELS}
    positive = sum(v for v in contrib.values() if v > 0)
    truth_share = {c: (contrib[c] / positive if contrib[c] > 0 else 0.0) for c in CHANNELS}
    return df, truth_share


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run():
    import pycausalsim
    from pycausalsim import MarketingAttribution
    from pycausalsim.discovery import CausalDiscovery

    result = {"version": pycausalsim.__version__, "python": sys.version.split()[0], "discovery": {}, "attribution": {}}
    warnings.simplefilter("ignore")

    def discover(df, method, **kw):
        t = time.perf_counter()
        try:
            g = CausalDiscovery(random_state=0).discover(df, method=method, **kw)
        except Exception as exc:  # report, do not hide
            return None, repr(exc), time.perf_counter() - t
        return g, None, time.perf_counter() - t

    # Equivalence-class methods on Gaussian data: graded against the class the data identify.
    for method, kw in [("pc", {"alpha": 0.01}), ("ges", {}), ("hybrid", {"alpha": 0.01})]:
        rows = {}
        for name, (df, parents) in named_graphs().items():
            g, err, secs = discover(df, method, **kw)
            truth_dag = edges_of(parents)
            rows[name] = {"error": err} if err else {
                **grade(edges_of(g), truth_dag, cpdag(list(parents), parents)),
                "graph": {k: sorted(v) for k, v in g.items()}, "seconds": round(secs, 2)}
        agg = {"cases": 0, "exact": 0, "wrong_arrows": 0, "unsupported_arrows": 0, "arrows": 0,
               "missing_edges": 0, "extra_edges": 0, "errors": 0}
        for names, parents, df in random_graphs():
            g, err, _ = discover(df, method, **kw)
            agg["cases"] += 1
            if err:
                agg["errors"] += 1
                continue
            r = grade(edges_of(g), edges_of(parents), cpdag(names, parents))
            for k in ["exact", "wrong_arrows", "unsupported_arrows", "arrows", "missing_edges", "extra_edges"]:
                agg[k] += int(r[k])
        result["discovery"][method] = {"named": rows, "random_5_node": agg}

    # Full-graph methods on non-Gaussian data, where direction is identified: graded against the DAG.
    for method in ["lingam", "notears"]:
        agg = {"cases": 0, "exact": 0, "wrong_arrows": 0, "missing_edges": 0, "extra_edges": 0,
               "self_loops": 0, "two_way_edges": 0, "errors": 0}
        for names, parents, df in random_graphs(noise="uniform"):
            g, err, _ = discover(df, method)
            agg["cases"] += 1
            if err:
                agg["errors"] += 1
                continue
            truth = edges_of(parents)
            r = grade(edges_of(g), truth, truth)
            agg["exact"] += int(r["exact"])
            for k in ["wrong_arrows", "missing_edges", "extra_edges", "two_way_edges"]:
                agg[k] += r[k]
            agg["self_loops"] += self_loops(g)
        result["discovery"][method] = {"random_5_node_uniform": agg}

    df, truth_share = attribution_data()
    result["attribution"]["truth_share"] = truth_share
    for method in ["shapley", "logistic", "linear", "uplift"]:
        try:
            attr = MarketingAttribution(df, "converted", CHANNELS, random_state=0).fit(method)
            shares = {k: float(v) for k, v in attr.get_attribution().items()}
            result["attribution"][method] = {
                "share": shares,
                "error_points": round(100 * sum(abs(shares.get(c, 0) - truth_share[c]) for c in CHANNELS) / 2, 1),
            }
        except Exception as exc:
            result["attribution"][method] = {"error": repr(exc)}
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(paths):
    runs = [json.load(open(p)) for p in paths]
    v = [r["version"] for r in runs]
    lines = [f"# Known-answer benchmark: PyCausalSim {' vs '.join(v)}", ""]

    lines += ["## Named graphs (Gaussian, n = 3,000): learned graph", "",
              "| Graph | Method | " + " | ".join(v) + " |", "|---|---|" + "---|" * len(v)]
    for name in runs[0]["discovery"]["pc"]["named"]:
        for m in ["pc", "ges", "hybrid"]:
            cells = []
            for r in runs:
                row = r["discovery"][m]["named"][name]
                if "error" in row:
                    cells.append("error")
                    continue
                g = row["graph"]
                es = edges_of(g)
                parts = []
                for a, b in sorted({tuple(sorted(e)) for e in es}):
                    parts.append(f"{a}-{b}" if (a, b) in es and (b, a) in es else (f"{a}->{b}" if (a, b) in es else f"{b}->{a}"))
                if row["exact"]:
                    mark = "correct"
                else:
                    bits = [f"{row[k]} {label}" for k, label in [
                        ("wrong_arrows", "wrong arrow"), ("unsupported_arrows", "unsupported arrow"),
                        ("extra_edges", "extra edge"), ("missing_edges", "missing edge")] if row[k]]
                    bits = [b + ("s" if not b.startswith("1 ") else "") for b in bits]
                    mark = ", ".join(bits) or "undirected where the data fix a direction"
                cells.append(f"{', '.join(parts) or 'none'} ({mark})")
            lines.append(f"| {name} | {m.upper()} | " + " | ".join(cells) + " |")

    lines += ["", "## 12 random 5-node graphs (Gaussian, n = 5,000)", "",
              "| Method | Measure | " + " | ".join(v) + " |", "|---|---|" + "---|" * len(v)]
    for m in ["pc", "ges", "hybrid"]:
        for k, label in [("exact", "Equivalence class exactly right (of 12)"),
                         ("wrong_arrows", "Arrows pointing the wrong way"),
                         ("unsupported_arrows", "Arrows the data cannot support"),
                         ("missing_edges", "Missing edges"), ("extra_edges", "Extra edges")]:
            lines.append(f"| {m.upper()} | {label} | " + " | ".join(str(r["discovery"][m]["random_5_node"][k]) for r in runs) + " |")

    lines += ["", "## 12 random 5-node graphs (uniform noise, n = 5,000)", "",
              "| Method | Measure | " + " | ".join(v) + " |", "|---|---|" + "---|" * len(v)]
    for m in ["lingam", "notears"]:
        for k, label in [("exact", "Graph exactly right (of 12)"), ("wrong_arrows", "Arrows pointing the wrong way"),
                         ("missing_edges", "Missing edges"), ("extra_edges", "Extra edges"),
                         ("self_loops", "Self-loops"), ("errors", "Errors")]:
            lines.append(f"| {m.upper()} | {label} | " + " | ".join(str(r["discovery"][m]["random_5_node_uniform"][k]) for r in runs) + " |")

    truth = runs[0]["attribution"]["truth_share"]
    lines += ["", "## Attribution, 40,000 simulated users, four channels with known effects", "",
              "True effects on conversion: email +10 points, search +4, display 0, social -3.", "",
              "| Method | Version | " + " | ".join(CHANNELS) + " | Credit misallocated |", "|---|---|" + "---|" * (len(CHANNELS) + 1)]
    lines.append("| Truth | | " + " | ".join(f"{100 * truth[c]:.0f}%" for c in CHANNELS) + " | 0 points |")
    for m in ["shapley", "logistic", "linear", "uplift"]:
        for r in runs:
            a = r["attribution"][m]
            if "error" in a:
                lines.append(f"| {m} | {r['version']} | error |" + " |" * len(CHANNELS))
                continue
            lines.append(f"| {m} | {r['version']} | " + " | ".join(f"{100 * a['share'].get(c, 0):.0f}%" for c in CHANNELS)
                         + f" | {a['error_points']} points |")
    lines += ["", "Credit misallocated: half the summed absolute gap between reported and true shares,",
              "in percentage points of total credit."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("results", nargs="+")
    p.add_argument("--out")
    args = ap.parse_args()
    if args.cmd == "run":
        res = run()
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"wrote {args.out} for pycausalsim {res['version']}")
    else:
        text = report(args.results)
        if args.out:
            open(args.out, "w").write(text)
        print(text)
