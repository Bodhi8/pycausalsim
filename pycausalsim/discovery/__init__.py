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
"""
Causal discovery: learning graph structure from observational data.

Methods
-------
- ``pc``       constraint-based (PC-stable with Fisher-z tests of partial correlation)
- ``ges``      score-based (greedy equivalence search with the BIC score; alias ``fges``)
- ``hybrid``   the GES graph without the edges that PC's independence tests reject
- ``lingam``   DirectLiNGAM, for linear models with non-Gaussian noise
- ``notears``  linear NOTEARS (continuous optimization with an acyclicity constraint)
- ``correlation``  a correlation threshold. Not a causal method; kept for quick demos.

What the output means
---------------------
Every method returns ``{variable: [its direct causes]}``.

Observational data often cannot tell which way an edge points: ``a -> b -> c``,
``a <- b <- c`` and ``a <- b -> c`` imply exactly the same independencies. ``pc``,
``ges`` and ``hybrid`` therefore return the *equivalence class* of graphs. An edge
whose direction the data do determine is listed once. An edge whose direction they
do not determine is listed **in both directions** (each variable appears among the
other's causes); :func:`undirected_edges` lists them. Earlier versions guessed a
direction for these edges from the variables' variances.

``lingam`` and ``notears`` always return a fully directed acyclic graph, because
they rest on stronger assumptions (non-Gaussian noise; comparable noise scales).
When those assumptions fail their directions are not reliable.

Background knowledge
--------------------
``discover(..., tiers={variable: tier})`` states an ordering: no edge runs from a
later tier to an earlier one. ``target=`` is shorthand for a last tier of its own.

All methods assume linear relationships and no unobserved common causes. FCI,
which relaxes the second assumption, is not implemented.
"""

from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

Graph = Dict[str, List[str]]

__all__ = ["CausalDiscovery", "undirected_edges", "is_dag"]


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def undirected_edges(graph: Graph) -> List[Tuple[str, str]]:
    """Edges listed in both directions, i.e. whose direction is not determined."""
    out = []
    for child, parents in graph.items():
        for parent in parents:
            if parent != child and child in graph.get(parent, []) and (child, parent) not in out:
                out.append((parent, child))
    return sorted(tuple(sorted(e)) for e in out)


def is_dag(graph: Graph) -> bool:
    """True if the graph has no cycle (two-way edges and self-loops count as cycles)."""
    state: Dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for parent in graph.get(node, []):
            if state.get(parent) == 1:
                return False
            if state.get(parent) is None and not visit(parent):
                return False
        state[node] = 2
        return True

    return all(visit(v) for v in list(graph) if state.get(v) is None)


class _PDAG:
    """A partially directed graph over named variables.

    ``adj[a]`` is the set of neighbours of ``a``; ``arrows`` holds ``(a, b)`` for
    every edge known to point from ``a`` to ``b``. An adjacent pair with no arrow
    is undirected.
    """

    def __init__(self, variables: List[str]):
        self.variables = list(variables)
        self.adj: Dict[str, Set[str]] = {v: set() for v in variables}
        self.arrows: Set[Tuple[str, str]] = set()

    def add_edge(self, a: str, b: str) -> None:
        self.adj[a].add(b)
        self.adj[b].add(a)

    def adjacent(self, a: str, b: str) -> bool:
        return b in self.adj[a]

    def directed(self, a: str, b: str) -> bool:
        return (a, b) in self.arrows

    def undirected(self, a: str, b: str) -> bool:
        return self.adjacent(a, b) and (a, b) not in self.arrows and (b, a) not in self.arrows

    def orient(self, a: str, b: str) -> bool:
        """Orient a - b as a -> b if it is still undirected."""
        if self.undirected(a, b):
            self.arrows.add((a, b))
            return True
        return False

    def _reaches(self, start: str, goal: str) -> bool:
        """Is there a directed path from start to goal?"""
        seen, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(b for (a, b) in self.arrows if a == node)
        return False

    def apply_meek_rules(self) -> None:
        """Orient every edge whose direction follows from the ones already known
        (Meek's rules 1 to 3): no new colliders and no cycles."""
        changed = True
        while changed:
            changed = False
            for a in self.variables:
                for b in list(self.adj[a]):
                    if not self.undirected(a, b):
                        continue
                    # Rule 1: c -> a, a - b, c and b not adjacent  =>  a -> b
                    if any(self.directed(c, a) and not self.adjacent(c, b)
                           for c in self.adj[a] if c != b):
                        changed |= self.orient(a, b)
                        continue
                    # Rule 2: a -> c -> b and a - b  =>  a -> b
                    if any(self.directed(a, c) and self.directed(c, b)
                           for c in self.adj[a] & self.adj[b]):
                        changed |= self.orient(a, b)
                        continue
                    # Rule 3: a - c -> b, a - d -> b, c and d not adjacent, a - b  =>  a -> b
                    mids = [c for c in self.adj[a] & self.adj[b]
                            if self.undirected(a, c) and self.directed(c, b)]
                    if any(not self.adjacent(c, d) for c, d in combinations(mids, 2)):
                        changed |= self.orient(a, b)

    def apply_tiers(self, tiers: Dict[str, int]) -> None:
        """Background knowledge: a variable cannot cause one in an earlier tier.
        Every edge between two tiers is pointed from the earlier to the later,
        overriding what the data suggested for that edge."""
        for a in self.variables:
            for b in self.adj[a]:
                if tiers.get(a, 0) < tiers.get(b, 0):
                    self.arrows.discard((b, a))
                    self.arrows.add((a, b))

    def to_graph(self) -> Graph:
        graph: Graph = {v: [] for v in self.variables}
        for a in self.variables:
            for b in sorted(self.adj[a], key=self.variables.index):
                if self.directed(b, a) or self.undirected(a, b):
                    graph[a].append(b)
        return graph


def _cpdag_from_dag(variables: List[str], parents: Dict[str, Set[str]]) -> _PDAG:
    """The equivalence class of a DAG: its skeleton, its colliders, and whatever
    else those force."""
    g = _PDAG(variables)
    for child, ps in parents.items():
        for p in ps:
            g.add_edge(p, child)
    for child, ps in parents.items():
        for p, q in combinations(sorted(ps), 2):
            if not g.adjacent(p, q):          # unshielded collider p -> child <- q
                g.orient(p, child)
                g.orient(q, child)
    g.apply_meek_rules()
    return g


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

class CausalDiscovery:
    """
    Causal structure discovery from observational data.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> discovery = CausalDiscovery()
    >>> graph = discovery.discover(data, method='pc')
    >>> undirected_edges(graph)     # edges the data could not orient
    """

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def discover(
        self,
        data: pd.DataFrame,
        method: str = 'pc',
        target: Optional[str] = None,
        tiers: Optional[Dict[str, int]] = None,
        **kwargs
    ) -> Graph:
        """
        Discover causal structure from data.

        Parameters
        ----------
        data : pd.DataFrame
            The observational data, numeric, one column per variable
        method : str
            'pc', 'ges' (alias 'fges'), 'hybrid', 'lingam', 'notears' or
            'correlation'
        target : str, optional
            The outcome variable. It is taken to be an effect, not a cause: every
            edge touching it points into it. Shorthand for putting the target
            alone in the last tier.
        tiers : dict, optional
            Background knowledge about ordering, as ``{variable: tier}``. A
            variable cannot cause one in an earlier tier, so an edge between two
            tiers always points from the earlier to the later, whatever the data
            suggest. Variables not listed are in tier 0. For example
            ``{'age': 0, 'dose': 1, 'recovery': 2}``.
        **kwargs
            Method-specific arguments: ``alpha`` (pc, hybrid, lingam),
            ``max_cond_size`` (pc, hybrid), ``max_parents`` (ges, hybrid),
            ``lambda1`` and ``w_threshold`` (notears), ``threshold`` (correlation)

        Returns
        -------
        dict
            ``{variable: [direct causes]}``. For 'pc', 'ges' and 'hybrid' an
            undetermined edge appears in both directions; see the module
            docstring and :func:`undirected_edges`.
        """
        methods = {
            'pc': self._pc_algorithm,
            'ges': self._ges_algorithm,
            'fges': self._ges_algorithm,
            'lingam': self._lingam_algorithm,
            'notears': self._notears_algorithm,
            'hybrid': self._hybrid_algorithm,
            'correlation': self._correlation_graph,
        }

        if not isinstance(method, str) or method.lower() not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if target is not None and target not in data.columns:
            raise ValueError(f"Target '{target}' is not a column of the data")
        self._check_data(data)

        order: Dict[str, int] = {}
        if tiers:
            unknown = [v for v in tiers if v not in data.columns]
            if unknown:
                raise ValueError(f"tiers names variables that are not columns: {unknown}")
            order = {v: int(t) for v, t in tiers.items()}
        if target is not None:
            others = [t for v, t in order.items() if v != target]
            order[target] = max(others, default=0) + 1

        return methods[method.lower()](data, tiers=order, **kwargs)

    @staticmethod
    def _check_data(data: pd.DataFrame) -> None:
        if data.shape[1] < 2:
            raise ValueError("Discovery needs at least two variables")
        non_numeric = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
        if non_numeric:
            raise ValueError(f"Columns must be numeric; encode these first: {non_numeric}")
        if data.isna().any().any():
            bad = [c for c in data.columns if data[c].isna().any()]
            raise ValueError(f"Data contain missing values in {bad}; drop or impute them first")
        constant = [c for c in data.columns if data[c].nunique() < 2]
        if constant:
            raise ValueError(f"Constant columns carry no information; remove them: {constant}")
        if len(data) < data.shape[1] + 4:
            raise ValueError(
                f"Too few rows ({len(data)}) for {data.shape[1]} variables"
            )

    # ------------------------------------------------------------------
    # PC
    # ------------------------------------------------------------------

    @staticmethod
    def _independent(corr: np.ndarray, n: int, i: int, j: int, cond: Tuple[int, ...],
                     alpha: float) -> bool:
        """Fisher-z test of the partial correlation of i and j given ``cond``."""
        idx = [i, j, *cond]
        sub = corr[np.ix_(idx, idx)]
        try:
            prec = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            prec = np.linalg.pinv(sub)
        denom = prec[0, 0] * prec[1, 1]
        if denom <= 0:
            return False
        r = float(np.clip(-prec[0, 1] / np.sqrt(denom), -0.999999, 0.999999))
        dof = n - len(cond) - 3
        if dof <= 0:
            return False          # not enough data to claim independence
        z = np.arctanh(r) * np.sqrt(dof)
        return bool(2 * stats.norm.sf(abs(z)) > alpha)

    def _skeleton(
        self,
        data: pd.DataFrame,
        alpha: float,
        max_cond_size: Optional[int]
    ) -> Tuple[_PDAG, Dict[FrozenSet[str], Set[str]]]:
        """PC-stable skeleton search. Returns the undirected graph and, for every
        pair found independent, the set that separated it."""
        variables = list(data.columns)
        index = {v: k for k, v in enumerate(variables)}
        corr = np.corrcoef(data.to_numpy(dtype=float), rowvar=False)
        n = len(data)

        g = _PDAG(variables)
        for a, b in combinations(variables, 2):
            g.add_edge(a, b)
        sepset: Dict[FrozenSet[str], Set[str]] = {}

        depth = 0
        limit = len(variables) - 2 if max_cond_size is None else max_cond_size
        while depth <= limit:
            # Neighbour sets are frozen for the whole level, so the result does
            # not depend on the order of the columns.
            frozen = {v: set(g.adj[v]) for v in variables}
            if all(len(frozen[v]) - 1 < depth for v in variables):
                break
            for x in variables:
                for y in sorted(frozen[x], key=index.get):
                    if not g.adjacent(x, y):
                        continue
                    others = sorted(frozen[x] - {y}, key=index.get)
                    if len(others) < depth:
                        continue
                    for cond in combinations(others, depth):
                        if self._independent(corr, n, index[x], index[y],
                                             tuple(index[c] for c in cond), alpha):
                            g.adj[x].discard(y)
                            g.adj[y].discard(x)
                            sepset[frozenset((x, y))] = set(cond)
                            break
            depth += 1
        return g, sepset

    def _pc_algorithm(
        self,
        data: pd.DataFrame,
        tiers: Optional[Dict[str, int]] = None,
        alpha: float = 0.05,
        max_cond_size: Optional[int] = None,
        **kwargs
    ) -> Graph:
        """
        PC algorithm (PC-stable).

        1. Start from the complete graph and remove an edge whenever the two
           variables are independent given some subset of their neighbours.
        2. For every chain ``x - y - z`` with x and z not adjacent, orient
           ``x -> y <- z`` if y was *not* in the set that separated x from z.
           Only then is y a collider; a chain or a fork is left alone.
        3. Orient whatever follows from that (Meek's rules).

        Edges that remain undetermined are returned in both directions.
        """
        tiers = tiers or {}
        g, sepset = self._skeleton(data, alpha, max_cond_size)
        g.apply_tiers(tiers)

        # Colliders. A proposed orientation that contradicts another one, or the
        # tiers, is dropped: conflicting evidence is treated as no evidence.
        proposals: Set[Tuple[str, str]] = set()
        for y in g.variables:
            for x, z in combinations(sorted(g.adj[y], key=g.variables.index), 2):
                if g.adjacent(x, z):
                    continue
                if y not in sepset.get(frozenset((x, z)), set()):
                    proposals.add((x, y))
                    proposals.add((z, y))
        for a, b in sorted(proposals):
            if (b, a) not in proposals:
                g.orient(a, b)

        g.apply_meek_rules()
        return g.to_graph()

    # ------------------------------------------------------------------
    # Score-based search (GES)
    # ------------------------------------------------------------------

    @staticmethod
    def _bic_scorer(data: pd.DataFrame):
        """Local BIC score of a variable given a parent set, for a linear
        Gaussian model. Higher is better."""
        variables = list(data.columns)
        index = {v: k for k, v in enumerate(variables)}
        X = data.to_numpy(dtype=float)
        X = X - X.mean(axis=0)
        n = len(X)
        cov = X.T @ X / n
        cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}

        def local(child: str, parents) -> float:
            c = index[child]
            ps = tuple(sorted(index[p] for p in parents))
            key = (c, ps)
            if key not in cache:
                var = cov[c, c]
                if ps:
                    spp = cov[np.ix_(ps, ps)]
                    spc = cov[np.ix_(ps, [c])]
                    var = var - (spc.T @ np.linalg.solve(spp + 1e-12 * np.eye(len(ps)), spc)).item()
                var = max(var, 1e-12)
                cache[key] = -0.5 * n * np.log(var) - 0.5 * np.log(n) * (len(ps) + 1)
            return cache[key]

        return local

    @staticmethod
    def _consistent_extension(g: _PDAG) -> Optional[Dict[str, Set[str]]]:
        """A DAG that keeps every arrow of ``g``, orients its undirected edges,
        and adds no new collider (Dor and Tarsi, 1992). None if there is none."""
        parents = {v: {a for (a, b) in g.arrows if b == v} for v in g.variables}
        children = {v: {b for (a, b) in g.arrows if a == v} for v in g.variables}
        und = {v: {u for u in g.adj[v] if g.undirected(v, u)} for v in g.variables}
        result = {v: set(parents[v]) for v in g.variables}
        remaining = set(g.variables)
        while remaining:
            chosen = None
            for x in g.variables:
                if x not in remaining or children[x] & remaining:
                    continue                                   # x must be a sink
                around = g.adj[x] & remaining
                if all((around - {y}) <= g.adj[y] for y in und[x] & remaining):
                    chosen = x
                    break
            if chosen is None:
                return None
            for y in und[chosen] & remaining:
                result[chosen].add(y)
            remaining.discard(chosen)
        return result

    def _ges_search(
        self,
        data: pd.DataFrame,
        tiers: Optional[Dict[str, int]] = None,
        max_parents: Optional[int] = None
    ) -> _PDAG:
        """Greedy equivalence search (Chickering, 2002).

        The forward phase adds, one at a time, the edge that most improves the
        score; the backward phase then removes edges while that improves it.
        Both phases move between equivalence classes of DAGs, so the search
        cannot be trapped by an arbitrary choice of edge direction.

        With ``tiers``, a step is allowed only if the resulting class has a
        member that respects the ordering; the ordering is then applied to the
        class before the next step, so the knowledge shapes the search instead
        of being imposed on its result.
        """
        variables = list(data.columns)
        local = self._bic_scorer(data)
        g = _PDAG(variables)
        tiers = tiers or {}

        def tier(v: str) -> int:
            return tiers.get(v, 0)

        def parents_of(graph: _PDAG, y: str) -> Set[str]:
            return {a for (a, b) in graph.arrows if b == y}

        def neighbours(graph: _PDAG, y: str) -> Set[str]:
            return {u for u in graph.adj[y] if graph.undirected(y, u)}

        def is_clique(graph: _PDAG, nodes: Set[str]) -> bool:
            return all(graph.adjacent(a, b) for a, b in combinations(nodes, 2))

        def semi_directed_path(graph: _PDAG, start: str, goal: str, blocked: Set[str]) -> bool:
            """A path from start to goal that never goes against an arrow and
            avoids ``blocked``."""
            seen, stack = {start}, [start]
            while stack:
                node = stack.pop()
                for nxt in graph.adj[node]:
                    if graph.directed(nxt, node) or nxt in blocked or nxt in seen:
                        continue
                    if nxt == goal:
                        return True
                    seen.add(nxt)
                    stack.append(nxt)
            return False

        def rebuild(candidate: _PDAG) -> Optional[_PDAG]:
            if tiers:
                candidate.apply_tiers(tiers)
            dag = self._consistent_extension(candidate)
            if dag is None:
                return None
            new = _cpdag_from_dag(variables, dag)
            if tiers:
                if any(tier(a) > tier(b) for (a, b) in new.arrows):
                    return None          # the data-implied arrow contradicts the ordering
                new.apply_tiers(tiers)
                new.apply_meek_rules()
            return new

        def copy(graph: _PDAG) -> _PDAG:
            c = _PDAG(variables)
            c.adj = {v: set(s) for v, s in graph.adj.items()}
            c.arrows = set(graph.arrows)
            return c

        # Forward phase
        while True:
            best_gain, best_graph = 1e-9, None
            for x in variables:
                for y in variables:
                    if x == y or g.adjacent(x, y) or tier(x) > tier(y):
                        continue
                    nb_y = neighbours(g, y)
                    na = {u for u in nb_y if g.adjacent(u, x)}
                    free = sorted(nb_y - na - {x})
                    pa_y = parents_of(g, y)
                    for size in range(len(free) + 1):
                        for T in combinations(free, size):
                            base = na | set(T)
                            if max_parents is not None and len(base | pa_y) + 1 > max_parents:
                                continue
                            if not is_clique(g, base):
                                continue
                            if semi_directed_path(g, y, x, base):
                                continue
                            gain = local(y, base | pa_y | {x}) - local(y, base | pa_y)
                            if gain <= best_gain:
                                continue
                            cand = copy(g)
                            cand.add_edge(x, y)
                            cand.arrows.add((x, y))
                            for t in T:
                                cand.arrows.add((t, y))
                            new = rebuild(cand)
                            if new is not None:
                                best_gain, best_graph = gain, new
            if best_graph is None:
                break
            g = best_graph

        # Backward phase
        while True:
            best_gain, best_graph = 1e-9, None
            for x in variables:
                for y in variables:
                    if x == y or not g.adjacent(x, y) or g.directed(y, x):
                        continue                    # considers x -> y and x - y
                    na = {u for u in neighbours(g, y) if g.adjacent(u, x)}
                    pa_y = parents_of(g, y) - {x}
                    members = sorted(na)
                    for size in range(len(members) + 1):
                        for H in combinations(members, size):
                            keep = na - set(H)
                            if not is_clique(g, keep):
                                continue
                            gain = local(y, keep | pa_y) - local(y, keep | pa_y | {x})
                            if gain <= best_gain:
                                continue
                            cand = copy(g)
                            cand.adj[x].discard(y)
                            cand.adj[y].discard(x)
                            cand.arrows.discard((x, y))
                            for h in H:
                                if cand.undirected(y, h):
                                    cand.arrows.add((y, h))
                                if cand.undirected(x, h):
                                    cand.arrows.add((x, h))
                            new = rebuild(cand)
                            if new is not None:
                                best_gain, best_graph = gain, new
            if best_graph is None:
                break
            g = best_graph
        return g

    def _ges_algorithm(
        self,
        data: pd.DataFrame,
        tiers: Optional[Dict[str, int]] = None,
        max_parents: Optional[int] = None,
        **kwargs
    ) -> Graph:
        """
        Greedy equivalence search with the BIC score of a linear Gaussian model.

        Returns the equivalence class: edges whose direction the score cannot
        distinguish come back in both directions.
        """
        return self._ges_search(data, tiers=tiers, max_parents=max_parents).to_graph()

    def _hybrid_algorithm(
        self,
        data: pd.DataFrame,
        tiers: Optional[Dict[str, int]] = None,
        alpha: float = 0.05,
        max_cond_size: Optional[int] = None,
        max_parents: Optional[int] = None,
        **kwargs
    ) -> Graph:
        """
        Hybrid search: an edge is kept only if both approaches support it.

        Greedy equivalence search proposes the graph; any edge between two
        variables that the PC independence tests separate is then removed, and
        the equivalence class of what remains is returned. This is stricter
        about adding edges than either method alone.
        """
        variables = list(data.columns)
        g = self._ges_search(data, tiers=tiers, max_parents=max_parents)
        skeleton, _ = self._skeleton(data, alpha, max_cond_size)
        dag = self._consistent_extension(g) or {v: set() for v in variables}
        dag = {child: {p for p in parents if skeleton.adjacent(p, child)}
               for child, parents in dag.items()}
        g = _cpdag_from_dag(variables, dag)
        if tiers:
            g.apply_tiers(tiers)
            g.apply_meek_rules()
        return g.to_graph()

    @staticmethod
    def _respect_tiers(graph: Graph, tiers: Optional[Dict[str, int]]) -> Graph:
        """Reverse any edge of a fully directed graph that points from a later
        tier to an earlier one."""
        if not tiers:
            return graph
        out: Graph = {v: [] for v in graph}
        for child, parents in graph.items():
            for parent in parents:
                if tiers.get(parent, 0) > tiers.get(child, 0):
                    if child not in out[parent]:
                        out[parent].append(child)
                elif parent not in out[child]:
                    out[child].append(parent)
        return out

    # ------------------------------------------------------------------
    # DirectLiNGAM
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(u: np.ndarray) -> float:
        """Maximum-entropy approximation of the differential entropy of a
        standardized variable (Hyvarinen, 1998)."""
        k1, k2, gamma = 79.047, 7.4129, 0.37457
        return float((1 + np.log(2 * np.pi)) / 2
                     - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2
                     - k2 * (np.mean(u * np.exp(-u ** 2 / 2))) ** 2)

    def _lingam_algorithm(
        self,
        data: pd.DataFrame,
        tiers: Optional[Dict[str, int]] = None,
        alpha: float = 0.01,
        **kwargs
    ) -> Graph:
        """
        DirectLiNGAM (Shimizu et al., 2011).

        In a linear model with non-Gaussian noise, a cause is independent of the
        residual of its effect regressed on it, but not the other way round.
        The algorithm repeatedly finds the variable most independent of all its
        residuals, places it next in the causal order, and removes its
        influence from the rest. Edges are then kept where a variable's
        coefficient on an earlier one is significant at ``alpha``.

        The direction of an edge is identified only through non-Gaussianity. If
        the noise looks Gaussian a warning is raised and the directions should
        not be relied on.
        """
        variables = list(data.columns)
        X = data.to_numpy(dtype=float)
        X = X - X.mean(axis=0)
        n, d = X.shape

        def std(v: np.ndarray) -> np.ndarray:
            s = v.std()
            return v / s if s > 0 else v

        remaining = list(range(d))
        R = X.copy()
        order: List[int] = []
        while len(remaining) > 1:
            best, best_score = remaining[0], -np.inf
            for i in remaining:
                xi = std(R[:, i])
                score = 0.0
                for j in remaining:
                    if j == i:
                        continue
                    xj = std(R[:, j])
                    rho = float(np.mean(xi * xj))
                    ri_j = xi - rho * xj          # residual of i regressed on j
                    rj_i = xj - rho * xi          # residual of j regressed on i
                    diff = ((self._entropy(xj) + self._entropy(std(ri_j)))
                            - (self._entropy(xi) + self._entropy(std(rj_i))))
                    score -= min(0.0, diff) ** 2
                if score > best_score:
                    best, best_score = i, score
            order.append(best)
            remaining.remove(best)
            xb = R[:, best]
            denom = float(xb @ xb)
            for j in remaining:
                if denom > 0:
                    R[:, j] = R[:, j] - (float(R[:, j] @ xb) / denom) * xb
        order.extend(remaining)

        graph: Graph = {v: [] for v in variables}
        residuals = {order[0]: X[:, order[0]]}
        for pos in range(1, d):
            child = order[pos]
            preds = order[:pos]
            Z = X[:, preds]
            beta, *_ = np.linalg.lstsq(Z, X[:, child], rcond=None)
            resid = X[:, child] - Z @ beta
            residuals[child] = resid
            dof = n - len(preds) - 1
            sigma2 = float(resid @ resid) / max(dof, 1)
            cov = sigma2 * np.linalg.pinv(Z.T @ Z)
            se = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
            p = 2 * stats.t.sf(np.abs(beta / se), max(dof, 1))
            graph[variables[child]] = [variables[q] for q, pv in zip(preds, p) if pv < alpha]

        gaussian = [variables[k] for k, r in residuals.items()
                    if stats.jarque_bera(r).pvalue > 0.05]
        if len(gaussian) >= 2:
            warnings.warn(
                "LiNGAM identifies direction from non-Gaussian noise, but the noise of "
                f"{gaussian} is consistent with a Gaussian. The directions of edges among "
                "these variables are not identified and may be arbitrary.",
                UserWarning, stacklevel=3
            )
        return self._respect_tiers(graph, tiers)

    # ------------------------------------------------------------------
    # NOTEARS
    # ------------------------------------------------------------------

    def _notears_algorithm(
        self,
        data: pd.DataFrame,
        tiers: Optional[Dict[str, int]] = None,
        lambda1: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
        **kwargs
    ) -> Graph:
        """
        Linear NOTEARS (Zheng et al., 2018).

        Minimizes the least-squares loss of ``X = XW`` plus an L1 penalty,
        subject to ``trace(exp(W * W)) - d = 0``, which holds exactly when W is
        acyclic. Solved with an augmented Lagrangian; weights below
        ``w_threshold`` in absolute value are then set to zero.

        NOTEARS is sensitive to the scale of the variables: it tends to place
        low-variance variables before high-variance ones. Its directions are
        reliable when noise scales are comparable, and should be checked
        against domain knowledge otherwise.
        """
        variables = list(data.columns)
        X = data.to_numpy(dtype=float)
        X = X - X.mean(axis=0)
        n, d = X.shape

        def adj(w: np.ndarray) -> np.ndarray:
            return (w[:d * d] - w[d * d:]).reshape(d, d)

        def h_and_grad(W: np.ndarray) -> Tuple[float, np.ndarray]:
            E = linalg.expm(W * W)
            return float(np.trace(E) - d), E.T * W * 2

        state = {'rho': 1.0, 'alpha': 0.0}

        def objective(w: np.ndarray) -> Tuple[float, np.ndarray]:
            W = adj(w)
            Rm = X - X @ W
            loss = 0.5 / n * float((Rm ** 2).sum())
            g_loss = -1.0 / n * X.T @ Rm
            h, g_h = h_and_grad(W)
            obj = loss + 0.5 * state['rho'] * h * h + state['alpha'] * h + lambda1 * float(w.sum())
            g_smooth = g_loss + (state['rho'] * h + state['alpha']) * g_h
            grad = np.concatenate([g_smooth + lambda1, -g_smooth + lambda1], axis=None)
            return obj, grad

        w_est = np.zeros(2 * d * d)
        h = np.inf
        # The diagonal is fixed at zero: a variable is never its own cause.
        bounds = [(0, 0) if i == j else (0, None)
                  for _ in range(2) for i in range(d) for j in range(d)]
        for _ in range(max_iter):
            w_new, h_new = w_est, h
            while state['rho'] < rho_max:
                sol = optimize.minimize(
                    objective, w_est, method='L-BFGS-B', jac=True, bounds=bounds
                )
                w_new = sol.x
                h_new, _ = h_and_grad(adj(w_new))
                if h_new > 0.25 * h:
                    state['rho'] *= 10
                else:
                    break
            w_est, h = w_new, h_new
            state['alpha'] += state['rho'] * h
            if h <= h_tol or state['rho'] >= rho_max:
                break

        W = adj(w_est)
        W[np.abs(W) < w_threshold] = 0.0
        np.fill_diagonal(W, 0.0)

        # W[i, j] != 0 means i -> j. Thresholding normally leaves a DAG; if a
        # cycle survives, drop the weakest remaining edge until none does.
        def to_graph(M: np.ndarray) -> Graph:
            return {
                variables[j]: [variables[i] for i in range(d) if M[i, j] != 0]
                for j in range(d)
            }

        while not is_dag(to_graph(W)):
            nz = np.argwhere(W != 0)
            i, j = min(nz, key=lambda ij: abs(W[ij[0], ij[1]]))
            W[i, j] = 0.0
        return self._respect_tiers(to_graph(W), tiers)

    # ------------------------------------------------------------------
    # Correlation heuristic
    # ------------------------------------------------------------------

    def _correlation_graph(
        self,
        data: pd.DataFrame,
        tiers: Optional[Dict[str, int]] = None,
        threshold: float = 0.1,
        **kwargs
    ) -> Graph:
        """
        Correlation-threshold graph. **Not a causal discovery method.**

        Connects every pair whose absolute correlation exceeds ``threshold`` and
        points the edge from the higher-variance variable to the lower, or as
        the tiers require. Correlation does not separate direct from
        indirect association, and variance says nothing about direction. This
        exists as a fast stand-in for demos and tests; use 'pc', 'ges', 'hybrid'
        or a graph from domain knowledge for analysis.
        """
        variables = list(data.columns)
        corr_matrix = data.corr()
        graph: Graph = {v: [] for v in variables}

        for i, v1 in enumerate(variables):
            for v2 in variables[i + 1:]:
                if abs(corr_matrix.loc[v1, v2]) <= threshold:
                    continue
                if data[v1].var() > data[v2].var():
                    graph[v2].append(v1)
                else:
                    graph[v1].append(v2)
        return self._respect_tiers(graph, tiers)
