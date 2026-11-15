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
CausalSimulator: Main class for causal discovery and simulation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from .results import CausalEffect, SensitivityResult, DriverRanking, OptimalPolicy
from .discovery import CausalDiscovery
from .models import StructuralCausalModel


class CausalSimulator:
    """
    Main class for causal discovery and intervention simulation.
    
    PyCausalSim's CausalSimulator learns causal structure from data and
    uses counterfactual simulation to estimate causal effects.
    
    Parameters
    ----------
    data : pd.DataFrame
        The observational data
    target : str
        The outcome variable of interest
    treatment_vars : list of str, optional
        Variables that can be intervened on (treatments)
    confounders : list of str, optional
        Known confounding variables to control for
    random_state : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> simulator = CausalSimulator(
    ...     data=df,
    ...     target='conversion_rate',
    ...     treatment_vars=['page_load_time', 'price'],
    ...     confounders=['traffic_source', 'device_type']
    ... )
    >>> simulator.discover_graph()
    >>> effect = simulator.simulate_intervention('page_load_time', 2.0)
    >>> print(effect.summary())
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        treatment_vars: Optional[List[str]] = None,
        confounders: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data.copy()
        self.target = target
        self.treatment_vars = treatment_vars or []
        self.confounders = confounders or []
        self.random_state = random_state
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize components
        self._graph = None
        self._scm = None
        self._discovery = CausalDiscovery(random_state=random_state)
        
        # Determine all variables
        self._all_vars = list(data.columns)
        self._feature_vars = [c for c in data.columns if c != target]
        
        # Bootstrap resampling uses its own generator, so results with a fixed
        # random_state do not depend on other code touching NumPy's global state.
        self._rng = np.random.default_rng(random_state)
        if random_state is not None:
            np.random.seed(random_state)
    
    def _validate_inputs(self):
        """Validate input data and parameters."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if self.target not in self.data.columns:
            raise ValueError(f"Target '{self.target}' not found in data columns")
        
        for var in self.treatment_vars:
            if var not in self.data.columns:
                raise ValueError(f"Treatment variable '{var}' not found in data")
        
        for var in self.confounders:
            if var not in self.data.columns:
                raise ValueError(f"Confounder '{var}' not found in data")
    
    @property
    def graph(self):
        """Return the learned causal graph."""
        return self._graph
    
    @property
    def scm(self) -> Optional[StructuralCausalModel]:
        """Return the structural causal model."""
        return self._scm
    
    def discover_graph(
        self,
        method: str = 'pc',
        **kwargs
    ) -> 'CausalSimulator':
        """
        Discover causal structure from data.
        
        Parameters
        ----------
        method : str
            Discovery method: 'pc', 'ges', 'fges', 'lingam', 'notears', 'hybrid'
        **kwargs
            Additional arguments passed to the discovery method
            
        Returns
        -------
        self
            Returns self for method chaining
            
        Examples
        --------
        >>> simulator.discover_graph(method='notears')
        >>> simulator.plot_graph()
        """
        # Prepare data for discovery
        discovery_vars = self._feature_vars + [self.target]
        discovery_data = self.data[discovery_vars]
        
        # Run discovery
        # Declared roles are background knowledge: a confounder is a cause of the
        # treatments and the target, not an effect of them, and the target causes
        # nothing. Other variables are left for the data to place.
        tiers = {v: (0 if v in self.confounders else 1) for v in discovery_vars}
        tiers[self.target] = 2
        self._graph = self._discovery.discover(
            data=discovery_data,
            method=method,
            tiers=tiers,
            **kwargs
        )
        
        # Discovery can return edges it could not orient, as a pair of variables
        # that are each listed as the other's parent. A structural model needs
        # one direction, so those are oriented from the declared roles.
        self._graph = self._check_graph(self._orient_two_way_edges(self._graph))

        # Fit SCM based on discovered graph
        self._scm = StructuralCausalModel(graph=self._graph)
        self._scm.fit(self.data)
        
        return self

    def _role_tier(self, var: str) -> Tuple[int, int]:
        """Ordering implied by the declared roles: confounders first, the target
        last, and among the rest a declared treatment after an undeclared
        variable."""
        if var == self.target:
            return (2, 0)
        if var in self.confounders:
            return (0, 0)
        return (1, 1 if var in self.treatment_vars else 0)

    def _orient_two_way_edges(self, graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Give one direction to every edge listed in both directions.

        These are edges whose direction neither the data nor the declared
        confounder and target roles settle. They are pointed from an undeclared
        variable to a declared treatment, and otherwise by column order. Either
        way the choice is an assumption, so it is reported.
        """
        graph = {v: list(dict.fromkeys(p)) for v, p in graph.items()}
        order = {c: i for i, c in enumerate(self.data.columns)}
        assumed = []
        for child in list(graph):
            for parent in list(graph[child]):
                if parent == child or child not in graph.get(parent, []):
                    continue
                a, b = sorted((child, parent), key=lambda v: (self._role_tier(v), order.get(v, 0)))
                # a is taken to be the cause, b the effect
                graph[a] = [p for p in graph.get(a, []) if p != b]
                if a not in graph.setdefault(b, []):
                    graph[b].append(a)
                assumed.append(f"{a} -> {b}")
        self._oriented_edges = sorted(set(assumed))
        if assumed:
            warnings.warn(
                "The data do not determine the direction of these edges; they were "
                f"oriented by declared role and column order: {self._oriented_edges}. "
                "Use set_graph() to state the direction you believe.",
                UserWarning, stacklevel=3
            )
        return graph

    def _check_graph(self, graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate a graph given as {variable: [its direct causes]}.

        Every name must be a column of the data, and the graph must have no
        cycles: a structural model computes each variable from its causes, which
        is undefined if a variable is among its own causes.
        """
        if not isinstance(graph, dict):
            raise TypeError("graph must be a dict of {variable: [direct causes]}")
        columns = set(self.data.columns)
        clean: Dict[str, List[str]] = {}
        for var, parents in graph.items():
            if isinstance(parents, str):
                raise TypeError(f"Parents of '{var}' must be a list, not a string")
            parents = list(dict.fromkeys(parents))
            for name in [var] + parents:
                if name not in columns:
                    raise ValueError(f"Graph variable '{name}' is not a column of the data")
            if var in parents:
                raise ValueError(f"Graph has a self-loop on '{var}'")
            clean[var] = parents

        # Depth-first search for a cycle along parent links
        state: Dict[str, int] = {}

        def visit(node, path):
            state[node] = 1
            for parent in clean.get(node, []):
                if state.get(parent) == 1:
                    cycle = path[path.index(parent):] + [parent] if parent in path else [parent, node, parent]
                    raise ValueError(
                        "Graph has a cycle: " + " <- ".join(cycle) + ". A causal graph must be acyclic."
                    )
                if state.get(parent) is None:
                    visit(parent, path + [parent])
            state[node] = 2

        for var in clean:
            if state.get(var) is None:
                visit(var, [var])
        return clean
    
    def set_graph(self, graph: Dict[str, List[str]]) -> 'CausalSimulator':
        """
        Manually set the causal graph.
        
        Parameters
        ----------
        graph : dict
            Adjacency list where keys are variables and values are lists
            of their direct causes (parents)
            
        Returns
        -------
        self
        """
        self._graph = self._check_graph(graph)
        self._scm = StructuralCausalModel(graph=self._graph)
        self._scm.fit(self.data)
        return self
    
    def simulate_intervention(
        self,
        variable: str,
        value: float,
        n_simulations: int = 1000,
        method: str = 'monte_carlo',
        condition: Optional[Dict[str, Any]] = None,
        n_bootstrap: Optional[int] = None
    ) -> CausalEffect:
        """
        Estimate the causal effect of setting a variable to a value.

        The estimate is ``E[target | do(variable = value)] - E[target]``. The
        intervention is propagated through every variable downstream of
        ``variable`` in the graph, so effects that travel through mediators are
        counted.

        Uncertainty comes from a bootstrap that refits the model on each
        resample, so the interval reflects how well the data pin down the
        mechanisms, not only which rows were drawn.

        Parameters
        ----------
        variable : str
            The variable to intervene on
        value : float
            The value to set the variable to
        n_simulations : int
            Upper bound on the number of bootstrap refits (see ``n_bootstrap``)
        method : str
            'monte_carlo' or 'scm': fit a mechanism for each variable and
            propagate the intervention through the graph (the two names are
            equivalent). 'backdoor': linear regression of the target on the
            variable and a backdoor adjustment set read from the graph.
        condition : dict, optional
            Conditions to apply (e.g., {'traffic_source': 'paid'})
        n_bootstrap : int, optional
            Number of bootstrap refits. Defaults to ``min(n_simulations, 100)``;
            each refit trains the mechanisms again, so large values are slow.

        Returns
        -------
        CausalEffect
            Point estimate from the full data, a 95% percentile interval from
            the bootstrap, and a two-sided p-value from the bootstrap standard
            error. ``n_simulations`` on the result is the number of refits used.

        Notes
        -----
        Mechanisms are tree ensembles, which cannot extrapolate: an
        intervention value outside the observed range of ``variable`` is
        treated as the nearest observed value, and a warning is raised.

        Examples
        --------
        >>> effect = simulator.simulate_intervention('page_load_time', 2.0)
        >>> print(f"Effect: {effect.point_estimate:.3f}")
        >>> print(f"95% CI: [{effect.ci_lower:.3f}, {effect.ci_upper:.3f}]")
        """
        if self._scm is None:
            raise RuntimeError("Must call discover_graph() first or set_graph()")

        if variable not in self.data.columns:
            raise ValueError(f"Variable '{variable}' not in data")

        if variable == self.target:
            raise ValueError("Cannot intervene on the target variable itself")

        if method not in ('monte_carlo', 'scm', 'backdoor'):
            raise ValueError(f"Unknown method: {method}")

        if n_bootstrap is None:
            n_bootstrap = min(int(n_simulations), 100)
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2")

        # Get original value
        original_value = self.data[variable].mean()

        # Apply conditions if specified
        data = self.data
        if condition is not None:
            mask = np.ones(len(data), dtype=bool)
            for k, v in condition.items():
                mask &= (data[k] == v)
            data = data[mask]

        if len(data) == 0:
            raise ValueError("No data matches the specified conditions")

        if method != 'backdoor':
            lo, hi = self.data[variable].min(), self.data[variable].max()
            if value < lo or value > hi:
                warnings.warn(
                    f"Intervention value {value} for '{variable}' is outside its observed "
                    f"range [{lo:.4g}, {hi:.4g}]. Tree-based mechanisms cannot extrapolate, "
                    "so the effect is estimated as if the value were at the nearest "
                    "observed extreme.",
                    UserWarning, stacklevel=2
                )

        if method == 'backdoor':
            adjustment_set = self._get_adjustment_set(variable)

            def estimate(frame):
                return self._backdoor_adjustment(frame, variable, value, adjustment_set)
        else:
            def estimate(frame):
                return self._scm_effect(frame, variable, value)

        point_estimate = float(estimate(data))

        n = len(data)
        effects = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            idx = self._rng.integers(0, n, size=n)
            effects[i] = estimate(data.iloc[idx])

        ci_lower, ci_upper = np.percentile(effects, [2.5, 97.5])

        # Two-sided p-value against zero effect. The bootstrap standard deviation
        # is itself the standard error of the estimate; it is not divided by the
        # number of replicates.
        se = float(np.std(effects, ddof=1))
        if se > 0:
            p_value = float(2 * stats.norm.sf(abs(point_estimate) / se))
        else:
            p_value = 1.0 if point_estimate == 0 else 0.0

        return CausalEffect(
            variable=variable,
            intervention_value=value,
            original_value=original_value,
            target=self.target,
            point_estimate=point_estimate,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=p_value,
            n_simulations=n_bootstrap,
            method=method,
            samples=effects,
            baseline_outcome=float(data[self.target].mean())
        )

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def _parents(self, var: str) -> List[str]:
        return list((self._graph or {}).get(var, []))

    def _descendants(self, var: str) -> set:
        """All variables downstream of ``var`` in the graph."""
        graph = self._graph or {}
        children: Dict[str, List[str]] = {}
        for child, parents in graph.items():
            for p in parents:
                children.setdefault(p, []).append(child)
        seen: set = set()
        stack = list(children.get(var, []))
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(children.get(v, []))
        return seen

    def _ancestors(self, var: str) -> set:
        """All variables upstream of ``var`` in the graph."""
        seen: set = set()
        stack = self._parents(var)
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(self._parents(v))
        return seen

    def _scm_effect(self, frame: pd.DataFrame, variable: str, value: float) -> float:
        """Effect of do(variable = value) on the target, with mechanisms fitted on ``frame``.

        Only the mechanisms on a causal path from ``variable`` to the target are
        fitted: those of the target and of the mediators between the two. If the
        target is not downstream of ``variable`` the effect is exactly zero.
        """
        downstream = self._descendants(variable)
        if self.target not in downstream:
            return 0.0

        on_path = (downstream & self._ancestors(self.target)) | {self.target}
        inputs = set()
        for v in on_path:
            inputs.update(self._parents(v))
        columns = [c for c in frame.columns if c in on_path or c in inputs or c == variable]

        # Variables off the path keep their observed values, so they are roots here.
        subgraph = {c: (self._parents(c) if c in on_path else []) for c in columns}
        scm = StructuralCausalModel(graph=subgraph)
        sub = frame[columns]
        scm.fit(sub)
        counterfactual = scm.counterfactual(intervention={variable: value}, data=sub)
        return float(np.mean(counterfactual[self.target].values) - np.mean(sub[self.target].values))

    def _get_adjustment_set(self, treatment: str) -> List[str]:
        """Backdoor adjustment set for the effect of ``treatment`` on the target.

        With confounders supplied by the user, those are used. Otherwise the
        parents of the treatment in the graph are used, which block every
        backdoor path when the graph has no hidden common causes. Variables
        downstream of the treatment (mediators, the target's other effects) are
        never adjusted for: doing so removes or distorts the effect.
        """
        downstream = self._descendants(treatment) if self._graph is not None else set()
        candidates = list(self.confounders) if self.confounders else self._parents(treatment)

        dropped = [c for c in candidates if c in downstream]
        if dropped:
            warnings.warn(
                f"Not adjusting for {dropped}: downstream of '{treatment}' in the graph.",
                UserWarning, stacklevel=3
            )
        return [c for c in candidates
                if c not in downstream and c != treatment and c != self.target]

    def _backdoor_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        value: float,
        adjustment_set: List[str]
    ) -> float:
        """Backdoor-adjusted effect from a linear outcome model."""
        from sklearn.linear_model import LinearRegression

        # Fit outcome model with treatment and the adjustment set
        X = data[[treatment] + adjustment_set].values
        y = data[self.target].values

        model = LinearRegression()
        model.fit(X, y)

        # Predict under intervention
        X_intervened = X.copy()
        X_intervened[:, 0] = value
        y_intervened = model.predict(X_intervened)

        return float(np.mean(y_intervened) - np.mean(y))

    def rank_drivers(
        self,
        standardize: bool = True,
        method: str = 'intervention',
        n_simulations: int = 500,
        n_bootstrap: Optional[int] = None
    ) -> DriverRanking:
        """
        Rank variables by their causal importance for the target.

        Each variable is shifted (by one standard deviation, or by 10% of its
        mean when ``standardize`` is False) and its total effect on the target
        is estimated with :meth:`simulate_intervention`, mediated paths included.
        Variables with no causal path to the target rank last with an effect of
        zero.

        Parameters
        ----------
        standardize : bool
            Shift each variable by one standard deviation (True) or by 10% of
            its mean (False)
        method : str
            Only 'intervention' is implemented
        n_simulations : int
            Upper bound on bootstrap refits per variable
        n_bootstrap : int, optional
            Bootstrap refits per variable. Defaults to ``min(n_simulations, 50)``.

        Returns
        -------
        DriverRanking
            Ranked list of (variable, effect) tuples

        Examples
        --------
        >>> drivers = simulator.rank_drivers()
        >>> for var, effect in drivers:
        ...     print(f"{var}: {effect:.3f}")
        """
        if self._scm is None:
            raise RuntimeError("Must call discover_graph() first")
        if method != 'intervention':
            raise ValueError(
                f"Unknown method: {method!r}. Only 'intervention' is implemented."
            )
        if n_bootstrap is None:
            n_bootstrap = min(int(n_simulations), 50)

        effects = {}
        effect_objects = {}

        for var in self._feature_vars:
            if var == self.target:
                continue

            # Compute standardized intervention
            current_mean = self.data[var].mean()
            current_std = self.data[var].std()

            if standardize and current_std > 0:
                # One standard deviation increase
                intervention_value = current_mean + current_std
            else:
                # 10% increase
                intervention_value = current_mean * 1.1

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    effect = self.simulate_intervention(
                        variable=var,
                        value=intervention_value,
                        method='monte_carlo',
                        n_bootstrap=n_bootstrap
                    )
                effects[var] = effect.point_estimate
                effect_objects[var] = effect
            except Exception as e:
                # A variable whose effect could not be estimated is left out of
                # the ranking; reporting it as zero would read as "no effect".
                warnings.warn(f"Could not compute effect for {var}: {e}; it is not ranked.")

        # Sort by absolute effect
        sorted_effects = sorted(
            effects.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return DriverRanking(
            target=self.target,
            drivers=sorted_effects,
            effects=effect_objects,
            method=method
        )

    def validate(
        self,
        variable: Optional[str] = None,
        n_simulations: int = 500,
        n_bootstrap: Optional[int] = None
    ) -> SensitivityResult:
        """
        Run sensitivity analysis on a causal effect.

        Four checks are run for the effect of raising ``variable`` by one
        standard deviation:

        - **Robustness value** (Cinelli and Hazlett, 2020): the share of residual
          variance an unobserved confounder would have to explain, in both the
          variable and the target, to bring the effect to zero. Near 0 means
          fragile, near 1 means very robust.
        - **Confounding bounds**: the range the effect could take if such a
          confounder explained 10%, 20%, 30% or 50% of both.
        - **Placebo treatment**: the variable is shuffled, which destroys any
          real effect, and the effect is estimated again. It should not be
          significant.
        - **Refutations**: adding an unrelated variable should not move the
          estimate, and half-samples should give estimates of the same sign.

        The robustness value, the bounds and the refutations use a linear
        outcome model with the backdoor adjustment set read from the graph.

        Parameters
        ----------
        variable : str, optional
            Variable to validate. If None, validates the top driver.
        n_simulations : int
            Upper bound on bootstrap refits
        n_bootstrap : int, optional
            Bootstrap refits for the effect estimates. Defaults to
            ``min(n_simulations, 50)``.

        Returns
        -------
        SensitivityResult
            Sensitivity analysis results

        Examples
        --------
        >>> sensitivity = simulator.validate('price')
        >>> print(sensitivity.summary())
        >>> sensitivity.robustness_value
        """
        if self._scm is None:
            raise RuntimeError("Must call discover_graph() first")
        if n_bootstrap is None:
            n_bootstrap = min(int(n_simulations), 50)

        # Get variable to validate
        if variable is None:
            drivers = self.rank_drivers(n_bootstrap=min(n_bootstrap, 20))
            if not len(drivers):
                raise ValueError("No variable with an estimable effect to validate")
            variable = drivers[0][0]
        if variable not in self.data.columns:
            raise ValueError(f"Variable '{variable}' not in data")
        if variable == self.target:
            raise ValueError("Cannot validate the target variable against itself")

        shift = float(self.data[variable].std())
        value = float(self.data[variable].mean()) + shift

        # Original effect
        original_effect = self.simulate_intervention(
            variable=variable, value=value, n_bootstrap=n_bootstrap
        )

        fit = self._linear_fit(self.data, variable, self._get_adjustment_set(variable))

        confounding_bounds = self._compute_confounding_bounds(fit, shift)
        robustness_value = self._compute_robustness_value(fit)
        placebo_results = self._run_placebo_tests(variable, value, n_bootstrap)
        refutation_results = self._run_refutations(variable, fit, shift, placebo_results)

        return SensitivityResult(
            original_effect=original_effect,
            confounding_bounds=confounding_bounds,
            placebo_results={k: v for k, v in placebo_results.items() if k != 'placebo_effect'},
            refutation_results=refutation_results,
            robustness_value=robustness_value
        )

    def _linear_fit(
        self,
        data: pd.DataFrame,
        variable: str,
        adjustment_set: List[str],
        extra: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Least-squares fit of the target on the variable and an adjustment set.

        Returns the variable's coefficient, its standard error, the residual
        degrees of freedom and the t statistic. ``extra`` is one more regressor.
        """
        y = data[self.target].to_numpy(dtype=float)
        cols = [data[variable].to_numpy(dtype=float)]
        cols += [data[c].to_numpy(dtype=float) for c in adjustment_set]
        if extra is not None:
            cols.append(np.asarray(extra, dtype=float))
        X = np.column_stack([np.ones(len(data))] + cols)
        n, k = X.shape
        df = n - k
        if df <= 0:
            raise ValueError("Not enough observations for the linear sensitivity model")
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        if rank < k:
            raise ValueError(
                f"'{variable}' is constant or collinear with its adjustment set; "
                "its effect is not identified by a linear model"
            )
        resid = y - X @ beta
        sigma2 = float(resid @ resid) / df
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(cov[1, 1]))
        coef = float(beta[1])
        t = coef / se if se > 0 else np.inf
        return {'coef': coef, 'se': se, 'df': float(df), 't': float(t)}

    def _compute_confounding_bounds(
        self,
        fit: Dict[str, float],
        shift: float
    ) -> Dict[str, Tuple[float, float]]:
        """Range of the effect under an unobserved confounder of a given strength.

        For a confounder that explains a share ``s`` of the residual variance of
        both the variable and the target, the largest possible bias in the
        coefficient is ``se * sqrt(df) * sqrt(s * s / (1 - s))`` (Cinelli and
        Hazlett, 2020, with equal partial R-squared on both sides).
        """
        bounds = {}
        for strength in [0.1, 0.2, 0.3, 0.5]:
            bias = fit['se'] * np.sqrt(fit['df']) * np.sqrt(strength * strength / (1 - strength))
            bounds[str(strength)] = (
                float((fit['coef'] - bias) * shift),
                float((fit['coef'] + bias) * shift)
            )
        return bounds

    def _run_placebo_tests(
        self,
        variable: str,
        value: float,
        n_bootstrap: int
    ) -> Dict[str, float]:
        """Placebo and stability checks, each estimated like the original effect."""
        results = {}

        # Placebo treatment: shuffling the variable breaks its link to everything
        # else, so the same estimator should report nothing significant.
        placebo_data = self.data.copy()
        placebo_data[variable] = self._rng.permutation(placebo_data[variable].to_numpy())
        placebo_sim = CausalSimulator(
            placebo_data, target=self.target, confounders=self.confounders,
            random_state=int(self._rng.integers(0, 2**31 - 1))
        )
        placebo_sim.set_graph(self._graph)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            placebo = placebo_sim.simulate_intervention(variable, value, n_bootstrap=n_bootstrap)
        results['placebo_treatment'] = float(placebo.p_value)
        results['placebo_effect'] = float(placebo.point_estimate)

        # Subset stability: share of half-samples whose estimate has the sign of
        # the full-sample estimate.
        adjustment_set = self._get_adjustment_set(variable)
        full = self._linear_fit(self.data, variable, adjustment_set)['coef']
        n = len(self.data)
        same_sign = []
        for _ in range(40):
            idx = self._rng.choice(n, size=max(n // 2, len(adjustment_set) + 3), replace=False)
            try:
                coef = self._linear_fit(self.data.iloc[idx], variable, adjustment_set)['coef']
            except ValueError:
                continue
            same_sign.append(np.sign(coef) == np.sign(full))
        results['subset_stability'] = float(np.mean(same_sign)) if same_sign else float('nan')

        return results

    def _run_refutations(
        self,
        variable: str,
        fit: Dict[str, float],
        shift: float,
        placebo_results: Dict[str, float]
    ) -> Dict[str, Any]:
        """Refutation checks on the linear backdoor estimate."""
        results = {}
        adjustment_set = self._get_adjustment_set(variable)

        # Random common cause: an unrelated variable added to the adjustment set
        # should move the coefficient by less than its own sampling error.
        noise = self._rng.standard_normal(len(self.data))
        with_noise = self._linear_fit(self.data, variable, adjustment_set, extra=noise)
        change = abs(with_noise['coef'] - fit['coef'])
        results['random_common_cause'] = {
            'original_coef': fit['coef'],
            'new_coef': with_noise['coef'],
            'change': float(change),
            'passed': bool(change < 2 * fit['se'])
        }

        # Placebo treatment: the shuffled variable should have no significant effect.
        results['placebo_treatment'] = {
            'effect': placebo_results['placebo_effect'],
            'p_value': placebo_results['placebo_treatment'],
            'passed': bool(placebo_results['placebo_treatment'] > 0.05)
        }

        # Data subsets: half-samples should agree with the full sample on sign.
        stability = placebo_results['subset_stability']
        results['data_subset'] = {
            'same_sign_share': stability,
            'passed': bool(stability >= 0.9)
        }

        return results

    def _compute_robustness_value(self, fit: Dict[str, float]) -> float:
        """Robustness value of the linear estimate (Cinelli and Hazlett, 2020).

        ``RV = (sqrt(f**4 + 4 f**2) - f**2) / 2`` with ``f = |t| / sqrt(df)``:
        the partial R-squared an unobserved confounder would need with both the
        variable and the target to reduce the estimate to zero.
        """
        if not np.isfinite(fit['t']):
            return 1.0
        f2 = fit['t'] ** 2 / fit['df']
        return float(0.5 * (np.sqrt(f2 * f2 + 4 * f2) - f2))

    def optimize_policy(
        self,
        objective: str = 'maximize',
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        n_iterations: int = 100,
        grid_points: int = 5
    ) -> OptimalPolicy:
        """
        Find the settings of the treatment variables that maximize or minimize the target.

        Every combination on a grid is evaluated by setting the variables jointly
        and propagating the intervention through the graph, so effects through
        mediators count. The search covers ``treatment_vars`` if they were given,
        otherwise up to five variables that cause the target according to the
        graph.

        Parameters
        ----------
        objective : str
            A string containing 'max' or 'min' (for example 'maximize')
        constraints : dict
            Variable constraints as {var: (min, max)}. Defaults to each
            variable's observed range, which is also as far as the model can be
            trusted.
        n_iterations : int
            Unused; kept for backward compatibility
        grid_points : int
            Grid values per variable

        Returns
        -------
        OptimalPolicy
            Optimal settings for treatment variables

        Examples
        --------
        >>> optimal = simulator.optimize_policy(
        ...     objective='maximize',
        ...     constraints={'price': (10, 50)}
        ... )
        >>> print(optimal.settings)
        """
        if self._scm is None:
            raise RuntimeError("Must call discover_graph() first")

        goal = objective.lower()
        if ('max' in goal) == ('min' in goal):
            raise ValueError("objective must contain either 'max' or 'min'")
        maximize = 'max' in goal

        # Variables to optimize
        if self.treatment_vars:
            vars_to_optimize = [v for v in self.treatment_vars if v != self.target]
        else:
            causes = self._ancestors(self.target)
            vars_to_optimize = [v for v in self._feature_vars if v in causes][:5]
        if not vars_to_optimize:
            raise ValueError(
                "No variable to optimize: pass treatment_vars, or use a graph in which "
                "some variable causes the target"
            )

        # Set default constraints
        constraints = dict(constraints or {})
        for var in vars_to_optimize:
            if var not in constraints:
                constraints[var] = (
                    float(self.data[var].min()),
                    float(self.data[var].max())
                )

        from itertools import product

        grids = {}
        for var in vars_to_optimize:
            low, high = constraints[var]
            if low > high:
                raise ValueError(f"Constraint for '{var}' has min above max")
            grids[var] = np.linspace(low, high, grid_points)

        baseline = float(self.data[self.target].mean())
        best_settings: Dict[str, float] = {}
        best_outcome = float('-inf') if maximize else float('inf')

        for values in product(*grids.values()):
            settings = {v: float(x) for v, x in zip(vars_to_optimize, values)}
            outcome = float(
                self._scm.counterfactual(intervention=settings, data=self.data)[self.target].mean()
            )
            if (outcome > best_outcome) if maximize else (outcome < best_outcome):
                best_outcome = outcome
                best_settings = settings

        return OptimalPolicy(
            objective=objective,
            settings=best_settings,
            expected_outcome=best_outcome,
            baseline_outcome=baseline,
            improvement=best_outcome - baseline,
            constraints=constraints
        )

    def plot_graph(self, ax=None, figsize=(12, 8)):
        """
        Visualize the learned causal graph.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        figsize : tuple
            Figure size
            
        Returns
        -------
        matplotlib.axes.Axes
        """
        if self._graph is None:
            raise RuntimeError("Must call discover_graph() first")
        
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Requires networkx and matplotlib")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        
        # Create networkx graph
        G = nx.DiGraph()
        
        for child, parents in self._graph.items():
            G.add_node(child)
            for parent in parents:
                G.add_edge(parent, child)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color nodes
        node_colors = []
        for node in G.nodes():
            if node == self.target:
                node_colors.append('lightcoral')
            elif node in self.treatment_vars:
                node_colors.append('lightgreen')
            elif node in self.confounders:
                node_colors.append('lightyellow')
            else:
                node_colors.append('lightblue')
        
        # Draw
        nx.draw(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=2000,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray'
        )
        
        ax.set_title('Learned Causal Graph')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightcoral', label='Target'),
            Patch(facecolor='lightgreen', label='Treatment'),
            Patch(facecolor='lightyellow', label='Confounder'),
            Patch(facecolor='lightblue', label='Other'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
        
        return ax
    
    def plot_effects(self, top_n: int = 10, ax=None):
        """Plot causal effects of top drivers."""
        drivers = self.rank_drivers()
        return drivers.plot(ax=ax, top_n=top_n)
    
    def to_dowhy(self):
        """Convert to DoWhy CausalModel."""
        from .adapters import to_dowhy
        return to_dowhy(self)
    
    def to_econml(self):
        """Convert to EconML estimator."""
        from .adapters import to_econml
        return to_econml(self)
    
    def summary(self) -> str:
        """Return summary of the simulator state."""
        lines = [
            "CausalSimulator Summary",
            "=" * 50,
            f"Target variable: {self.target}",
            f"Treatment variables: {self.treatment_vars}",
            f"Confounders: {self.confounders}",
            f"Total observations: {len(self.data)}",
            f"Total variables: {len(self._all_vars)}",
            "",
            f"Graph discovered: {'Yes' if self._graph else 'No'}",
            f"SCM fitted: {'Yes' if self._scm else 'No'}",
        ]
        
        if self._graph:
            n_edges = sum(len(parents) for parents in self._graph.values())
            lines.append(f"Causal edges: {n_edges}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"CausalSimulator(target='{self.target}', n_obs={len(self.data)})"
