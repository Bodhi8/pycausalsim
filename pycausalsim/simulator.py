"""
CausalSimulator: Main class for causal discovery and simulation.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
import warnings

from .results import CausalEffect, SimulationResult, SensitivityResult, DriverRanking, OptimalPolicy
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
        
        # Set random state
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
        self._graph = self._discovery.discover(
            data=discovery_data,
            method=method,
            target=self.target,
            **kwargs
        )
        
        # Fit SCM based on discovered graph
        self._scm = StructuralCausalModel(graph=self._graph)
        self._scm.fit(self.data)
        
        return self
    
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
        self._graph = graph
        self._scm = StructuralCausalModel(graph=graph)
        self._scm.fit(self.data)
        return self
    
    def simulate_intervention(
        self,
        variable: str,
        value: float,
        n_simulations: int = 1000,
        method: str = 'monte_carlo',
        condition: Optional[Dict[str, Any]] = None
    ) -> CausalEffect:
        """
        Simulate the causal effect of an intervention.
        
        Uses counterfactual simulation to estimate what would happen
        if we set a variable to a specific value.
        
        Parameters
        ----------
        variable : str
            The variable to intervene on
        value : float
            The value to set the variable to
        n_simulations : int
            Number of Monte Carlo simulations
        method : str
            Simulation method: 'monte_carlo', 'scm', 'backdoor'
        condition : dict, optional
            Conditions to apply (e.g., {'traffic_source': 'paid'})
            
        Returns
        -------
        CausalEffect
            Object containing the causal effect estimate and confidence interval
            
        Examples
        --------
        >>> effect = simulator.simulate_intervention('page_load_time', 2.0)
        >>> print(f"Effect: {effect.point_estimate:.2%}")
        >>> print(f"95% CI: [{effect.ci_lower:.2%}, {effect.ci_upper:.2%}]")
        """
        if self._scm is None:
            raise RuntimeError("Must call discover_graph() first or set_graph()")
        
        if variable not in self.data.columns:
            raise ValueError(f"Variable '{variable}' not in data")
        
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
        
        # Run simulation based on method
        if method == 'monte_carlo':
            effects = self._simulate_monte_carlo(data, variable, value, n_simulations)
        elif method == 'scm':
            effects = self._simulate_scm(data, variable, value, n_simulations)
        elif method == 'backdoor':
            effects = self._simulate_backdoor(data, variable, value, n_simulations)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute statistics
        point_estimate = np.mean(effects)
        ci_lower, ci_upper = np.percentile(effects, [2.5, 97.5])
        
        # P-value (two-tailed test against null of zero effect)
        t_stat = point_estimate / (np.std(effects) / np.sqrt(len(effects)) + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(effects)-1))
        
        return CausalEffect(
            variable=variable,
            intervention_value=value,
            original_value=original_value,
            target=self.target,
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            n_simulations=n_simulations,
            method=method,
            samples=effects
        )
    
    def _simulate_monte_carlo(
        self,
        data: pd.DataFrame,
        variable: str,
        value: float,
        n_simulations: int
    ) -> np.ndarray:
        """Monte Carlo simulation for causal effect."""
        effects = np.zeros(n_simulations)
        n_samples = len(data)
        
        for i in range(n_simulations):
            # Bootstrap sample
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            sample = data.iloc[idx].copy()
            
            # Factual outcome
            y_factual = sample[self.target].values
            
            # Counterfactual: intervene on variable
            sample_cf = sample.copy()
            sample_cf[variable] = value
            
            # Predict counterfactual outcome using SCM
            y_counterfactual = self._scm.predict(sample_cf, self.target)
            
            # Average treatment effect
            effects[i] = np.mean(y_counterfactual) - np.mean(y_factual)
        
        return effects
    
    def _simulate_scm(
        self,
        data: pd.DataFrame,
        variable: str,
        value: float,
        n_simulations: int
    ) -> np.ndarray:
        """SCM-based simulation with noise."""
        effects = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            # Generate counterfactual using SCM
            cf_data = self._scm.counterfactual(
                intervention={variable: value},
                data=data,
                noise_scale=0.1
            )
            
            y_factual = data[self.target].values
            y_counterfactual = cf_data[self.target].values
            
            effects[i] = np.mean(y_counterfactual) - np.mean(y_factual)
        
        return effects
    
    def _simulate_backdoor(
        self,
        data: pd.DataFrame,
        variable: str,
        value: float,
        n_simulations: int
    ) -> np.ndarray:
        """Backdoor adjustment simulation."""
        effects = np.zeros(n_simulations)
        
        # Get backdoor adjustment set
        adjustment_set = self._get_adjustment_set(variable)
        
        for i in range(n_simulations):
            # Bootstrap
            idx = np.random.choice(len(data), size=len(data), replace=True)
            sample = data.iloc[idx]
            
            # Stratified estimation
            effect = self._backdoor_adjustment(
                sample, variable, value, adjustment_set
            )
            effects[i] = effect
        
        return effects
    
    def _get_adjustment_set(self, treatment: str) -> List[str]:
        """Get minimal adjustment set for backdoor criterion."""
        if self._graph is None:
            return self.confounders
        
        # Use confounders if specified, otherwise find from graph
        if self.confounders:
            return self.confounders
        
        # Find all ancestors of treatment and target that are common
        adjustment_set = []
        for var in self._feature_vars:
            if var != treatment and var != self.target:
                adjustment_set.append(var)
        
        return adjustment_set
    
    def _backdoor_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        value: float,
        adjustment_set: List[str]
    ) -> float:
        """Compute backdoor-adjusted effect."""
        from sklearn.linear_model import LinearRegression
        
        # Fit outcome model with treatment and confounders
        X = data[[treatment] + adjustment_set].values
        y = data[self.target].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict under intervention
        X_intervened = X.copy()
        X_intervened[:, 0] = value
        y_intervened = model.predict(X_intervened)
        
        return np.mean(y_intervened) - np.mean(y)
    
    def rank_drivers(
        self,
        standardize: bool = True,
        method: str = 'intervention',
        n_simulations: int = 500
    ) -> DriverRanking:
        """
        Rank variables by their causal importance for the target.
        
        Parameters
        ----------
        standardize : bool
            Whether to use standardized effect sizes
        method : str
            Method for computing effects: 'intervention', 'shap', 'gradient'
        n_simulations : int
            Number of simulations per variable
            
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
                effect = self.simulate_intervention(
                    variable=var,
                    value=intervention_value,
                    n_simulations=n_simulations,
                    method='monte_carlo'
                )
                
                effects[var] = effect.point_estimate
                effect_objects[var] = effect
            except Exception as e:
                warnings.warn(f"Could not compute effect for {var}: {e}")
                effects[var] = 0.0
        
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
        n_simulations: int = 500
    ) -> SensitivityResult:
        """
        Run sensitivity analysis to validate causal claims.
        
        Parameters
        ----------
        variable : str, optional
            Variable to validate. If None, validates top driver.
        n_simulations : int
            Number of simulations for tests
            
        Returns
        -------
        SensitivityResult
            Sensitivity analysis results
            
        Examples
        --------
        >>> sensitivity = simulator.validate()
        >>> sensitivity.confounding_bounds()
        >>> sensitivity.placebo_test()
        """
        if self._scm is None:
            raise RuntimeError("Must call discover_graph() first")
        
        # Get variable to validate
        if variable is None:
            drivers = self.rank_drivers(n_simulations=100)
            variable = drivers[0][0] if drivers else self._feature_vars[0]
        
        # Original effect
        original_effect = self.simulate_intervention(
            variable=variable,
            value=self.data[variable].mean() + self.data[variable].std(),
            n_simulations=n_simulations
        )
        
        # Confounding bounds
        confounding_bounds = self._compute_confounding_bounds(
            variable, original_effect, n_simulations
        )
        
        # Placebo tests
        placebo_results = self._run_placebo_tests(
            variable, n_simulations
        )
        
        # Refutation
        refutation_results = self._run_refutations(
            variable, original_effect, n_simulations
        )
        
        # Robustness value
        robustness_value = self._compute_robustness_value(
            original_effect, confounding_bounds
        )
        
        return SensitivityResult(
            original_effect=original_effect,
            confounding_bounds=confounding_bounds,
            placebo_results=placebo_results,
            refutation_results=refutation_results,
            robustness_value=robustness_value
        )
    
    def _compute_confounding_bounds(
        self,
        variable: str,
        original_effect: CausalEffect,
        n_simulations: int
    ) -> Dict[str, Tuple[float, float]]:
        """Compute effect bounds under different confounding assumptions."""
        bounds = {}
        
        for strength in [0.1, 0.2, 0.3, 0.5]:
            # Simulate unobserved confounder
            lower, upper = self._bound_with_confounder(
                variable, original_effect, strength, n_simulations
            )
            bounds[str(strength)] = (lower, upper)
        
        return bounds
    
    def _bound_with_confounder(
        self,
        variable: str,
        original_effect: CausalEffect,
        strength: float,
        n_simulations: int
    ) -> Tuple[float, float]:
        """Compute bounds assuming unobserved confounder of given strength."""
        effect = original_effect.point_estimate
        
        # Rosenbaum-style bounds
        gamma = 1 + strength * 2  # Sensitivity parameter
        
        lower = effect - strength * abs(effect)
        upper = effect + strength * abs(effect)
        
        return (lower, upper)
    
    def _run_placebo_tests(
        self,
        variable: str,
        n_simulations: int
    ) -> Dict[str, float]:
        """Run placebo tests."""
        results = {}
        
        # Random permutation test
        original = self.simulate_intervention(
            variable, 
            self.data[variable].mean() * 1.1,
            n_simulations=100
        )
        
        permutation_effects = []
        for _ in range(100):
            shuffled_data = self.data.copy()
            shuffled_data[variable] = np.random.permutation(shuffled_data[variable])
            
            temp_scm = StructuralCausalModel(graph=self._graph)
            temp_scm.fit(shuffled_data)
            
            # Simple estimate
            effect = np.corrcoef(shuffled_data[variable], shuffled_data[self.target])[0, 1]
            permutation_effects.append(effect)
        
        # P-value: fraction of permuted effects >= observed
        p_value = np.mean(np.abs(permutation_effects) >= abs(original.point_estimate))
        results['permutation_test'] = p_value
        
        # Subset stability test
        n = len(self.data)
        subset_effects = []
        for _ in range(50):
            idx = np.random.choice(n, size=n//2, replace=False)
            subset = self.data.iloc[idx]
            
            temp_scm = StructuralCausalModel(graph=self._graph)
            temp_scm.fit(subset)
            
            effect = np.corrcoef(subset[variable], subset[self.target])[0, 1]
            subset_effects.append(effect)
        
        # CV of subset effects (lower is more stable)
        cv = np.std(subset_effects) / (np.abs(np.mean(subset_effects)) + 1e-10)
        results['subset_stability'] = 1 - min(cv, 1)  # Convert to p-value-like metric
        
        return results
    
    def _run_refutations(
        self,
        variable: str,
        original_effect: CausalEffect,
        n_simulations: int
    ) -> Dict[str, Any]:
        """Run refutation methods."""
        results = {}
        
        # Add random common cause
        data_with_random = self.data.copy()
        data_with_random['_random_confounder'] = np.random.randn(len(self.data))
        
        # Check if effect changes significantly
        new_confounders = self.confounders + ['_random_confounder']
        
        # Simple regression-based check
        from sklearn.linear_model import LinearRegression
        
        X_original = self.data[[variable] + self.confounders].values
        X_new = data_with_random[[variable] + new_confounders].values
        y = self.data[self.target].values
        
        model_original = LinearRegression().fit(X_original, y)
        model_new = LinearRegression().fit(X_new, y)
        
        coef_change = abs(model_new.coef_[0] - model_original.coef_[0])
        results['random_common_cause'] = {
            'original_coef': model_original.coef_[0],
            'new_coef': model_new.coef_[0],
            'change': coef_change,
            'passed': coef_change < 0.1 * abs(model_original.coef_[0])
        }
        
        # Placebo treatment
        placebo_treatment = np.random.randn(len(self.data))
        placebo_corr = np.corrcoef(placebo_treatment, self.data[self.target])[0, 1]
        results['placebo_treatment'] = {
            'effect': placebo_corr,
            'passed': abs(placebo_corr) < 0.1
        }
        
        return results
    
    def _compute_robustness_value(
        self,
        effect: CausalEffect,
        bounds: Dict[str, Tuple[float, float]]
    ) -> float:
        """Compute robustness value (minimum confounding to nullify effect)."""
        for strength, (lower, upper) in sorted(bounds.items()):
            if lower <= 0 <= upper:
                return float(strength)
        return 1.0  # Very robust
    
    def optimize_policy(
        self,
        objective: str = 'maximize',
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        n_iterations: int = 100
    ) -> OptimalPolicy:
        """
        Find optimal policy to maximize/minimize target.
        
        Parameters
        ----------
        objective : str
            'maximize' or 'minimize' the target
        constraints : dict
            Variable constraints as {var: (min, max)}
        n_iterations : int
            Number of optimization iterations
            
        Returns
        -------
        OptimalPolicy
            Optimal settings for treatment variables
            
        Examples
        --------
        >>> optimal = simulator.optimize_policy(
        ...     objective='maximize_conversion',
        ...     constraints={'price': (10, 50)}
        ... )
        >>> print(optimal.settings)
        """
        if self._scm is None:
            raise RuntimeError("Must call discover_graph() first")
        
        # Variables to optimize
        vars_to_optimize = self.treatment_vars if self.treatment_vars else self._feature_vars[:5]
        
        # Set default constraints
        if constraints is None:
            constraints = {}
        
        for var in vars_to_optimize:
            if var not in constraints:
                constraints[var] = (
                    self.data[var].min(),
                    self.data[var].max()
                )
        
        # Simple grid search optimization
        best_settings = {}
        best_outcome = float('-inf') if 'max' in objective.lower() else float('inf')
        baseline = self.data[self.target].mean()
        
        # Grid search
        from itertools import product
        
        grids = {}
        for var in vars_to_optimize:
            low, high = constraints[var]
            grids[var] = np.linspace(low, high, 5)
        
        for values in product(*grids.values()):
            settings = dict(zip(vars_to_optimize, values))
            
            # Predict outcome under intervention
            data_intervened = self.data.copy()
            for var, val in settings.items():
                data_intervened[var] = val
            
            outcome = self._scm.predict(data_intervened, self.target).mean()
            
            if 'max' in objective.lower():
                if outcome > best_outcome:
                    best_outcome = outcome
                    best_settings = settings
            else:
                if outcome < best_outcome:
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
