"""
Marketing Attribution with Causal Inference.

Move beyond last-touch attribution to understand true incremental value
of each marketing channel.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings


class MarketingAttribution:
    """
    Causal marketing attribution using multiple methods.
    
    Unlike last-touch or first-touch attribution, this uses causal inference
    to determine the true incremental contribution of each touchpoint.
    
    Parameters
    ----------
    data : pd.DataFrame
        Touchpoint data with conversions
    conversion_col : str
        Name of conversion column
    touchpoint_cols : list of str
        Names of touchpoint columns
    user_col : str, optional
        User identifier column
        
    Examples
    --------
    >>> attr = MarketingAttribution(
    ...     data=touchpoint_data,
    ...     conversion_col='converted',
    ...     touchpoint_cols=['email', 'display', 'search', 'social', 'direct']
    ... )
    >>> attr.fit(method='shapley')
    >>> weights = attr.get_attribution()
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        conversion_col: str = 'converted',
        touchpoint_cols: Optional[List[str]] = None,
        user_col: Optional[str] = None,
        random_state: Optional[int] = None
    ):
        self.data = data.copy()
        self.conversion_col = conversion_col
        self.user_col = user_col
        self.touchpoint_cols = touchpoint_cols or self._detect_touchpoints()
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Validate
        self._validate_inputs()
        
        # Fitted components
        self._attribution = None
        self._method = None
        self._model = None
        self._incremental_effects = None
    
    def _detect_touchpoints(self) -> List[str]:
        """Auto-detect touchpoint columns."""
        exclude = {self.conversion_col, self.user_col}
        binary_cols = []
        for col in self.data.columns:
            if col not in exclude:
                unique = self.data[col].nunique()
                if unique == 2 or (unique <= 10 and self.data[col].dtype in ['int64', 'float64']):
                    binary_cols.append(col)
        return binary_cols
    
    def _validate_inputs(self):
        """Validate input data."""
        if self.conversion_col not in self.data.columns:
            raise ValueError(f"Conversion column '{self.conversion_col}' not found")
        
        for col in self.touchpoint_cols:
            if col not in self.data.columns:
                raise ValueError(f"Touchpoint column '{col}' not found")
    
    def fit(
        self,
        method: str = 'shapley',
        **kwargs
    ) -> 'MarketingAttribution':
        """
        Fit attribution model.
        
        Parameters
        ----------
        method : str
            Attribution method:
            - 'shapley': Causal Shapley values
            - 'logistic': Logistic regression coefficients
            - 'uplift': Uplift-based attribution
            - 'markov': Markov chain attribution
            - 'linear': Linear regression
            
        Returns
        -------
        self
        """
        methods = {
            'shapley': self._fit_shapley,
            'logistic': self._fit_logistic,
            'uplift': self._fit_uplift,
            'markov': self._fit_markov,
            'linear': self._fit_linear,
            'last_touch': self._fit_last_touch,
            'first_touch': self._fit_first_touch,
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Unknown method: {method}")
        
        self._method = method.lower()
        methods[self._method](**kwargs)
        
        return self
    
    def _fit_shapley(self, n_samples: int = 1000, **kwargs):
        """
        Compute Shapley values for attribution.
        
        Uses cooperative game theory to fairly distribute credit.
        """
        n_touchpoints = len(self.touchpoint_cols)
        n_data = len(self.data)
        
        # Compute characteristic function for all coalitions
        def v(coalition: set) -> float:
            """Value function: conversion rate with these touchpoints."""
            if not coalition:
                return 0.0
            
            mask = np.ones(n_data, dtype=bool)
            for tp in coalition:
                mask &= (self.data[tp] > 0)
            
            if mask.sum() == 0:
                return 0.0
            
            return self.data.loc[mask, self.conversion_col].mean()
        
        # Monte Carlo Shapley approximation
        shapley_values = {tp: 0.0 for tp in self.touchpoint_cols}
        
        for _ in range(n_samples):
            # Random permutation
            perm = np.random.permutation(self.touchpoint_cols)
            
            coalition = set()
            prev_value = 0.0
            
            for tp in perm:
                coalition.add(tp)
                curr_value = v(coalition)
                
                # Marginal contribution
                shapley_values[tp] += (curr_value - prev_value) / n_samples
                prev_value = curr_value
        
        # Normalize to sum to 1
        total = sum(shapley_values.values())
        if total > 0:
            shapley_values = {k: v / total for k, v in shapley_values.items()}
        
        self._attribution = shapley_values
        self._incremental_effects = self._compute_incremental_effects()
    
    def _fit_logistic(self, **kwargs):
        """Logistic regression-based attribution."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        X = self.data[self.touchpoint_cols].values
        y = self.data[self.conversion_col].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        model = LogisticRegression(random_state=self.random_state)
        model.fit(X_scaled, y)
        
        self._model = model
        
        # Convert coefficients to attribution weights
        coefs = np.abs(model.coef_[0])
        total = coefs.sum()
        
        if total > 0:
            weights = coefs / total
        else:
            weights = np.ones(len(self.touchpoint_cols)) / len(self.touchpoint_cols)
        
        self._attribution = dict(zip(self.touchpoint_cols, weights))
        self._incremental_effects = self._compute_incremental_effects()
    
    def _fit_uplift(self, **kwargs):
        """Uplift-based attribution."""
        attribution = {}
        
        for tp in self.touchpoint_cols:
            # Treatment: exposed to touchpoint
            treated = self.data[self.data[tp] > 0]
            control = self.data[self.data[tp] == 0]
            
            if len(treated) > 0 and len(control) > 0:
                uplift = treated[self.conversion_col].mean() - control[self.conversion_col].mean()
                attribution[tp] = max(0, uplift)  # Only positive uplift
            else:
                attribution[tp] = 0.0
        
        # Normalize
        total = sum(attribution.values())
        if total > 0:
            attribution = {k: v / total for k, v in attribution.items()}
        
        self._attribution = attribution
        self._incremental_effects = self._compute_incremental_effects()
    
    def _fit_markov(self, **kwargs):
        """
        Markov chain attribution.
        
        Models customer journey as Markov chain and computes
        removal effects for each channel.
        """
        # Build transition matrix
        touchpoints = self.touchpoint_cols + ['conversion', 'null']
        n_states = len(touchpoints)
        state_idx = {s: i for i, s in enumerate(touchpoints)}
        
        transitions = np.zeros((n_states, n_states))
        
        # Count transitions from data
        for _, row in self.data.iterrows():
            journey = []
            for tp in self.touchpoint_cols:
                if row[tp] > 0:
                    journey.append(tp)
            
            if row[self.conversion_col] > 0:
                journey.append('conversion')
            else:
                journey.append('null')
            
            # Add transitions
            for i in range(len(journey) - 1):
                from_state = journey[i]
                to_state = journey[i + 1]
                transitions[state_idx[from_state], state_idx[to_state]] += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        trans_probs = transitions / row_sums
        
        # Compute baseline conversion probability
        baseline_conv = self._markov_conversion_prob(trans_probs, state_idx)
        
        # Compute removal effect for each channel
        removal_effects = {}
        for tp in self.touchpoint_cols:
            # Remove channel (set its transitions to null)
            modified = trans_probs.copy()
            tp_idx = state_idx[tp]
            
            # Redirect all traffic through this channel to null
            modified[:, tp_idx] = 0
            modified[tp_idx, :] = 0
            modified[tp_idx, state_idx['null']] = 1
            
            # Renormalize
            row_sums = modified.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            modified = modified / row_sums
            
            modified_conv = self._markov_conversion_prob(modified, state_idx)
            removal_effects[tp] = max(0, baseline_conv - modified_conv)
        
        # Normalize
        total = sum(removal_effects.values())
        if total > 0:
            attribution = {k: v / total for k, v in removal_effects.items()}
        else:
            attribution = {k: 1/len(self.touchpoint_cols) for k in self.touchpoint_cols}
        
        self._attribution = attribution
        self._incremental_effects = self._compute_incremental_effects()
    
    def _markov_conversion_prob(
        self,
        trans_probs: np.ndarray,
        state_idx: Dict[str, int]
    ) -> float:
        """Compute conversion probability from Markov chain."""
        n_states = trans_probs.shape[0]
        conv_idx = state_idx['conversion']
        null_idx = state_idx['null']
        
        # Start from equal distribution over touchpoints
        start_dist = np.zeros(n_states)
        for tp in self.touchpoint_cols:
            start_dist[state_idx[tp]] = 1.0 / len(self.touchpoint_cols)
        
        # Run chain until convergence
        state = start_dist.copy()
        for _ in range(100):
            state = state @ trans_probs
        
        return state[conv_idx]
    
    def _fit_linear(self, **kwargs):
        """Linear regression attribution."""
        from sklearn.linear_model import LinearRegression
        
        X = self.data[self.touchpoint_cols].values
        y = self.data[self.conversion_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        self._model = model
        
        # Use absolute coefficients
        coefs = np.abs(model.coef_)
        total = coefs.sum()
        
        if total > 0:
            weights = coefs / total
        else:
            weights = np.ones(len(self.touchpoint_cols)) / len(self.touchpoint_cols)
        
        self._attribution = dict(zip(self.touchpoint_cols, weights))
        self._incremental_effects = self._compute_incremental_effects()
    
    def _fit_last_touch(self, **kwargs):
        """Last-touch attribution (baseline)."""
        attribution = {tp: 0.0 for tp in self.touchpoint_cols}
        
        # For each conversion, credit the last touchpoint
        conversions = self.data[self.data[self.conversion_col] > 0]
        
        for _, row in conversions.iterrows():
            # Find last touchpoint (rightmost in list)
            for tp in reversed(self.touchpoint_cols):
                if row[tp] > 0:
                    attribution[tp] += 1
                    break
        
        # Normalize
        total = sum(attribution.values())
        if total > 0:
            attribution = {k: v / total for k, v in attribution.items()}
        
        self._attribution = attribution
        self._incremental_effects = self._compute_incremental_effects()
    
    def _fit_first_touch(self, **kwargs):
        """First-touch attribution (baseline)."""
        attribution = {tp: 0.0 for tp in self.touchpoint_cols}
        
        # For each conversion, credit the first touchpoint
        conversions = self.data[self.data[self.conversion_col] > 0]
        
        for _, row in conversions.iterrows():
            # Find first touchpoint
            for tp in self.touchpoint_cols:
                if row[tp] > 0:
                    attribution[tp] += 1
                    break
        
        # Normalize
        total = sum(attribution.values())
        if total > 0:
            attribution = {k: v / total for k, v in attribution.items()}
        
        self._attribution = attribution
        self._incremental_effects = self._compute_incremental_effects()
    
    def _compute_incremental_effects(self) -> Dict[str, float]:
        """Compute incremental effect of each touchpoint."""
        effects = {}
        
        for tp in self.touchpoint_cols:
            treated = self.data[self.data[tp] > 0][self.conversion_col].mean()
            control = self.data[self.data[tp] == 0][self.conversion_col].mean()
            effects[tp] = treated - control if not np.isnan(treated - control) else 0.0
        
        return effects
    
    def get_attribution(self) -> Dict[str, float]:
        """
        Get attribution weights.
        
        Returns
        -------
        dict
            Attribution weights for each touchpoint (sum to 1)
        """
        if self._attribution is None:
            raise RuntimeError("Must call fit() first")
        
        return self._attribution.copy()
    
    def get_incremental_effects(self) -> Dict[str, float]:
        """
        Get incremental effects of each touchpoint.
        
        Returns
        -------
        dict
            Incremental conversion lift per touchpoint
        """
        if self._incremental_effects is None:
            raise RuntimeError("Must call fit() first")
        
        return self._incremental_effects.copy()
    
    def optimize_budget(
        self,
        total_budget: float,
        cost_per_touchpoint: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, float]:
        """
        Optimize budget allocation across touchpoints.
        
        Parameters
        ----------
        total_budget : float
            Total budget to allocate
        cost_per_touchpoint : dict, optional
            Cost per unit for each touchpoint
        constraints : dict, optional
            Min/max constraints: {touchpoint: (min, max)}
            
        Returns
        -------
        dict
            Optimal budget allocation
        """
        if self._attribution is None:
            raise RuntimeError("Must call fit() first")
        
        # Default: equal costs
        if cost_per_touchpoint is None:
            cost_per_touchpoint = {tp: 1.0 for tp in self.touchpoint_cols}
        
        # Simple allocation proportional to attribution
        allocation = {}
        
        # Adjust for cost efficiency
        efficiency = {}
        for tp in self.touchpoint_cols:
            efficiency[tp] = self._attribution.get(tp, 0) / cost_per_touchpoint.get(tp, 1)
        
        total_efficiency = sum(efficiency.values())
        
        if total_efficiency > 0:
            for tp in self.touchpoint_cols:
                allocation[tp] = total_budget * efficiency[tp] / total_efficiency
        else:
            # Equal allocation
            per_tp = total_budget / len(self.touchpoint_cols)
            allocation = {tp: per_tp for tp in self.touchpoint_cols}
        
        # Apply constraints
        if constraints:
            for tp, (min_val, max_val) in constraints.items():
                if tp in allocation:
                    allocation[tp] = np.clip(allocation[tp], min_val, max_val)
        
        return allocation
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare attribution across different methods.
        
        Returns
        -------
        pd.DataFrame
            Comparison of attribution weights by method
        """
        methods = ['shapley', 'logistic', 'uplift', 'markov', 'last_touch', 'first_touch']
        results = {}
        
        for method in methods:
            try:
                self.fit(method=method)
                results[method] = self._attribution.copy()
            except Exception as e:
                warnings.warn(f"Method {method} failed: {e}")
        
        return pd.DataFrame(results)
    
    def plot_attribution(self, ax=None):
        """Plot attribution weights."""
        import matplotlib.pyplot as plt
        
        if self._attribution is None:
            raise RuntimeError("Must call fit() first")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        touchpoints = list(self._attribution.keys())
        weights = list(self._attribution.values())
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(touchpoints)))
        ax.barh(touchpoints, weights, color=colors)
        ax.set_xlabel('Attribution Weight')
        ax.set_title(f'Marketing Attribution ({self._method})')
        
        return ax
    
    def summary(self) -> str:
        """Return attribution summary."""
        if self._attribution is None:
            return "Attribution not fitted. Call fit() first."
        
        lines = [
            "Marketing Attribution Summary",
            "=" * 50,
            f"Method: {self._method}",
            f"Touchpoints: {len(self.touchpoint_cols)}",
            f"Total records: {len(self.data)}",
            f"Conversion rate: {self.data[self.conversion_col].mean():.2%}",
            "",
            "Attribution Weights:",
        ]
        
        sorted_attr = sorted(self._attribution.items(), key=lambda x: x[1], reverse=True)
        for tp, weight in sorted_attr:
            inc_effect = self._incremental_effects.get(tp, 0)
            lines.append(f"  {tp}: {weight:.1%} (incremental: {inc_effect:+.2%})")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        status = "fitted" if self._attribution else "unfitted"
        return f"MarketingAttribution(touchpoints={len(self.touchpoint_cols)}, {status})"
