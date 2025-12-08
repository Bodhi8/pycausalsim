"""
Result classes for PyCausalSim outputs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


@dataclass
class CausalEffect:
    """
    Represents the causal effect of an intervention.
    
    Attributes:
        variable: The variable that was intervened on
        intervention_value: The value set for the intervention
        original_value: The original/baseline value
        target: The outcome variable
        point_estimate: The estimated causal effect
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        p_value: Statistical significance
        n_simulations: Number of simulations run
        method: The method used for estimation
    """
    variable: str
    intervention_value: float
    original_value: float
    target: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_simulations: int
    method: str = "simulation"
    effect_type: str = "ate"  # ate, att, cate
    samples: Optional[np.ndarray] = None
    
    def summary(self) -> str:
        """Return a formatted summary of the causal effect."""
        lines = [
            f"Causal Effect Summary",
            f"=" * 50,
            f"Intervention: {self.variable} = {self.intervention_value}",
            f"Original value: {self.original_value:.4f}",
            f"Target variable: {self.target}",
            f"",
            f"Effect on {self.target}: {self.point_estimate:+.4f}",
            f"  ({self.point_estimate:+.2%} relative change)" if abs(self.original_value) > 1e-10 else "",
            f"95% CI: [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]",
            f"P-value: {self.p_value:.4f}",
            f"",
            f"Simulations: {self.n_simulations}",
            f"Method: {self.method}",
        ]
        return "\n".join(line for line in lines if line or line == "")
    
    def __repr__(self) -> str:
        return (f"CausalEffect(variable='{self.variable}', "
                f"effect={self.point_estimate:.4f}, "
                f"ci=[{self.ci_lower:.4f}, {self.ci_upper:.4f}])")
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if the effect is statistically significant."""
        return self.p_value < alpha
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'variable': self.variable,
            'intervention_value': self.intervention_value,
            'original_value': self.original_value,
            'target': self.target,
            'point_estimate': self.point_estimate,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'p_value': self.p_value,
            'n_simulations': self.n_simulations,
            'method': self.method,
            'effect_type': self.effect_type,
        }


@dataclass
class SimulationResult:
    """
    Results from a simulation run.
    
    Attributes:
        factual: Observed/factual outcomes
        counterfactual: Simulated counterfactual outcomes
        intervention: The intervention applied
        metadata: Additional simulation metadata
    """
    factual: np.ndarray
    counterfactual: np.ndarray
    intervention: Dict[str, float]
    effect: CausalEffect
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def treatment_effect(self) -> np.ndarray:
        """Individual treatment effects."""
        return self.counterfactual - self.factual
    
    @property
    def ate(self) -> float:
        """Average treatment effect."""
        return float(np.mean(self.treatment_effect))
    
    @property
    def att(self) -> float:
        """Average treatment effect on the treated."""
        if 'treated_mask' in self.metadata:
            mask = self.metadata['treated_mask']
            return float(np.mean(self.treatment_effect[mask]))
        return self.ate
    
    def summary(self) -> str:
        """Return simulation summary."""
        return self.effect.summary()
    
    def plot_counterfactual(self, ax=None):
        """Plot factual vs counterfactual distributions."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.factual, bins=50, alpha=0.5, label='Factual', density=True)
        ax.hist(self.counterfactual, bins=50, alpha=0.5, label='Counterfactual', density=True)
        ax.axvline(np.mean(self.factual), color='blue', linestyle='--', 
                   label=f'Factual mean: {np.mean(self.factual):.4f}')
        ax.axvline(np.mean(self.counterfactual), color='orange', linestyle='--',
                   label=f'Counterfactual mean: {np.mean(self.counterfactual):.4f}')
        ax.set_xlabel('Outcome')
        ax.set_ylabel('Density')
        ax.set_title('Factual vs Counterfactual Distribution')
        ax.legend()
        
        return ax


@dataclass
class SensitivityResult:
    """
    Results from sensitivity analysis.
    
    Attributes:
        original_effect: The original causal effect estimate
        confounding_bounds: Bounds under different confounding assumptions
        placebo_results: Results from placebo tests
        refutation_results: Results from various refutation methods
    """
    original_effect: CausalEffect
    confounding_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    placebo_results: Optional[Dict[str, float]] = None
    refutation_results: Optional[Dict[str, Any]] = None
    robustness_value: Optional[float] = None
    
    def confounding_bounds_analysis(self) -> str:
        """Summarize confounding bounds analysis."""
        if self.confounding_bounds is None:
            return "No confounding bounds computed."
        
        lines = ["Confounding Bounds Analysis", "=" * 40]
        for confounder_strength, (lower, upper) in self.confounding_bounds.items():
            lines.append(f"  Confounder strength {confounder_strength}: [{lower:.4f}, {upper:.4f}]")
        return "\n".join(lines)
    
    def placebo_test_summary(self) -> str:
        """Summarize placebo test results."""
        if self.placebo_results is None:
            return "No placebo tests run."
        
        lines = ["Placebo Test Results", "=" * 40]
        for test_name, p_value in self.placebo_results.items():
            status = "PASS" if p_value > 0.05 else "FAIL"
            lines.append(f"  {test_name}: p={p_value:.4f} [{status}]")
        return "\n".join(lines)
    
    def summary(self) -> str:
        """Full sensitivity analysis summary."""
        sections = [
            "Sensitivity Analysis Summary",
            "=" * 50,
            "",
            "Original Effect:",
            f"  Point estimate: {self.original_effect.point_estimate:.4f}",
            f"  95% CI: [{self.original_effect.ci_lower:.4f}, {self.original_effect.ci_upper:.4f}]",
            "",
        ]
        
        if self.robustness_value is not None:
            sections.append(f"Robustness Value (RV): {self.robustness_value:.4f}")
            sections.append("  (Minimum confounding strength to nullify effect)")
            sections.append("")
        
        if self.confounding_bounds:
            sections.append(self.confounding_bounds_analysis())
            sections.append("")
        
        if self.placebo_results:
            sections.append(self.placebo_test_summary())
            sections.append("")
        
        return "\n".join(sections)
    
    def plot_bounds(self, ax=None):
        """Plot confounding bounds."""
        import matplotlib.pyplot as plt
        
        if self.confounding_bounds is None:
            raise ValueError("No confounding bounds to plot")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        strengths = list(self.confounding_bounds.keys())
        lowers = [self.confounding_bounds[s][0] for s in strengths]
        uppers = [self.confounding_bounds[s][1] for s in strengths]
        
        ax.fill_between(range(len(strengths)), lowers, uppers, alpha=0.3, label='Effect bounds')
        ax.axhline(0, color='red', linestyle='--', label='Null effect')
        ax.axhline(self.original_effect.point_estimate, color='blue', 
                   linestyle='-', label='Original estimate')
        ax.set_xticks(range(len(strengths)))
        ax.set_xticklabels([str(s) for s in strengths])
        ax.set_xlabel('Unobserved Confounder Strength')
        ax.set_ylabel('Causal Effect')
        ax.set_title('Sensitivity to Unobserved Confounding')
        ax.legend()
        
        return ax


@dataclass
class DriverRanking:
    """
    Ranking of causal drivers for an outcome.
    """
    target: str
    drivers: List[Tuple[str, float]]
    effects: Dict[str, CausalEffect]
    method: str = "intervention"
    
    def __iter__(self):
        return iter(self.drivers)
    
    def __len__(self):
        return len(self.drivers)
    
    def __getitem__(self, idx):
        return self.drivers[idx]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.drivers, columns=['variable', 'causal_effect'])
    
    def plot(self, ax=None, top_n: int = 10):
        """Plot driver ranking."""
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        top_drivers = self.drivers[:top_n]
        vars_, effects = zip(*top_drivers) if top_drivers else ([], [])
        
        colors = ['green' if e > 0 else 'red' for e in effects]
        ax.barh(range(len(vars_)), effects, color=colors, alpha=0.7)
        ax.set_yticks(range(len(vars_)))
        ax.set_yticklabels(vars_)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Causal Effect')
        ax.set_title(f'Causal Drivers of {self.target}')
        ax.invert_yaxis()
        
        return ax


@dataclass 
class OptimalPolicy:
    """
    Optimal policy from policy optimization.
    """
    objective: str
    settings: Dict[str, float]
    expected_outcome: float
    baseline_outcome: float
    improvement: float
    constraints: Dict[str, Tuple[float, float]]
    
    def summary(self) -> str:
        """Summarize optimal policy."""
        lines = [
            "Optimal Policy",
            "=" * 50,
            f"Objective: {self.objective}",
            "",
            "Recommended Settings:",
        ]
        for var, val in self.settings.items():
            lines.append(f"  {var}: {val:.4f}")
        
        lines.extend([
            "",
            f"Expected outcome: {self.expected_outcome:.4f}",
            f"Baseline outcome: {self.baseline_outcome:.4f}",
            f"Expected improvement: {self.improvement:+.4f} ({self.improvement/self.baseline_outcome*100:+.2f}%)",
        ])
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"OptimalPolicy(improvement={self.improvement:+.4f}, settings={self.settings})"
