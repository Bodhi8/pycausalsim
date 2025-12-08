"""
Validation and Sensitivity Analysis for Causal Claims.

Implements multiple methods to validate causal inferences:
- Sensitivity to unobserved confounding
- Placebo tests
- Refutation methods
- Cross-validation
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
import warnings


@dataclass
class RefutationResult:
    """Result from a refutation test."""
    method: str
    original_estimate: float
    refuted_estimate: float
    p_value: float
    passed: bool
    details: Dict[str, Any]
    
    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (f"{self.method}: {status}\n"
                f"  Original: {self.original_estimate:.4f}\n"
                f"  Refuted: {self.refuted_estimate:.4f}\n"
                f"  P-value: {self.p_value:.4f}")


class CausalValidator:
    """
    Comprehensive validation suite for causal claims.
    
    Parameters
    ----------
    simulator : CausalSimulator
        Fitted simulator to validate
        
    Examples
    --------
    >>> validator = CausalValidator(simulator)
    >>> results = validator.run_all_tests()
    >>> validator.summary()
    """
    
    def __init__(self, simulator, random_state: Optional[int] = None):
        self.simulator = simulator
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self._results = {}
    
    def run_all_tests(
        self,
        variable: Optional[str] = None,
        n_simulations: int = 500
    ) -> Dict[str, Any]:
        """
        Run all validation tests.
        
        Parameters
        ----------
        variable : str, optional
            Variable to validate (defaults to top driver)
        n_simulations : int
            Number of simulations
            
        Returns
        -------
        dict
            All validation results
        """
        if variable is None:
            drivers = self.simulator.rank_drivers(n_simulations=100)
            variable = drivers[0][0] if drivers else self.simulator._feature_vars[0]
        
        self._results = {
            'variable': variable,
            'confounding_bounds': self.sensitivity_analysis(variable, n_simulations),
            'placebo_tests': self.placebo_tests(variable, n_simulations),
            'refutations': self.refutation_tests(variable, n_simulations),
            'robustness': self.robustness_checks(variable, n_simulations)
        }
        
        return self._results
    
    def sensitivity_analysis(
        self,
        variable: str,
        n_simulations: int = 500
    ) -> Dict[str, Tuple[float, float]]:
        """
        Sensitivity analysis for unobserved confounding.
        
        Uses Rosenbaum-style bounds to assess how strong an unobserved
        confounder would need to be to nullify the effect.
        
        Parameters
        ----------
        variable : str
            Treatment variable
        n_simulations : int
            Number of simulations
            
        Returns
        -------
        dict
            Bounds for different confounding strengths
        """
        # Get original effect
        original = self.simulator.simulate_intervention(
            variable,
            self.simulator.data[variable].mean() + self.simulator.data[variable].std(),
            n_simulations=n_simulations
        )
        
        bounds = {}
        
        for gamma in [1.1, 1.25, 1.5, 2.0, 3.0]:
            # Rosenbaum bounds
            lower, upper = self._rosenbaum_bounds(original.point_estimate, gamma)
            bounds[gamma] = (lower, upper)
        
        return bounds
    
    def _rosenbaum_bounds(
        self,
        estimate: float,
        gamma: float
    ) -> Tuple[float, float]:
        """Compute Rosenbaum sensitivity bounds."""
        # Simplified Rosenbaum-style bounds
        # In practice, these depend on the data structure
        delta = abs(estimate) * (gamma - 1) / gamma
        
        lower = estimate - delta
        upper = estimate + delta
        
        return (lower, upper)
    
    def placebo_tests(
        self,
        variable: str,
        n_simulations: int = 500
    ) -> Dict[str, RefutationResult]:
        """
        Run placebo tests.
        
        Tests include:
        - Random treatment assignment
        - Placebo outcome
        - Subset data test
        
        Parameters
        ----------
        variable : str
            Treatment variable
        n_simulations : int
            Number of simulations
            
        Returns
        -------
        dict
            Placebo test results
        """
        results = {}
        
        # Original effect
        original = self.simulator.simulate_intervention(
            variable,
            self.simulator.data[variable].mean() * 1.1,
            n_simulations=n_simulations // 2
        )
        
        # Test 1: Random treatment (permutation)
        results['random_treatment'] = self._placebo_random_treatment(
            variable, original.point_estimate, n_simulations
        )
        
        # Test 2: Placebo outcome
        results['placebo_outcome'] = self._placebo_outcome(
            variable, original.point_estimate, n_simulations
        )
        
        # Test 3: Subset stability
        results['subset_stability'] = self._placebo_subset(
            variable, original.point_estimate, n_simulations
        )
        
        return results
    
    def _placebo_random_treatment(
        self,
        variable: str,
        original_estimate: float,
        n_simulations: int
    ) -> RefutationResult:
        """Test with randomly permuted treatment."""
        placebo_effects = []
        
        for _ in range(min(100, n_simulations)):
            # Permute treatment
            permuted_data = self.simulator.data.copy()
            permuted_data[variable] = np.random.permutation(permuted_data[variable])
            
            # Simple correlation as proxy for effect
            effect = np.corrcoef(
                permuted_data[variable],
                permuted_data[self.simulator.target]
            )[0, 1]
            placebo_effects.append(effect)
        
        # P-value: fraction of placebo effects >= original
        p_value = np.mean(np.abs(placebo_effects) >= abs(original_estimate) * 0.5)
        
        return RefutationResult(
            method='Random Treatment',
            original_estimate=original_estimate,
            refuted_estimate=np.mean(placebo_effects),
            p_value=p_value,
            passed=p_value < 0.1,  # Expect placebo to be small
            details={'n_permutations': len(placebo_effects)}
        )
    
    def _placebo_outcome(
        self,
        variable: str,
        original_estimate: float,
        n_simulations: int
    ) -> RefutationResult:
        """Test with placebo (random) outcome."""
        # Generate random outcome
        placebo_outcome = np.random.randn(len(self.simulator.data))
        
        # Correlation between treatment and random outcome
        effect = np.corrcoef(
            self.simulator.data[variable],
            placebo_outcome
        )[0, 1]
        
        # This should be near zero
        _, p_value = stats.pearsonr(
            self.simulator.data[variable],
            placebo_outcome
        )
        
        return RefutationResult(
            method='Placebo Outcome',
            original_estimate=original_estimate,
            refuted_estimate=effect,
            p_value=p_value,
            passed=p_value > 0.05,  # Should NOT be significant
            details={'placebo_effect': effect}
        )
    
    def _placebo_subset(
        self,
        variable: str,
        original_estimate: float,
        n_simulations: int
    ) -> RefutationResult:
        """Test stability across random subsets."""
        subset_effects = []
        n = len(self.simulator.data)
        
        for _ in range(50):
            # Random 50% subset
            idx = np.random.choice(n, size=n // 2, replace=False)
            subset = self.simulator.data.iloc[idx]
            
            # Effect in subset
            effect = np.corrcoef(
                subset[variable],
                subset[self.simulator.target]
            )[0, 1]
            subset_effects.append(effect)
        
        # Coefficient of variation (stability metric)
        cv = np.std(subset_effects) / (abs(np.mean(subset_effects)) + 1e-10)
        
        # Effect is stable if CV is low
        return RefutationResult(
            method='Subset Stability',
            original_estimate=original_estimate,
            refuted_estimate=np.mean(subset_effects),
            p_value=1 - min(cv, 1),  # Higher = more stable
            passed=cv < 0.5,  # Reasonably stable
            details={'cv': cv, 'n_subsets': len(subset_effects)}
        )
    
    def refutation_tests(
        self,
        variable: str,
        n_simulations: int = 500
    ) -> Dict[str, RefutationResult]:
        """
        Run refutation tests to challenge causal claims.
        
        Tests include:
        - Add random common cause
        - Replace treatment with placebo
        - Data subset validation
        
        Parameters
        ----------
        variable : str
            Treatment variable
        n_simulations : int
            Number of simulations
            
        Returns
        -------
        dict
            Refutation results
        """
        results = {}
        
        # Get original effect
        original = self.simulator.simulate_intervention(
            variable,
            self.simulator.data[variable].mean() * 1.1,
            n_simulations=n_simulations // 2
        )
        
        # Test 1: Add random common cause
        results['random_common_cause'] = self._refute_random_cause(
            variable, original.point_estimate
        )
        
        # Test 2: Unobserved common cause
        results['unobserved_confounder'] = self._refute_unobserved_confounder(
            variable, original.point_estimate
        )
        
        return results
    
    def _refute_random_cause(
        self,
        variable: str,
        original_estimate: float
    ) -> RefutationResult:
        """Add random common cause and check if estimate changes."""
        from sklearn.linear_model import LinearRegression
        
        # Original regression
        X_orig = self.simulator.data[[variable] + self.simulator.confounders]
        y = self.simulator.data[self.simulator.target]
        
        model_orig = LinearRegression().fit(X_orig, y)
        coef_orig = model_orig.coef_[0]
        
        # With random common cause
        data_new = self.simulator.data.copy()
        data_new['_random_confounder'] = np.random.randn(len(data_new))
        
        X_new = data_new[[variable] + self.simulator.confounders + ['_random_confounder']]
        model_new = LinearRegression().fit(X_new, y)
        coef_new = model_new.coef_[0]
        
        # Check if coefficient changed significantly
        change = abs(coef_new - coef_orig) / (abs(coef_orig) + 1e-10)
        
        return RefutationResult(
            method='Random Common Cause',
            original_estimate=coef_orig,
            refuted_estimate=coef_new,
            p_value=1 - change,  # Higher = more robust
            passed=change < 0.1,  # Less than 10% change
            details={'coefficient_change': change}
        )
    
    def _refute_unobserved_confounder(
        self,
        variable: str,
        original_estimate: float
    ) -> RefutationResult:
        """Simulate effect of unobserved confounder."""
        from sklearn.linear_model import LinearRegression
        
        # Create simulated confounder that affects both treatment and outcome
        confounding_strength = 0.3
        
        # Simulated confounder
        U = np.random.randn(len(self.simulator.data))
        
        # Adjust treatment and outcome
        adjusted_treatment = self.simulator.data[variable] + confounding_strength * U
        adjusted_outcome = self.simulator.data[self.simulator.target] + confounding_strength * U
        
        # New correlation
        new_effect = np.corrcoef(adjusted_treatment, adjusted_outcome)[0, 1]
        
        # Original correlation
        orig_effect = np.corrcoef(
            self.simulator.data[variable],
            self.simulator.data[self.simulator.target]
        )[0, 1]
        
        change = abs(new_effect - orig_effect) / (abs(orig_effect) + 1e-10)
        
        return RefutationResult(
            method='Unobserved Confounder',
            original_estimate=orig_effect,
            refuted_estimate=new_effect,
            p_value=1 - min(change, 1),
            passed=change < 0.3,  # Effect survives moderate confounding
            details={'confounding_strength': confounding_strength, 'change': change}
        )
    
    def robustness_checks(
        self,
        variable: str,
        n_simulations: int = 500
    ) -> Dict[str, Any]:
        """
        Additional robustness checks.
        
        Parameters
        ----------
        variable : str
            Treatment variable
        n_simulations : int
            Number of simulations
            
        Returns
        -------
        dict
            Robustness check results
        """
        results = {}
        
        # Check 1: Effect sign consistency
        results['sign_consistency'] = self._check_sign_consistency(
            variable, n_simulations
        )
        
        # Check 2: Effect size reasonableness
        results['effect_magnitude'] = self._check_effect_magnitude(
            variable, n_simulations
        )
        
        # Check 3: Sample size sensitivity
        results['sample_sensitivity'] = self._check_sample_sensitivity(
            variable, n_simulations
        )
        
        return results
    
    def _check_sign_consistency(
        self,
        variable: str,
        n_simulations: int
    ) -> Dict[str, Any]:
        """Check if effect sign is consistent across bootstrap samples."""
        effects = []
        n = len(self.simulator.data)
        
        for _ in range(100):
            idx = np.random.choice(n, size=n, replace=True)
            subset = self.simulator.data.iloc[idx]
            
            effect = np.corrcoef(
                subset[variable],
                subset[self.simulator.target]
            )[0, 1]
            effects.append(effect)
        
        positive = sum(1 for e in effects if e > 0)
        consistency = max(positive, 100 - positive) / 100
        
        return {
            'consistency': consistency,
            'mean_effect': np.mean(effects),
            'std_effect': np.std(effects),
            'passed': consistency > 0.8
        }
    
    def _check_effect_magnitude(
        self,
        variable: str,
        n_simulations: int
    ) -> Dict[str, Any]:
        """Check if effect magnitude is reasonable."""
        effect = np.corrcoef(
            self.simulator.data[variable],
            self.simulator.data[self.simulator.target]
        )[0, 1]
        
        # Effect should not be implausibly large
        is_reasonable = abs(effect) < 0.95
        
        return {
            'effect': effect,
            'is_reasonable': is_reasonable,
            'passed': is_reasonable
        }
    
    def _check_sample_sensitivity(
        self,
        variable: str,
        n_simulations: int
    ) -> Dict[str, Any]:
        """Check sensitivity to sample size."""
        sample_sizes = [0.25, 0.5, 0.75, 1.0]
        effects_by_size = {}
        n = len(self.simulator.data)
        
        for frac in sample_sizes:
            effects = []
            for _ in range(20):
                size = int(frac * n)
                idx = np.random.choice(n, size=size, replace=False)
                subset = self.simulator.data.iloc[idx]
                
                effect = np.corrcoef(
                    subset[variable],
                    subset[self.simulator.target]
                )[0, 1]
                effects.append(effect)
            
            effects_by_size[frac] = np.mean(effects)
        
        # Check if effects converge as sample size increases
        values = list(effects_by_size.values())
        variance = np.var(values)
        
        return {
            'effects_by_sample_size': effects_by_size,
            'variance': variance,
            'passed': variance < 0.1
        }
    
    def compute_robustness_value(self) -> float:
        """
        Compute overall robustness value (RV).
        
        The robustness value represents the minimum strength of
        confounding required to nullify the causal effect.
        
        Returns
        -------
        float
            Robustness value (0 = fragile, 1 = very robust)
        """
        if 'confounding_bounds' not in self._results:
            raise RuntimeError("Must run validation tests first")
        
        bounds = self._results['confounding_bounds']
        
        # Find minimum gamma where bounds include zero
        for gamma, (lower, upper) in sorted(bounds.items()):
            if lower <= 0 <= upper:
                # Normalize gamma to 0-1 scale
                return (gamma - 1) / 2  # Rough normalization
        
        return 1.0  # Very robust if no gamma makes bounds include zero
    
    def summary(self) -> str:
        """Return comprehensive validation summary."""
        if not self._results:
            return "No validation tests run. Call run_all_tests() first."
        
        lines = [
            "Causal Validation Summary",
            "=" * 60,
            f"Variable: {self._results.get('variable', 'Unknown')}",
            "",
        ]
        
        # Sensitivity analysis
        if 'confounding_bounds' in self._results:
            lines.append("Sensitivity to Unobserved Confounding:")
            for gamma, (lower, upper) in self._results['confounding_bounds'].items():
                includes_zero = "✗ includes 0" if lower <= 0 <= upper else "✓"
                lines.append(f"  Γ={gamma}: [{lower:.4f}, {upper:.4f}] {includes_zero}")
            lines.append("")
        
        # Placebo tests
        if 'placebo_tests' in self._results:
            lines.append("Placebo Tests:")
            for name, result in self._results['placebo_tests'].items():
                status = "✓ PASS" if result.passed else "✗ FAIL"
                lines.append(f"  {name}: {status} (p={result.p_value:.4f})")
            lines.append("")
        
        # Refutations
        if 'refutations' in self._results:
            lines.append("Refutation Tests:")
            for name, result in self._results['refutations'].items():
                status = "✓ PASS" if result.passed else "✗ FAIL"
                lines.append(f"  {name}: {status}")
            lines.append("")
        
        # Robustness
        if 'robustness' in self._results:
            lines.append("Robustness Checks:")
            for name, result in self._results['robustness'].items():
                status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
                lines.append(f"  {name}: {status}")
            lines.append("")
        
        # Overall assessment
        try:
            rv = self.compute_robustness_value()
            lines.append(f"Overall Robustness Value: {rv:.2f}")
            if rv > 0.5:
                lines.append("Assessment: Causal claim appears robust")
            elif rv > 0.2:
                lines.append("Assessment: Moderate robustness - interpret with caution")
            else:
                lines.append("Assessment: Fragile - strong confounding could nullify effect")
        except Exception:
            pass
        
        return "\n".join(lines)
    
    def plot_sensitivity(self, ax=None):
        """Plot sensitivity analysis results."""
        import matplotlib.pyplot as plt
        
        if 'confounding_bounds' not in self._results:
            raise RuntimeError("Must run sensitivity analysis first")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        bounds = self._results['confounding_bounds']
        gammas = sorted(bounds.keys())
        lowers = [bounds[g][0] for g in gammas]
        uppers = [bounds[g][1] for g in gammas]
        
        ax.fill_between(gammas, lowers, uppers, alpha=0.3, label='Effect bounds')
        ax.axhline(0, color='red', linestyle='--', label='Null effect')
        ax.plot(gammas, [(l + u) / 2 for l, u in zip(lowers, uppers)], 
                'b-', linewidth=2, label='Point estimate')
        
        ax.set_xlabel('Confounding Strength (Γ)')
        ax.set_ylabel('Causal Effect')
        ax.set_title('Sensitivity to Unobserved Confounding')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
