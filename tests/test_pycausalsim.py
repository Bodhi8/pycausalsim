"""
Test suite for PyCausalSim.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch


class TestCausalSimulator:
    """Tests for the main CausalSimulator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n = 500
        
        # Confounders
        X = np.random.randn(n)
        
        # Treatment affected by confounder
        T = 0.5 * X + np.random.randn(n) * 0.5
        
        # Outcome affected by treatment and confounder
        Y = 0.3 * T + 0.4 * X + np.random.randn(n) * 0.3
        
        return pd.DataFrame({
            'confounder': X,
            'treatment': T,
            'outcome': Y
        })
    
    def test_initialization(self, sample_data):
        """Test simulator initialization."""
        from pycausalsim import CausalSimulator
        
        sim = CausalSimulator(
            data=sample_data,
            target='outcome',
            treatment_vars=['treatment'],
            confounders=['confounder']
        )
        
        assert sim.target == 'outcome'
        assert sim.treatment_vars == ['treatment']
        assert sim.confounders == ['confounder']
        assert len(sim.data) == 500
    
    def test_invalid_target(self, sample_data):
        """Test error on invalid target."""
        from pycausalsim import CausalSimulator
        
        with pytest.raises(ValueError, match="Target.*not found"):
            CausalSimulator(
                data=sample_data,
                target='nonexistent'
            )
    
    def test_discover_graph(self, sample_data):
        """Test graph discovery."""
        from pycausalsim import CausalSimulator
        
        sim = CausalSimulator(
            data=sample_data,
            target='outcome'
        )
        
        sim.discover_graph(method='correlation')
        
        assert sim.graph is not None
        assert 'outcome' in sim.graph
        assert sim.scm is not None
    
    def test_simulate_intervention(self, sample_data):
        """Test intervention simulation."""
        from pycausalsim import CausalSimulator
        
        sim = CausalSimulator(
            data=sample_data,
            target='outcome',
            treatment_vars=['treatment']
        )
        
        sim.discover_graph(method='correlation')
        
        effect = sim.simulate_intervention(
            variable='treatment',
            value=1.0,
            n_simulations=100
        )
        
        assert effect is not None
        assert hasattr(effect, 'point_estimate')
        assert hasattr(effect, 'ci_lower')
        assert hasattr(effect, 'ci_upper')
        assert hasattr(effect, 'p_value')
    
    def test_rank_drivers(self, sample_data):
        """Test driver ranking."""
        from pycausalsim import CausalSimulator
        
        sim = CausalSimulator(
            data=sample_data,
            target='outcome'
        )
        
        sim.discover_graph(method='correlation')
        drivers = sim.rank_drivers(n_simulations=50)
        
        assert len(drivers) > 0
        assert all(isinstance(d, tuple) and len(d) == 2 for d in drivers)


class TestStructuralCausalModel:
    """Tests for StructuralCausalModel."""
    
    @pytest.fixture
    def simple_graph(self):
        """Simple causal graph."""
        return {
            'Y': ['X', 'Z'],
            'X': ['Z'],
            'Z': []
        }
    
    @pytest.fixture
    def sample_data(self):
        """Generate data matching the simple graph."""
        np.random.seed(42)
        n = 500
        
        Z = np.random.randn(n)
        X = 0.5 * Z + np.random.randn(n) * 0.3
        Y = 0.4 * X + 0.3 * Z + np.random.randn(n) * 0.3
        
        return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    def test_scm_initialization(self, simple_graph):
        """Test SCM initialization."""
        from pycausalsim.models import StructuralCausalModel
        
        scm = StructuralCausalModel(graph=simple_graph)
        
        assert scm.graph == simple_graph
        assert not scm.is_fitted
    
    def test_scm_fit(self, simple_graph, sample_data):
        """Test SCM fitting."""
        from pycausalsim.models import StructuralCausalModel
        
        scm = StructuralCausalModel(graph=simple_graph)
        scm.fit(sample_data)
        
        assert scm.is_fitted
        assert len(scm._mechanisms) == 3
    
    def test_scm_predict(self, simple_graph, sample_data):
        """Test SCM prediction."""
        from pycausalsim.models import StructuralCausalModel
        
        scm = StructuralCausalModel(graph=simple_graph)
        scm.fit(sample_data)
        
        predictions = scm.predict(sample_data, 'Y')
        
        assert len(predictions) == len(sample_data)
    
    def test_scm_counterfactual(self, simple_graph, sample_data):
        """Test counterfactual generation."""
        from pycausalsim.models import StructuralCausalModel
        
        scm = StructuralCausalModel(graph=simple_graph)
        scm.fit(sample_data)
        
        cf = scm.counterfactual(
            intervention={'X': 1.0},
            data=sample_data
        )
        
        assert len(cf) == len(sample_data)
        assert 'Y' in cf.columns
    
    def test_scm_sample(self, simple_graph, sample_data):
        """Test SCM sampling."""
        from pycausalsim.models import StructuralCausalModel
        
        scm = StructuralCausalModel(graph=simple_graph)
        scm.fit(sample_data)
        
        samples = scm.sample(n_samples=100)
        
        assert len(samples) == 100
        assert set(samples.columns) == set(sample_data.columns)


class TestMarketingAttribution:
    """Tests for MarketingAttribution."""
    
    @pytest.fixture
    def touchpoint_data(self):
        """Generate synthetic touchpoint data."""
        np.random.seed(42)
        n = 1000
        
        return pd.DataFrame({
            'email': np.random.binomial(1, 0.3, n),
            'display': np.random.binomial(1, 0.4, n),
            'search': np.random.binomial(1, 0.5, n),
            'social': np.random.binomial(1, 0.2, n),
            'converted': np.random.binomial(1, 0.1, n)
        })
    
    def test_attribution_initialization(self, touchpoint_data):
        """Test attribution initialization."""
        from pycausalsim import MarketingAttribution
        
        attr = MarketingAttribution(
            data=touchpoint_data,
            conversion_col='converted',
            touchpoint_cols=['email', 'display', 'search', 'social']
        )
        
        assert attr.conversion_col == 'converted'
        assert len(attr.touchpoint_cols) == 4
    
    def test_shapley_attribution(self, touchpoint_data):
        """Test Shapley attribution."""
        from pycausalsim import MarketingAttribution
        
        attr = MarketingAttribution(
            data=touchpoint_data,
            conversion_col='converted'
        )
        
        attr.fit(method='shapley')
        weights = attr.get_attribution()
        
        assert len(weights) > 0
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Sums to 1
    
    def test_logistic_attribution(self, touchpoint_data):
        """Test logistic regression attribution."""
        from pycausalsim import MarketingAttribution
        
        attr = MarketingAttribution(
            data=touchpoint_data,
            conversion_col='converted'
        )
        
        attr.fit(method='logistic')
        weights = attr.get_attribution()
        
        assert len(weights) > 0


class TestExperimentAnalysis:
    """Tests for ExperimentAnalysis."""
    
    @pytest.fixture
    def experiment_data(self):
        """Generate synthetic experiment data."""
        np.random.seed(42)
        n = 1000
        
        treatment = np.random.binomial(1, 0.5, n)
        covariate = np.random.randn(n)
        
        # True effect = 0.3
        outcome = 0.3 * treatment + 0.2 * covariate + np.random.randn(n) * 0.5
        
        return pd.DataFrame({
            'treatment': treatment,
            'covariate': covariate,
            'outcome': outcome
        })
    
    def test_experiment_initialization(self, experiment_data):
        """Test experiment analysis initialization."""
        from pycausalsim import ExperimentAnalysis
        
        exp = ExperimentAnalysis(
            data=experiment_data,
            treatment='treatment',
            outcome='outcome'
        )
        
        assert exp.treatment == 'treatment'
        assert exp.outcome == 'outcome'
    
    def test_difference_in_means(self, experiment_data):
        """Test difference in means estimator."""
        from pycausalsim import ExperimentAnalysis
        
        exp = ExperimentAnalysis(
            data=experiment_data,
            treatment='treatment',
            outcome='outcome'
        )
        
        effect = exp.estimate_effect(method='difference')
        
        # True effect is 0.3
        assert abs(effect.estimate - 0.3) < 0.15
        assert effect.ci_lower < effect.estimate < effect.ci_upper
    
    def test_doubly_robust(self, experiment_data):
        """Test doubly robust estimator."""
        from pycausalsim import ExperimentAnalysis
        
        exp = ExperimentAnalysis(
            data=experiment_data,
            treatment='treatment',
            outcome='outcome',
            covariates=['covariate']
        )
        
        effect = exp.estimate_effect(method='dr')
        
        assert abs(effect.estimate - 0.3) < 0.15


class TestUpliftModeler:
    """Tests for UpliftModeler."""
    
    @pytest.fixture
    def uplift_data(self):
        """Generate synthetic uplift data."""
        np.random.seed(42)
        n = 1000
        
        feature = np.random.randn(n)
        treatment = np.random.binomial(1, 0.5, n)
        
        # Heterogeneous effect: positive for high feature, negative for low
        base_prob = 0.3
        uplift = 0.2 * feature * treatment
        outcome = np.random.binomial(1, np.clip(base_prob + uplift, 0, 1))
        
        return pd.DataFrame({
            'feature': feature,
            'treatment': treatment,
            'outcome': outcome
        })
    
    def test_uplift_initialization(self, uplift_data):
        """Test uplift modeler initialization."""
        from pycausalsim.uplift import UpliftModeler
        
        uplift = UpliftModeler(
            data=uplift_data,
            treatment='treatment',
            outcome='outcome'
        )
        
        assert uplift.treatment == 'treatment'
        assert uplift.outcome == 'outcome'
    
    def test_two_model_uplift(self, uplift_data):
        """Test two-model uplift approach."""
        from pycausalsim.uplift import UpliftModeler
        
        uplift = UpliftModeler(
            data=uplift_data,
            treatment='treatment',
            outcome='outcome',
            features=['feature']
        )
        
        uplift.fit(method='two_model')
        scores = uplift.predict_uplift()
        
        assert len(scores) == len(uplift_data)
    
    def test_uplift_segments(self, uplift_data):
        """Test uplift segmentation."""
        from pycausalsim.uplift import UpliftModeler
        
        uplift = UpliftModeler(
            data=uplift_data,
            treatment='treatment',
            outcome='outcome',
            features=['feature']
        )
        
        uplift.fit(method='two_model')
        segments = uplift.segment_by_effect(n_segments=4)
        
        assert len(segments) <= 4


class TestCausalDiscovery:
    """Tests for causal discovery algorithms."""
    
    @pytest.fixture
    def chain_data(self):
        """Generate chain graph data: Z -> X -> Y."""
        np.random.seed(42)
        n = 500
        
        Z = np.random.randn(n)
        X = 0.7 * Z + np.random.randn(n) * 0.3
        Y = 0.6 * X + np.random.randn(n) * 0.3
        
        return pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
    
    def test_pc_algorithm(self, chain_data):
        """Test PC algorithm."""
        from pycausalsim.discovery import CausalDiscovery
        
        cd = CausalDiscovery(random_state=42)
        graph = cd.discover(chain_data, method='pc', target='Y')
        
        assert graph is not None
        assert 'Y' in graph
    
    def test_ges_algorithm(self, chain_data):
        """Test GES algorithm."""
        from pycausalsim.discovery import CausalDiscovery
        
        cd = CausalDiscovery(random_state=42)
        graph = cd.discover(chain_data, method='ges', target='Y')
        
        assert graph is not None
    
    def test_correlation_graph(self, chain_data):
        """Test correlation-based graph."""
        from pycausalsim.discovery import CausalDiscovery
        
        cd = CausalDiscovery(random_state=42)
        graph = cd.discover(chain_data, method='correlation', target='Y')
        
        assert graph is not None
        # X should be parent of Y
        assert 'X' in graph.get('Y', [])


class TestUtils:
    """Tests for utility functions."""
    
    def test_simulate_data(self):
        """Test data simulation."""
        from pycausalsim.utils import simulate_data
        
        data, true_graph = simulate_data(
            n_samples=500,
            n_features=3,
            treatment_effect=0.5,
            random_state=42
        )
        
        assert len(data) == 500
        assert 'treatment' in data.columns
        assert 'outcome' in data.columns
        assert 'outcome' in true_graph
    
    def test_evaluate_graph(self):
        """Test graph evaluation."""
        from pycausalsim.utils import evaluate_graph
        
        true = {'Y': ['X', 'Z'], 'X': ['Z'], 'Z': []}
        discovered = {'Y': ['X', 'Z'], 'X': [], 'Z': []}  # Missing X <- Z
        
        metrics = evaluate_graph(discovered, true)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'shd' in metrics
    
    def test_bootstrap_effect(self):
        """Test bootstrap effect estimation."""
        from pycausalsim.utils import bootstrap_effect
        
        np.random.seed(42)
        n = 500
        T = np.random.randn(n)
        Y = 0.5 * T + np.random.randn(n)
        data = pd.DataFrame({'treatment': T, 'outcome': Y})
        
        est, lower, upper = bootstrap_effect(
            data, 'treatment', 'outcome', n_bootstrap=100
        )
        
        assert lower < est < upper
        assert abs(est - 0.5) < 0.2


class TestValidation:
    """Tests for validation module."""
    
    @pytest.fixture
    def fitted_simulator(self):
        """Create fitted simulator for validation tests."""
        from pycausalsim import CausalSimulator
        
        np.random.seed(42)
        n = 500
        X = np.random.randn(n)
        T = 0.5 * X + np.random.randn(n) * 0.5
        Y = 0.3 * T + 0.4 * X + np.random.randn(n) * 0.3
        
        data = pd.DataFrame({
            'confounder': X,
            'treatment': T,
            'outcome': Y
        })
        
        sim = CausalSimulator(
            data=data,
            target='outcome',
            treatment_vars=['treatment'],
            confounders=['confounder']
        )
        sim.discover_graph(method='correlation')
        
        return sim
    
    def test_validator_initialization(self, fitted_simulator):
        """Test validator initialization."""
        from pycausalsim.validation import CausalValidator
        
        validator = CausalValidator(fitted_simulator)
        
        assert validator.simulator is fitted_simulator
    
    def test_run_all_tests(self, fitted_simulator):
        """Test running all validation tests."""
        from pycausalsim.validation import CausalValidator
        
        validator = CausalValidator(fitted_simulator, random_state=42)
        results = validator.run_all_tests(n_simulations=50)
        
        assert 'confounding_bounds' in results
        assert 'placebo_tests' in results
        assert 'refutations' in results


class TestResults:
    """Tests for result classes."""
    
    def test_causal_effect(self):
        """Test CausalEffect class."""
        from pycausalsim.results import CausalEffect
        
        effect = CausalEffect(
            variable='treatment',
            intervention_value=1.0,
            original_value=0.5,
            target='outcome',
            point_estimate=0.25,
            ci_lower=0.1,
            ci_upper=0.4,
            p_value=0.01,
            n_simulations=1000
        )
        
        assert effect.is_significant(alpha=0.05)
        assert 'treatment' in effect.summary()
        
        d = effect.to_dict()
        assert d['point_estimate'] == 0.25
    
    def test_simulation_result(self):
        """Test SimulationResult class."""
        from pycausalsim.results import SimulationResult, CausalEffect
        
        effect = CausalEffect(
            variable='X',
            intervention_value=1.0,
            original_value=0.0,
            target='Y',
            point_estimate=0.5,
            ci_lower=0.3,
            ci_upper=0.7,
            p_value=0.001,
            n_simulations=100
        )
        
        result = SimulationResult(
            factual=np.array([1, 2, 3]),
            counterfactual=np.array([2, 3, 4]),
            intervention={'X': 1.0},
            effect=effect
        )
        
        assert result.ate == 1.0
        np.testing.assert_array_equal(result.treatment_effect, [1, 1, 1])


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
