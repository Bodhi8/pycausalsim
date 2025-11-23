# PyCausalSim Architecture

## Vision

PyCausalSim is a Python framework for causal discovery through simulation, specifically designed for digital metrics optimization (conversion rates, engagement, retention, etc.). Unlike correlation-based approaches, PyCausalSim uses simulation to establish true causal relationships from observational data.

## Core Philosophy

1. **Simulation-First**: Use counterfactual simulation to test causal hypotheses
2. **Multi-Method**: Support multiple causal discovery approaches
3. **Practitioner-Friendly**: Designed for data scientists working on real business problems
4. **Integration-Ready**: Works with existing causal inference libraries (DoWhy, EconML, etc.)

## Architecture Overview

```
PyCausalSim
├── Core Engine
│   ├── Structural Causal Models (SCM)
│   ├── Counterfactual Generator
│   └── Intervention Simulator
├── Discovery Methods
│   ├── Graph Learning
│   ├── Uplift Modeling
│   └── Sensitivity Analysis
├── Simulation Backends
│   ├── Agent-Based Models
│   ├── Monte Carlo Simulation
│   └── Neural Causal Models
└── Utilities
    ├── Visualization
    ├── Validation
    └── Integration Adapters
```

## Core API Design

### 1. Main Entry Point: `CausalSimulator`

The primary interface for users to conduct causal analysis.

```python
from pycausalsim import CausalSimulator

# Initialize with data
simulator = CausalSimulator(
    data=df,
    target='conversion_rate',
    treatment_vars=['page_load_time', 'price', 'design_variant'],
    confounders=['traffic_source', 'device_type', 'time_of_day']
)

# Discover causal structure
simulator.discover_graph(method='pc')  # or 'ges', 'lingam', 'notears'

# Simulate interventions
results = simulator.simulate_intervention(
    variable='page_load_time',
    value=2.0,
    n_simulations=1000
)

# Rank causal drivers
drivers = simulator.rank_drivers()
```

### 2. Structural Causal Model (SCM)

Core causal modeling component.

```python
from pycausalsim.models import StructuralCausalModel

# Define or learn SCM
scm = StructuralCausalModel()
scm.fit(data)

# Generate counterfactuals
counterfactuals = scm.counterfactual(
    intervention={'page_load_time': 2.0},
    evidence={'traffic_source': 'paid'},
    n_samples=1000
)

# Compute effects
ate = scm.average_treatment_effect('page_load_time', 'conversion_rate')
cate = scm.conditional_average_treatment_effect(
    'page_load_time', 
    'conversion_rate',
    conditions={'device_type': 'mobile'}
)
```

### 3. Graph Discovery

Learn causal structure from data.

```python
from pycausalsim.discovery import CausalGraphDiscovery

# Initialize discovery
discovery = CausalGraphDiscovery(data)

# Learn graph structure
graph = discovery.learn(
    method='notears',  # Neural network-based
    lambda_param=0.1
)

# Validate with domain knowledge
graph.add_forbidden_edge('conversion_rate', 'page_load_time')
graph.add_required_edge('price', 'conversion_rate')

# Refine
refined_graph = discovery.refine(graph, method='bootstrap', n_iter=100)
```

### 4. Intervention Simulator

Simulate what-if scenarios.

```python
from pycausalsim.simulation import InterventionSimulator

sim = InterventionSimulator(causal_model=scm)

# Single intervention
result = sim.do(
    intervention={'page_load_time': 2.0},
    n_simulations=10000
)

# Multiple interventions (joint)
results = sim.do_multiple([
    {'page_load_time': 2.0},
    {'page_load_time': 2.0, 'price': 0.9 * current_price},
    {'design_variant': 'B'}
])

# Policy simulation
policy_results = sim.simulate_policy(
    policy=lambda x: x['page_load_time'] * 0.8 if x['traffic_source'] == 'mobile' else x['page_load_time'],
    n_simulations=10000
)
```

### 5. Uplift Modeling

Identify heterogeneous treatment effects.

```python
from pycausalsim.uplift import UpliftModeler

uplift = UpliftModeler(data, treatment='campaign_variant', outcome='conversion')

# Fit uplift model
uplift.fit(method='causal_forest')

# Predict individual treatment effects
ite = uplift.predict_ite(X_new)

# Segment by uplift
segments = uplift.segment_by_effect(n_segments=4)
# Returns: ['persuadables', 'sure_things', 'lost_causes', 'sleeping_dogs']

# Optimize targeting
optimal_treatment = uplift.recommend_treatment(X_new)
```

### 6. Sensitivity Analysis

Test robustness of causal conclusions.

```python
from pycausalsim.validation import SensitivityAnalysis

sensitivity = SensitivityAnalysis(simulator)

# Test sensitivity to unmeasured confounding
bounds = sensitivity.confounding_bounds(
    treatment='page_load_time',
    outcome='conversion_rate',
    r_squared_range=(0, 0.3)
)

# Placebo tests
placebo_results = sensitivity.placebo_test(
    pre_period=(0, 100),
    post_period=(101, 200),
    n_permutations=1000
)

# Refutation tests
refutation = sensitivity.refute(
    method='random_common_cause',
    n_iterations=100
)
```

### 7. Agent-Based Simulation (Advanced)

For complex systems with interaction effects.

```python
from pycausalsim.agents import AgentBasedSimulator

# Define agent behavior
class UserAgent:
    def __init__(self, attributes):
        self.attributes = attributes
        
    def decide(self, environment):
        # Decision logic based on causal model
        pass

# Create simulator
abs_sim = AgentBasedSimulator(
    n_agents=10000,
    agent_class=UserAgent,
    causal_model=scm
)

# Run simulation
results = abs_sim.simulate(
    steps=100,
    interventions={'page_load_time': 2.0}
)

# Analyze emergent effects
network_effects = abs_sim.analyze_contagion()
```

## Key Design Principles

### 1. Composability
All components work together but can be used independently:
```python
# Can use just graph discovery
from pycausalsim.discovery import CausalGraphDiscovery
graph = CausalGraphDiscovery(data).learn()

# Or just simulation
from pycausalsim.simulation import InterventionSimulator
sim = InterventionSimulator(my_custom_scm)
```

### 2. Integration with Ecosystem
```python
# Import from DoWhy
from dowhy import CausalModel as DoWhyModel
from pycausalsim.adapters import from_dowhy

simulator = from_dowhy(dowhy_model)

# Export to EconML
econml_model = simulator.to_econml()

# Work with Papilon
from papilon import ComplexSystem
from pycausalsim.adapters import from_papilon

papilon_results = ComplexSystem().run()
causal_analysis = from_papilon(papilon_results)
```

### 3. Transparency & Interpretability
```python
# Every result includes diagnostics
result = simulator.simulate_intervention('price', 0.9)

print(result.summary())
# Shows: effect size, confidence intervals, assumptions, sensitivity

result.plot()  # Visualize counterfactual vs. factual
result.validate()  # Run automatic validation checks
result.explain()  # Plain English explanation
```

### 4. Validation-First
```python
# Built-in validation pipeline
validator = simulator.validate()

validator.check_assumptions()  # Positivity, unconfoundedness, etc.
validator.check_overlap()  # Covariate balance
validator.cross_validate()  # Out-of-sample performance
validator.refutation_tests()  # Add random confounders, etc.

# Get validation report
report = validator.generate_report()
```

## Implementation Phases

### Phase 1: Core Foundation (Weeks 1-4)
- [ ] Structural Causal Model implementation
- [ ] Basic graph discovery (PC, GES algorithms)
- [ ] Monte Carlo counterfactual simulation
- [ ] Intervention simulator
- [ ] Basic visualization

### Phase 2: Advanced Discovery (Weeks 5-8)
- [ ] Neural causal models (NOTEARS, DAG-GNN)
- [ ] Variational Bayesian graph learning
- [ ] Constraint-based methods
- [ ] Hybrid approaches

### Phase 3: Uplift & Heterogeneity (Weeks 9-12)
- [ ] Uplift modeling (T-learner, S-learner, X-learner)
- [ ] Causal forests
- [ ] Meta-learners
- [ ] Individual treatment effect estimation

### Phase 4: Agent-Based & Complex Systems (Weeks 13-16)
- [ ] Agent-based simulation engine
- [ ] Network effects modeling
- [ ] Time-series causal discovery
- [ ] Dynamic treatment regimes

### Phase 5: Ecosystem Integration (Weeks 17-20)
- [ ] DoWhy adapter
- [ ] EconML adapter
- [ ] CausalML adapter
- [ ] Papilon integration
- [ ] Export to common formats

## Technical Stack

### Core Dependencies
```python
# Numerical & Data
numpy >= 1.21.0
pandas >= 1.3.0
scipy >= 1.7.0

# ML & Causal
scikit-learn >= 1.0.0
torch >= 1.10.0  # For neural causal models
dowhy >= 0.8.0  # Optional, for integration
econml >= 0.13.0  # Optional, for integration

# Graph & Optimization
networkx >= 2.6.0
cvxpy >= 1.1.0  # For constrained optimization
pgmpy >= 0.1.18  # For Bayesian networks

# Visualization
matplotlib >= 3.4.0
seaborn >= 0.11.0
plotly >= 5.0.0  # For interactive plots
```

### Optional Dependencies
```python
# For agent-based models
mesa >= 0.9.0

# For large-scale simulation
dask >= 2021.10.0
ray >= 1.8.0

# For GPU acceleration
cupy >= 9.0.0  # CUDA arrays
```

## Code Organization

```
pycausalsim/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── simulator.py          # Main CausalSimulator class
│   ├── scm.py                 # Structural Causal Model
│   └── graph.py               # Causal graph representation
├── discovery/
│   ├── __init__.py
│   ├── constraint_based.py    # PC, FCI algorithms
│   ├── score_based.py         # GES, FGES algorithms
│   ├── functional.py          # LiNGAM, ANM
│   └── neural.py              # NOTEARS, DAG-GNN
├── simulation/
│   ├── __init__.py
│   ├── intervention.py        # Intervention simulator
│   ├── counterfactual.py      # Counterfactual generation
│   └── monte_carlo.py         # Monte Carlo methods
├── uplift/
│   ├── __init__.py
│   ├── meta_learners.py       # T, S, X learners
│   ├── causal_forest.py       # Causal forest implementation
│   └── segmentation.py        # Effect segmentation
├── agents/
│   ├── __init__.py
│   ├── simulator.py           # Agent-based simulator
│   ├── agent.py               # Base agent class
│   └── environment.py         # Environment class
├── validation/
│   ├── __init__.py
│   ├── sensitivity.py         # Sensitivity analysis
│   ├── refutation.py          # Refutation tests
│   └── diagnostics.py         # Diagnostic checks
├── adapters/
│   ├── __init__.py
│   ├── dowhy_adapter.py       # DoWhy integration
│   ├── econml_adapter.py      # EconML integration
│   └── papilon_adapter.py     # Papilon integration
├── viz/
│   ├── __init__.py
│   ├── graphs.py              # Graph visualization
│   ├── effects.py             # Effect visualization
│   └── diagnostics.py         # Diagnostic plots
└── utils/
    ├── __init__.py
    ├── data.py                # Data utilities
    ├── metrics.py             # Evaluation metrics
    └── io.py                  # I/O utilities
```

## Example Use Cases

### Use Case 1: E-commerce Conversion Optimization
```python
# Load data
data = pd.read_csv('ecommerce_data.csv')

# Initialize
sim = CausalSimulator(
    data=data,
    target='conversion',
    treatment_vars=['price', 'load_time', 'design', 'shipping_cost'],
    confounders=['traffic_source', 'device', 'hour', 'day_of_week']
)

# Discover structure
sim.discover_graph(method='notears')

# Simulate interventions
price_effect = sim.simulate_intervention('price', current * 0.9)
load_effect = sim.simulate_intervention('load_time', 2.0)

# Find optimal policy
optimal = sim.optimize_policy(
    objective='maximize_conversion',
    constraints={'price': (min_price, max_price)}
)
```

### Use Case 2: Marketing Attribution
```python
from pycausalsim import MarketingAttribution

# Initialize with multi-touch data
attr = MarketingAttribution(
    data=touchpoint_data,
    conversion_col='converted',
    touchpoint_cols=['email', 'display', 'search', 'social']
)

# Learn causal structure
attr.fit(method='shapley')

# Get true attribution (not just last-touch)
attribution_weights = attr.get_attribution()

# Simulate budget reallocation
new_budget = attr.optimize_budget(
    total_budget=100000,
    objective='maximize_conversions'
)
```

### Use Case 3: Product Feature Impact
```python
# A/B test analysis with causal lens
from pycausalsim import ExperimentAnalysis

exp = ExperimentAnalysis(
    data=ab_test_data,
    treatment='new_feature',
    outcome='engagement'
)

# Get causal effect
effect = exp.estimate_effect(method='dr')  # Doubly robust

# Check for heterogeneous effects
heterogeneity = exp.analyze_heterogeneity(
    covariates=['user_tenure', 'activity_level']
)

# Simulate long-term impact
longterm = exp.simulate_longterm(periods=90)
```

## Performance Considerations

### Scalability
- Support for large datasets via Dask integration
- GPU acceleration for neural methods
- Parallel simulation via Ray/multiprocessing
- Lazy evaluation where possible

### Optimization
- Cached graph computations
- Efficient tensor operations (PyTorch)
- Sparse matrix representations
- Incremental learning for streaming data

## Documentation Strategy

1. **API Reference**: Auto-generated from docstrings
2. **User Guide**: Tutorial-style documentation
3. **Examples**: Jupyter notebooks for common use cases
4. **Theory**: Background on causal inference concepts
5. **Comparison**: vs. correlation methods, vs. other causal libraries

## Testing Strategy

1. **Unit Tests**: Each component tested independently
2. **Integration Tests**: End-to-end workflows
3. **Simulation Tests**: Validate on synthetic data with known causal structure
4. **Benchmark Tests**: Compare against published results
5. **Performance Tests**: Ensure scalability

## Success Metrics

- **Adoption**: PyPI downloads, GitHub stars
- **Usage**: Active users, questions/issues
- **Accuracy**: Validated on benchmark datasets
- **Performance**: Speed vs. alternatives
- **Community**: Contributors, integrations

## Future Directions

1. **Time-Series Causal Discovery**: Granger causality, VAR models
2. **Fairness**: Counterfactual fairness analysis
3. **Reinforcement Learning**: Causal RL for policy learning
4. **NLP Integration**: Text-based confounders
5. **AutoML**: Automatic method selection
6. **Real-Time**: Streaming causal inference

---

This architecture provides a solid foundation for PyCausalSim while remaining flexible enough to evolve as the field advances.
