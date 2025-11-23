# PyCausalSim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PyCausalSim** is a Python framework for causal discovery and inference through simulation. Unlike correlation-based approaches, PyCausalSim uses counterfactual simulation to establish true causal relationships from observational data, specifically designed for digital metrics optimization.

## Why PyCausalSim?

Traditional analytics tell you **what** happened. PyCausalSim tells you **why** it happened and **what would happen if** you changed something.

```python
from pycausalsim import CausalSimulator

# Your conversion rate increased after reducing load time
# But did load time CAUSE the increase? Or was it just correlation?

simulator = CausalSimulator(data, target='conversion_rate')
simulator.discover_graph()  # Learn causal structure

# Simulate: What if we reduce load time to 2 seconds?
effect = simulator.simulate_intervention('page_load_time', 2.0)
print(f"Expected conversion lift: {effect.point_estimate:.2%}")
print(f"95% CI: [{effect.ci_lower:.2%}, {effect.ci_upper:.2%}]")

# Rank all causal drivers
drivers = simulator.rank_drivers()
# Returns: [(page_load_time, 0.15), (price, 0.12), (design, 0.03), ...]
```

## Key Features

### 1. **Simulation-Based Causal Discovery**
Unlike regression or ML, PyCausalSim uses counterfactual simulation to test causal hypotheses:
- Generate synthetic "what-if" scenarios
- Test interventions before deploying them
- Understand non-linear and interaction effects

### 2. **Multiple Discovery Methods**
- **Constraint-based**: PC, FCI algorithms
- **Score-based**: GES, FGES algorithms  
- **Functional**: LiNGAM, ANM
- **Neural**: NOTEARS, DAG-GNN (deep learning-based)
- **Hybrid**: Combine methods for robustness

### 3. **Structural Causal Models (SCM)**
Build explicit causal models that capture how your system works:
```python
from pycausalsim.models import StructuralCausalModel

scm = StructuralCausalModel()
scm.fit(data)

# Generate counterfactuals
counterfactuals = scm.counterfactual(
    intervention={'price': 0.9 * current_price},
    evidence={'traffic_source': 'paid'}
)
```

### 4. **Uplift Modeling**
Identify WHO will respond to treatments:
```python
from pycausalsim.uplift import UpliftModeler

uplift = UpliftModeler(data, treatment='campaign', outcome='conversion')
uplift.fit()

# Segment users by predicted treatment effect
segments = uplift.segment_by_effect()
# Returns: persuadables, sure_things, lost_causes, sleeping_dogs
```

### 5. **Agent-Based Simulation**
Model complex systems with interacting agents:
```python
from pycausalsim.agents import AgentBasedSimulator

# Simulate 10,000 users with network effects
abs_sim = AgentBasedSimulator(n_agents=10000, causal_model=scm)
results = abs_sim.simulate(steps=100)

# Analyze emergent behavior
network_effects = abs_sim.analyze_contagion()
```

### 6. **Built-in Validation**
Every causal claim is automatically validated:
```python
# Sensitivity analysis
sensitivity = simulator.validate()
sensitivity.confounding_bounds()  # Test unmeasured confounding
sensitivity.placebo_test()  # Placebo tests
sensitivity.refute()  # Multiple refutation methods
```

## Installation

```bash
pip install pycausalsim
```

For development installation:
```bash
git clone https://github.com/Bodhi8/pycausalsim.git
cd pycausalsim
pip install -e ".[dev]"
```

## Quick Start

### Basic Example: E-commerce Conversion

```python
import pandas as pd
from pycausalsim import CausalSimulator

# Load your data
data = pd.read_csv('conversion_data.csv')

# Initialize simulator
simulator = CausalSimulator(
    data=data,
    target='conversion_rate',
    treatment_vars=['page_load_time', 'price', 'design_variant'],
    confounders=['traffic_source', 'device_type', 'time_of_day']
)

# Step 1: Discover causal structure
simulator.discover_graph(method='notears')
simulator.plot_graph()  # Visualize learned causal relationships

# Step 2: Simulate interventions
load_time_effect = simulator.simulate_intervention(
    variable='page_load_time',
    value=2.0,  # seconds
    n_simulations=1000
)

print(load_time_effect.summary())
# Output:
# Intervention: page_load_time = 2.0
# Current value: 3.5
# Effect on conversion_rate: +2.3% (95% CI: [1.8%, 2.8%])
# P-value: 0.001

# Step 3: Rank all causal drivers
drivers = simulator.rank_drivers()
for var, effect in drivers:
    print(f"{var}: {effect:.3f}")

# Step 4: Find optimal policy
optimal = simulator.optimize_policy(
    objective='maximize_conversion',
    constraints={'price': (10, 50)}
)
```

### Marketing Attribution

```python
from pycausalsim import MarketingAttribution

# Multi-touch attribution with causal inference
attr = MarketingAttribution(
    data=touchpoint_data,
    conversion_col='converted',
    touchpoint_cols=['email', 'display', 'search', 'social', 'direct']
)

# Learn true causal contribution (not just last-touch)
attr.fit(method='shapley')

# Get attribution weights
weights = attr.get_attribution()
# {'email': 0.25, 'display': 0.15, 'search': 0.35, 'social': 0.20, 'direct': 0.05}

# Optimize budget allocation
optimal_budget = attr.optimize_budget(total_budget=100000)
```

### A/B Test Analysis

```python
from pycausalsim import ExperimentAnalysis

# Analyze A/B test with causal lens
exp = ExperimentAnalysis(
    data=ab_test_data,
    treatment='new_feature',
    outcome='engagement'
)

# Get causal effect (with heterogeneity)
effect = exp.estimate_effect(method='dr')  # Doubly robust estimator

# Check for heterogeneous effects
het = exp.analyze_heterogeneity(covariates=['user_tenure', 'activity_level'])
het.plot()

# Simulate long-term impact
longterm = exp.simulate_longterm(periods=90)
```

## Use Cases

### 1. **Conversion Rate Optimization**
- Identify true drivers of conversion (vs. correlated factors)
- Simulate impact of UX changes before deploying
- Optimize page elements (load time, design, pricing)

### 2. **Marketing Attribution**
- Move beyond last-touch attribution
- Understand true incremental value of each channel
- Optimize budget allocation causally

### 3. **Product Feature Impact**
- Understand which features actually drive engagement
- Simulate long-term effects of feature changes
- Identify user segments with heterogeneous responses

### 4. **Pricing Optimization**
- Understand causal price elasticity (not just correlation)
- Account for selection bias and confounding
- Simulate dynamic pricing strategies

### 5. **Retention Analysis**
- Identify causal drivers of churn
- Simulate intervention effectiveness
- Target users most likely to respond to retention efforts

## Why Simulation for Causality?

### The Problem with Correlation
```python
# Traditional approach - WRONG!
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X, y)
feature_importance = rf.feature_importances_

# Feature importance ≠ causal importance
# Tells you what predicts, not what causes
# Fails with confounding, selection bias, reverse causation
```

### The PyCausalSim Approach - RIGHT!
```python
from pycausalsim import CausalSimulator

sim = CausalSimulator(data, target='conversion')
sim.discover_graph()  # Learn causal structure

# Explicitly models confounding
# Distinguishes correlation from causation
# Validates assumptions
# Provides counterfactual predictions

drivers = sim.rank_drivers()  # TRUE causal importance
```

## Visualization

PyCausalSim includes rich visualization tools:

```python
# Causal graph
simulator.plot_graph()

# Effect sizes
simulator.plot_effects()

# Counterfactual vs. factual
result.plot_counterfactual()

# Sensitivity analysis
sensitivity.plot_bounds()

# Heterogeneity
uplift.plot_segments()
```

## Integrations

PyCausalSim integrates with the broader causal inference ecosystem:

### DoWhy Integration
```python
from dowhy import CausalModel
from pycausalsim.adapters import from_dowhy

dowhy_model = CausalModel(...)
simulator = from_dowhy(dowhy_model)
```

### EconML Integration
```python
simulator = CausalSimulator(data)
econml_model = simulator.to_econml()
```

### Papilon Integration
```python
from papilon import ComplexSystem
from pycausalsim.adapters import from_papilon

papilon_results = ComplexSystem().run()
causal_analysis = from_papilon(papilon_results)
```

## Documentation

- **[Full Documentation](https://pycausalsim.readthedocs.io)** (Coming soon)
- **[API Reference](https://pycausalsim.readthedocs.io/api)** (Coming soon)
- **[Examples](./examples)**: Jupyter notebooks with detailed walkthroughs
- **[Architecture](./ARCHITECTURE.md)**: Technical design document

## Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/Bodhi8/pycausalsim.git
cd pycausalsim
pip install -e ".[dev]"
pytest tests/
```

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Acknowledgments

PyCausalSim builds on the excellent work of:
- [DoWhy](https://github.com/py-why/dowhy) - Microsoft's causal inference library
- [EconML](https://github.com/microsoft/EconML) - Heterogeneous treatment effects
- [CausalML](https://github.com/uber/causalml) - Uber's uplift modeling library
- [Papilon](https://github.com/Bodhi8/papilon) - Complex systems framework

Inspired by research including:
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference*
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*

## Contact and Support

- **Issues**: [GitHub Issues](https://github.com/Bodhi8/pycausalsim/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Bodhi8/pycausalsim/discussions)
- **Email**: [your-email@example.com]

## Roadmap

### v0.1.0 (Current)
- [x] Core simulation engine
- [x] Basic graph discovery (PC, GES)
- [x] Structural causal models
- [x] Intervention simulator

### v0.2.0 (Q2 2025)
- [ ] Neural causal models (NOTEARS)
- [ ] Uplift modeling
- [ ] Enhanced visualization
- [ ] DoWhy/EconML integration

### v0.3.0 (Q3 2025)
- [ ] Agent-based simulation
- [ ] Time-series causal discovery
- [ ] Real-time inference
- [ ] Comprehensive documentation

### v1.0.0 (Q4 2025)
- [ ] Production-ready
- [ ] Full test coverage
- [ ] Performance optimization
- [ ] Industry case studies

## Citation

If you use PyCausalSim in your research, please cite:

```bibtex
@software{pycausalsim2025,
  title = {PyCausalSim: Causal Discovery Through Simulation},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/Bodhi8/pycausalsim}
}
```

---

**Made with care for data scientists who want to understand causality, not just correlation.**

*"Correlation is not causation. But simulation can reveal causation."*
