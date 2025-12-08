# PyCausalSim

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**PyCausalSim** is a Python framework for causal discovery and inference through simulation. Unlike correlation-based approaches, PyCausalSim uses counterfactual simulation to establish true causal relationships from observational data, specifically designed for digital metrics optimization.

## Why PyCausalSim?

Traditional analytics tell you *what* happened. PyCausalSim tells you *why* it happened and *what would happen if* you changed something.

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

### 1. Simulation-Based Causal Discovery
- Generate synthetic "what-if" scenarios
- Test interventions before deploying them
- Understand non-linear and interaction effects

### 2. Multiple Discovery Methods
- **Constraint-based**: PC, FCI algorithms
- **Score-based**: GES, FGES algorithms
- **Functional**: LiNGAM for non-Gaussian data
- **Neural**: NOTEARS (deep learning-based)
- **Hybrid**: Combine methods for robustness

### 3. Structural Causal Models (SCM)
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

### 4. Marketing Attribution
```python
from pycausalsim import MarketingAttribution

attr = MarketingAttribution(data, conversion_col='converted')
attr.fit(method='shapley')  # Causal Shapley values

weights = attr.get_attribution()
# {'email': 0.25, 'display': 0.15, 'search': 0.35, 'social': 0.20, 'direct': 0.05}

optimal_budget = attr.optimize_budget(total_budget=100000)
```

### 5. A/B Test Analysis
```python
from pycausalsim import ExperimentAnalysis

exp = ExperimentAnalysis(data, treatment='new_feature', outcome='engagement')
effect = exp.estimate_effect(method='dr')  # Doubly robust estimator

# Check for heterogeneous effects
het = exp.analyze_heterogeneity(covariates=['user_tenure', 'activity_level'])
```

### 6. Uplift Modeling
```python
from pycausalsim.uplift import UpliftModeler

uplift = UpliftModeler(data, treatment='campaign', outcome='conversion')
uplift.fit()

# Segment users: persuadables, sure_things, lost_causes, sleeping_dogs
segments = uplift.segment_by_effect()
```

### 7. Built-in Validation
```python
# Sensitivity analysis
sensitivity = simulator.validate()
sensitivity.confounding_bounds()  # Test unmeasured confounding
sensitivity.placebo_test()  # Placebo tests
sensitivity.refute()  # Multiple refutation methods
```

## Installation

```bash
pip install git+https://github.com/Bodhi8/pycausalsim.git
```

Or clone and install locally:

```bash
git clone https://github.com/Bodhi8/pycausalsim.git
cd pycausalsim
pip install -e ".[dev]"
```

## Quick Start

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
simulator.discover_graph(method='ges')
simulator.plot_graph()  # Visualize learned causal relationships

# Step 2: Simulate interventions
load_time_effect = simulator.simulate_intervention(
    variable='page_load_time',
    value=2.0,  # seconds
    n_simulations=1000
)
print(load_time_effect.summary())

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

## Use Cases

1. **Conversion Rate Optimization**: Identify true drivers of conversion vs. correlated factors
2. **Marketing Attribution**: Move beyond last-touch to understand true incremental value
3. **Product Feature Impact**: Understand which features actually drive engagement
4. **Pricing Optimization**: Understand causal price elasticity (not just correlation)
5. **Retention Analysis**: Identify causal drivers of churn

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

### The PyCausalSim Approach

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

## Project Structure

```
pycausalsim/
├── pycausalsim/
│   ├── __init__.py         # Package exports
│   ├── simulator.py        # Main CausalSimulator class
│   ├── attribution.py      # Marketing Attribution
│   ├── experiment.py       # A/B Test Analysis
│   ├── results.py          # Result classes
│   ├── models/             # Structural Causal Models
│   ├── discovery/          # Causal discovery algorithms
│   ├── uplift/             # Uplift modeling
│   ├── agents/             # Agent-based simulation
│   ├── validation/         # Sensitivity analysis
│   ├── adapters/           # DoWhy/EconML integration
│   ├── utils/              # Utilities
│   └── visualization/      # Plotting
├── tests/                  # Test suite
├── examples/               # Example scripts
├── pyproject.toml          # Package configuration
└── README.md
```

## Dependencies

**Core:**
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

**Optional:**
- matplotlib, networkx, seaborn (visualization)
- dowhy, econml (integrations)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/Bodhi8/pycausalsim.git
cd pycausalsim
pip install -e ".[dev]"
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

PyCausalSim builds on research from:
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference*
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*
- Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*

And integrates with:
- [DoWhy](https://github.com/py-why/dowhy) - Microsoft's causal inference library
- [EconML](https://github.com/py-why/EconML) - Heterogeneous treatment effects
- [CausalML](https://github.com/uber/causalml) - Uber's uplift modeling library

## Citation

```bibtex
@software{pycausalsim2025,
  title = {PyCausalSim: Causal Discovery Through Simulation},
  author = {Brian Curry},
  year = {2025},
  url = {https://github.com/Bodhi8/pycausalsim}
}
```

## Contact

- GitHub: [github.com/Bodhi8/pycausalsim](https://github.com/Bodhi8/pycausalsim)
- Email: brian@vector1.ai
- Website: [Vector1 AI](https://vector1.ai)
