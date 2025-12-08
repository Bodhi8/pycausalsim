"""
PyCausalSim - Complete Example
==============================

This script demonstrates all major features of PyCausalSim:
1. Causal Discovery - Learning causal structure from data
2. Intervention Simulation - Estimating causal effects
3. Marketing Attribution - Assigning credit to touchpoints
4. A/B Test Analysis - Robust experiment analysis
5. Uplift Modeling - Identifying treatment responders

Run this example:
    python examples/complete_example.py
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '..')

# =============================================================================
# 1. CAUSAL SIMULATOR - Core Functionality
# =============================================================================
print("=" * 70)
print("1. CAUSAL SIMULATOR - Discovering Causal Relationships")
print("=" * 70)

from pycausalsim import CausalSimulator

# Generate synthetic e-commerce data
np.random.seed(42)
n = 1000

# True causal structure:
# traffic_source -> page_load_time -> conversion_rate
# traffic_source -> conversion_rate (confounding)
# price -> conversion_rate

traffic_source = np.random.choice([0, 1], n, p=[0.6, 0.4])  # 0=organic, 1=paid
page_load_time = 3.0 + 0.5 * traffic_source + np.random.exponential(0.5, n)
price = 20 + np.random.normal(0, 5, n)
device_type = np.random.choice([0, 1, 2], n)  # 0=mobile, 1=desktop, 2=tablet

# True conversion model (with causal effects)
conversion_prob = 0.15 - 0.03 * page_load_time + 0.02 * traffic_source - 0.005 * price
conversion_prob = np.clip(conversion_prob, 0.01, 0.99)
conversion_rate = np.random.binomial(1, conversion_prob) * 100  # Convert to percentage

data = pd.DataFrame({
    'traffic_source': traffic_source,
    'page_load_time': page_load_time,
    'price': price,
    'device_type': device_type,
    'conversion_rate': conversion_rate
})

print(f"\nDataset: {len(data)} observations")
print(f"Variables: {list(data.columns)}")
print(f"Conversion rate: {data['conversion_rate'].mean():.1f}%")

# Initialize simulator
simulator = CausalSimulator(
    data=data,
    target='conversion_rate',
    treatment_vars=['page_load_time', 'price'],
    confounders=['traffic_source', 'device_type']
)

# Discover causal graph
print("\nDiscovering causal structure...")
simulator.discover_graph(method='ges')
print(f"Learned graph: {simulator.graph}")

# Simulate intervention: What if we reduce load time to 2 seconds?
print("\nSimulating intervention: page_load_time = 2.0 seconds")
effect = simulator.simulate_intervention(
    variable='page_load_time',
    value=2.0,
    n_simulations=500
)

print(f"\n{effect.summary()}")

# Rank all causal drivers
print("\n" + "-" * 50)
print("Ranking Causal Drivers of Conversion Rate:")
print("-" * 50)
drivers = simulator.rank_drivers(n_simulations=200)
for var, effect_size in drivers:
    print(f"  {var}: {effect_size:+.4f}")

# Validate causal claims
print("\n" + "-" * 50)
print("Validating Causal Claims:")
print("-" * 50)
sensitivity = simulator.validate(variable='page_load_time', n_simulations=200)
print(sensitivity.summary())


# =============================================================================
# 2. MARKETING ATTRIBUTION
# =============================================================================
print("\n" + "=" * 70)
print("2. MARKETING ATTRIBUTION - True Channel Value")
print("=" * 70)

from pycausalsim import MarketingAttribution

# Generate marketing touchpoint data
np.random.seed(42)
n = 2000

# Simulate customer journeys with multiple touchpoints
touchpoint_data = pd.DataFrame({
    'email': np.random.binomial(1, 0.3, n),
    'display': np.random.binomial(1, 0.4, n),
    'search': np.random.binomial(1, 0.5, n),
    'social': np.random.binomial(1, 0.25, n),
    'direct': np.random.binomial(1, 0.15, n)
})

# True causal contribution (search > email > display > social > direct)
base_conversion = 0.05
conversion_prob = (
    base_conversion + 
    0.15 * touchpoint_data['search'] +
    0.10 * touchpoint_data['email'] +
    0.05 * touchpoint_data['display'] +
    0.03 * touchpoint_data['social'] +
    0.01 * touchpoint_data['direct']
)
touchpoint_data['converted'] = np.random.binomial(1, np.clip(conversion_prob, 0, 1))

print(f"\nMarketing data: {len(touchpoint_data)} customers")
print(f"Conversion rate: {touchpoint_data['converted'].mean():.1%}")

# Fit attribution model
attr = MarketingAttribution(
    data=touchpoint_data,
    conversion_col='converted',
    touchpoint_cols=['email', 'display', 'search', 'social', 'direct']
)

# Compare attribution methods
print("\n" + "-" * 50)
print("Attribution Weights by Method:")
print("-" * 50)

for method in ['shapley', 'logistic', 'last_touch']:
    attr.fit(method=method)
    weights = attr.get_attribution()
    print(f"\n{method.upper()}:")
    for channel, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {channel}: {weight:.1%}")

# Optimize budget allocation
print("\n" + "-" * 50)
print("Optimized Budget Allocation ($100,000):")
print("-" * 50)
attr.fit(method='shapley')
optimal = attr.optimize_budget(total_budget=100000)
for channel, budget in sorted(optimal.items(), key=lambda x: x[1], reverse=True):
    print(f"  {channel}: ${budget:,.0f}")


# =============================================================================
# 3. A/B TEST ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("3. A/B TEST ANALYSIS - Robust Experiment Analysis")
print("=" * 70)

from pycausalsim import ExperimentAnalysis

# Generate A/B test data with heterogeneous effects
np.random.seed(42)
n = 2000

user_tenure = np.random.exponential(12, n)  # months
activity_level = np.random.uniform(0, 10, n)
treatment = np.random.binomial(1, 0.5, n)

# True effect: 0.5 overall, but varies by tenure
true_effect = 0.5 + 0.02 * user_tenure - 0.03 * activity_level
engagement = 5 + treatment * true_effect + 0.1 * activity_level + np.random.normal(0, 1, n)

ab_data = pd.DataFrame({
    'new_feature': treatment,
    'engagement': engagement,
    'user_tenure': user_tenure,
    'activity_level': activity_level
})

print(f"\nA/B test: {len(ab_data)} users")
print(f"Treatment group: {ab_data['new_feature'].sum()} users")
print(f"Control group: {(1 - ab_data['new_feature']).sum()} users")

# Analyze experiment
exp = ExperimentAnalysis(
    data=ab_data,
    treatment='new_feature',
    outcome='engagement',
    covariates=['user_tenure', 'activity_level']
)

print("\n" + "-" * 50)
print("Treatment Effect Estimates:")
print("-" * 50)

for method in ['difference', 'ols', 'dr']:
    effect = exp.estimate_effect(method=method)
    sig = "***" if effect.p_value < 0.001 else "**" if effect.p_value < 0.01 else "*" if effect.p_value < 0.05 else ""
    print(f"  {method.upper():12} Effect: {effect.estimate:.4f} (p={effect.p_value:.4f}) {sig}")

# Analyze heterogeneous effects
print("\n" + "-" * 50)
print("Heterogeneous Effects by User Tenure:")
print("-" * 50)
het = exp.analyze_heterogeneity(covariates=['user_tenure'])
for h in het:
    print(f"\nBy {h.covariate}:")
    for seg, eff in h.segments.items():
        ci = h.segment_cis.get(seg, (0, 0))
        print(f"  Segment {seg}: {eff:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")


# =============================================================================
# 4. UPLIFT MODELING
# =============================================================================
print("\n" + "=" * 70)
print("4. UPLIFT MODELING - Who Responds to Treatment?")
print("=" * 70)

from pycausalsim.uplift import UpliftModeler

# Generate data with heterogeneous treatment effects
np.random.seed(42)
n = 3000

feature1 = np.random.randn(n)  # responsiveness indicator
feature2 = np.random.randn(n)  # baseline propensity
treatment = np.random.binomial(1, 0.5, n)

# True heterogeneous effect: positive for high feature1, negative for low
base_prob = 0.3 + 0.1 * feature2
uplift_effect = 0.15 * feature1 * treatment  # Only affects treated
outcome_prob = np.clip(base_prob + uplift_effect, 0.01, 0.99)
outcome = np.random.binomial(1, outcome_prob)

uplift_data = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'campaign': treatment,
    'converted': outcome
})

print(f"\nUplift data: {len(uplift_data)} users")
print(f"Treatment: {uplift_data['campaign'].sum()} users")
print(f"Overall conversion: {uplift_data['converted'].mean():.1%}")

# Fit uplift model
uplift = UpliftModeler(
    data=uplift_data,
    treatment='campaign',
    outcome='converted',
    features=['feature1', 'feature2']
)

uplift.fit(method='two_model')

# Get segments
print("\n" + "-" * 50)
print("User Segments by Treatment Response:")
print("-" * 50)
segments = uplift.segment_by_effect()
for seg in segments:
    print(f"\n  {seg.name}:")
    print(f"    Size: {seg.size} users ({seg.size/len(uplift_data):.1%})")
    print(f"    Predicted uplift: {seg.predicted_uplift:.4f}")
    print(f"    Actual uplift: {seg.actual_uplift:.4f}")
    print(f"    Conversion (treated): {seg.conversion_treated:.1%}")
    print(f"    Conversion (control): {seg.conversion_control:.1%}")

# Evaluate model
print("\n" + "-" * 50)
print("Model Performance:")
print("-" * 50)
metrics = uplift.evaluate()
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")


# =============================================================================
# 5. STRUCTURAL CAUSAL MODEL
# =============================================================================
print("\n" + "=" * 70)
print("5. STRUCTURAL CAUSAL MODEL - Explicit Causal Mechanisms")
print("=" * 70)

from pycausalsim.models import StructuralCausalModel

# Define a known causal graph
# Price -> Demand -> Revenue
# Advertising -> Demand
graph = {
    'revenue': ['demand', 'price'],
    'demand': ['price', 'advertising'],
    'price': [],
    'advertising': []
}

# Generate data according to this structure
np.random.seed(42)
n = 500

price = np.random.uniform(10, 50, n)
advertising = np.random.uniform(0, 100, n)
demand = 100 - 1.5 * price + 0.5 * advertising + np.random.normal(0, 10, n)
revenue = demand * price / 100

scm_data = pd.DataFrame({
    'price': price,
    'advertising': advertising, 
    'demand': demand,
    'revenue': revenue
})

print(f"\nSCM data: {len(scm_data)} observations")
print(f"Causal graph: {graph}")

# Fit SCM
scm = StructuralCausalModel(graph=graph)
scm.fit(scm_data)

print("\n" + "-" * 50)
print("Fitted Structural Causal Model")
print("-" * 50)
print(scm.summary())

# Compute ATE for price reduction
print("\n" + "-" * 50)
print("Average Treatment Effect: 10% Price Reduction")
print("-" * 50)
ate = scm.ate(
    treatment='price',
    outcome='revenue',
    treatment_value=27,  # 10% below mean
    control_value=30,    # mean price
    n_samples=5000
)
print(f"  ATE on revenue: {ate:.4f}")

# Generate counterfactuals
print("\n" + "-" * 50)
print("Counterfactual Analysis: What if advertising = 80?")
print("-" * 50)
cf = scm.counterfactual(
    intervention={'advertising': 80},
    data=scm_data.head(10)
)
print(f"  Original avg demand: {scm_data.head(10)['demand'].mean():.2f}")
print(f"  Counterfactual avg demand: {cf['demand'].mean():.2f}")
print(f"  Difference: {cf['demand'].mean() - scm_data.head(10)['demand'].mean():+.2f}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("PYCAUSALSIM DEMONSTRATION COMPLETE")
print("=" * 70)
print("""
PyCausalSim provides:

1. CausalSimulator    - Learn causal graphs, simulate interventions
2. MarketingAttribution - True incremental channel value  
3. ExperimentAnalysis   - Robust A/B test analysis with heterogeneity
4. UpliftModeler        - Identify treatment responders
5. StructuralCausalModel - Explicit causal mechanisms

Key advantages over correlation-based approaches:
  ✓ Distinguishes correlation from causation
  ✓ Accounts for confounding
  ✓ Enables counterfactual reasoning
  ✓ Validates causal assumptions
  ✓ Handles treatment heterogeneity

For more information:
  GitHub: https://github.com/Bodhi8/pycausalsim
  Docs:   https://pycausalsim.readthedocs.io
""")
