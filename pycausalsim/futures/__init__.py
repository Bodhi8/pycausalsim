"""
pycausalsim.futures

PyCausalSim is retrodictive by design: it learns a graph from existing data and
estimates interventions within that data's support. Forecasting inverts both
halves. The graph is specified rather than discovered, and every intervention
of interest lies far outside observed support.

This module supplies what that inversion needs:

  TemporalSCM              a graph unrolled over a horizon, with lags and state
  MechanismPrior           declared extrapolation, with out-of-support flagging
  StructuralIntervention   interventions that add, remove and rewire edges
  Scenario                 a declarative bundle of timed interventions
  rank_drivers             single-lever do() at matched intensity
  ClaimSet                 dated, checkable claims with a Brier score
  backtest                 retrodiction against observed anchors

The simulation and validation machinery of the core library still applies.
The discovery layer does not.
"""

from .temporal import TemporalSCM, RunResult, Context
from .mechanism import MechanismPrior, SupportReport, OutOfSupport
from .structural import StructuralIntervention, StructuralRule
from .scenario import Scenario, Intervention, ramp, step
from .results import (Comparison, rank_drivers, DriverEffect,
                      FalsifiableClaim, ClaimSet)
from .backtest import backtest, BacktestResult, AnchorScore

SI = StructuralIntervention

__all__ = [
    "TemporalSCM", "RunResult", "Context",
    "MechanismPrior", "SupportReport", "OutOfSupport",
    "StructuralIntervention", "StructuralRule", "SI",
    "Scenario", "Intervention", "ramp", "step",
    "Comparison", "rank_drivers", "DriverEffect",
    "FalsifiableClaim", "ClaimSet",
    "backtest", "BacktestResult", "AnchorScore",
]

__version__ = "0.2.0.dev0"


def compare(scm, scenarios, target, draws=None, params=None, baseline=None):
    """Run several scenarios on the same parameter draws and compare them."""
    p = params if params is not None else scm.sample_params(draws or scm.draws)
    runs = [scm.run(s, params=p) for s in scenarios]
    return Comparison(scm, runs, target,
                      baseline=baseline or scenarios[0].name)
