#   Copyright 2025 - 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
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
  rolling_backtest         the same from many forecast origins, pooled
  empirical_path           resample whole episodes from history, optionally weighted
  marketing_worlds         many simulated brands with known media returns, for
                           grading marketing mix models
  media_plan_scm           play a media plan forward from posterior response curves
  compare_plans            quarter totals, gains and extrapolation flags per plan
  lift_test                a simulated geo holdout on any world: measured return, standard error, truth
  worlds_from_graph        worlds whose structure is a declared knowledge graph
  grade, leaderboard       score any marketing mix model, from any library, on the same worlds

The simulation and validation machinery of the core library still applies.
The discovery layer does not.
"""

from .temporal import TemporalSCM, RunResult, Context
from .mechanism import MechanismPrior, SupportReport, OutOfSupport
from .structural import StructuralIntervention, StructuralRule
from .scenario import Scenario, Intervention, ramp, step
from .results import (Comparison, rank_drivers, DriverEffect,
                      FalsifiableClaim, ClaimSet)
from .backtest import backtest, rolling_backtest, BacktestResult, AnchorScore
from .marketing import (marketing_worlds, MarketingWorlds, media_plan_scm, compare_plans,
                        lift_test, worlds_from_graph)
from .grading import grade, leaderboard, papilon_estimator, ols_estimator

SI = StructuralIntervention

__all__ = [
    "TemporalSCM", "RunResult", "Context",
    "MechanismPrior", "SupportReport", "OutOfSupport",
    "StructuralIntervention", "StructuralRule", "SI",
    "Scenario", "Intervention", "ramp", "step",
    "Comparison", "rank_drivers", "DriverEffect",
    "FalsifiableClaim", "ClaimSet",
    "backtest", "rolling_backtest", "BacktestResult", "AnchorScore",
    "marketing_worlds", "MarketingWorlds", "media_plan_scm", "compare_plans",
    "lift_test", "worlds_from_graph", "grade", "leaderboard", "papilon_estimator", "ols_estimator",
]

__version__ = "0.5.0"


def compare(scm, scenarios, target, draws=None, params=None, baseline=None):
    """Run several scenarios on the same parameter draws and compare them."""
    p = params if params is not None else scm.sample_params(draws or scm.draws)
    runs = [scm.run(s, params=p) for s in scenarios]
    return Comparison(scm, runs, target,
                      baseline=baseline or scenarios[0].name)
