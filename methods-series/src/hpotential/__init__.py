"""Reproducible code for the H-potential health-capability methods series.

The package holds the machinery behind six manuscripts: a second-order
measurement model and its factor-score determinacy, a two-stage prognostic
pipeline, a modified Delphi consensus rule, a continuous-time state model, a
costed access-versus-efficacy rule, and an early health technology assessment
with value of information.

Every number this package produces is computed from synthetic data or from
closed-form algebra. No participant has been recruited, no Delphi panel has been
run, and nothing here reports an empirical finding about any patient group.
Papers 1 to 3 are protocols.

Layout
------
``hpotential.core``
    The methods, one module per area.
``hpotential.analyses``
    One module per paper, each exposing ``run(cfg=None, outdir=None)``.
``hpotential.config``
    Frozen dataclasses holding the exact published parameters.
``hpotential.utils``
    Figure style, numerical checks, summary statistics.

Determinism
-----------
Every stochastic function takes an explicit ``seed`` or ``rng``. Nothing in the
library touches the global numpy random state, so the same seed gives the same
output, and two analyses running in the same process cannot disturb each other.

Examples
--------
>>> from hpotential import determinacy_ceiling
>>> rho, c = determinacy_ceiling()
>>> round(rho, 6), round(c, 4)
(0.877819, 3.3586)
"""

from __future__ import annotations

__version__ = "1.1.0"

from . import analyses, config, core, utils
from .core.access import AccessModel, costed_statistic
from .core.consensus import DelphiConsensusEngine, DelphiRule, kendalls_w, scale_cvi
from .core.dynamical import FokkerPlanck, LinearSDE, euler_maruyama, rk4
from .core.hta import HTAModel, evpi, evppi
from .core.itembank import DOMAINS, LIFE_STAGES, HPotentialItemBank
from .core.measurement import (
    MeasurementModel,
    determinacy_ceiling,
    required_second_order_signal,
    rho_determinacy,
    second_order_determinacy,
)
from .core.prognostic import cox_fit, events_inflation

__all__ = [
    "__version__",
    "analyses",
    "config",
    "core",
    "utils",
    "AccessModel",
    "DOMAINS",
    "DelphiConsensusEngine",
    "DelphiRule",
    "FokkerPlanck",
    "HPotentialItemBank",
    "HTAModel",
    "LIFE_STAGES",
    "LinearSDE",
    "MeasurementModel",
    "costed_statistic",
    "cox_fit",
    "determinacy_ceiling",
    "euler_maruyama",
    "events_inflation",
    "evpi",
    "evppi",
    "kendalls_w",
    "required_second_order_signal",
    "rho_determinacy",
    "rk4",
    "scale_cvi",
    "second_order_determinacy",
]
