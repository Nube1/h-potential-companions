"""Competing-risk pieces: the model CIF stays in [0, 1] and agrees with Aalen-Johansen."""

import numpy as np
import pandas as pd

from hpotential.evaluation import fit_predict_cox, horizon_status
from hpotential.figures.stats import cif_at, weighted_cif


def _frame(n=2000, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t1 = rng.exponential(900 / np.exp(0.5 * x))
    t2 = rng.exponential(1500, n)
    cens = rng.uniform(300, 2500, n)
    t = np.minimum.reduce([t1, t2, cens])
    return pd.DataFrame({"x": x, "t": t, "e": (t == t1).astype(int), "c": (t == t2).astype(int)})


def test_cif_bounded_and_matches_aalen_johansen():
    df = _frame()
    risk, _ = fit_predict_cox(df, df, "t", "e", ["x"], 1825.0, competing_col="c")
    assert np.all((risk >= 0) & (risk <= 1))
    aj = cif_at(weighted_cif(df["t"], df["e"], df["c"]), 1825.0)
    assert abs(risk.mean() - aj) < 0.03
    naive, _ = fit_predict_cox(df, df, "t", "e", ["x"], 1825.0)
    assert naive.mean() > risk.mean()  # censoring at death overstates risk


def test_competing_event_counts_as_known_non_case():
    known, y = horizon_status(pd.Series([100.0]), pd.Series([0]), 365.0, pd.Series([1]))
    assert known[0] and y[0] == 0
