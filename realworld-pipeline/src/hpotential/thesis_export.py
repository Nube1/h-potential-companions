"""Fill the thesis results chapters from a finished pipeline run.

    hpotential thesis-export --run outputs/nhats --thesis ../thesis

writes ``thesis/generated/``:

* ``values.tex``: table rows and single values, as macros;
* ``figures/<key>.tex``: the \\includegraphics lines for each reserved figure,
  with the PDFs beside them;
* ``text/<key>.tex``: short results paragraphs that state the numbers and
  nothing more, each ending with a marker where the interpretation goes;
* ``EXPORT.json``: what was filled, what was not, and why.

The thesis reads these through ``hpresults.sty``. Anything not generated keeps
its placeholder, so the chapters compile at every stage. Every number here is
aggregate, and counts below the disclosure threshold are suppressed.

A run made by ``hpotential demo`` carries a SYNTHETIC marker. The export
refuses it unless ``allow_synthetic`` is set, and then stamps every generated
table, figure and paragraph as synthetic.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .data import load_and_validate
from .evaluation import fit_predict_cox, metrics
from .figures.stats import weighted_km
from .measurement import candidate_scores, fit_measurement, transform_measurement

MIN_CELL = 11
PRIMARY = ("mortality", "hospitalisation")
NAMES = {
    "mortality": "all-cause mortality",
    "hospitalisation": "first hospitalisation",
    "combined": "the combined endpoint",
}
# Rows of the comparison table, in the thesis order. None: not built by the pipeline.
COMPARATORS = [
    (r"$H_{\text{potential}}$", "geometric_crf"),
    (r"\quad $H$, fixed floor (sensitivity)", "geometric_crf_floor"),
    (r"\quad Additive score (sensitivity)", "additive"),
    (r"\quad Second-order factor score", "factor_score"),
    ("QALY proxy (EQ-5D crosswalk)", "qaly_proxy"),
    ("Frailty index", "frailty_index"),
    ("Charlson-type index", "charlson_index"),
    ("ADL/IADL-only", "adl_iadl_count"),
    ("Age and comorbidity count", "comorbidity_index"),
    ("Age and sex only", "baseline"),
]
DASH = r"[\,--\,]"
# Comparators tested for incremental value: each model, then the same model with H.
INCREMENTAL = [
    ("Frailty index", "frailty_index"),
    ("Charlson-type index", "charlson_index"),
    ("Comorbidity count", "comorbidity_index"),
    ("ADL/IADL count", "adl_iadl_count"),
    ("QALY proxy (EQ-5D crosswalk)", "qaly_proxy"),
]


# ------------------------------------------------------------------ formatting


def _n(k: int, total: int) -> str:
    if 0 < k < MIN_CELL:
        return f"$<${MIN_CELL}"
    return f"{k:,} ({100 * k / total:.1f})"


def _ci(est: float, lo: float, hi: float, d: int = 2) -> str:
    if not np.isfinite(est):
        return DASH
    return f"{est:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


def _p(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _p_text(p: float) -> str:
    return "$p < 0.001$" if p < 0.001 else f"$p = {p:.3f}$"


def _pct(lo: float, hi: float, x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (np.nan, np.nan) if not len(x) else tuple(np.percentile(x, [lo, hi]))


# ---------------------------------------------------------------- statistics


def harrell_c(time, event, risk) -> float:
    from lifelines.utils import concordance_index

    t, e, r = _finite(time, event, risk)
    return float(concordance_index(t, -r, e))


def _finite(time, event, risk):
    t, e, r = (np.asarray(x, float) for x in (time, event, risk))
    ok = np.isfinite(t) & np.isfinite(e) & np.isfinite(r)
    return t[ok], e[ok].astype(int), r[ok]


def uno_c(time, event, risk, tau: float) -> float:
    """Uno's IPCW concordance truncated at ``tau``.

    Each case i with T_i < tau is compared with everyone still event-free
    after T_i and weighted by 1/G(T_i-)^2, G being Kaplan-Meier for censoring.
    """
    t, e, r = _finite(time, event, risk)
    km = weighted_km(t, 1 - e)  # censoring distribution
    g_t, g_s = km["time"].to_numpy(), km["survival"].to_numpy()
    idx = np.searchsorted(g_t, t, side="left") - 1  # G just before t
    g = np.where(idx >= 0, g_s[np.clip(idx, 0, None)], 1.0)
    cases = np.flatnonzero((e == 1) & (t < tau) & (g > 0))
    num = den = 0.0
    for chunk in np.array_split(cases, max(1, len(cases) // 256)):
        later = t[None, :] > t[chunk, None]
        w = 1 / g[chunk] ** 2
        conc = (r[chunk, None] > r[None, :]) + 0.5 * (r[chunk, None] == r[None, :])
        num += (w * (later * conc).sum(axis=1)).sum()
        den += (w * later.sum(axis=1)).sum()
    return float(num / den) if den else np.nan


def _performance(frame, spec, risk, horizon) -> dict:
    """Harrell's C, Uno's C and calibration for one risk vector."""
    comp = frame[spec["competing"]] if spec.get("competing") else None
    m = metrics(frame[spec["time"]], frame[spec["event"]], risk, horizon, (), comp)
    return {
        "harrell": harrell_c(frame[spec["time"]], frame[spec["event"]], risk),
        "uno": uno_c(frame[spec["time"]], frame[spec["event"]], risk, horizon),
        "slope": m.get("calibration_slope", np.nan),
        "citl": m.get("calibration_intercept", np.nan),
    }


def _cox_hr(frame, spec, cols, weight):
    from lifelines import CoxPHFitter

    use = frame[[spec["time"], spec["event"], *cols, weight]].dropna()
    use = use[use[spec["time"]] > 0]
    cph = CoxPHFitter().fit(
        use, duration_col=spec["time"], event_col=spec["event"], weights_col=weight, robust=True
    )
    s = cph.summary.loc["h_sd"]
    return (
        float(s["exp(coef)"]),
        float(s["exp(coef) lower 95%"]),
        float(s["exp(coef) upper 95%"]),
        float(s["p"]),
    )


def _info_criteria(frame, spec, cols) -> tuple[float, float]:
    """Unweighted partial-likelihood AIC and BIC (BIC uses the number of events)."""
    from lifelines import CoxPHFitter

    use = frame[[spec["time"], spec["event"], *cols]].dropna()
    use = use[use[spec["time"]] > 0]
    cph = CoxPHFitter().fit(use, duration_col=spec["time"], event_col=spec["event"])
    k, ll, d = len(cols), cph.log_likelihood_, max(int(use[spec["event"]].sum()), 1)
    return -2 * ll + 2 * k, -2 * ll + k * np.log(d)


def _loglik(frame, spec, cols) -> float:
    """Unweighted Cox partial log-likelihood (cause-specific where death competes)."""
    from lifelines import CoxPHFitter

    use = frame[[spec["time"], spec["event"], *cols]].dropna()
    use = use[use[spec["time"]] > 0]
    return float(
        CoxPHFitter().fit(use, duration_col=spec["time"], event_col=spec["event"]).log_likelihood_
    )


# ----------------------------------------------------- bootstrap workers

_BOOT: dict = {}


def _invariance_one(args):
    from .cfa import invariance

    X, mask, pattern, w = args
    return invariance([X[~mask], X[mask]], pattern, [w[~mask], w[mask]])


def _pmap(fn, items, jobs):
    """Map over items in worker processes when jobs > 1, in order."""
    items = list(items)
    if jobs > 1 and len(items) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(jobs, len(items))) as ex:
            return list(ex.map(fn, items))
    return [fn(i) for i in items]


def _wlsmv_one(args):
    """One imputed copy through WLSMV; None if it fails to converge."""
    from .wlsmv import fit_wlsmv

    raw, pattern, w, seed, ordinal = args
    try:
        return fit_wlsmv(raw, pattern, w, reps=100, seed=seed, ordinal=ordinal)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _boot_init(base, cfg, endpoints, horizon, weight):
    """Per-process state for the bootstrap refits (set once per worker)."""
    _BOOT.update(base=base, cfg=cfg, endpoints=endpoints, horizon=horizon, weight=weight)


def _boot_scored(model, rows: pd.DataFrame) -> pd.DataFrame:
    cfg = _BOOT["cfg"]
    dom = transform_measurement(rows, model, cfg).reset_index(drop=True)
    h = candidate_scores(dom, cfg.raw["measurement"]["alpha"], [])["geometric_crf"]
    out = rows.reset_index(drop=True).assign(geometric_crf=h.to_numpy())
    out["female"] = (out["sex"].astype(str).str.lower() == "female").astype(float)
    return out


def _boot_perf(train, test, spec):
    risk, _ = fit_predict_cox(
        train,
        test,
        spec["time"],
        spec["event"],
        ["geometric_crf", "age", "female"],
        _BOOT["horizon"],
        _BOOT["weight"],
        spec.get("competing"),
    )
    return _performance(test, spec, risk, _BOOT["horizon"])


def _boot_rep(seed):
    """One resample: refit everything, return optimism per endpoint and measure."""
    base = _BOOT["base"]
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(base), len(base))
    b_rows = base.iloc[idx].reset_index(drop=True)
    try:
        model = fit_measurement(b_rows, _BOOT["cfg"], se=False)
        b, orig = _boot_scored(model, b_rows), _boot_scored(model, base)
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return None
    out = {}
    for endpoint in PRIMARY:
        spec = _BOOT["endpoints"][endpoint]
        try:
            on_boot, on_orig = _boot_perf(b, b, spec), _boot_perf(b, orig, spec)
        except (ValueError, ZeroDivisionError):
            continue
        out[endpoint] = {k: on_boot[k] - on_orig[k] for k in on_boot}
    return out


# ---------------------------------------------------------------- the export


class Export:
    def __init__(
        self, run_dir, thesis_dir, config_path=None, reps=200, allow_synthetic=False, jobs=0
    ):
        from .api import DEFAULT_CONFIG, PACKAGE_ROOT

        self.run = Path(run_dir)
        self.out = Path(thesis_dir) / "generated"
        self.reps = int(reps)
        import os

        self.jobs = int(jobs) if jobs and int(jobs) > 0 else (os.cpu_count() or 1)
        self.synthetic = (self.run / "SYNTHETIC").exists()
        if self.synthetic and not allow_synthetic:
            raise ValueError(
                f"{self.run} holds made-up demo data. Export a real NHATS run, or pass "
                "--allow-synthetic to test the wiring (every item is then stamped synthetic)."
            )
        self.cfg = load_config(config_path or DEFAULT_CONFIG)
        audit = json.loads((self.run / "run_audit.json").read_text())
        data_path = Path(audit["data_path"])
        for candidate in (data_path, PACKAGE_ROOT / data_path, self.run / data_path.name):
            if candidate.exists():
                data_path = candidate
                break
        df, _ = load_and_validate(data_path, self.cfg)
        self.raw = df.reset_index(drop=True)
        idc = self.cfg.raw["data"]["id"]
        scores = pd.read_csv(self.run / "participant_scores_RESTRICTED.csv")
        # Comparators in the scores file carry the pipeline's imputed values.
        dup = [c for c in scores if c != idc and c in self.raw]
        self.data = self.raw.drop(columns=dup).merge(
            scores, on=idc, how="inner", validate="one_to_one"
        )
        preds = pd.read_csv(self.run / "predictions_RESTRICTED.csv")
        self.test = self.data.merge(preds, on=idc, how="inner", validate="one_to_one")
        self.perf = pd.read_csv(self.run / "model_performance.csv")
        self.weight = self.cfg.raw["data"]["weight"]
        self.endpoints = self.cfg.raw["outcomes"]["endpoints"]
        self.horizon = float(self.cfg.raw["outcomes"]["horizons_days"][-1])
        self.rng = np.random.default_rng(self.cfg.seed)
        self.rows: dict[str, str] = {}
        self.values: dict[str, str] = {}
        self.filled: list[str] = []
        self.unfilled: dict[str, str] = {}
        sd = self.data["geometric_crf"].std()
        self.data["h_sd"] = (self.data["geometric_crf"] - self.data["geometric_crf"].mean()) / sd
        self.data["female"] = (self.data["sex"].astype(str).str.lower() == "female").astype(float)

    # -------------------------------------------------------------- helpers

    def risk(self, endpoint: str, model: str) -> np.ndarray:
        return self.test[f"risk__{endpoint}__{model}__{self.horizon:g}"].to_numpy()

    def _stamp_rows(self, ncol: int) -> str:
        if not self.synthetic:
            return ""
        return rf"\midrule\multicolumn{{{ncol}}}{{l}}{{\hpstamp}} \\" + "\n"

    def boot(self, frame: pd.DataFrame, stat) -> np.ndarray:
        n = len(frame)
        out = []
        for _ in range(self.reps):
            idx = self.rng.integers(0, n, n)
            try:
                out.append(stat(frame.iloc[idx].reset_index(drop=True), idx))
            except (ValueError, ZeroDivisionError):
                continue
        return np.array(out, dtype=float)

    # --------------------------------------------------------------- tables

    # ------------------------------------------------------------ Chapter 5

    def measurement_chapter(self) -> None:
        """Fill Chapter 5 from the joint CFA refitted on the development rows.

        The refit uses the same seed and rows as the pipeline, so it reproduces
        the pipeline's measurement model exactly.
        """
        import yaml
        from scipy.stats import chi2 as chi2_dist
        from scipy.stats import t as t_dist

        from .api import DEFAULT_LAYOUT
        from .cfa import rmsea_interval, rubin
        from .measurement import DOMAINS, fit_measurement

        cfg = self.cfg
        idc = cfg.raw["data"]["id"]
        train_ids = set(self.data.loc[self.data["split"] == "train", idc])
        train = self.raw[self.raw[idc].isin(train_ids)].reset_index(drop=True)
        mm = fit_measurement(train, cfg)
        pooled = mm.pooled()
        self.measurement = pooled
        labels = {}
        try:
            layout = yaml.safe_load(Path(DEFAULT_LAYOUT).read_text())
            labels = {k: v.get("label", k) for k, v in layout["indicators"].items()}
        except (OSError, KeyError):
            pass
        titles = {
            "biomedical": "Biomedical Potential",
            "social": "Social Modifiers",
            "systemic": "Systemic Burdens (reverse coded)",
        }

        # WLSMV, the estimator Chapter 5 specifies as the check on ML. Ordinal
        # indicators are classified on the observed data, and each imputed set is
        # put back on its original scale so they can be rounded to categories.
        from .measurement import _indicator_frame
        from .wlsmv import classify

        observed = _indicator_frame(train, cfg).to_numpy(float)
        ordinal = classify(observed)
        lo_obs, hi_obs = np.nanmin(observed, axis=0), np.nanmax(observed, axis=0)
        w_tr = pd.to_numeric(train[self.weight], errors="coerce").fillna(0).to_numpy()
        jobs = []
        for Xi, c, sc in zip(mm.imputed(train, cfg), mm.centre, mm.scale, strict=True):
            raw = Xi * sc + c
            raw[:, ordinal] = np.clip(np.round(raw[:, ordinal]), lo_obs[ordinal], hi_obs[ordinal])
            jobs.append((raw, mm.pattern, w_tr, cfg.seed, ordinal))
        wl = [r for r in _pmap(_wlsmv_one, jobs, self.jobs) if r is not None]
        self.wlsmv = None
        if wl:
            pooled_w = rubin(
                np.array([np.abs(r.loadings) for r in wl]),
                np.array([r.loadings_se**2 for r in wl]),
            )
            self.wlsmv = {
                "fit": {
                    k: float(np.mean([r.fit_indices()[k] for r in wl])) for k in wl[0].fit_indices()
                },
                "loadings": pooled_w["estimate"],
                "n_ordinal": int(ordinal.sum()),
                "imputations": len(wl),
            }

        # Fit
        f = pooled["fit"]
        n = pooled["n"]
        p_chi = float(chi2_dist.sf(f["chi2"], f["df"]))
        lo, hi = rmsea_interval(f["chi2"], int(f["df"]), n)
        g = self.wlsmv["fit"] if self.wlsmv else None
        if g:
            p_w = float(chi2_dist.sf(g["chi2"], g["df"]))
            wl_lo, wl_hi = rmsea_interval(g["chi2"], int(g["df"]), n)
            wcol = {
                "chi2": f"{g['chi2']:.1f} ({g['df']:.0f}), {_p(p_w)}",
                "cfi": f"{g['cfi']:.3f}",
                "tli": f"{g['tli']:.3f}",
                "rmsea": f"{g['rmsea']:.3f} ({wl_lo:.3f} to {wl_hi:.3f})",
                "srmr": f"{g['srmr']:.3f}",
            }
        else:
            wcol = dict.fromkeys(("chi2", "cfi", "tli", "rmsea", "srmr"), DASH)
        self.rows["fit"] = (
            rf"$\chi^2$ (df), $p$ & {f['chi2']:.1f} ({f['df']:.0f}), {_p(p_chi)} & {wcol['chi2']} & -- \\"
            "\n"
            rf"CFI & {f['cfi']:.3f} & {wcol['cfi']} & $\geq 0.95$ \\"
            "\n"
            rf"TLI & {f['tli']:.3f} & {wcol['tli']} & $\geq 0.95$ \\"
            "\n"
            rf"RMSEA (90\% CI) & {f['rmsea']:.3f} ({lo:.3f} to {hi:.3f}) & {wcol['rmsea']} & $\leq 0.06$ \\"
            "\n"
            rf"SRMR & {f['srmr']:.3f} & {wcol['srmr']} & $\leq 0.08$ \\"
            "\n" + self._stamp_rows(4)
        )

        # Loadings
        rows = []
        for d in DOMAINS:
            rows.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{titles[d]}}}}} \\")
            for c in cfg.indicators[d]["indicators"]:
                ld = pooled["loadings"][c]
                z = ld["estimate"] / ld["se"] if ld["se"] > 0 else np.inf
                dof = ld["df"] if np.isfinite(ld["df"]) else 1e6
                pv = float(2 * t_dist.sf(abs(z), dof))
                rows.append(
                    rf"\quad {labels.get(c, c)} & {ld['estimate']:.3f} ({ld['se']:.3f}) & "
                    rf"{_p(pv)} & {ld['estimate'] ** 2:.3f} \\"
                )
            if d != DOMAINS[-1]:
                rows.append(r"\addlinespace")
        self.rows["loadings"] = "\n".join(rows) + "\n" + self._stamp_rows(4)

        # Reliability
        rel = pooled["reliability"]
        self.rows["reliability"] = (
            "\n".join(
                rf"{titles[d]} & {rel[d]['omega']:.3f} & {rel[d]['alpha']:.3f} & {rel[d]['ave']:.3f} \\"
                for d in DOMAINS
            )
            + "\n"
            + self._stamp_rows(4)
        )

        # Discriminant validity: sqrt(AVE) on the diagonal, correlation [HTMT] off it
        R = np.array(pooled["factor_correlations"])
        H = np.array(pooled["htmt"])
        rows = []
        for i, d in enumerate(DOMAINS):
            cells = []
            for j in range(3):
                if j == i:
                    cells.append(rf"\textbf{{{np.sqrt(rel[d]['ave']):.3f}}}")
                elif j < i:
                    cells.append(f"{R[i, j]:.3f} [{H[i, j]:.3f}]")
                else:
                    cells.append("")
            rows.append(rf"{titles[d]} & " + " & ".join(cells) + r" \\")
        self.rows["discriminant"] = "\n".join(rows) + "\n" + self._stamp_rows(4)

        # Second order
        so = pooled["second_order"]
        adm = pooled["second_order_admissible"]
        if adm > 0:
            gam = np.array([so[d]["estimate"] for d in DOMAINS])
            w = gam / gam.sum()
            self.rows["secondorder"] = (
                "\n".join(
                    rf"{titles[d]} & {so[d]['estimate']:.3f} ({so[d]['se']:.3f}) & "
                    rf"{so[d]['estimate']:.3f} & {w[i]:.3f} \\"
                    for i, d in enumerate(DOMAINS)
                )
                + "\n"
                + self._stamp_rows(4)
            )
        else:
            self.unfilled["tab:secondorder"] = "second-order model not admissible in any imputation"

        # Invariance across sex and across age band, averaged over imputations
        Xs = mm.imputed(train, cfg)
        w_all = pd.to_numeric(train[self.weight], errors="coerce").fillna(0).to_numpy()
        female = train["sex"].astype(str).str.lower().eq("female").to_numpy()
        old = pd.to_numeric(train["age"], errors="coerce").ge(80).to_numpy()
        inv = {}
        for gname, mask in (("sex", female), ("age band (80 and over)", old)):
            if min(mask.sum(), (~mask).sum()) < 100:
                continue
            per = _pmap(_invariance_one, [(X, mask, mm.pattern, w_all) for X in Xs], self.jobs)
            inv[gname] = {
                lvl: {
                    k: float(np.mean([r[lvl][k] for r in per]))
                    for k in per[0][lvl]
                    if k in ("cfi", "rmsea", "delta_cfi", "delta_rmsea")
                }
                for lvl in ("configural", "metric", "scalar")
            }
        self.invariance = inv

        # Leave one indicator out (first imputation)
        from .cfa import fit_cfa

        X0 = Xs[0]
        base = mm.fits[0]
        base_std = np.array([e for e, _ in base.standardised_loadings()])
        shifts, dcfi = [], []
        for j in range(X0.shape[1]):
            keep = [i for i in range(X0.shape[1]) if i != j]
            pat = {}
            for d, cols in mm.pattern.items():
                pat[d] = [keep.index(i) for i in cols if i != j]
            if any(len(v) < 2 for v in pat.values()):
                continue
            fj = fit_cfa(X0[:, keep], pat, w_all, se=False)
            std_j = np.array([e for e, _ in fj.standardised_loadings()])
            shifts.append(float(np.max(np.abs(std_j - base_std[keep]))))
            dcfi.append(fj.cfi - base.cfi)
        self.loo = {"max_shift": max(shifts) if shifts else np.nan, "dcfi": dcfi}

        # Sensitivity: people a proxy said could not be tested on cognition are
        # imputed in the primary analysis; here their cognition is set to the
        # lowest possible score instead (thesis appendix on data checks).
        self.floor_sens = None
        flag = "cognition_not_asked"
        cog = [c for d in DOMAINS for c in cfg.indicators[d]["indicators"] if "cognition" in c]
        if flag in train and cog and pd.to_numeric(train[flag], errors="coerce").eq(1).any():
            import copy

            from .config import StudyConfig

            raw2 = copy.deepcopy(cfg.raw)
            raw2["measurement"]["not_asked_floor"] = {"flag": flag, "indicators": cog, "value": 0}
            cfg2 = StudyConfig(raw2, cfg.path)
            mm2 = fit_measurement(train, cfg2, se=False)

            def mean_loadings(m):
                return np.mean([[e for e, _ in f.standardised_loadings()] for f in m.fits], axis=0)

            s1 = mm.raw_scores(train, cfg)["biomedical"]
            s2 = mm2.raw_scores(train, cfg2)["biomedical"]
            self.floor_sens = {
                "n": int(pd.to_numeric(train[flag], errors="coerce").eq(1).sum()),
                "max_shift": float(np.max(np.abs(mean_loadings(mm2) - mean_loadings(mm)))),
                "corr": float(np.corrcoef(s1, s2)[0, 1]),
            }
        self.filled += [
            "tab:fit (ML column)",
            "tab:loadings",
            "tab:reliability",
            "tab:discriminant",
        ] + (["tab:secondorder"] if adm > 0 else [])
        if self.wlsmv:
            self.filled.append("tab:fit (WLSMV column)")
        else:
            self.unfilled["tab:fit WLSMV column"] = "WLSMV did not converge in any imputation."

    def _floor_sentence(self) -> str:
        s = self.floor_sens
        if not s:
            return ""
        who = (
            f"the {s['n']:,} development participants"
            if s["n"] >= 11
            else "the development participants (fewer than 11)"
        )
        return (
            f" Setting cognition to the lowest possible score for {who} whom a proxy said could "
            "not be tested, instead of imputing it, changed the standardised loadings by at most "
            f"{s['max_shift']:.3f}, and the biomedical domain scores under the two rules correlated "
            f"at {s['corr']:.3f}."
        )

    def _wlsmv_sentence(self) -> str:
        if not self.wlsmv:
            return "WLSMV did not converge, so the estimators could not be compared."
        ml = np.array(
            [self.measurement["loadings"][c]["estimate"] for c in self.measurement["loadings"]]
        )
        diff = np.abs(ml - self.wlsmv["loadings"])
        return (
            f"Under WLSMV, with {self.wlsmv['n_ordinal']} of the {len(ml)} indicators treated as "
            "ordinal, the standardised loadings differed from the ML estimates by at most "
            f"{diff.max():.3f} (mean {diff.mean():.3f}), and CFI was "
            f"{self.wlsmv['fit']['cfi']:.3f} (Table~\\ref{{tab:fit}}). The asymptotic covariance "
            "of the polychoric and polyserial correlations was estimated by bootstrap."
        )

    def measurement_text(self, tdir: Path) -> dict:
        pooled, paras = self.measurement, {}
        f = pooled["fit"]
        stamp = "\\hpstamp\\ " if self.synthetic else ""
        paras["estimation"] = (
            f"The correlated three-domain model was fitted by maximum likelihood, with the survey "
            f"weights, to each of {pooled['imputations']} imputed copies of the development split "
            f"(n = {pooled['n']:,}). Missing indicators were imputed by chained equations from the "
            "indicators, age and sex; outcomes were not used. Loadings and second-order loadings "
            "were pooled by Rubin's rules; fit indices are averages over imputations. The second-"
            "order model was admissible in "
            f"{pooled['second_order_admissible']:.0%} of imputations.".replace("%", "\\%")
        )
        paras["fit"] = (
            f"Averaged over imputations, CFI was {f['cfi']:.3f}, TLI {f['tli']:.3f}, RMSEA "
            f"{f['rmsea']:.3f} and SRMR {f['srmr']:.3f}, against the thresholds in "
            "Table~\\ref{tab:fit}."
        )
        lds = pooled["loadings"]
        lo_c = min(lds, key=lambda c: lds[c]["estimate"])
        paras["loadings"] = (
            "Standardised loadings ranged from "
            f"{min(v['estimate'] for v in lds.values()):.3f} to "
            f"{max(v['estimate'] for v in lds.values()):.3f} "
            f"(Table~\\ref{{tab:loadings}}); the lowest was for {lo_c.replace('_', ' ')}."
        )
        rel = pooled["reliability"]
        R = np.array(pooled["factor_correlations"])
        H = np.array(pooled["htmt"])
        iu = np.triu_indices(3, 1)
        paras["reliability"] = (
            "Composite reliability ranged from "
            f"{min(r['omega'] for r in rel.values()):.3f} to {max(r['omega'] for r in rel.values()):.3f} "
            f"and AVE from {min(r['ave'] for r in rel.values()):.3f} to "
            f"{max(r['ave'] for r in rel.values()):.3f}. The largest factor correlation was "
            f"{np.abs(R[iu]).max():.3f} and the largest HTMT ratio {H[iu].max():.3f} "
            "(Tables~\\ref{tab:reliability} and~\\ref{tab:discriminant})."
        )
        if pooled["second_order_admissible"] > 0:
            so = pooled["second_order"]
            paras["secondorder"] = (
                "The second-order loadings were "
                + ", ".join(f"{so[d]['estimate']:.3f} for the {d} domain" for d in so)
                + f". The general factor score had determinacy {pooled['determinacy']['general']:.3f}, "
                "against domain-score determinacies of "
                + ", ".join(f"{pooled['determinacy'][d]:.3f}" for d in so)
                + " (Table~\\ref{tab:secondorder}). With three domains the second-order model fits "
                "exactly as well as the correlated model, so its fit is not separate evidence for "
                "a general factor."
            )
        inv_txt = []
        for g, r in self.invariance.items():
            inv_txt.append(
                f"across {g}, CFI changed by {r['metric']['delta_cfi']:+.3f} from configural to "
                f"metric and {r['scalar']['delta_cfi']:+.3f} from metric to scalar"
            )
        paras["robustness"] = (
            ("Measurement invariance: " + "; ".join(inv_txt) + ". " if inv_txt else "")
            + "Removing each indicator in turn changed the remaining standardised loadings by at "
            f"most {self.loo['max_shift']:.3f}, and CFI by at most "
            f"{max((abs(x) for x in self.loo['dcfi']), default=float('nan')):.3f}. "
            + self._wlsmv_sentence()
            + self._floor_sentence()
        )
        for k, body in paras.items():
            (tdir / f"{k}.tex").write_text(
                f"\\noindent {stamp}{body}\n\\hpinterpret\n", encoding="utf-8"
            )
            self.filled.append(f"text:{k}")

    def sample_table(self) -> None:
        d, n = self.data, len(self.data)
        female = int(d["female"].sum())
        mm = d["comorbidity_index"].dropna()
        q1, q3 = np.percentile(mm, [25, 75])
        rows = [
            rf"Age, years, mean (SD) & {d['age'].mean():.1f} ({d['age'].std():.1f}) \\",
            rf"Female, n (\%) & {_n(female, n)} \\",
            rf"Multimorbidity, count of conditions, median (IQR) & {mm.median():.0f} ({q1:.0f} to {q3:.0f}) \\",
        ]
        for label in [
            "Any ADL limitation",
            "Any IADL limitation",
            "Cognitive impairment",
            "Positive depression screen (PHQ-2)",
            "Low social participation",
            "Dual Medicare/Medicaid eligibility",
            "Less than high-school education",
            "Reported care delayed for cost or transport",
        ]:
            rows.append(rf"{label}, n (\%) & {DASH} \\")
        self.rows["sample"] = "\n".join(rows) + "\n" + self._stamp_rows(2)
        self.values["N"] = f"$N = {n:,}$".replace(",", r"\,")
        self.filled.append("tab:sample (age, sex, multimorbidity)")
        self.unfilled["tab:sample other rows"] = (
            "ADL, IADL, cognition, PHQ-2 screen, participation, dual eligibility, education and "
            "delayed care need item-level variables and cut-points; the analytic extract holds "
            "the eleven indicator scores, not these binary summaries."
        )

    def events_table(self) -> None:
        d, n = self.data, len(self.data)
        deaths = int(d["mortality_event"].sum())
        hosp = int(d["hospital_event"].sum())
        comp = int(d["hospital_competing"].sum())
        comb = int(d["combined_event"].sum())
        # Reverse Kaplan-Meier: potential follow-up, treating death as censoring.
        km = weighted_km(d["mortality_time"], 1 - d["mortality_event"])

        def q(p):
            hit = km.loc[km["survival"] <= p, "time"]
            return hit.iloc[0] / 365.25 if len(hit) else np.nan

        med, lo, hi = q(0.5), q(0.75), q(0.25)
        fu = f"{med:.1f} ({lo:.1f} to {hi:.1f})" if np.isfinite(med) else "not reached"
        rows = [
            rf"All-cause deaths, n (\%) & {_n(deaths, n)} \\",
            rf"First hospitalisations, n (\%) & {_n(hosp, n)} \\",
            rf"\quad Deaths before any stay (competing event), n (\%) & {_n(comp, n)} \\",
            rf"Combined endpoint (sensitivity), n (\%) & {_n(comb, n)} \\",
            rf"Median follow-up, years (IQR), reverse Kaplan--Meier & {fu} \\",
            rf"Total person-years & {d['mortality_time'].sum() / 365.25:,.0f} \\",
            rf"Alive at last contact (censored for death), n (\%) & {_n(n - deaths, n)} \\",
        ]
        self.rows["events"] = "\n".join(rows) + "\n" + self._stamp_rows(2)
        self.ev = {"n": n, "deaths": deaths, "hosp": hosp, "comp": comp, "comb": comb, "fu": fu}
        self.filled.append("tab:events")

    def hr_table(self) -> None:
        rows, self.hr = [], {}
        for endpoint in (*PRIMARY, "combined"):
            spec = self.endpoints[endpoint]
            un = _cox_hr(self.data, spec, ["h_sd"], self.weight)
            adj = _cox_hr(self.data, spec, ["h_sd", "age", "female"], self.weight)
            self.hr[endpoint] = (un, adj)
            if endpoint == "combined":
                continue
            kind = " (cause-specific)" if spec.get("competing") else ""
            rows += [
                rf"\multicolumn{{3}}{{l}}{{\textit{{{NAMES[endpoint].capitalize()}{kind}}}}} \\",
                rf"\quad Unadjusted & {_ci(*un[:3])} & {_p(un[3])} \\",
                rf"\quad Adjusted for age and sex & {_ci(*adj[:3])} & {_p(adj[3])} \\",
            ]
            if endpoint == "mortality":
                rows.append(r"\addlinespace")
        self.rows["hr"] = "\n".join(rows) + "\n" + self._stamp_rows(3)
        self.filled.append("tab:hr")

    def discrimination_table(self) -> None:
        self.disc = {}
        cols = {}
        for endpoint in (*PRIMARY, "combined"):
            spec = self.endpoints[endpoint]
            risk = self.risk(endpoint, "geometric_crf")
            point = _performance(self.test, spec, risk, self.horizon)
            boots = self.boot(
                self.test,
                lambda f, idx, s=spec, r=risk: list(
                    _performance(f, s, r[idx], self.horizon).values()
                ),
            )
            cis = {k: _pct(2.5, 97.5, boots[:, i]) for i, k in enumerate(point)}
            self.disc[endpoint] = (point, cis)
            cols[endpoint] = {k: _ci(point[k], *cis[k]) for k in point}
        labels = [
            (r"Harrell's C \citep{harrell_1996}", "harrell"),
            (rf"Uno's C (IPCW), truncated at {self.horizon / 365.25:.0f} years", "uno"),
            ("Calibration-in-the-large", "citl"),
            ("Calibration slope", "slope"),
        ]
        self.rows["discrimination"] = (
            "\n".join(
                rf"{lab} & {cols['mortality'][k]} & {cols['hospitalisation'][k]} \\"
                for lab, k in labels
            )
            + "\n"
            + self._stamp_rows(3)
        )
        self.filled.append("tab:discrimination")

    def comparison_table(self) -> None:
        train = self.data[self.data["split"] == "train"]
        rows, self.comp = [], {}
        for e_i, endpoint in enumerate(PRIMARY):
            spec = self.endpoints[endpoint]
            rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{NAMES[endpoint].capitalize()}}}}} \\")
            h_risk = self.risk(endpoint, "geometric_crf")
            for label, model in COMPARATORS:
                if model is None or f"risk__{endpoint}__{model}__{self.horizon:g}" not in self.test:
                    rows.append(rf"{label} & {DASH} & {DASH} & {DASH} & {DASH} \\")
                    continue
                r = self.risk(endpoint, model)
                c = harrell_c(self.test[spec["time"]], self.test[spec["event"]], r)
                cols = ["age", "female"] + ([] if model == "baseline" else [model])
                aic, bic = _info_criteria(train, spec, cols)
                if model == "geometric_crf":
                    delta = "--"
                else:
                    t, ev = self.test[spec["time"]].to_numpy(), self.test[spec["event"]].to_numpy()
                    b = self.boot(
                        self.test,
                        lambda f, idx, t=t, ev=ev, r=r, h=h_risk: (
                            harrell_c(t[idx], ev[idx], r[idx]) - harrell_c(t[idx], ev[idx], h[idx])
                        ),
                    )
                    lo, hi = _pct(2.5, 97.5, b)
                    delta = _ci(c - harrell_c(t, ev, h_risk), lo, hi, 3)
                self.comp[(endpoint, model)] = (c, delta, aic, bic)
                rows.append(rf"{label} & {c:.3f} & {delta} & {aic:,.1f} & {bic:,.1f} \\")
            if e_i == 0:
                rows.append(r"\addlinespace")
        self.rows["comparison"] = "\n".join(rows) + "\n" + self._stamp_rows(5)
        built = [lab for lab, m in COMPARATORS if m and (PRIMARY[0], m) in self.comp]
        missing = [lab for lab, m in COMPARATORS if m and (PRIMARY[0], m) not in self.comp]
        self.filled.append(f"tab:comparison ({len(built)} models)")
        if missing:
            self.unfilled["tab:comparison"] = (
                "not in this run (optional comparator skipped at ingest, or model omitted): "
                + "; ".join(missing)
            )

    def incremental_table(self) -> None:
        """Each comparator model against the same model with H added.

        Change in hold-out Harrell's C with a paired bootstrap interval, and a
        likelihood-ratio test and change in AIC from nested Cox models in the
        development split.
        """
        from scipy.stats import chi2

        train = self.data[self.data["split"] == "train"]
        rows, self.incr = [], {}
        for e_i, endpoint in enumerate(PRIMARY):
            spec = self.endpoints[endpoint]
            rows.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{NAMES[endpoint].capitalize()}}}}} \\")
            t = self.test[spec["time"]].to_numpy()
            ev = self.test[spec["event"]].to_numpy()
            for label, c in INCREMENTAL:
                cols = [f"risk__{endpoint}__{m}__{self.horizon:g}" for m in (c, f"{c}_plus_h")]
                if not all(col in self.test for col in cols) or c not in train:
                    rows.append(rf"{label} & {DASH} & {DASH} & {DASH} & {DASH} \\")
                    continue
                r0, r1 = (self.test[col].to_numpy() for col in cols)
                dc = harrell_c(t, ev, r1) - harrell_c(t, ev, r0)
                b = self.boot(
                    self.test,
                    lambda f, idx, r0=r0, r1=r1, t=t, ev=ev: (
                        harrell_c(t[idx], ev[idx], r1[idx]) - harrell_c(t[idx], ev[idx], r0[idx])
                    ),
                )
                lo, hi = _pct(2.5, 97.5, b)
                base_cols = [c, "age", "female"]
                lr = 2 * (
                    _loglik(train, spec, [*base_cols, "geometric_crf"])
                    - _loglik(train, spec, base_cols)
                )
                pval = float(chi2.sf(lr, 1))
                daic = 2 - lr
                self.incr[(endpoint, c)] = (dc, lo, hi, lr, pval, daic)
                rows.append(
                    rf"{label} & {_ci(dc, lo, hi, 3)} & {lr:.1f} & {_p(pval)} & {daic:.1f} \\"
                )
            if e_i == 0:
                rows.append(r"\addlinespace")
        self.rows["increment"] = "\n".join(rows) + "\n" + self._stamp_rows(5)
        self.filled.append("tab:increment")

    def secondary_table(self) -> None:
        un, adj = self.hr["combined"]
        point, cis = self.disc["combined"]
        rows = [
            rf"Hazard ratio per SD (adjusted for age and sex) & {_ci(*adj[:3])} \\",
            rf"Harrell's C & {_ci(point['harrell'], *cis['harrell'])} \\",
            rf"Uno's C (IPCW) & {_ci(point['uno'], *cis['uno'])} \\",
            rf"Calibration slope & {_ci(point['slope'], *cis['slope'])} \\",
            rf"Calibration-in-the-large & {_ci(point['citl'], *cis['citl'])} \\",
        ]
        self.rows["secondary"] = "\n".join(rows) + "\n" + self._stamp_rows(2)
        self.filled.append("tab:secondary")

    def internal_table(self) -> None:
        """Harrell's bootstrap optimism, refitting the whole pipeline in each resample.

        In each resample the imputation models, the joint CFA, the domain bounds
        and the Cox models are refitted on the bootstrap rows, then evaluated on
        those rows and on the original rows; the mean difference is the optimism.
        Resamples run in parallel across ``jobs`` processes. Each draws from its
        own pre-generated seed, so the result does not depend on the number of
        processes.
        """
        state = (self.raw, self.cfg, self.endpoints, self.horizon, self.weight)
        _boot_init(*state)
        full_model = fit_measurement(self.raw, self.cfg, se=False)
        full = _boot_scored(full_model, self.raw)
        apparent = {e: _boot_perf(full, full, self.endpoints[e]) for e in PRIMARY}
        seeds = self.rng.integers(0, 2**31 - 1, self.reps)
        if self.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(self.jobs, initializer=_boot_init, initargs=state) as ex:
                results = list(
                    ex.map(_boot_rep, seeds, chunksize=max(1, self.reps // (4 * self.jobs)))
                )
        else:
            results = [_boot_rep(sd) for sd in seeds]
        optimism = {e: {k: [] for k in apparent[e]} for e in PRIMARY}
        for res in results:
            for endpoint, diffs in (res or {}).items():
                for k, v in diffs.items():
                    optimism[endpoint][k].append(v)
        rows, self.internal = [], {}
        for e_i, endpoint in enumerate(PRIMARY):
            app, opt = apparent[endpoint], optimism[endpoint]
            corrected = {k: app[k] - np.nanmean(opt[k]) for k in app}
            self.internal[endpoint] = (app, corrected, len(opt["harrell"]))
            rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{NAMES[endpoint].capitalize()}}}}} \\")
            for lab, k in [
                ("Harrell's C", "harrell"),
                ("Uno's C (IPCW)", "uno"),
                ("Calibration slope", "slope"),
                ("Calibration-in-the-large", "citl"),
            ]:
                rows.append(rf"\quad {lab} & {app[k]:.3f} & {corrected[k]:.3f} \\")
            if e_i == 0:
                rows.append(r"\addlinespace")
        self.rows["internal"] = "\n".join(rows) + "\n" + self._stamp_rows(3)
        self.filled.append(f"tab:internal ({self.reps} bootstrap refits of the whole pipeline)")

    # -------------------------------------------------------------- figures

    def figures(self) -> None:
        src = self.run / "figures"
        plan = {
            "score_dist": ["bio_03_domain_distributions"],
            "km_overall": ["bio_04_km_mortality", "bio_05_cuminc_hospitalisation"],
            "schoenfeld": ["bio_10_schoenfeld"],
            "calibration_plot": ["bio_06_calibration"],
            "tdauc": ["bio_11_time_auc"],
            "dca": ["bio_07_decision_curves"],
            "calibration_secondary": ["bio_06b_calibration_secondary"],
        }
        fig_dir = self.out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for key, names in plan.items():
            missing = [n for n in names if not (src / f"{n}.pdf").exists()]
            if missing:
                self.unfilled[f"fig:{key}"] = f"figure files not in the run: {', '.join(missing)}"
                continue
            width = 0.49 if len(names) > 1 else 0.95
            lines = []
            for n in names:
                shutil.copy2(src / f"{n}.pdf", fig_dir / f"{n}.pdf")
                lines.append(
                    rf"\includegraphics[width={width}\linewidth]{{\hpgendir/figures/{n}.pdf}}%"
                )
            body = "\n\\hfill\n".join(lines) if len(lines) > 1 else lines[0]
            if self.synthetic:
                body += "\n\\par\\smallskip\\hpstamp"
            (fig_dir / f"{key}.tex").write_text(body + "\n", encoding="utf-8")
            self.filled.append(f"fig:{key}")

    # ----------------------------------------------------------------- text

    def text(self) -> None:
        tdir = self.out / "text"
        tdir.mkdir(parents=True, exist_ok=True)
        paras = {}
        d = self.data
        h = d["geometric_crf"]
        near_zero = int((h < 0.05).sum())
        skew = float(((h - h.mean()) ** 3).mean() / h.std() ** 3)
        paras["score_dist"] = (
            f"Across the {len(d):,} people in the analytic sample, $H_{{\\text{{potential}}}}$ "
            f"had mean {h.mean():.3f} (SD {h.std():.3f}) and median {h.median():.3f} "
            f"(IQR {h.quantile(0.25):.3f} to {h.quantile(0.75):.3f}), with skewness "
            f"{skew:.2f}. {_n(near_zero, len(d)) if near_zero else 'No one'} "
            "had a score below 0.05, the region where the geometric form is most sensitive "
            "to the lowest domain (Figure~\\ref{fig:score_dist})."
        )
        sents = []
        for endpoint in PRIMARY:
            un, adj = self.hr[endpoint]
            kind = (
                ", cause-specific with death as a competing event,"
                if endpoint != "mortality"
                else ""
            )
            sents.append(
                f"For {NAMES[endpoint]}{kind} the hazard ratio per standard deviation of "
                f"$H_{{\\text{{potential}}}}$ was {_ci(*un[:3])} unadjusted ({_p_text(un[3])}) "
                f"and {_ci(*adj[:3])} after adjustment for age and sex."
            )
        paras["assoc"] = (
            " ".join(sents)
            + " Hazard ratios use the survey weights with robust standard errors (Table~\\ref{tab:hr})."
        )
        ph = self._ph_numbers()
        if ph:
            paras["ph"] = (
                "The rank-time test of proportional hazards for $H_{\\text{potential}}$ gave "
                + " and ".join(f"$p = {v['ph_test_p']:.2f}$ for {NAMES[k]}" for k, v in ph.items())
                + " (Figure~\\ref{fig:schoenfeld})."
            )
        disc = []
        for endpoint in PRIMARY:
            pt, ci = self.disc[endpoint]
            disc.append(
                f"for {NAMES[endpoint]}, Harrell's C was {_ci(pt['harrell'], *ci['harrell'])} "
                f"and Uno's C {_ci(pt['uno'], *ci['uno'])}"
            )
        paras["discrimination"] = (
            f"In the hold-out split (n = {len(self.test):,}), "
            + "; ".join(disc)
            + f". Intervals are percentile intervals from {self.reps} bootstrap resamples of the "
            "hold-out rows (Table~\\ref{tab:discrimination}). For hospitalisation, deaths before a "
            "first stay are censored in the concordance statistics, which therefore describe the "
            "cause-specific ranking."
        )
        cal = []
        for endpoint in PRIMARY:
            pt, ci = self.disc[endpoint]
            cal.append(
                f"for {NAMES[endpoint]} the calibration slope was {_ci(pt['slope'], *ci['slope'])} "
                f"and calibration-in-the-large {_ci(pt['citl'], *ci['citl'])}"
            )
        paras["calibration"] = (
            "Recalibrating the predicted five-year risk on the logit scale among people whose "
            "status at five years is known, "
            + "; ".join(cal)
            + ". A slope of one and an intercept of zero mean perfect calibration "
            "(Figure~\\ref{fig:calibration_plot})."
        )
        comp = []
        for endpoint in PRIMARY:
            parts = [
                f"{lab.replace(chr(92) + 'quad ', '')} {v[0]:.3f}"
                for lab, m in COMPARATORS
                if (v := self.comp.get((endpoint, m)))
            ]
            comp.append(
                f"For {NAMES[endpoint]}, hold-out Harrell's C was " + "; ".join(parts) + "."
            )
        paras["comparison"] = (
            " ".join(comp)
            + " Every model also contains age and sex. $\\Delta$C and its interval in "
            "Table~\\ref{tab:comparison} compare each model with $H_{\\text{potential}}$ on the "
            "same resampled rows; AIC and BIC are from unweighted fits in the development split."
        )
        nb = []
        for endpoint in PRIMARY:
            sub = self.perf[(self.perf["endpoint"] == endpoint)].set_index("model")
            cols = [c for c in sub.columns if c.startswith("net_benefit_")]
            if "geometric_crf" not in sub.index or not cols:
                continue
            bits = []
            for c in cols:
                t = float(c.split("_")[-1])
                bits.append(
                    f"{sub.loc['geometric_crf', c]:.3f} at a threshold of {t:.0%} "
                    f"(age and sex only {sub.loc['baseline', c]:.3f})".replace("%", "\\%")
                )
            nb.append(
                f"For {NAMES[endpoint]} the net benefit of the model with $H$ was "
                + "; ".join(bits)
                + "."
            )
        if nb:
            paras["net_benefit"] = " ".join(nb) + " (Figure~\\ref{fig:dca})."
        un, adj = self.hr["combined"]
        pt, ci = self.disc["combined"]
        paras["combined"] = (
            f"For the combined endpoint ({self.ev['comb']:,} events) the adjusted hazard ratio per "
            f"standard deviation was {_ci(*adj[:3])}, hold-out Harrell's C "
            f"{_ci(pt['harrell'], *ci['harrell'])} and calibration slope "
            f"{_ci(pt['slope'], *ci['slope'])} (Table~\\ref{{tab:secondary}})."
        )
        if hasattr(self, "internal"):
            bits = []
            for endpoint in PRIMARY:
                app, cor, n_ok = self.internal[endpoint]
                bits.append(
                    f"for {NAMES[endpoint]}, apparent Harrell's C {app['harrell']:.3f} against "
                    f"{cor['harrell']:.3f} after correction, and calibration slope "
                    f"{app['slope']:.3f} against {cor['slope']:.3f}"
                )
            paras["optimism"] = (
                f"Refitting the multiple imputation, the joint factor model, the domain scaling "
                f"and the Cox models in each of {n_ok} bootstrap samples gave "
                + "; ".join(bits)
                + " (Table~\\ref{tab:internal})."
            )
        if getattr(self, "measurement", None):
            self.measurement_text(tdir)
        for key, body in paras.items():
            stamp = "\\hpstamp\\ " if self.synthetic else ""
            (tdir / f"{key}.tex").write_text(
                f"\\noindent {stamp}{body}\n\\hpinterpret\n", encoding="utf-8"
            )
            self.filled.append(f"text:{key}")
        if getattr(self, "incr", None):
            n_sig = sum(1 for v in self.incr.values() if v[4] < 0.05)
            dcs = [v[0] for v in self.incr.values()]
            body = (
                "Table~\\ref{tab:increment} adds $H_{\\text{potential}}$ to each comparator model "
                "(with age and sex). Across endpoints and comparators the change in hold-out "
                f"Harrell's C ranged from {min(dcs):.3f} to {max(dcs):.3f}, and the "
                f"likelihood-ratio test had $p<0.05$ in {n_sig} of {len(self.incr)} comparisons. "
                "The change in C uses paired bootstrap resamples of the hold-out rows; the "
                "likelihood-ratio test and the change in AIC come from nested Cox models in the "
                "development split, so a negative change in AIC favours adding $H$."
            )
            table = (
                "\\begin{table}[htbp]\n\\centering\n"
                "\\caption{Incremental value of $H_{\\text{potential}}$ added to each comparator model.}\n"
                "\\label{tab:increment}\n\\footnotesize\n\\renewcommand{\\arraystretch}{1.2}\n"
                "\\begin{tabular}{p{4.4cm} p{3.6cm} p{1.9cm} p{1.4cm} p{1.4cm}}\n\\toprule\n"
                "\\textbf{Comparator} & \\textbf{$\\Delta$C (95\\% CI)} & "
                "\\textbf{LR $\\chi^2$ (1 df)} & \\textbf{$p$} & \\textbf{$\\Delta$AIC} \\\\\n"
                "\\midrule\n"
                + self.rows["increment"]
                + "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
            )
            stamp = "\\hpstamp\\ " if self.synthetic else ""
            (tdir / "increment.tex").write_text(
                f"\\noindent {stamp}{body}\n\n{table}\\hpinterpret\n", encoding="utf-8"
            )
            self.filled.append("text:increment")
        else:
            self.unfilled["text:increment"] = "no comparator-plus-H models in this run."

    def _ph_numbers(self) -> dict:
        index = self.run / "figures" / "figure_index.json"
        if not index.exists():
            return {}
        for e in json.loads(index.read_text()):
            if e["name"] == "bio_10_schoenfeld":
                return e["numbers"]
        return {}

    # ---------------------------------------------------------------- write

    def write(self, internal: bool = True) -> dict:
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        self.sample_table()
        self.measurement_chapter()
        self.events_table()
        self.hr_table()
        self.discrimination_table()
        self.comparison_table()
        self.incremental_table()
        self.secondary_table()
        if internal:
            self.internal_table()
        else:
            self.unfilled["tab:internal"] = "skipped (--no-internal)"
        self.figures()
        self.text()
        lines = [
            "% Written by `hpotential thesis-export`. Do not edit; rerun the export.",
            f"% Run: {self.run.resolve().name}",
        ]
        if self.synthetic:
            lines.append(r"\hpsynthetictrue")
        for key, value in self.values.items():
            lines.append(rf"\hpsetval{{{key}}}{{{value}}}")
        for key, body in self.rows.items():
            lines.append(rf"\hpsetrows{{{key}}}{{%" + "\n" + body + "}")
        (self.out / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = {
            "run": str(self.run.resolve()),
            "synthetic": self.synthetic,
            "bootstrap_replicates": self.reps,
            "filled": self.filled,
            "not_filled": self.unfilled,
            "differs_from_protocol": [
                "NHATS asks about any overnight hospital stay since the last interview; it does "
                "not separate planned from unplanned stays.",
                "A death in the interval straight after the last stay-free interview counts as "
                "a competing event; a death after missed interviews is censored at the last "
                "answer.",
            ],
        }
        (self.out / "EXPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def export(
    run_dir, thesis_dir, config_path=None, reps=200, allow_synthetic=False, internal=True, jobs=0
):
    """Write thesis/generated from a run. ``jobs`` = 0 uses every processor core."""
    return Export(run_dir, thesis_dir, config_path, reps, allow_synthetic, jobs).write(internal)


__all__ = ["export", "harrell_c", "uno_c"]
