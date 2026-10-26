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
    ("QALY proxy", None),
    ("Frailty index", "frailty_index"),
    ("Charlson index", None),
    ("ADL/IADL-only", None),
    ("Age and comorbidity count", "comorbidity_index"),
    ("Age and sex only", "baseline"),
]
DASH = r"[\,--\,]"


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


# ---------------------------------------------------------------- the export


class Export:
    def __init__(self, run_dir, thesis_dir, config_path=None, reps=200, allow_synthetic=False):
        from .api import DEFAULT_CONFIG, PACKAGE_ROOT

        self.run = Path(run_dir)
        self.out = Path(thesis_dir) / "generated"
        self.reps = int(reps)
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
        self.data = self.raw.merge(scores, on=idc, how="inner", validate="one_to_one")
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
            "delayed care are not in the analytic extract yet; add them to nhats_layout.yaml."
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
        self.filled.append("tab:comparison (rows the pipeline builds)")
        self.unfilled["tab:comparison QALY proxy, Charlson index, ADL/IADL-only"] = (
            "these comparators are not built from the NHATS extract yet."
        )

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

        The measurement model (domain bounds) and the Cox models are refitted
        on each bootstrap sample, evaluated there and on the original rows; the
        mean difference is the optimism.
        """
        cfg, alpha = self.cfg, self.cfg.raw["measurement"]["alpha"]
        base = self.raw

        def scored(fit_rows: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
            model = fit_measurement(fit_rows, cfg)
            dom = transform_measurement(rows, model, cfg).reset_index(drop=True)
            h = candidate_scores(dom, alpha, [])["geometric_crf"]
            out = rows.reset_index(drop=True).assign(geometric_crf=h.to_numpy())
            out["female"] = (out["sex"].astype(str).str.lower() == "female").astype(float)
            return out

        def perf(train, test, spec):
            risk, _ = fit_predict_cox(
                train,
                test,
                spec["time"],
                spec["event"],
                ["geometric_crf", "age", "female"],
                self.horizon,
                self.weight,
                spec.get("competing"),
            )
            return _performance(test, spec, risk, self.horizon)

        rows, self.internal = [], {}
        full = scored(base, base)
        for e_i, endpoint in enumerate(PRIMARY):
            spec = self.endpoints[endpoint]
            apparent = perf(full, full, spec)
            optimism = {k: [] for k in apparent}
            n = len(base)
            for _ in range(self.reps):
                idx = self.rng.integers(0, n, n)
                b_rows = base.iloc[idx].reset_index(drop=True)
                try:
                    b = scored(b_rows, b_rows)
                    orig = scored(b_rows, base)
                    on_boot, on_orig = perf(b, b, spec), perf(b, orig, spec)
                except (ValueError, ZeroDivisionError):
                    continue
                for k in apparent:
                    optimism[k].append(on_boot[k] - on_orig[k])
            corrected = {k: apparent[k] - np.nanmean(optimism[k]) for k in apparent}
            self.internal[endpoint] = (apparent, corrected, len(optimism["harrell"]))
            rows.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{NAMES[endpoint].capitalize()}}}}} \\")
            for lab, k in [
                ("Harrell's C", "harrell"),
                ("Uno's C (IPCW)", "uno"),
                ("Calibration slope", "slope"),
                ("Calibration-in-the-large", "citl"),
            ]:
                rows.append(rf"\quad {lab} & {apparent[k]:.3f} & {corrected[k]:.3f} \\")
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
                f"Refitting the domain factor models, the median imputation, the domain scaling "
                f"and the Cox models in each of {n_ok} bootstrap samples gave "
                + "; ".join(bits)
                + " (Table~\\ref{tab:internal})."
            )
        for key, body in paras.items():
            stamp = "\\hpstamp\\ " if self.synthetic else ""
            (tdir / f"{key}.tex").write_text(
                f"\\noindent {stamp}{body}\n\\hpinterpret\n", encoding="utf-8"
            )
            self.filled.append(f"text:{key}")
        self.unfilled["text:increment"] = (
            "incremental value needs models that add H to each comparator; not built yet."
        )

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
        self.events_table()
        self.hr_table()
        self.discrimination_table()
        self.comparison_table()
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
                "The protocol scores H from a confirmatory factor model with multiple imputation. "
                "The pipeline fits a one-factor model to each domain separately in the development "
                "split, fills missing indicators with the development-split median, and scales each "
                "domain between its 1st and 99th percentiles. The bootstrap refits all of that.",
                "NHATS asks about any overnight hospital stay since the last interview; it does "
                "not separate planned from unplanned stays.",
                "A death in the interval straight after the last stay-free interview counts as "
                "a competing event; a death after missed interviews is censored at the last "
                "answer.",
            ],
            "never_filled_here": (
                "Chapter 5 measurement-model results (estimation, fit, loadings, reliability, "
                "second-order structure, invariance) come from the CFA in thesis/analysis/, "
                "not from this pipeline."
            ),
        }
        (self.out / "EXPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def export(run_dir, thesis_dir, config_path=None, reps=200, allow_synthetic=False, internal=True):
    return Export(run_dir, thesis_dir, config_path, reps, allow_synthetic).write(internal)


__all__ = ["export", "harrell_c", "uno_c"]
