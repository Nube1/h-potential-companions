"""Read NHATS public-use files and build the canonical analytic extract.

The layout file (configs/nhats_layout.yaml) names every source variable and
code. Nothing here guesses a variable: a name that is not in the files stops
the build, so a codebook mismatch cannot pass silently.

The output has one row per sample person at the baseline round and the
columns of docs/DATA_CONTRACT.md, plus condition flags and household income
for the descriptive figures. Participant-level output stays on the machine
that holds the data; see the repository guard in tools/guard_data.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

READERS = {
    ".dta": lambda p, cols: pd.read_stata(p, columns=cols, convert_categoricals=False),
    ".sas7bdat": lambda p, cols: _subset(pd.read_sas(p, format="sas7bdat"), cols),
    ".csv": lambda p, cols: _subset(pd.read_csv(p, low_memory=False), cols),
    ".parquet": lambda p, cols: pd.read_parquet(p, columns=cols),
}

MIN_OBSERVED_SHARE = 0.5  # multi-item scores need at least half their items


def _subset(df: pd.DataFrame, cols: list[str] | None) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    return df if cols is None else df[[c for c in cols if c in df.columns]]


# --------------------------------------------------------------------- layout


@dataclass
class Layout:
    raw: dict
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> Layout:
        path = Path(path)
        return cls(yaml.safe_load(path.read_text(encoding="utf-8")), path)

    @property
    def r(self) -> int:
        return int(self.raw["release"]["baseline_round"])

    @property
    def follow_ups(self) -> list[int]:
        return [int(k) for k in self.raw["release"]["follow_up_rounds"]]

    @property
    def days_per_round(self) -> float:
        return float(self.raw["release"].get("days_per_round", 365.25))

    @property
    def special_missing(self) -> list[float]:
        return [float(x) for x in self.raw.get("special_missing", [])]

    def name(self, template: str, k: int | None = None) -> str:
        return template.format(r=self.r, k=k if k is not None else self.r).lower()

    # Variables needed from each file, by file key.
    def baseline_variables(self) -> list[str]:
        names = {self.name(self.raw["ids"]["person"])}
        names |= {self.name(v) for v in self.raw["design"].values()}
        for spec in self.raw["covariates"].values():
            names.add(self.name(spec["source"]))
        for block in ("indicators", "comparators"):
            for spec in self.raw[block].values():
                if not spec.get("optional"):
                    names |= {self.name(s) for s in _sources(spec)}
        names |= {self.name(s) for s in self.raw.get("conditions", {}).values()}
        sample = self.raw.get("sample")
        if sample:
            names.add(self.name(sample["residence"]))
        return sorted(names)

    def preferred_variables(self) -> list[str]:
        """Variables used in place of a required set when the release has them.

        NHATS publishes 20 income imputations in a separate file; the five in
        the SP file are used when that file is absent.
        """
        names: set[str] = set()
        for spec in self.raw["indicators"].values():
            names |= {self.name(s) for s in spec.get("imputations", [])}
        return sorted(names)

    def fallback_sources(self) -> dict[int, list[str]]:
        """Templates filled from an earlier round's SP file where baseline is missing.

        NHATS asks some items (education, for one) only at a person's first
        interview, so continuing sample members have them in the Round 1 file.
        """
        out: dict[int, set[str]] = {}
        for block in ("covariates", "indicators"):
            for spec in self.raw[block].values():
                if "fallback_round" in spec:
                    out.setdefault(int(spec["fallback_round"]), set()).update(_sources(spec))
        return {k: sorted(v) for k, v in out.items()}

    def optional_variables(self) -> dict[str, list[str]]:
        """Source variables of optional comparators, by comparator.

        An optional comparator whose variables are absent from the release is
        skipped and reported; it never blocks the build.
        """
        return {
            name: sorted({self.name(s) for s in _sources(spec)})
            for name, spec in self.raw["comparators"].items()
            if spec.get("optional")
        }

    def tracker_variables(self, latest: int) -> list[str]:
        status = self.raw["outcomes"]["mortality"]["status"]
        ks = [k for k in self.follow_ups if k <= latest]
        return [self.name(self.raw["ids"]["person"])] + [self.name(status, k) for k in ks]

    def follow_up_sp_variables(self, k: int) -> list[str]:
        item = self.raw["outcomes"]["hospitalisation"]["sp_item"]
        return [self.name(self.raw["ids"]["person"]), self.name(item, k)]


def _sources(spec: dict) -> list[str]:
    if spec.get("kind") == "neg_log_income":
        return list(spec["fallback_imputations"]) + [spec["couple"]["source"]]
    if "source" in spec:
        return [spec["source"]]
    if "sources" in spec:
        return list(spec["sources"])
    if "weights" in spec:  # charlson
        return list(spec["weights"])
    if "dimensions" in spec:  # eq5d_crosswalk
        return [s for d in spec["dimensions"].values() for s in d["sources"]]
    return [s for g in spec.get("groups", []) for s in g["sources"]]


# ---------------------------------------------------------------------- files


def find_file(data_dir: Path, stem: str) -> Path | None:
    """The file named ``stem`` with a readable extension, any case.

    NHATS releases add a version tag to file names (``..._SP_File_V3``), so a
    trailing ``_v<n>`` is accepted; with several versions the highest wins.
    Stata is preferred over SAS, CSV and Parquet when one release ships more
    than one format.
    """
    pattern = re.compile(re.escape(stem.lower()) + r"(?:_v(\d+))?$")
    order = {".dta": 0, ".sas7bdat": 1, ".parquet": 2, ".csv": 3}
    best: tuple[int, int, Path] | None = None
    for path in Path(data_dir).rglob("*"):
        if path.suffix.lower() not in READERS:
            continue
        m = pattern.match(path.stem.lower())
        if m:
            key = (-int(m.group(1) or 0), order[path.suffix.lower()], path)
            if best is None or key < best:
                best = key
    return best[2] if best else None


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    reader = READERS[path.suffix.lower()]
    if path.suffix.lower() in (".dta", ".parquet") and columns is not None:
        # Column selection is case-sensitive in these readers: resolve first.
        available = _columns(path)
        lookup = {c.lower(): c for c in available}
        columns = [lookup[c] for c in columns if c in lookup]
    return reader(path, columns).rename(columns=str.lower)


def _columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        with pd.io.stata.StataReader(path) as reader:
            return list(reader.variable_labels())
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    if suffix == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    return list(pd.read_sas(path, format="sas7bdat", chunksize=1).read(1).columns)


def available_columns(path: Path) -> set[str]:
    return {c.lower() for c in _columns(path)}


# ---------------------------------------------------------------------- check


@dataclass
class CheckReport:
    files: dict[str, str | None] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)
    missing_variables: dict[str, list[str]] = field(default_factory=dict)
    unexpected_codes: dict[str, list[float]] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    optional_missing: dict[str, list[str]] = field(default_factory=dict)
    tracker_status_values: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing_files and not any(self.missing_variables.values())

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "files": self.files,
            "missing_files": self.missing_files,
            "missing_variables": self.missing_variables,
            "unexpected_codes": {k: v[:20] for k, v in self.unexpected_codes.items()},
            "rows": self.rows,
            "optional_comparators_missing_variables": self.optional_missing,
            # Only code 62 (deceased) is confirmed in the NHATS User Guide; review
            # the other codes here against the tracker codebook before analysis.
            "tracker_status_values": self.tracker_status_values,
        }


def _latest_tracker(data_dir: Path, layout: Layout) -> tuple[int, Path] | None:
    for k in sorted(layout.follow_ups, reverse=True):
        path = find_file(data_dir, layout.name(layout.raw["files"]["tracker"], k))
        if path is not None:
            return k, path
    return None


def _baseline_files(data_dir: Path, layout: Layout) -> list[tuple[str, Path]]:
    """Files that may hold baseline variables, in search order.

    NHATS keeps instrument items in the sample-person file, but the analytic
    weights, design variables, derived race and vital status in the tracker
    file, and imputed income in a separate income file. A baseline variable is
    taken from the first of these files that holds it.
    """
    out = []
    sp = find_file(data_dir, layout.name(layout.raw["files"]["sp"]))
    if sp is not None:
        out.append(("sp", sp))
    tr = _latest_tracker(data_dir, layout)
    if tr is not None:
        out.append(("tracker", tr[1]))
    stem = layout.raw["files"].get("income")
    if stem:
        inc = find_file(data_dir, layout.name(stem))
        if inc is not None:
            out.append(("income", inc))
    return out


def read_baseline(data_dir: Path, layout: Layout, columns: list[str]) -> pd.DataFrame:
    """The requested baseline columns, gathered across files and joined on the person id."""
    pid = layout.name(layout.raw["ids"]["person"])
    frame, got = None, {pid}
    for _, path in _baseline_files(data_dir, layout):
        have = available_columns(path)
        take = [c for c in columns if c in have and c not in got]
        if pid not in have or (frame is not None and not take):
            continue
        part = read_table(path, [pid, *take]).set_index(pid)
        got |= set(take)
        frame = part if frame is None else frame.join(part, how="left")
    return frame


def check(data_dir: str | Path, layout: Layout) -> CheckReport:
    """Report absent files, absent variables and codes outside the layout."""
    data_dir = Path(data_dir)
    report = CheckReport()
    sp_stem = layout.name(layout.raw["files"]["sp"])
    sp = find_file(data_dir, sp_stem)
    report.files["sp"] = str(sp) if sp else None
    if sp is None:
        report.missing_files.append(sp_stem)
    else:
        need = layout.baseline_variables()
        have = set()
        for label, path in _baseline_files(data_dir, layout):
            have |= available_columns(path)
            report.files.setdefault(label, str(path))
        report.missing_variables["sp"] = [v for v in need if v not in have]
        for comp, names in layout.optional_variables().items():
            gone = [v for v in names if v not in have]
            if gone:
                report.optional_missing[comp] = gone
        frame = read_baseline(data_dir, layout, [v for v in need if v in have]).reset_index()
        report.rows["sp"] = len(frame)
        report.unexpected_codes.update(_unexpected_codes(frame, layout))
    tracker = _latest_tracker(data_dir, layout)
    if tracker is None:
        report.missing_files.append(
            layout.name(layout.raw["files"]["tracker"], max(layout.follow_ups))
        )
    else:
        k, path = tracker
        report.files["tracker"] = str(path)
        have = available_columns(path)
        report.missing_variables["tracker"] = [
            v for v in layout.tracker_variables(k) if v not in have
        ]
        status_cols = [v for v in layout.tracker_variables(k)[1:] if v in have]
        if status_cols:
            st = read_table(path, status_cols)
            report.tracker_status_values = {
                c: {str(int(v)): int(n) for v, n in st[c].value_counts().sort_index().items()}
                for c in status_cols
            }
    for rnd, templates in layout.fallback_sources().items():
        stem = layout.raw["files"]["sp"].format(r=rnd, k=rnd).lower()
        path = find_file(data_dir, stem)
        report.files[f"sp_round_{rnd}_fallback"] = str(path) if path else None
        if path is None:
            report.missing_files.append(stem)
            continue
        have = available_columns(path)
        need = [t.format(r=rnd, k=rnd).lower() for t in templates]
        report.missing_variables[f"sp_round_{rnd}_fallback"] = [v for v in need if v not in have]
    for k in layout.follow_ups:
        stem = layout.raw["files"]["sp"].format(r=k, k=k).lower()
        path = find_file(data_dir, stem)
        report.files[f"sp_round_{k}"] = str(path) if path else None
        if path is not None:
            have = available_columns(path)
            report.missing_variables[f"sp_round_{k}"] = [
                v for v in layout.follow_up_sp_variables(k) if v not in have
            ]
    return report


def _declared_codes(spec: dict) -> set[float] | None:
    codes = _declared_answers(spec)
    if codes is not None:
        codes |= {float(x) for x in spec.get("missing_codes", [])}
    return codes


def _declared_answers(spec: dict) -> set[float] | None:
    if "codes" in spec:  # the full set of valid answers, where the layout declares it
        return {float(x) for x in spec["codes"]}
    if "map" in spec:
        return {float(x) for x in spec["map"]}
    if "range" in spec:
        lo, hi = spec["range"]
        return {float(x) for x in range(int(lo), int(hi) + 1)}
    if "present" in spec and spec.get("kind") in ("binary", "count_present", "count_absent"):
        return None  # any non-yes answer counts as no
    return None


def _unexpected_codes(frame: pd.DataFrame, layout: Layout) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    specs = (
        list(layout.raw["covariates"].values())
        + list(layout.raw["indicators"].values())
        + list(layout.raw.get("comparators", {}).values())
    )
    special = set(layout.special_missing)
    for spec in specs:
        codes = _declared_codes(spec)
        if codes is None:
            continue
        for src in _sources(spec):
            col = layout.name(src)
            if col not in frame:
                continue
            values = set(pd.to_numeric(frame[col], errors="coerce").dropna().unique())
            odd = sorted(float(v) for v in values - codes - special)
            if odd:
                out[col] = odd
    return out


# ---------------------------------------------------------------------- build


def _clean(series: pd.Series, special: list[float]) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    return x.mask(x.isin(special))


def _items(
    frame: pd.DataFrame, layout: Layout, sources: list[str], spec: dict | None = None
) -> pd.DataFrame:
    """Source items as floats, special codes missing.

    ``recode`` maps a code to a value before cleaning (NHATS -1 "inapplicable"
    is sometimes a real answer: no floors to climb in a one-floor home);
    ``recode_by_source`` does the same for one item. ``missing_codes`` are
    valid NHATS codes that carry no answer here (don't know, not done).
    """
    spec = spec or {}
    out = {}
    for s in sources:
        c = layout.name(s)
        x = pd.to_numeric(frame[c], errors="coerce").astype(float)
        rec = {**spec.get("recode", {}), **spec.get("recode_by_source", {}).get(s, {})}
        if rec:
            x = x.replace({float(k): float(v) for k, v in rec.items()})
        x = x.mask(x.isin(layout.special_missing))
        miss = spec.get("missing_codes")
        if miss:
            x = x.mask(x.isin([float(v) for v in miss]))
        out[c] = x
    return pd.DataFrame(out)


def _prorated(values: pd.DataFrame, how: str) -> pd.Series:
    n_items = values.shape[1]
    observed = values.notna().sum(axis=1)
    enough = observed >= np.ceil(MIN_OBSERVED_SHARE * n_items)
    if how == "sum":
        result = values.sum(axis=1) * n_items / observed.replace(0, np.nan)
    else:
        result = values.mean(axis=1)
    return result.where(enough)


def score_item(frame: pd.DataFrame, layout: Layout, spec: dict) -> pd.Series:
    """One canonical indicator from its NHATS source items."""
    kind = spec.get("kind", "value")
    if kind == "neg_log_income":
        return neg_log_income(frame, layout, spec)
    items = _items(frame, layout, _sources(spec), spec)
    if kind == "value":
        return items.iloc[:, 0]
    if kind == "reverse":
        lo, hi = spec["range"]
        return lo + hi - items.iloc[:, 0]
    if kind in ("reverse_sum", "reverse_mean"):
        lo, hi = spec["range"]
        flipped = lo + hi - items
        return _prorated(flipped, "sum" if kind == "reverse_sum" else "mean")
    if kind == "sum":
        return _prorated(items, "sum")
    if kind in ("count_present", "count_absent", "binary"):
        if "present_by_source" in spec:  # item-specific codes for "present"
            yes = (
                pd.DataFrame(
                    {
                        layout.name(s): items[layout.name(s)].isin([float(v) for v in codes])
                        for s, codes in spec["present_by_source"].items()
                    }
                )
                .astype(float)
                .where(items.notna())
            )
        else:
            yes = items.isin([float(v) for v in spec["present"]]).astype(float).where(items.notna())
        if kind == "binary":
            return yes.iloc[:, 0]
        present = _prorated(yes, "sum")
        return present if kind == "count_present" else items.shape[1] - present
    if kind == "neg_log":
        x = items.iloc[:, 0]
        return -np.log1p(x.where(x >= 0))
    if kind == "deficit_fraction":
        flags = []
        for group in spec["groups"]:
            g = _items(frame, layout, group["sources"], group)
            flags.append(
                g.isin([float(v) for v in group["deficit"]]).astype(float).where(g.notna())
            )
        all_flags = pd.concat(flags, axis=1)
        return _prorated(all_flags, "mean")
    if kind == "charlson":
        # Weighted sum of conditions present (Charlson et al. 1987), prorated
        # over the weight observed when some items are missing.
        w = pd.Series({layout.name(s): float(v) for s, v in spec["weights"].items()})
        yes = items.isin([float(v) for v in spec["present"]]).astype(float).where(items.notna())
        seen = (yes.notna() * w).sum(axis=1)
        score = (yes.fillna(0) * w).sum(axis=1) * w.sum() / seen.replace(0, np.nan)
        return score.where(seen >= MIN_OBSERVED_SHARE * w.sum())
    if kind == "eq5d_crosswalk":
        return eq5d_crosswalk(frame, layout, spec)
    raise ValueError(f"Unknown indicator kind: {kind}")


def neg_log_income(frame: pd.DataFrame, layout: Layout, spec: dict) -> pd.Series:
    """Income shortfall: minus the mean log equivalised income over NHATS imputations.

    NHATS total income is for the person, or for the couple when married or
    living with a partner. Couple income is divided by the square root of two
    (the square-root equivalence scale) so the two are comparable. The 20
    interval-regression imputations are used when the release has them, else
    the five hot-deck imputations in the SP file. Averaging the log over
    imputations gives one value per person; the between-imputation spread is
    not carried forward (see the thesis appendix on data checks).
    """
    pref = [layout.name(s) for s in spec.get("imputations", [])]
    cols = pref if pref and all(c in frame for c in pref) else None
    sources = spec.get("imputations") if cols else spec["fallback_imputations"]
    x = _items(frame, layout, sources)
    x = x.where(x >= 0)
    c = spec["couple"]
    status = _clean(frame[layout.name(c["source"])], layout.special_missing)
    scale = np.where(status.isin([float(v) for v in c["codes"]]), np.sqrt(2.0), 1.0)
    logs = np.log1p(x.div(scale, axis=0))
    return -logs.mean(axis=1)


def eq5d_levels(frame: pd.DataFrame, layout: Layout, spec: dict) -> pd.DataFrame:
    """EQ-5D-3L levels (1 to 3) for each dimension from NHATS items.

    Each dimension counts its items showing a problem (``present`` codes), or
    sums the item values when ``sum: true``, prorated over the items answered
    as for the other indicators, and maps the result to a level by two
    cut-points: below the first is level 1, below the second level 2,
    otherwise level 3. A dimension with too few items answered is missing.
    """
    levels = {}
    for dim, d in spec["dimensions"].items():
        items = _items(frame, layout, d["sources"], d)
        if d.get("sum"):
            value = _prorated(items, "sum")
        else:
            yes = items.isin([float(v) for v in d["present"]]).astype(float).where(items.notna())
            value = _prorated(yes, "sum")
        a, b = d["cuts"]
        lvl = np.where(value >= b, 3.0, np.where(value >= a, 2.0, 1.0))
        levels[dim] = pd.Series(lvl, index=frame.index).where(value.notna())
    return pd.DataFrame(levels)


def eq5d_crosswalk(frame: pd.DataFrame, layout: Layout, spec: dict) -> pd.Series:
    """A health-utility proxy: crosswalked EQ-5D-3L levels valued by an additive tariff.

    utility = 1 - constant (any level above 1) - sum of dimension decrements
              - N3 (any level 3)
    The tariff is declared in the layout and must be checked against its source.
    """
    lv = eq5d_levels(frame, layout, spec)
    tar = spec["tariff"]
    dec = pd.Series(0.0, index=lv.index)
    for dim in lv:
        l2, l3 = tar[dim]
        dec += np.where(lv[dim] == 2, l2, 0.0) + np.where(lv[dim] == 3, l3, 0.0)
    any_problem = (lv > 1).any(axis=1)
    any_three = (lv == 3).any(axis=1)
    util = (
        1 - np.where(any_problem, tar["constant"], 0.0) - dec - np.where(any_three, tar["n3"], 0.0)
    )
    return pd.Series(util, index=lv.index).where(lv.notna().all(axis=1))


def _outcome_times(
    alive_or_event: pd.DataFrame, event_flags: pd.DataFrame, layout: Layout
) -> tuple[pd.Series, pd.Series]:
    """First event round -> midpoint time; else censor at last known round.

    ``alive_or_event`` marks rounds where the person's status is known;
    ``event_flags`` marks rounds where the event is first reported.
    """
    days = layout.days_per_round
    rounds = np.array([int(c) for c in event_flags.columns])
    gap = rounds - layout.r
    ev = event_flags.to_numpy(dtype=bool)
    known = alive_or_event.to_numpy(dtype=bool)
    has_event = ev.any(axis=1)
    first = np.where(has_event, ev.argmax(axis=1), -1)
    event_time = np.where(has_event, (gap[np.maximum(first, 0)] - 0.5) * days, np.nan)
    last_known = np.where(known.any(axis=1), known.shape[1] - 1 - known[:, ::-1].argmax(axis=1), -1)
    censor_time = np.where(last_known >= 0, gap[np.maximum(last_known, 0)] * days, np.nan)
    time = np.where(has_event, event_time, censor_time)
    return pd.Series(time, index=event_flags.index), pd.Series(
        has_event.astype(float), index=event_flags.index
    ).where(~np.isnan(time))


def _fill_from_earlier_rounds(sp: pd.DataFrame, data_dir: Path, layout: Layout) -> dict:
    """Fill baseline items NHATS asks only at a first interview from that round's file.

    Changes ``sp`` in place and returns, per variable, how many values were filled.
    """
    pid = layout.name(layout.raw["ids"]["person"])
    counts = {}
    for rnd, templates in layout.fallback_sources().items():
        path = find_file(data_dir, layout.raw["files"]["sp"].format(r=rnd, k=rnd).lower())
        if path is None:
            continue
        cols = [t.format(r=rnd, k=rnd).lower() for t in templates]
        early = read_table(path, [pid, *cols]).set_index(pid).reindex(sp.index)
        for t, col in zip(templates, cols, strict=True):
            now = layout.name(t)
            if col not in early or now not in sp:
                continue
            gap = _clean(sp[now], layout.special_missing).isna()
            prior = _clean(early[col], layout.special_missing)
            take = gap & prior.notna()
            sp.loc[take, now] = prior[take]
            counts[now] = int(take.sum())
    return counts


def build(data_dir: str | Path, layout: Layout) -> tuple[pd.DataFrame, dict]:
    """Canonical analytic extract and a disclosure-safe flow summary."""
    data_dir = Path(data_dir)
    report = check(data_dir, layout)
    if not report.ok:
        raise ValueError("NHATS layout check failed: " + json.dumps(report.to_dict(), indent=2))
    pid = layout.name(layout.raw["ids"]["person"])
    have = set()
    for _, path in _baseline_files(data_dir, layout):
        have |= available_columns(path)
    optional = {v for names in layout.optional_variables().values() for v in names if v in have}
    optional |= {v for v in layout.preferred_variables() if v in have}
    sp = read_baseline(data_dir, layout, sorted(set(layout.baseline_variables()) | optional))
    flow = {"sp_file_rows": len(sp)}
    sample = layout.raw.get("sample")
    if sample:
        res = _clean(sp[layout.name(sample["residence"])], layout.special_missing)
        sp = sp[res.isin([float(v) for v in sample["keep"]])].copy()
        flow["with_sp_interview"] = len(sp)
    filled = _fill_from_earlier_rounds(sp, data_dir, layout)
    if filled:
        flow["filled_from_earlier_round"] = filled

    out = pd.DataFrame(index=sp.index)
    out["participant_id"] = sp.index.astype(str)
    d = layout.raw["design"]
    out["survey_weight"] = _clean(sp[layout.name(d["weight"])], layout.special_missing)
    out["stratum"] = sp[layout.name(d["stratum"])]
    out["psu"] = sp[layout.name(d["psu"])]
    for name, spec in layout.raw["covariates"].items():
        x = _clean(sp[layout.name(spec["source"])], layout.special_missing)
        out[name] = x.map({float(k): v for k, v in spec["map"].items()}) if "map" in spec else x
    for name, spec in layout.raw["indicators"].items():
        out[name] = score_item(sp, layout, spec)
    skipped = {}
    for name, spec in layout.raw["comparators"].items():
        absent = [layout.name(s) for s in _sources(spec) if layout.name(s) not in sp]
        if absent:  # only optional comparators can get here; check() guards the rest
            out[name] = np.nan
            skipped[name] = absent
            continue
        out[name] = score_item(sp, layout, spec)
    if skipped:
        flow["comparators_skipped"] = {k: f"{len(v)} variables absent" for k, v in skipped.items()}
    for name, src in layout.raw.get("conditions", {}).items():
        x = _clean(sp[layout.name(src)], layout.special_missing)
        out[f"cond_{name}"] = x.isin([1.0, 7.0]).astype(float).where(x.notna())
    for name, spec in layout.raw["indicators"].items():
        if spec.get("kind") == "neg_log_income":  # equivalised income for the figures
            out["income"] = np.expm1(-out[name])
            break
        if spec.get("kind") == "neg_log":
            out["income"] = _clean(sp[layout.name(spec["source"])], layout.special_missing)
            break

    # Mortality from the latest tracker.
    latest, tracker_path = _latest_tracker(data_dir, layout)
    ks = [k for k in layout.follow_ups if k <= latest]
    mort = layout.raw["outcomes"]["mortality"]
    tracker = read_table(tracker_path, layout.tracker_variables(latest)).set_index(pid)
    tracker = tracker.reindex(out.index)
    status = pd.DataFrame(
        {k: _clean(tracker[layout.name(mort["status"], k)], layout.special_missing) for k in ks}
    )
    dead = status.isin([float(c) for c in mort["event_codes"]])
    dead = dead & ~dead.shift(1, axis=1, fill_value=False).cummax(axis=1)
    known = status.notna()
    out["mortality_time"], out["mortality_event"] = _outcome_times(known, dead, layout)

    # First overnight hospital stay from follow-up SP files, censored at death.
    hosp = layout.raw["outcomes"]["hospitalisation"]
    answered, stays = {}, {}
    for k in ks:
        path = find_file(data_dir, layout.raw["files"]["sp"].format(r=k, k=k).lower())
        if path is None:
            continue
        f = read_table(path, layout.follow_up_sp_variables(k)).set_index(pid).reindex(out.index)
        x = _clean(f[layout.name(hosp["sp_item"], k)], layout.special_missing)
        answered[k] = x.notna()
        stays[k] = x.isin([float(v) for v in hosp["present"]])
    if stays:
        stays_df = pd.DataFrame(stays)
        first_stay = stays_df & ~stays_df.shift(1, axis=1, fill_value=False).cummax(axis=1)
        t, e = _outcome_times(pd.DataFrame(answered), first_stay, layout)
        # A death in the interval straight after the last stay-free interview is
        # a competing event: the person was seen stay-free, then died before the
        # next interview could ask. A death after missed interviews stays
        # censored at the last answer, since the gap was not observed. No
        # answered follow-up at all (death before the first follow-up) counts too.
        seen_until = t.where(e.notna(), 0.0).fillna(0.0)
        died_first = (out["mortality_event"] == 1) & (
            out["mortality_time"] <= seen_until + layout.days_per_round
        )
        censor_at_death = died_first & (e != 1)
        t = t.where(~censor_at_death, out["mortality_time"])
        e = e.where(~censor_at_death, 0.0)
        out["hospital_time"], out["hospital_event"] = t, e
        # Death before a first stay is a competing event, not ordinary censoring.
        out["hospital_competing"] = censor_at_death.astype(float).where(e.notna())
    else:
        out["hospital_time"], out["hospital_event"] = np.nan, np.nan
        out["hospital_competing"] = np.nan
    # Secondary combined endpoint: death or first overnight stay, whichever first.
    out["combined_time"] = out[["mortality_time", "hospital_time"]].min(axis=1)
    out["combined_event"] = (
        (
            (out["mortality_event"] == 1) & (out["mortality_time"] <= out["combined_time"])
            | (out["hospital_event"] == 1)
        )
        .astype(float)
        .where(out["mortality_time"].notna())
    )

    keep = out["survey_weight"].gt(0)
    flow["positive_weight"] = int(keep.sum())
    keep &= out["mortality_time"].notna()
    flow["with_follow_up"] = int(keep.sum())
    out = out[keep].reset_index(drop=True)
    flow["analytic_rows"] = len(out)
    flow["mortality_events"] = int(out["mortality_event"].sum())
    flow["hospital_events"] = int(out["hospital_event"].fillna(0).sum())
    flow["indicator_missing_fraction"] = {
        c: round(float(out[c].isna().mean()), 4) for c in layout.raw["indicators"]
    }
    flow["baseline_round"] = layout.r
    flow["follow_up_rounds"] = ks
    return out, flow
