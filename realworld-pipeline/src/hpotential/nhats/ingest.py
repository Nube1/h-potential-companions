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
                names |= {self.name(s) for s in _sources(spec)}
        names |= {self.name(s) for s in self.raw.get("conditions", {}).values()}
        return sorted(names)

    def tracker_variables(self, latest: int) -> list[str]:
        status = self.raw["outcomes"]["mortality"]["status"]
        ks = [k for k in self.follow_ups if k <= latest]
        return [self.name(self.raw["ids"]["person"])] + [self.name(status, k) for k in ks]

    def follow_up_sp_variables(self, k: int) -> list[str]:
        item = self.raw["outcomes"]["hospitalisation"]["sp_item"]
        return [self.name(self.raw["ids"]["person"]), self.name(item, k)]


def _sources(spec: dict) -> list[str]:
    if "source" in spec:
        return [spec["source"]]
    if "sources" in spec:
        return list(spec["sources"])
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
        }


def _latest_tracker(data_dir: Path, layout: Layout) -> tuple[int, Path] | None:
    for k in sorted(layout.follow_ups, reverse=True):
        path = find_file(data_dir, layout.name(layout.raw["files"]["tracker"], k))
        if path is not None:
            return k, path
    return None


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
        have = available_columns(sp)
        report.missing_variables["sp"] = [v for v in need if v not in have]
        frame = read_table(sp, [v for v in need if v in have])
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
    specs = list(layout.raw["covariates"].values()) + list(layout.raw["indicators"].values())
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
            odd = sorted(values - codes - special)
            if odd:
                out[col] = odd
    return out


# ---------------------------------------------------------------------- build


def _clean(series: pd.Series, special: list[float]) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    return x.mask(x.isin(special))


def _items(frame: pd.DataFrame, layout: Layout, sources: list[str]) -> pd.DataFrame:
    cols = [layout.name(s) for s in sources]
    return pd.DataFrame({c: _clean(frame[c], layout.special_missing) for c in cols})


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
    items = _items(frame, layout, _sources(spec))
    if kind == "value":
        return items.iloc[:, 0]
    if kind == "reverse":
        lo, hi = spec["range"]
        return lo + hi - items.iloc[:, 0]
    if kind in ("reverse_sum", "reverse_mean"):
        lo, hi = spec["range"]
        flipped = lo + hi - items
        return _prorated(flipped, "sum" if kind == "reverse_sum" else "mean")
    if kind in ("count_present", "count_absent", "binary"):
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
            g = _items(frame, layout, group["sources"])
            flags.append(
                g.isin([float(v) for v in group["deficit"]]).astype(float).where(g.notna())
            )
        all_flags = pd.concat(flags, axis=1)
        return _prorated(all_flags, "mean")
    raise ValueError(f"Unknown indicator kind: {kind}")


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


def build(data_dir: str | Path, layout: Layout) -> tuple[pd.DataFrame, dict]:
    """Canonical analytic extract and a disclosure-safe flow summary."""
    data_dir = Path(data_dir)
    report = check(data_dir, layout)
    if not report.ok:
        raise ValueError("NHATS layout check failed: " + json.dumps(report.to_dict(), indent=2))
    pid = layout.name(layout.raw["ids"]["person"])
    sp = read_table(Path(report.files["sp"]), layout.baseline_variables()).set_index(pid)
    flow = {"sp_file_rows": len(sp)}

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
    for name, spec in layout.raw["comparators"].items():
        out[name] = score_item(sp, layout, spec)
    for name, src in layout.raw.get("conditions", {}).items():
        x = _clean(sp[layout.name(src)], layout.special_missing)
        out[f"cond_{name}"] = x.isin([1.0, 7.0]).astype(float).where(x.notna())
    income = layout.raw["indicators"].get("systemic_indicator_1", {})
    if income.get("kind") == "neg_log":
        out["income"] = _clean(sp[layout.name(income["source"])], layout.special_missing)

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
        died_first = (out["mortality_event"] == 1) & (out["mortality_time"] < t.fillna(np.inf))
        censor_at_death = died_first & (e != 1)
        t = t.where(~censor_at_death, out["mortality_time"])
        e = e.where(~censor_at_death, 0.0)
        out["hospital_time"], out["hospital_event"] = t, e
    else:
        out["hospital_time"], out["hospital_event"] = np.nan, np.nan

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
