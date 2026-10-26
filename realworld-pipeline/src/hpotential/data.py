from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def acquire(url: str, destination: str | Path, expected_sha256: str) -> Path:
    """Download only when an immutable checksum is supplied and matches."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("A 64-character SHA-256 is mandatory for remote data")
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    actual = sha256_file(destination)
    if actual.lower() != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch: expected {expected_sha256}, got {actual}")
    return destination

def required_columns(cfg) -> set[str]:
    r = cfg.raw
    columns = {r["data"]["id"], r["data"]["weight"], *r["data"].get("design", [])}
    columns |= set(r["data"]["covariates"])
    for spec in r["measurement"]["domains"].values(): columns |= set(spec["indicators"])
    for spec in r["outcomes"]["endpoints"].values(): columns |= {spec["time"], spec["event"]}
    columns |= set(r["validation"].get("equity_groups", []))
    columns |= set(r["validation"].get("comparators", []))
    return columns

def load_and_validate(path: str | Path, cfg) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if path.suffix.lower() == ".csv": df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}: df = pd.read_parquet(path)
    else: raise ValueError("Input must be CSV or Parquet")
    missing = sorted(required_columns(cfg) - set(df.columns))
    if missing: raise ValueError("Missing required columns: " + ", ".join(missing))
    id_col, weight = cfg.raw["data"]["id"], cfg.raw["data"]["weight"]
    if df.empty or df[id_col].isna().any() or df[id_col].duplicated().any():
        raise ValueError("Participant IDs must be present and unique")
    if (pd.to_numeric(df[weight], errors="coerce") <= 0).any(): raise ValueError("Weights must be positive")
    for endpoint, spec in cfg.raw["outcomes"]["endpoints"].items():
        t = pd.to_numeric(df[spec["time"]], errors="coerce")
        e = pd.to_numeric(df[spec["event"]], errors="coerce")
        if (t.dropna() < 0).any() or not set(e.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Invalid time/event coding for {endpoint}")
    missingness = df[list(required_columns(cfg))].isna().mean()
    max_missing = float(cfg.raw["data"].get("max_column_missing", .60))
    bad = missingness[missingness > max_missing]
    if len(bad): raise ValueError("Missingness threshold exceeded: " + ", ".join(bad.index))
    manifest = {"path": str(path), "sha256": sha256_file(path), "rows": len(df),
                "columns": len(df.columns), "required_columns": sorted(required_columns(cfg)),
                "missing_fraction": missingness.round(6).to_dict()}
    return df, manifest

def write_manifest(manifest: dict, path: str | Path):
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

def synthetic_dataset(cfg, n=750, seed=20260905):
    """Clearly labelled CI/tutorial data; never substantive evidence."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(3), [[1,.25,.15],[.25,1,.2],[.15,.2,1]], n)
    df = pd.DataFrame({cfg.raw["data"]["id"]: [f"S{i:05d}" for i in range(n)],
                       cfg.raw["data"]["weight"]: rng.uniform(.5, 2, n),
                       "psu": rng.integers(1, 61, n), "stratum": rng.integers(1, 16, n),
                       "age": rng.integers(65, 96, n), "sex": rng.choice(["female","male"], n),
                       "race_group": rng.choice(["Black","White","Other"], n, p=[.22,.65,.13])})
    for j, (_, spec) in enumerate(cfg.indicators.items()):
        for k, col in enumerate(spec["indicators"]):
            x = .75*z[:,j] + rng.normal(0,.65,n)
            df[col] = x if spec.get("direction", "higher_better") == "higher_better" else -x
    potential = z[:,0] + .45*z[:,1] + .45*z[:,2] - .025*(df.age-65)
    for name in cfg.raw["validation"].get("comparators", []): df[name] = -potential + rng.normal(0,1,n)
    for endpoint, spec in cfg.raw["outcomes"]["endpoints"].items():
        rate = np.exp(-4.8 - .35*potential + (.25 if endpoint == "hospitalisation" else 0))
        event_t = rng.exponential(1/rate); censor_t = rng.uniform(800, 2400, n)
        df[spec["time"]] = np.minimum(event_t, censor_t).round(1)
        df[spec["event"]] = (event_t <= censor_t).astype(int)
    for domain in cfg.indicators.values():
        for col in domain["indicators"]:
            df.loc[rng.random(n) < .04, col] = np.nan
    return df
