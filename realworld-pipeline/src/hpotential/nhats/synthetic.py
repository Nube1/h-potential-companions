"""Made-up files laid out like NHATS public-use files, for tests and demos.

They carry the variable names, codes and special missing values of the layout,
and a latent structure so the figures have something to show. They are not
NHATS data and nothing computed from them is evidence about anyone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import Layout


def _categorical(rng, latent, cuts):
    """Ordered codes 1..len(cuts)+1 from a latent score and cut points."""
    return 1 + np.searchsorted(np.sort(cuts), latent)


def _with_missing(rng, x, share=0.03, codes=(-7, -8)):
    x = np.asarray(x, dtype=float).copy()
    hit = rng.random(len(x)) < share
    x[hit] = rng.choice(codes, hit.sum())
    return x


def write_synthetic_nhats(
    out_dir: str | Path, layout: Layout, n: int = 1500, seed: int = 20261026, fmt: str = ".dta"
) -> list[Path]:
    """Write the baseline SP file, the Round 1 SP file, the income file,
    follow-up SP files and the latest tracker.

    The coding follows the NHATS documents checked for the thesis: derived
    activity items use the "self and difficulty" codes, some home items are
    skipped (-1) by design, education and gender are asked only at a first
    interview, and a few baseline rows are decedents or facility-only cases.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    r = layout.r
    name = layout.name

    bio, soc, sysb = rng.multivariate_normal(
        [0, 0, 0], [[1, 0.5, -0.45], [0.5, 1, -0.4], [-0.45, -0.4, 1]], n
    ).T
    age_band = np.clip(1 + rng.poisson(1.6, n), 1, 6)
    bio = bio - 0.35 * (age_band - 3)
    sp = pd.DataFrame({name(layout.raw["ids"]["person"]): 10_000_000 + np.arange(n)})
    d = layout.raw["design"]
    sp[name(d["weight"])] = np.round(rng.lognormal(8.8, 0.6, n), 2)
    sp.loc[rng.random(n) < 0.04, name(d["weight"])] = 0.0  # out of the analytic population
    sp[name(d["stratum"])] = rng.integers(1, 57, n)
    sp[name(d["psu"])] = rng.integers(1, 3, n)
    # Residence and interview type: mostly community; a few decedents (with the
    # positive weight NHATS wrongly gives them) and facility-only cases.
    dresid = rng.choice([1, 2, 4, 6, 3, 5], n, p=[0.88, 0.04, 0.03, 0.03, 0.01, 0.01])
    sp[name("r{r}dresid")] = dresid
    no_sp = ~np.isin(dresid, [1, 2, 4])
    continuing = rng.random(n) < 0.6  # in the 2011 cohort: first interview in Round 1
    sp[name("r{r}d2intvrage")] = age_band
    gender = rng.choice([1, 2], n, p=[0.43, 0.57])
    sp[name("r{r}dgender")] = np.where(continuing, -1, gender)
    sp[name("rl{r}dracehisp")] = rng.choice(
        [1, 2, 3, 4, 5, 6], n, p=[0.67, 0.2, 0.04, 0.07, 0.01, 0.01]
    )

    sp[name("hc{r}health")] = _with_missing(rng, _categorical(rng, -bio, [-1.3, -0.4, 0.5, 1.3]))
    # Biomedical: conditions, activities, cognition, mood.
    for j in range(1, 11):
        p = 1 / (1 + np.exp(1.3 + 0.8 * bio - 0.15 * j / 10))
        sp[name(f"hc{{r}}disescn{j}")] = _with_missing(rng, np.where(rng.random(n) < p, 1, 2), 0.01)
    for item in (
        "sc{r}deatsfdf",
        "sc{r}dbathsfdf",
        "sc{r}dtoilsfdf",
        "sc{r}ddressfdf",
        "mo{r}dbedsfdf",
        "mo{r}dinsdsfdf",
        "mo{r}doutsfdf",
        "ha{r}dlaunsfdf",
        "ha{r}dshopsfdf",
        "ha{r}dmealsfdf",
        "ha{r}dbanksfdf",
        "mc{r}dmedssfdf",
    ):
        # 2 no difficulty, 3 difficulty, 1 did not do by self; a few 6-8.
        z = -1.1 * bio + rng.normal(0, 0.8, n)
        code = np.where(z > 2.0, 1, np.where(z > 0.9, 3, 2))
        odd = rng.random(n) < 0.02
        code = np.where(odd, rng.choice([6, 7, 8], n), code)
        if item.startswith("mc"):
            code = np.where(rng.random(n) < 0.05, 9, code)  # takes no medications
        sp[name(item)] = _with_missing(rng, code)
    for item, top in (("cg{r}dwrdimmrc", 10), ("cg{r}dwrddlyrc", 10), ("cg{r}dclkdraw", 5)):
        mean = top * (0.5 + 0.12 * bio)
        sp[name(item)] = _with_missing(
            rng, np.clip(np.round(mean + rng.normal(0, top * 0.18, n)), 0, top)
        )
    for j in range(1, 5):
        sp[name(f"hc{{r}}depresan{j}")] = _with_missing(
            rng, _categorical(rng, -0.6 * bio + rng.normal(0, 0.9, n), [0.3, 1.2, 2.0])
        )
    pain = -0.8 * bio + rng.normal(0, 0.9, n)
    sp[name("ss{r}painbothr")] = _with_missing(rng, np.where(pain > 0.1, 1, 2))
    sp[name("ss{r}painlimts")] = _with_missing(rng, np.where(pain > 0.9, 1, 2))
    # Social: participation, network, communication.
    for item in ("vistfrfam", "attrelser", "clbmtgrac", "outfrenjy"):
        p = 1 / (1 + np.exp(-(0.4 + 0.9 * soc)))
        sp[name(f"pa{{r}}{item}")] = _with_missing(rng, np.where(rng.random(n) < p, 1, 2))
    sp[name("sn{r}dnumsn")] = _with_missing(
        rng, np.clip(np.round(2 + 1.2 * soc + rng.normal(0, 1, n)), 0, 5)
    )
    for item in ("ss{r}hearphone", "ss{r}convquiet", "ss{r}seewellst", "ss{r}glrednewp"):
        p = 1 / (1 + np.exp(-(1.8 + 0.7 * soc + 0.5 * bio)))
        sp[name(item)] = _with_missing(rng, np.where(rng.random(n) < p, 1, 2))
    # Systemic: income, schooling, home access, care access (higher = more burden).
    income = np.exp(10.4 - 0.7 * sysb + rng.normal(0, 0.5, n))
    sp[name("ia{r}totinc")] = _with_missing(rng, np.round(income, -2), 0.3)
    sp[name("hh{r}dmarstat")] = rng.choice(
        [1, 2, 3, 4, 5, 6], n, p=[0.5, 0.03, 0.02, 0.1, 0.3, 0.05]
    )
    for j in range(1, 6):
        sp[name(f"ia{{r}}toincim{j}")] = np.round(income * np.exp(rng.normal(0, 0.1, n)), -2)
    inc = pd.DataFrame({name(layout.raw["ids"]["person"]): sp.iloc[:, 0]})
    for j in range(1, 21):
        inc[name(f"ia{{r}}dtoincimi{j}")] = np.round(income * np.exp(rng.normal(0, 0.1, n)), -2)
    school = _categorical(rng, -sysb + rng.normal(0, 0.6, n), np.linspace(-1.6, 1.6, 8))
    sp[name("el{r}higstschl")] = np.where(continuing, -1, _with_missing(rng, school))
    for item, yes_is_barrier in (
        ("ho{r}entrstair", True),
        ("ho{r}homeamen1", False),
        ("ho{r}entrnramp", False),
    ):
        p = 1 / (1 + np.exp(-(-0.3 + 0.8 * sysb)))
        barrier = rng.random(n) < p
        sp[name(item)] = _with_missing(rng, np.where(barrier == yes_is_barrier, 1, 2))
    levels = _categorical(rng, 0.6 * sysb + rng.normal(0, 1, n), [0.0, 1.0, 1.8])
    multi_unit = rng.random(n) < 0.25
    sp[name("ho{r}levelsflr")] = np.where(multi_unit, -1, _with_missing(rng, levels))
    # HO8 is skipped (-1) for one-floor homes and for multi-unit buildings.
    skip8 = multi_unit | (levels == 1)
    sp[name("ho{r}homeamen1")] = np.where(skip8, -1, sp[name("ho{r}homeamen1")])
    for item in ("mc{r}havregdoc", "mc{r}regdoclyr"):
        p = 1 / (1 + np.exp(-(-2.0 + 0.9 * sysb)))
        sp[name(item)] = _with_missing(rng, np.where(rng.random(n) < p, 2, 1))
    for item in ("walk6blks", "up20stair", "car20pnds", "geonknees", "bendover", "rechovrhd"):
        p = 1 / (1 + np.exp(-(1.0 + 1.1 * bio)))
        sp[name(f"pc{{r}}{item}")] = _with_missing(rng, np.where(rng.random(n) < p, 1, 2))

    # No SP interview: every instrument item is inapplicable.
    keep = {
        name(layout.raw["ids"]["person"]),
        *(name(v) for v in layout.raw["design"].values()),
        name("r{r}dresid"),
        name("r{r}d2intvrage"),
        name("r{r}dgender"),
        name("rl{r}dracehisp"),
    }
    for col in sp.columns:
        if col not in keep:
            sp.loc[no_sp, col] = -1
    # Round 1 SP file: first-interview items for the continuing sample.
    r1 = pd.DataFrame({name(layout.raw["ids"]["person"]): sp.iloc[:, 0][continuing]})
    r1["el1higstschl"] = school[continuing]
    r1["r1dgender"] = gender[continuing]

    # Outcomes from a latent hazard; capability = biomedical x context.
    potential = bio + 0.4 * soc - 0.4 * sysb
    death_years = rng.exponential(1 / np.exp(-2.9 - 0.55 * potential + 0.25 * (age_band - 3)))
    hosp_years = rng.exponential(1 / np.exp(-1.6 - 0.35 * potential))
    attrition_years = rng.exponential(18, n)

    ks = layout.follow_ups
    tracker = pd.DataFrame({name(layout.raw["ids"]["person"]): sp.iloc[:, 0]})
    status_code = layout.raw["outcomes"]["mortality"]["event_codes"][0]
    paths = []
    for k in ks:
        years = k - r
        dead = death_years <= years
        lost = (attrition_years <= years) & ~dead
        status = np.where(dead, status_code, 60)  # 60: interviewed (made-up code)
        status = np.where(lost, -9, status)
        tracker[name("r{k}status", k)] = status
        fu = pd.DataFrame({name(layout.raw["ids"]["person"]): sp.iloc[:, 0]})
        alive = ~dead & ~lost
        stayed = (hosp_years <= years) & (hosp_years > years - 1)
        item = np.where(stayed, 1, 2).astype(float)
        item[~alive] = -1
        fu[name("hc{k}hosptstay", k)] = item
        path = out_dir / (layout.raw["files"]["sp"].format(r=k, k=k) + fmt)
        _write(fu, path)
        paths.append(path)
    tpath = out_dir / (layout.raw["files"]["tracker"].format(r=r, k=max(ks)) + fmt)
    _write(tracker, tpath)
    spath = out_dir / (layout.raw["files"]["sp"].format(r=r, k=r) + fmt)
    _write(sp, spath)
    r1path = out_dir / (layout.raw["files"]["sp"].format(r=1, k=1) + fmt)
    _write(r1, r1path)
    paths.append(r1path)
    if layout.raw["files"].get("income"):
        ipath = out_dir / (layout.raw["files"]["income"].format(r=r, k=r) + fmt)
        _write(inc, ipath)
        paths.append(ipath)
    return [spath, tpath, *paths]


def _write(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".dta":
        frame.to_stata(path, write_index=False, version=118)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
