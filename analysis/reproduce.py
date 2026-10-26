"""Regenerate every synthetic result table reported in the papers.

Usage::

    python analysis/reproduce.py            # writes results/
    python analysis/reproduce.py --out DIR  # writes somewhere else

Floats are written to 12 significant digits. Last-digit differences between
NumPy/BLAS builds then disappear, so the CSV files and their hashes are the
same on Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hpotential.crf import Action, H, allocate, ede, gap_decomposition  # noqa: E402
from hpotential.measurement import GaussianMeasurement  # noqa: E402

SEED = 20261025
N_SYNTHETIC = 4000
SIGNIFICANT_DIGITS = 12
Z_95 = 1.959963984540054  # standard normal 97.5% quantile

PROFILES = [
    ("balanced", 1, 0.45, 0.45),
    ("moderate", 1, 0.60, 0.30),
    ("unbalanced", 1, 0.70, 0.20),
    ("extreme", 1, 0.89, 0.01),
]


def _round(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.{SIGNIFICANT_DIGITS}g}")
    return value


def write_csv(out: Path, name: str, rows: list[dict[str, object]]) -> None:
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: _round(v) for k, v in row.items()} for row in rows)


def aggregation_tables() -> tuple[list[dict], list[dict]]:
    """Paper 1: equal arithmetic context, unequal bottlenecks."""
    aggregation = [
        dict(
            profile=name,
            P=p,
            S=s,
            R=r,
            arithmetic=H(p, s, r, 1),
            geometric=H(p, s, r),
            harmonic=H(p, s, r, -1),
            bottleneck=H(p, s, r, -math.inf),
        )
        for name, p, s, r in PROFILES
    ]
    sensitivity = [
        dict(profile=name, alpha=alpha, rho=rho, H=H(p, s, r, rho, alpha))
        for alpha in (0.25, 0.5, 0.75)
        for rho in (-20, -5, -2, -1, -0.5, 0, 1)
        for name, p, s, r in PROFILES
    ]
    return aggregation, sensitivity


def measurement_tables() -> tuple[list[dict], list[dict]]:
    """Paper 2: known Gaussian model under missingness and intercept drift."""
    rng = np.random.default_rng(SEED)
    phi = np.array([[1, 0.3, 0.2], [0.3, 1, 0.4], [0.2, 0.4, 1]])
    loadings = np.zeros((9, 3))
    for j in range(9):
        loadings[j, j // 3] = (0.6, 0.75, 0.9)[j % 3]
    psi = 1 - np.sum(loadings**2, axis=1)
    names = [f"indicator_{j + 1}" for j in range(9)]
    model = GaussianMeasurement(names, loadings, phi, psi)

    # The draw order below fixes the published numbers; do not reorder.
    z = rng.multivariate_normal(np.zeros(3), phi, size=N_SYNTHETIC)
    x = z @ loadings.T + rng.normal(size=(N_SYNTHETIC, 9)) * np.sqrt(psi)
    missing = rng.random(x.shape) < 0.3
    shifted = x.copy()
    shifted[:, 3:6] += 0.5  # constant drift in the social indicators

    scenarios = [
        ("complete", x),
        ("missing_30pct", np.where(missing, np.nan, x)),
        ("social_intercept_shift", shifted),
    ]
    rows = []
    for label, data in scenarios:
        means, covs = model.score(dict(zip(names, data.T)))
        for k in range(3):
            errors = means[:, k] - z[:, k]
            sd = np.sqrt(covs[:, k, k])
            rows.append(
                dict(
                    scenario=label,
                    domain=k + 1,
                    n=N_SYNTHETIC,
                    correlation=float(np.corrcoef(means[:, k], z[:, k])[0, 1]),
                    rmse=float(np.sqrt(np.mean(errors**2))),
                    bias=float(np.mean(errors)),
                    posterior_sd=float(np.mean(sd)),
                    coverage95=float(np.mean(np.abs(errors) <= Z_95 * sd)),
                )
            )
    determinacy = [
        dict(domain=k + 1, implied=float(d)) for k, d in enumerate(model.implied_determinacy())
    ]
    return rows, determinacy


def policy_tables() -> tuple[list[dict], list[dict]]:
    """Paper 3: aversion to inequality reverses a two-action choice."""
    population = [(1, 0.2, 0.2), (1, 0.8, 0.8)]
    actions = [
        Action("support_low", 1, ((0, 0.1, 0.1), (0, 0, 0))),
        Action("support_high", 1, ((0, 0, 0), (0, 0.2, 0.2))),
    ]
    policy = []
    for eps in (0, 0.5, 1, 2, 3):
        best = allocate(population, actions, 1, eps)
        for a in actions:
            post = [
                tuple(p[k] + a.changes[i][k] for k in range(3)) for i, p in enumerate(population)
            ]
            values = [H(*p) for p in post]
            policy.append(
                dict(
                    epsilon=eps,
                    action=a.name,
                    cost=a.cost,
                    mean=sum(values) / len(values),
                    ede=ede(values, eps),
                    optimal=a.name in best["selected"],
                )
            )
    gap = [
        dict(
            zip(
                ("biomedical", "social", "structural", "total"),
                gap_decomposition((0.3, 0.7, 0.85), (0.3, 0.7, 0.2)),
            )
        )
    ]
    return policy, gap


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    aggregation, sensitivity = aggregation_tables()
    measurement, determinacy = measurement_tables()
    policy, gap = policy_tables()
    tables = {
        "aggregation.csv": aggregation,
        "sensitivity.csv": sensitivity,
        "measurement.csv": measurement,
        "determinacy.csv": determinacy,
        "policy.csv": policy,
        "gap.csv": gap,
    }
    for name, rows in tables.items():
        write_csv(out, name, rows)

    manifest = {
        "command": "python analysis/reproduce.py",
        "seed": SEED,
        "n_synthetic": N_SYNTHETIC,
        "significant_digits": SIGNIFICANT_DIGITS,
        "clinical_data": False,
        "csv_sha256": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in sorted(tables)
        },
        # Recorded for information; the CSV hashes should not depend on it.
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.system(),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps({"tables": len(tables), "synthetic_n": N_SYNTHETIC, "out": str(out)}))
    return manifest


if __name__ == "__main__":
    main()
