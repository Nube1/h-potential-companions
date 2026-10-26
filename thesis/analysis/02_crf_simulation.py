"""Reproducible mathematical stress tests for the Capability Realisation Function."""
from pathlib import Path
import csv
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "Figures"
OUT = ROOT / "outputs" / "crf_simulation"
FIG.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.5
PROFILES = [
    {"profile": "Balanced", "P": 80.0, "S": 0.60, "B": 0.40},
    {"profile": "Social bottleneck", "P": 80.0, "S": 0.30, "B": 0.10},
    {"profile": "Structural bottleneck", "P": 95.0, "S": 0.95, "B": 0.75},
    {"profile": "Moderate and enabled", "P": 65.0, "S": 0.80, "B": 0.20},
]


def context_mean(S, R, rho, alpha=ALPHA):
    if S == 0 or R == 0:
        return 0.0
    if rho == 0:
        return S**alpha * R ** (1 - alpha)
    return (alpha * S**rho + (1 - alpha) * R**rho) ** (1 / rho)


def realised(P, S, B, rho=0, alpha=ALPHA):
    return P * context_mean(S, 1 - B, rho, alpha)


def ede(values, epsilon):
    values = np.asarray(values, dtype=float)
    if epsilon == 1:
        return float(np.exp(np.mean(np.log(values))))
    return float(np.mean(values ** (1 - epsilon)) ** (1 / (1 - epsilon)))


models = {
    "Additive": lambda p: p["P"] * (ALPHA * p["S"] + (1 - ALPHA) * (1 - p["B"])),
    "Geometric CRF": lambda p: realised(p["P"], p["S"], p["B"], 0),
    "Harmonic CRF": lambda p: realised(p["P"], p["S"], p["B"], -1),
    "Bottleneck limit": lambda p: p["P"] * min(p["S"], 1 - p["B"]),
}


def candidate_family(P, S, B, rho=-1.0, alpha=ALPHA, beta=(0, 1, 1, 1, 0)):
    """Chapter-2 candidate forms; illustrative, never validation results."""
    if P < 0 or not 0 <= S <= 1 or not 0 <= B <= 1:
        raise ValueError("Require P >= 0 and S,B in [0,1]")
    b0, b_p, b_s, b_b, b_ps = beta
    additive = b0 + b_p * P + b_s * S - b_b * B
    rel_beta = (1.0, 0.5, 0.5, 0.25, 0.25)
    r = 1 - B
    exponent = (rel_beta[0] * P + rel_beta[1] * S + rel_beta[2] * r +
                rel_beta[3] * P * S + rel_beta[4] * P * r)
    relational = (1 - math.exp(-exponent)) / (1 - math.exp(-sum(rel_beta)))
    return {
        "additive_regression": additive,
        "interaction_regression": additive + b_ps * P * S,
        "multiplicative": realised(P, S, B, 0, alpha),
        "ces": realised(P, S, B, rho, alpha),
        "relational": relational,
    }

rows = []
for p in PROFILES:
    row = dict(p)
    row.update({name: fn(p) for name, fn in models.items()})
    H = row["Geometric CRF"]
    row.update({
        "marginal_P": H / p["P"],
        "marginal_S": ALPHA * H / p["S"],
        "marginal_burden_reduction": (1 - ALPHA) * H / (1 - p["B"]),
    })
    rows.append(row)

with (OUT / "profile_comparison.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 5.8))
x = np.arange(len(PROFILES))
width = 0.19
for j, name in enumerate(models):
    ax.bar(x + (j - 1.5) * width, [r[name] for r in rows], width, label=name)
ax.set_xticks(x, [p["profile"] for p in PROFILES], rotation=12, ha="right")
ax.set_ylabel("Realised potential (illustrative units)")
ax.set_title("Compensation changes profile rankings")
ax.legend(frameon=False, ncols=2)
fig.tight_layout()
fig.savefig(FIG / "crf_profile_comparison.png", dpi=220)
plt.close(fig)

rhos = np.array([-8, -4, -2, -1, -0.5, -0.2, 0.0])
rho_rows = []
fig, ax = plt.subplots(figsize=(9, 5.5))
for p in PROFILES:
    vals = [realised(p["P"], p["S"], p["B"], float(r)) for r in rhos]
    ax.plot(rhos, vals, marker="o", label=p["profile"])
    rho_rows.extend({"profile": p["profile"], "rho": r, "H": h}
                    for r, h in zip(rhos, vals))
ax.set_xlabel("Substitution parameter rho")
ax.set_ylabel("Realised potential")
ax.set_title("Stronger non-compensation penalises bottlenecks")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "crf_rho_sensitivity.png", dpi=220)
plt.close(fig)

with (OUT / "rho_sensitivity.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["profile", "rho", "H"])
    writer.writeheader()
    writer.writerows(rho_rows)

epsilons = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3], dtype=float)
base = [r["Geometric CRF"] for r in rows]
equity_rows = [{"epsilon": e, "equivalent_potential": ede(base, e)} for e in epsilons]
fig, ax = plt.subplots(figsize=(8.5, 5.2))
ax.plot(epsilons, [r["equivalent_potential"] for r in equity_rows], marker="o")
ax.set_xlabel("Inequality aversion epsilon")
ax.set_ylabel("Equally distributed equivalent potential")
ax.set_title("Equity weighting prioritises low-capability profiles")
fig.tight_layout()
fig.savefig(FIG / "crf_equity_aversion.png", dpi=220)
plt.close(fig)

with (OUT / "equity_aversion.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epsilon", "equivalent_potential"])
    writer.writeheader()
    writer.writerows(equity_rows)

# Exact decomposition check for the first two profiles.
a, b = PROFILES[0], PROFILES[1]
lhs = math.log(realised(a["P"], a["S"], a["B"]) /
               realised(b["P"], b["S"], b["B"]))
rhs = (math.log(a["P"] / b["P"]) + ALPHA * math.log(a["S"] / b["S"]) +
       (1 - ALPHA) * math.log((1 - a["B"]) / (1 - b["B"])))
assert abs(lhs - rhs) < 1e-12
assert all(0 <= r["Geometric CRF"] <= r["P"] for r in rows)
assert candidate_family(10, 0.8, 0.2)["multiplicative"] == realised(10, 0.8, 0.2)

print(f"Wrote {len(rows)} profiles, {len(rho_rows)} sensitivity rows, and {len(equity_rows)} equity rows.")
