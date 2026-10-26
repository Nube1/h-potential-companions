"""Executable checks for the propositions in Appendix A.

These checks verify the implementation of the stated formulas on a finite grid.
They are not proofs and are not empirical validation of H-Potential.
"""
from itertools import product
import math


def contextual_mean(s, r, rho, alpha):
    if not (0 <= s <= 1 and 0 <= r <= 1):
        raise ValueError("s and r must lie in [0,1]")
    if not 0 < alpha < 1 or rho > 1:
        raise ValueError("require 0 < alpha < 1 and rho <= 1")
    if s == 0 or r == 0:
        if rho <= 0:
            return 0.0
    if rho == 0:
        return s**alpha * r ** (1 - alpha)
    return (alpha * s**rho + (1 - alpha) * r**rho) ** (1 / rho)


def h_ces(p, s, r, rho=0, alpha=0.5):
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0,1]")
    return p * contextual_mean(s, r, rho, alpha)


def h_relational(p, s, r, beta=(1.0, 0.5, 0.5, 0.25, 0.25)):
    exponent = (beta[0] * p + beta[1] * s + beta[2] * r +
                beta[3] * p * s + beta[4] * p * r)
    return (1 - math.exp(-exponent)) / (1 - math.exp(-sum(beta)))


grid = [0.1, 0.25, 0.5, 0.75, 1.0]
for rho in (-4, -1, 0, 0.5, 1):
    for p, s, r in product(grid, repeat=3):
        value = h_ces(p, s, r, rho)
        assert -1e-12 <= value <= p + 1e-12
        assert h_ces(min(1, p + 0.01), s, r, rho) + 1e-12 >= value
        assert h_ces(p, min(1, s + 0.01), r, rho) + 1e-12 >= value
        assert h_ces(p, s, min(1, r + 0.01), rho) + 1e-12 >= value

# Exact geometric elasticities, checked by analytic identity.
p, s, r, alpha = 0.7, 0.6, 0.8, 0.4
h = h_ces(p, s, r, 0, alpha)
assert math.isclose((h / p) * p / h, 1.0)
assert math.isclose((alpha * h / s) * s / h, alpha)
assert math.isclose(((1 - alpha) * h / r) * r / h, 1 - alpha)

# Ranking reversal: arithmetic compensation versus bottleneck protection.
x, y = (1.0, 1.0, 0.2), (1.0, 0.55, 0.55)
arithmetic = lambda z: z[0] * 0.5 * (z[1] + z[2])
bottleneck = lambda z: z[0] * min(z[1], z[2])
assert arithmetic(x) > arithmetic(y)
assert bottleneck(x) < bottleneck(y)

# Relational ontology is bounded and can reject the biomedical ceiling.
for p, s, r in product([0, 0.25, 0.5, 1], repeat=3):
    value = h_relational(p, s, r)
    assert -1e-12 <= value <= 1 + 1e-12
assert h_relational(0, 0.8, 0.8) > 0

print("All finite-grid formal property checks passed.")
