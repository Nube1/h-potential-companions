"""Factor kinetics, immune tolerance induction and gene therapy: Appendix C."""
from __future__ import annotations
import math


def hill_fraction(Y, K_I, beta):
    """Neutralisation fraction. Sigmoid in Y when beta > 1."""
    if Y == 0:
        return 0.0
    return Y ** beta / (K_I ** beta + Y ** beta)


def steady_state(Y, *, numerator, alpha, K_I, beta, lam):
    """Proposition C.2: S_T*(Y) = (f + h) / (alpha*Hill(Y) + lambda)."""
    return numerator / (alpha * hill_fraction(Y, K_I, beta) + lam)


def inhibitor_threshold(*, numerator, alpha, K_I, beta, lam, S_prot):
    """Proposition C.3: the titre at which the regimen stops protecting."""
    h_need = (numerator / S_prot - lam) / alpha
    if not 0.0 < h_need < 1.0:
        raise ValueError(f"h_need={h_need:.4f} outside (0,1); no finite threshold")
    return K_I * (h_need / (1.0 - h_need)) ** (1.0 / beta)


def linear_reference(Y, *, numerator, alpha, K_I, lam):
    """Definition C.2: the classical linear model, beta = 1 small-Y limit."""
    return numerator / (alpha * Y / K_I + lam)


def iti_equilibrium(I, *, rho, kappa, lam_Y, Y_max):
    """Proposition C.4: extinction above a threshold suppression, else a
    positive steady titre. Immune tolerance induction is all or nothing."""
    if kappa * I + lam_Y >= rho:
        return 0.0
    return Y_max * (1.0 - (kappa * I + lam_Y) / rho)


def iti_critical_suppression(*, rho, kappa, lam_Y):
    return (rho - lam_Y) / kappa


def gene_therapy_equilibrium(A, eta, Y):
    """Proposition C.5: F* = A / (eta * Y). The inhibitor is in the
    denominator, so halving the titre is worth as much as doubling expression."""
    if Y <= 0:
        raise ValueError("Y must be positive for a finite equilibrium")
    return A / (eta * Y)
