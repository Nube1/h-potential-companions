crf <- function(P, S, B, alpha = 0.5, rho = 0) {
  stopifnot(all(P >= 0), all(S >= 0 & S <= 1), all(B >= 0 & B <= 1),
            alpha > 0, alpha < 1, rho <= 0)
  R <- 1 - B
  if (rho == 0) return(P * S^alpha * R^(1 - alpha))
  out <- numeric(length(P + S + B))
  boundary <- S == 0 | R == 0
  out[boundary] <- 0
  out[!boundary] <- P[!boundary] *
    (alpha * S[!boundary]^rho + (1 - alpha) * R[!boundary]^rho)^(1 / rho)
  out
}

crf_gap <- function(P_a, S_a, B_a, P_b, S_b, B_b, alpha = 0.5) {
  c(biomedical = log(P_a / P_b),
    social = alpha * log(S_a / S_b),
    structural = (1 - alpha) * log((1 - B_a) / (1 - B_b)))
}

hp_candidates <- function(P, S, B, alpha = 0.5, rho = 0,
                          beta = c(0, 1, 1, 1, 0)) {
  stopifnot(length(P) == length(S), length(S) == length(B),
            all(is.finite(c(P, S, B))), all(P >= 0),
            all(S >= 0 & S <= 1), all(B >= 0 & B <= 1),
            length(beta) == 5)
  R <- 1 - B
  additive <- beta[1] + beta[2] * P + beta[3] * S - beta[4] * B
  interaction <- additive + beta[5] * P * S
  multiplicative <- P * S^alpha * R^(1 - alpha)
  ces <- crf(P, S, B, alpha = alpha, rho = rho)
  rel_beta <- c(1, 0.5, 0.5, 0.25, 0.25)
  exponent <- rel_beta[1] * P + rel_beta[2] * S + rel_beta[3] * R +
    rel_beta[4] * P * S + rel_beta[5] * P * R
  relational <- (1 - exp(-exponent)) / (1 - exp(-sum(rel_beta)))
  data.frame(additive, interaction, multiplicative, ces, relational)
}

# Dependency-free verification of axiomatic boundary cases.
stopifnot(crf(10, 1, 0) == 10)
stopifnot(crf(10, 0, 0) == 0)
stopifnot(crf(10, 1, 1) == 0)
stopifnot(crf(10, 0.8, 0.2) <= 10)
stopifnot(all.equal(hp_candidates(10, 0.8, 0.2)$multiplicative,
                    crf(10, 0.8, 0.2)))
