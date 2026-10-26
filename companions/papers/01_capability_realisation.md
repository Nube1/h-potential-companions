# Capability realisation under contextual bottlenecks: a bounded model for multidimensional health potential

Ronald Rethabile Malamule Mahomane

Research manuscript draft. Author affiliation, co-authorship and submission declarations require author confirmation. Companion to H_Potential_Thesis_V12_4; no empirical NHATS or haemophilia validation is claimed.

## Abstract

Health capacity and the conditions under which it can be exercised are distinct objects. A scalar that combines them therefore commits to a rule about substitution, whether that rule is stated or hidden in a weighted sum. This paper specifies a bounded capability-realisation function with biomedical capacity P, social enablement S and structural freedom R. We derive the geometric form from proportionality in capacity, normalisation and constant contextual elasticities, then embed it in a non-compensatory constant-elasticity-of-substitution family. The derivations establish boundary behaviour, complementary marginal effects and an exact decomposition of individual log-score differences. Reproducible numerical examples show how equal contextual averages conceal unequal bottlenecks. With equal contextual weights, profiles (0.70, 0.20) and (0.45, 0.45) have the same arithmetic contextual score, but geometric scores of 0.3742 and 0.4500. The accompanying implementation evaluates the family in the log domain and handles exact zeros without an arbitrary floor. The contribution is a transparent specification and verification framework, using established aggregation tools, rather than a claim to have discovered a new mean. Cardinal units, normative weights, measurement validity and causal intervention effects remain separate empirical obligations. The model provides a falsifiable representation of capacity conversion; it does not by itself identify capability sets, establish clinical validity or justify treatment allocation.

Keywords: capability; health measurement; contextual bottlenecks; geometric aggregation; non-compensation; reproducibility.

## 1. Introduction

Health measurement serves several purposes that should not be conflated. Describing current functioning, predicting death, valuing a health gain and representing opportunities for functioning are different inferential tasks. The World Health Organization's healthy-ageing framework connects intrinsic capacity, environment and functional ability [1]. That relationship motivates a representation in which capacity and context remain distinguishable. It does not establish a particular equation, nor does it guarantee that capacity and context can each be measured on ratio scales.

The thesis underlying this paper [3] proposes biomedical, social and systemic domains. Its earlier heuristic, (LQ)S-B, expresses a useful concern about conversion but mixes subtraction and multiplication without supplying compatible scales. It can also produce negative values. Replacing that heuristic requires more than attaching a new name to a composite. The replacement must state what complete obstruction means, whether good context may exceed a biomedical ceiling, and which trade-offs are permitted.

We study a capacity-ceiling specification: context changes the proportion of a defined capacity that is realised. This is an assumption about the operational construct, not a biological law. Assistive technologies that alter capacity itself may change P; a relational theory in which context constitutes capability may require a different model altogether. The present paper makes the narrower specification explicit and supplies proofs and executable checks. Companion papers examine measurement uncertainty and policy aggregation as separate problems.

## 2. Objects, scales and assumptions

Let P be nonnegative capacity measured in an explicitly declared unit U. Let S and R lie in [0,1], with larger values denoting more social enablement and structural freedom. If B is normalised structural burden, R=1-B. All three quantities must refer to the same measurement occasion and target construct. A domain score of zero means a substantively defined limiting state; a sample minimum is not automatically that state.

We require dimensional coherence, H(P,1,1)=P, monotonicity in each input, a capacity ceiling 0<=H<=P, and essential contextual domains: H(P,0,R)=H(P,S,0)=0. The last condition imposes total obstruction at a contextual boundary. It is contestable. A domain interpreted merely as one source of support may not be essential if alternatives exist. Evidence of functioning despite a measured zero may indicate incorrect anchors or failure of the axiom.

Unit equivariance means H(cP,S,R)=cH(P,S,R) for c>0. It does not mean that arbitrary monotone transformations of S and R preserve scores or rankings. For example, squaring one domain changes relative shortfalls. Replicating every population observation is a separate invariance of a population aggregator, not a synonym for changing measurement units.

P can be assigned QALY units only when its underlying life-duration and health-utility construction supports that interpretation. A standardised factor score has no such unit. In this paper's computational examples P=1 in arbitrary capacity units. Scores from different disease-specific instruments cannot be compared merely because both have been rescaled to [0,1].

## 3. Characterisation and admissible family

### 3.1 Geometric realisation

Assume H is positive and continuously differentiable on P>0 and S,R>0. Require proportionality in P and constant contextual elasticities alpha and 1-alpha, where 0<alpha<1. Then:

```equation
H(P,S,R) = P S^alpha R^(1-alpha).                                      (1)
```

To prove the claim, proportionality gives H=P g(S,R). The restriction d log(g)/d log(S)=alpha integrates to log(g)=alpha log(S)+c(R). The second elasticity gives c(R)=(1-alpha)log(R)+C. Normalisation at S=R=1 makes C=0. Exponentiation establishes (1), uniquely under these additional restrictions. The boundary extension follows by continuity.

The basic axioms alone do not identify this form. Constant elasticities do the identifying work. Nor does constant elasticity imply constant marginal returns: for 0<alpha<1 the marginal social return decreases with S, holding other inputs fixed. A test of constant elasticity must examine proportional response, rather than use the presence of diminishing marginal returns as a rejection criterion.

### 3.2 Substitution sensitivity

The CES construction uses an established generalised mean [2]:

```equation
M_(alpha,rho)(S,R) = [alpha S^rho + (1-alpha)R^rho]^(1/rho), rho<0.
H_(alpha,rho) = P M_(alpha,rho);  M_(alpha,0) = S^alpha R^(1-alpha).     (2)
```

For positive inputs, generalised-mean inequalities place M between min(S,R) and max(S,R). It therefore lies in [0,1]. Monotonicity follows by differentiating the powered sum; multiplying by P preserves it. As either contextual input tends to zero with rho<0, the corresponding negative power diverges and M tends to zero. The geometric limit has the same property. Normalisation follows by setting both inputs to one. These observations prove the stated axioms for rho<=0. As rho approaches negative infinity the mean tends to min(S,R).

Figure 1 draws the level sets of the contextual mean for four members of the family. For rho<=0 every level set bends towards the axes and the mean is zero on them, so a shortfall in one contextual domain cannot be offset by the other; only the arithmetic comparator, rho=1, has straight level sets that cross the axes. Figure 2 plots the mean against rho for the three Table 1 profiles. Each curve is non-decreasing, passes through the geometric mean at rho=0 and approaches min(S,R) as rho becomes large and negative. The balanced profile is unaffected by rho, while the most unbalanced profile changes most, which is why the choice of rho must be declared rather than tuned.

![Figure 1. Level sets of the contextual mean M(S, R) with equal weights for four members of the CES family. Markers are the three Table 1 profiles with the same arithmetic mean of 0.45. Curved level sets that meet the axes at zero show that a shortfall in one domain cannot be offset by the other.](../figures/proof_01_level_sets)

![Figure 2. The CES contextual mean against rho for the three Table 1 profiles. Dotted lines mark min(S, R), the bottleneck limit. The balanced profile is flat; the more unbalanced the profile, the more the choice of rho matters.](../figures/proof_02_ces_family)

The case rho=1 is the weighted arithmetic comparator. It deliberately violates essential enablement: at S=0 and R>0 it remains positive. Including this comparator is necessary to expose the consequence of the boundary axiom. Choosing rho on mortality outcomes changes the function into an outcome-selected predictor; it cannot then be presented as an independently specified measurement rule.

## 4. Consequences and interpretive limits

### 4.1 Marginal effects

On the strictly positive interior, the geometric derivatives are:

```equation
H_P = H/P;  H_S = alpha H/S;  H_R = (1-alpha)H/R.
H_PS = alpha S^(alpha-1) R^(1-alpha) > 0.
H_PR = (1-alpha) S^alpha R^(-alpha) > 0.                               (3)
```

Capacity and context are complementary in the precise sense of positive cross-partials. This does not prove intersectionality as a social theory. Intersectional inquiry also concerns institutions, power, identity, histories and heterogeneous mechanisms that cannot be inferred from a positive interaction derivative. The formula is one representation of compounding disadvantage, with a scope narrower than the theory that motivated it.

Figure 3 shows the ceiling: the realised share H/P never exceeds one, reaches one only when S=R=1 and is zero whenever either contextual domain is zero. Because the figure is drawn with P=30 in arbitrary units, it also illustrates unit equivariance; the plotted share is the same for any unit of P. Figure 4 shows the marginal effect of social enablement, H_S=alpha H/S, for three levels of structural freedom. The analytic curve and central finite differences agree to within 1e-6, and the gain from a unit of social enablement is larger when structural freedom is higher, which is the positive cross-partial in (3).

![Figure 3. Share of biomedical capacity realised, H/P, against social enablement for four levels of structural freedom (P = 30 in arbitrary units). The ceiling at one is reached only with both domains at one.](../figures/proof_03_capacity_ceiling)

![Figure 4. Analytic marginal effect of social enablement (lines) and central finite differences (open markers) for three levels of structural freedom, P = 1 and alpha = 0.5. The same gain in S is worth more when structural freedom is high.](../figures/proof_05_marginal_effects)

At S=0 or R=0 some marginal derivatives are singular or direction dependent. The software refuses interior-gradient requests there. It returns a zero score at the boundary and leaves finite intervention analysis to explicit before-and-after calculations. Silently replacing zeros with a small positive constant would change the model and conceal the singularity.

### 4.2 Balance penalty

Weighted arithmetic-geometric mean inequality gives:

```equation
P S^alpha R^(1-alpha) <= P[alpha S+(1-alpha)R].                         (4)
```

For P>0 equality holds exactly when S=R. For P=0 both sides vanish whatever the context. The difference measures the geometric penalty relative to this particular arithmetic comparator. It is not a loss measured against an observed counterfactual and must not be interpreted as an identified health gain available to a service.

Figure 5 makes the penalty concrete. Holding the arithmetic mean of S and R at 0.45, it moves the two domains apart by d. The arithmetic score stays at 0.45, while the geometric score falls along sqrt(0.45^2-d^2), and the harmonic and bottleneck scores fall faster. The marked profiles are the rows of Table 1.

![Figure 5. H against the gap between the two context domains when their arithmetic mean is held at 0.45. Markers are the Table 1 profiles. Every non-compensatory member penalises imbalance.](../figures/proof_04_balance_penalty)

### 4.3 Exact log decomposition

For two positive, comparable profiles a and b:

```equation
log(H_a/H_b) = log(P_a/P_b) + alpha log(S_a/S_b)
                             + (1-alpha)log(R_a/R_b).                 (5)
```

The result is algebraic and contains no residual. Components can oppose one another, making percentage contributions negative or greater than 100%; when the total gap is zero percentages are undefined. The decomposition describes a scoring identity rather than causal attribution.

Figure 6 applies (5) to two profiles that differ only in structural freedom, 0.85 against 0.20, with the same P and S. The biomedical and social parts are zero and the whole gap of 0.723 in log H is structural; the parts sum to the directly computed log ratio to twelve decimal places.

![Figure 6. Exact log decomposition of the gap between two profiles that differ only in structural freedom (0.85 against 0.20), from results/gap.csv. The whole gap is structural, as it should be.](../figures/proof_06_gap_decomposition)

For populations, a valid direct extension decomposes differences in weighted mean log scores, provided weights sum to one in each group and scales are comparable. It does not generally decompose the log ratio of arithmetic mean H values by substituting arithmetic mean domains. Nonlinearity and within-group dependence prevent that shortcut. Group-average policy conclusions require individual calculations followed by the intended population aggregation.

## 5. Computational verification and results

The implementation hpotential/crf.py evaluates negative-rho means with a log-sum-exp calculation. Near rho=0 it uses a second-order cumulant expansion of the log mean to avoid division of two cancellation-prone small quantities. Exact zero inputs are handled by their mathematical limits. rho=-infinity requests the exact minimum rather than a large finite approximation. Positive rho remains available as an explicitly compensatory comparator.

Table 1 is generated from results/aggregation.csv. It holds P=1, alpha=0.5 and an arithmetic contextual average of 0.45 in every row. All rows are constructed mathematical profiles, not participants.

| S | R | Arithmetic H | Geometric H | Harmonic H | Bottleneck H |
|---|---|---|---|---|---|
| 0.45 | 0.45 | 0.4500 | 0.4500 | 0.4500 | 0.4500 |
| 0.60 | 0.30 | 0.4500 | 0.4243 | 0.4000 | 0.3000 |
| 0.70 | 0.20 | 0.4500 | 0.3742 | 0.3111 | 0.2000 |
| 0.89 | 0.01 | 0.4500 | 0.0943 | 0.0198 | 0.0100 |

An independent analytical check reproduces the harmonic formula 1/(alpha/S+(1-alpha)/R). Grid checks cover normalisation, essential boundaries, capacity limits, monotonicity and ordering across rho. Central finite differences verify geometric gradients. Unit conversion and the exact log decomposition are checked separately. These tests establish implementation consistency for their cases, not exhaustive proof or empirical validity.

The sensitivity table varies alpha over 0.25, 0.50 and 0.75 and rho over seven declared values. Reporting that surface exposes an ethical and empirical modelling choice that a single headline score would hide. Its purpose is robustness assessment, not selection of the parameter values yielding the most compelling narrative.

## 6. Empirical falsification and discussion

A future evaluation must freeze indicator directions, anchors and aggregation parameters before outcome analysis. It must compare the domain profile, an additive composite, a defensible latent score and the CRF on the same cohort and validation resamples. A prognostic coefficient alone cannot establish measurement validity. Failure to find useful prognostic differences narrows an applied claim; it does not disprove the mathematical identities.

Zero anchors deserve special scrutiny. If S means a limited support questionnaire rather than all effective enablement, its zero may not imply complete exclusion. The essential-domain axiom would then be substantively inappropriate. Likewise, redefining P to absorb every environmental aid would preserve the ceiling by definition while weakening the framework's ability to distinguish domains. The operational dictionary must expose these decisions.

The evidence here supports boundedness, sensitivity and reproducible computation. It does not establish that the constructs exist as reflective factors, that domain scores have meaningful ratios, or that improving one measured input causes the corresponding model gain. The specification's value lies in making those obligations visible. Established geometric and CES mathematics supplies the tools; the candidate contribution is their explicit, testable use within this health-capability model, with boundary and interpretive failures treated as scientific outcomes.

## 7. Conclusion

A capacity-ceiling CRF separates aggregation commitments from measurement and prognosis. Constant contextual elasticities identify the geometric form only after additional assumptions; essential boundaries identify a non-compensatory family without identifying a unique member. Exact log decompositions and executable verification make the consequences inspectable. Clinical and policy use requires independent evidence about measurement scales, transportability and intervention responses.

## Declarations and reproducibility

No patient records were analysed. Numerical examples are deterministic synthetic constructions. Faculty, Senate and ethics approvals are not asserted. This manuscript is a research draft for supervisory review; institutional requirements and authorship must be confirmed before submission. Code, tests and generated tables accompany the paper. Run python analysis/reproduce.py and python -m unittest discover -s tests -v. No clinical dosing function is supplied.

## References

[1] World Health Organization. World report on ageing and health. Geneva: WHO; 2015. ISBN 9789241565042. https://www.who.int/publications/i/item/9789241565042

[2] Arrow KJ, Chenery HB, Minhas BS, Solow RM. Capital-labor substitution and economic efficiency. Review of Economics and Statistics. 1961;43(3):225-250. https://doi.org/10.2307/1927286

[3] Mahomane RRM. Development and Internal Validation of the H-potential: A Multidimensional Health Metric for Predicting Health Outcomes. H_Potential_Thesis_V12_4, supplied thesis draft, 283 pages. Unpublished; title-page date is not treated as evidence of completed research.
