# H-Potential Thesis — V3 Academic Revision Memo

## What was changed

### Formal mathematical reconstruction
Appendix A has been replaced with an axiomatic foundation grounded in
measurement theory, representation theorems, multi-attribute value theory,
CES production mathematics, health-capital theory, multidimensional
deprivation measurement and latent-variable theory. Unsupported claims that
the conceptual score was measured in years or machine-verified by absent code
were removed. Definitions, axioms, propositions, proofs, counterexamples,
failure conditions and plain-language explanations are now paired throughout.

### Philosophical, economic and policy integration
Chapter 2 now compares naturalist, action-theoretic, capability and
opportunity-based philosophies of health and derives two competing ontologies:
a biomedical capacity-ceiling model and a relational capability model. The
mathematical appendix formalises both. Economic analysis separates need,
effectiveness, cost-effectiveness and affordability, connects the framework to
distributional cost-effectiveness analysis, and prohibits interpreting the
latent score as a QALY or economic threshold. Policy implications are aligned
with WHO health-equity and global HTA principles. The revision separates a
globally proposed conceptual framework from its United States empirical
development setting and from jurisdiction-specific policy appraisal. South
Africa is retained as a demanding future translation case, not treated as the
boundary of the theory. Explicit safeguards address automated denial and the
individualisation of structural injustice.

### Global scope and jurisdictional translation
The thesis now distinguishes conceptual, measurement, predictive and decision
transportability. It does not infer global validity from NHATS. The formal
appendix indexes the policy evidence vector by jurisdiction and proves that
equal effects do not entail equal policy rankings when opportunity costs,
budgets, distributional priorities or legal duties differ. This makes the
global ambition conditional and empirically falsifiable.

Chapter 2 now introduces the jurisdiction-indexed evidence vector in full
rather than requiring the reader to reconstruct it from Appendix A. It defines
each coordinate, derives profile change, discounted incremental outcomes and
costs, incremental net monetary benefit, budget impact and group-disaggregated
distributional evidence, and works through a two-jurisdiction numerical
example. Appendix A retains its proper role by extending this argument through
the formal non-transportability proposition and proof.

### 0. Chapter 2 expanded as the thesis's critical intellectual foundation
The literature review now presents supporting and opposing positions in direct
dialogue. It adds verified scholarship on the empirical limits of capability
measurement, the contested operationalisation and clinical added value of
intrinsic capacity, reflective/formative misspecification, and the normative
effects of weighting and aggregation. It also states five specific gaps that
H-Potential attempts to address and gives an explicit failure condition for
each. These are conditional research contributions, not claims of established
superiority.

This revision moves the thesis away from a protocol-like, formula-first presentation and towards a defensible doctoral research argument.

### 1. Literature review rebuilt around an intellectual argument
The literature chapter now:
- starts from the distinction between health state, functioning, intrinsic capacity, capability and future outcome;
- positions QALYs, DALYs and health expectancy as established traditions rather than straw-man comparators;
- treats frailty and intrinsic capacity as major conceptual predecessors;
- incorporates capability measures (ICECAP) and social vulnerability indices;
- explicitly engages the ATHLOS latent health metrics and their use of CFA/IRT;
- incorporates life-course and social-determinants theory;
- introduces intersectionality as a measurement and interpretation problem rather than a decorative citation;
- narrows the novelty claim to a testable integration of intrinsic capacity, enabling social conditions and structural constraints.

### 2. The novelty claim has been deliberately narrowed
The earlier claim that existing measures do not integrate biological, social and structural determinants was too broad. The literature contains multidimensional health metrics, latent health scores, intrinsic-capacity measures, capability instruments and social-vulnerability indices. The revised thesis acknowledges these directly and makes the contribution conditional on empirical evidence of incremental construct and prognostic value.

### 3. The mathematical formula has been demoted from “the metric” to a heuristic
The expression `(L × Q)S − B` contains strong assumptions about commensurability, multiplicativity and compensation that are not established by the literature. It is therefore retained as a conceptual mathematical representation, not as the empirical estimator.

The empirical $H_{potential}$ is now a latent score. It is not described as literal years of healthy life unless a separate calibration study establishes that interpretation.

### 4. Measurement theory has been strengthened
The methods now explicitly distinguish reflective from formative measurement. This is essential because biomedical manifestations such as function and cognition are conceptually different from education, income, housing and access barriers. The thesis no longer assumes that every indicator is a reflective manifestation of one latent trait.

Competing structures are now pre-specified:
1. correlated three-factor model;
2. second-order model;
3. bifactor/general-specific sensitivity model where identification permits.

A higher-order score is retained only if the empirical evidence supports it.

### 5. NHATS survey design is now explicit
The methods now require the correct NHATS weights, strata and clustering variables and acknowledge the special issues associated with Round 5 replenishment and longitudinal follow-up.

### 6. Prognostic interpretation is more disciplined
Hospitalisation is no longer treated as a simple proxy for worsening health. Death is recognised as a competing event for hospitalisation, and the distinction between association, prediction and causation is made explicit.

### 7. Equity is treated as a validation question
Measurement invariance, calibration and discrimination across relevant demographic and socioeconomic groups are now part of the empirical logic rather than an afterthought.

## What cannot honestly be completed yet

The archive does **not contain the approved NHATS analysis extract**. The preflight script explicitly expects an external `nhats_analysis.csv` file. The analysis scripts are also scaffolds and state that survey design, multiple imputation, competing-risk analysis, full-pipeline bootstrap, calibration and subgroup analyses still need implementation.

Therefore no numerical result, hazard ratio, model-fit statistic or claim of successful validation has been invented in this revision.

Before submission, the following must be completed from the real analysis:

1. freeze the NHATS cohort and variable dictionary;
2. execute the data-quality and overlap audits;
3. fit and compare the measurement models;
4. assess reliability, factor determinacy and invariance;
5. lock the measurement model before accessing prognostic results;
6. run the two endpoint-specific survival analyses;
7. perform full-pipeline internal validation;
8. generate calibration and decision-curve outputs;
9. complete subgroup/equity analyses;
10. replace every remaining results placeholder with actual output;
11. rewrite the abstract, discussion and conclusions from the final results;
12. perform a clean-environment reproducibility run;
13. conduct a final reference, quotation, citation and similarity audit.

## Turnitin position

The objective is **not** to manipulate Turnitin or write around a similarity detector. The academically defensible route is to produce an independently reasoned thesis, cite the underlying literature accurately, use quotation only where genuinely necessary, and distinguish the author's synthesis from prior claims. A Turnitin score cannot be guaranteed from this environment because the institutional Turnitin corpus and settings are not available here.
