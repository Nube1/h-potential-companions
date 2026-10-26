# Chapter formatting and expansion protocol

A repeatable way to give every chapter of the thesis the structure, format and
readability of the anisotropic-star paper (the "Mahomane et al. 2026 Formalism").
Chapter 4 (the Capability Realisation Model) is the **worked template**: open it to
see each device in place. This file is the recipe for applying the same treatment
to the other chapters.

The scaffold is `hpformat.sty` (already loaded in `thesis.tex`). It gives you:

- `\begin{ataglance}[The argument in outline] ... \end{ataglance}` — the roadmap box.
- `\begin{reading}[What this says] ... \end{reading}` — the per-result plain-English box.
- `\begin{notationtable} Symbol & Name & Meaning \\ ... \end{notationtable}` — the definitions table.
- `\hpstep{Seed}{...}` — an inline labelled step, when a full box is too heavy.

## The three rules (from the exemplar)

1. **Definition before use.** No symbol, construct or measure appears before it has
   been defined. Every chapter that uses mathematics opens with a `notationtable`.
2. **Every equation is one of three things** — a *definition* (introduces a symbol),
   a *structural relation* (states a property the model must have), or a
   *consequence* (follows from earlier lines, naming which). State this once in the
   chapter's roadmap box, then keep to it.
3. **A reading after every result.** Immediately after each displayed equation or
   theorem, a `reading` box says what it means in the domain: "Clinically, this
   means…", "Statistically, this means…", "For policy, this means…". This is the
   single biggest readability gain.

## The five-part arc (Seed → Junction → Perturbation → Stability, plus a reading)

Map each chapter onto the same skeleton the paper uses. The health-measurement
translation of each physics term:

| Paper term | Meaning here |
|---|---|
| **Seed / baseline** | The null or equilibrium: the system with the effect switched off. |
| **Junction / boundary** | Where the model meets the world, and the condition it must satisfy there. |
| **Perturbation** | The first-order change: add a domain, a covariate, an intervention, a shock. |
| **Order parameter** | The one quantity that says how far the system has moved from the baseline. |
| **Stability check** | Does it survive: fit, identification, invariance, calibration, falsification. |

## How to find what to expand (mechanical, not a matter of taste)

Walk each chapter and flag, for expansion, anything that meets one of these:

- a displayed equation with no plain-English sentence within two lines of it → add a `reading` box;
- a paragraph longer than about eight lines of unbroken abstraction → split it, and add a worked one- or two-person example;
- a symbol used before it is defined → move its definition up, or into the `notationtable`;
- a claim stated without its condition ("this holds") → state the condition;
- a result whose consequence is not named ("it follows that") → name the equations used.

---

## Per-chapter application

Each entry gives the chapter's one job, its Seed/Junction/Perturbation/Stability,
the format devices to add, and the specific areas to expand.

### Chapter 1 — Introduction and Background
- **One job:** state the problem and the central claim, and promise the arc.
- **Seed:** health measured as current disease status. **Perturbation:** add context (enablement, structural freedom). **Junction:** the definition/boundary between a *determinant* of health and a *component* of capability. **Stability:** the conditions under which the proposed construct would be rejected.
- **Devices:** one `ataglance` box near the end giving the six-step map of the whole thesis (it currently ends with a prose "structure of the thesis" — convert that to the box). No notation table needed.
- **Expand:** the "mathematical proposition" section already forward-references the CRF; add a one-line `reading` after Eq. (1) `H=(L·Q)S−B` saying in words why it is only a heuristic.

### Chapter 2 — Theoretical Framework and Literature Review
- **One job:** the "taxonomy of solutions" — group the field into a few schools and locate the gap.
- **Seed:** the established traditions (QALY/DALY, health expectancy, frailty, intrinsic capacity, capability) each answering a different question. **Perturbation:** the specific integration the thesis proposes. **Junction:** the discriminant-validity boundary (the construct must not collapse into wellbeing, frailty or vulnerability). **Stability:** the five gaps, each with a failure condition.
- **Devices:** `ataglance` box listing the four/five schools and the gap; a small comparison **table** (tradition | question it answers | what it omits); `reading` boxes are optional here since it is prose.
- **Expand:** wherever a school is named, add one sentence on *why it does not isolate the tested variable* (the taxonomy's narrative function). Keep each school to a short paragraph.

### Chapter 3 — Formalism and Mathematical Specification
- **One job:** turn the construct into measurable objects and state the measurement assumptions.
- **Seed:** the correlated three-domain model (not a general factor a priori). **Junction:** the outcome-independence boundary — measurement is fixed before, and without, the outcomes. **Perturbation:** admitting a higher-order or bifactor structure. **Stability:** identification, factor determinacy, reflective-vs-formative correctness.
- **Devices:** open with a `notationtable` (domains, indicators, loadings, factor score); a `reading` box after each measurement equation ("Statistically, this means…"); state the definition/structural/consequence convention.
- **Expand:** the reflective-vs-formative passage — convert to a two-column contrast and a one-line example (function is reflective; income is formative). Add a worked two-person illustration of a domain score.

### Chapter 4 — Axiomatic Capability Realisation Model (DONE — the template)
- Already carries the `ataglance` roadmap, the `notationtable`, and five `reading` boxes (form, complementarity, gap decomposition, policy gradient, equity functional). Use it as the reference for the others.

### Chapter 5 — Results I: Measurement Model
- **One job:** report whether the construct can be measured.
- **Seed:** the correlated-domains baseline fit. **Perturbation:** the higher-order/bifactor alternatives. **Junction:** the fit cut-offs (CFI/TLI/RMSEA/SRMR). **Stability:** reliability, factor determinacy, measurement invariance across groups.
- **Devices:** `ataglance` box (what a reader will and will not learn here, given results are pending); a `reading` box after each fit/reliability table saying what the numbers would mean; keep the placeholder discipline.
- **Expand:** add a short "how to read this table" sentence under each results table; define every fit index at first use (or point to the notation table).

### Chapter 6 — Prognostic Performance
- **One job:** test whether the measured construct predicts, and compare representations head-to-head.
- **Seed:** age-and-sex baseline. **Perturbation:** add the construct / the CRF. **Junction:** the two co-primary endpoints and family-wise error. **Stability:** calibration, optimism correction, subgroup error, the CRF-vs-additive-vs-factor-score comparison.
- **Devices:** `notationtable` (hazard ratio, C, calibration slope, net benefit); a `reading` box after the Cox equation and after each performance measure; an `ataglance` box listing the comparison ladder (baseline → comparators → additive → factor score → CRF).
- **Expand:** state, in one plain sentence each, what discrimination, calibration and net benefit *mean* before the tables; make the head-to-head protocol a numbered list.

### Chapter 7 — Translational Application to Haemophilia
- **One job:** show the model travels by indicator substitution, as a proof of concept.
- **Seed:** the general-population construct. **Perturbation:** substitute disease-relevant indicators. **Junction:** measurement invariance (the invariant core that anchors the scale). **Stability:** the recalibration/validation roadmap.
- **Devices:** `ataglance` box; the mapping **table** (already present) kept and labelled representative; a `reading` box after the invariance statement explaining "recalibration, not a new construct".
- **Expand:** add the one-line condition for when substitution preserves the construct (≥2 invariant indicators per domain); keep every Delphi statement conditional.

### Chapter 8 — Discussion
- **One job:** say what the evidence permits and what it does not.
- **Seed:** the hypotheses as stated. **Perturbation:** the results (pending). **Junction:** the boundary between what is shown and what is claimed. **Stability:** limitations, alternative explanations.
- **Devices:** an `ataglance` box mapping each finding to the hypothesis it bears on; `reading` boxes optional. No new math.
- **Expand:** for each interpretation, add the sentence "this would follow only if …"; keep result-dependent passages as marked placeholders.

### Chapter 9 — Conclusions
- **One job:** name the central contribution and the aim-by-aim verdict.
- Already reframed to lead with the realisation model. **Devices:** one `ataglance` box summarising the four contributions with the CRF at the centre; keep the placeholders on empirical verdicts.

### Appendices A–D
- Heavy proofs live here, as in the paper. Each appendix: open with one line saying which main-text result it supports; keep the definition/structural/consequence labelling; a `reading` box is unnecessary in an appendix but a one-line "what this buys the main text" at the end of each is worth adding.

---

## Order of work (so two people do not collide)

1. Chapter 4 is done; use it as the reference.
2. Do the mathematical chapters next (3, then 6), because the `reading`-box treatment
   is where they gain the most and because their notation feeds the others.
3. Then the results and translational chapters (5, 7), which mostly need "how to read
   this" sentences and one `ataglance` box each.
4. Then the framing chapters (1, 2, 8, 9), which need roadmap boxes and taxonomy
   tables rather than equations.
5. Rebuild after each chapter (`pdflatex → bibtex → pdflatex ×2`) and read the
   changed pages, not just the log.

## What not to do

- Do not add a `reading` box that merely restates the equation in symbols; it must
  say what the result *means* for a patient, a population, or a budget.
- Do not let a device replace content: the boxes are for translation and navigation,
  not for hiding an unfinished derivation.
- Do not insert numbers into any `reading` box in a results chapter until the locked
  analysis provides them; keep the placeholders.
