# Strengthening map: folding the master document into the thesis

The master document *"The Axiomatic and Empirical Foundations of the H-potential
Framework"* is the definitive integrated statement. This file records what has been
folded in, and where the remaining sections strengthen each chapter, so the rollout
can proceed chapter by chapter using the format in
`CHAPTER_FORMAT_AND_EXPANSION_PROTOCOL.md`.

## Done in this pass (central Chapter 4)

Chapter 4 (the Capability Realisation Model and Integrated Policy Calculation) now
carries, in the readable format (roadmap box, notation table, per-result reading
boxes):

- **The identification problem** (§4.2): $F_i=F_j \nRightarrow \mathcal{C}_i=\mathcal{C}_j$ — why a model is needed at all.
- **Two ontologies** (§4.4): capacity-ceiling $H^C=P\,M(S,R)$ vs relational $H^R=G$, making Axiom 4 an explicit, defended choice.
- **The integrated policy calculation** (§4.10): the eight-coordinate evidence vector $\mathcal{D}_j(a)$ with its coordinate table — the "policy calculation" the chapter title promises.
- **The Non-Transportability Theorem** (§4.11) with proof: equal effects do not entail equal policy rankings.
- **Explicit falsification thresholds** (§4.14.3): HTMT, factor-score determinacy, $\Delta C$, calibration slope, invariance.

The thesis builds clean: `pdflatex → bibtex → pdflatex ×2`, 188 pages, no undefined
references or citations.

## Remaining, by chapter (highest value first)

### Chapter 6 — Prognostic Performance (strongest remaining upgrade)
From master §4.3. Replace the plain Cox treatment of hospitalisation with a
**competing-risks** formulation, since death is a competing event:
- cause-specific hazards $h_{ik}(t)=h_{0k}(t)\exp(\beta_{\zeta,k}\hat\zeta_i+\gamma_k^\top Z_i)$ for $k\in\{$hospitalisation, death$\}$;
- cumulative incidence via the **Fine–Gray** subdistribution model, to avoid overestimating hospitalisation risk in frail cohorts.
Add the survey-design apparatus from §4.1 (PSU/strata/weights; Taylor linearisation or BRR) and the two-stage firewall wording from §4.2. Add a `reading` box after each of Uno's C, calibration and net benefit.

### Chapter 2 — Literature (conceptual spine)
From master §1.2. The **five-tradition taxonomy** (Biostatistical Naturalism, Action-Theoretic Holism, Capability, Deficit-Accumulation/Frailty, Intrinsic Capacity) with the comparison table (construct target | structure | limits | role in the argument). V9 already has a policy vector at §2.14.1; keep it as the introduction and let Chapter 4 be the formal home, cross-referencing rather than repeating.

### Chapter 3 — Formalism / Chapter 5 — Measurement
From master §3. Add: the **reflective-vs-formative** contrast table; the SEM equations $y_i=\tau+\Lambda\eta_i+\varepsilon_i$, $\eta_i=\alpha+\Gamma\zeta_i+\delta_i$, and $\Sigma(\theta)=\Lambda(\Gamma\Gamma^\top+\Psi_\delta)\Lambda^\top+\Theta$; the **three competing structures** table (correlated / second-order / bifactor); the **invariance pipeline** (configural → metric → scalar → strict) with the rule that scalar invariance is required before comparing latent means. A `reading` box after each.

### Chapter 7 — Translational (haemophilia)
From master §6. Add the **indicator table** with units and expected loading directions (HJHS, ABR, inhibitor titre, FISH; centre proximity, literacy, caregiver support; stockout days, per-capita IU, out-of-pocket share); the **Delphi consensus thresholds** ($\ge$80% relevance $\ge$7 and IQR $\le$2 to include; $\ge$70% $\le$3 to exclude); and, in Appendix C, the **SDE kinetics** (factor activity, inhibitor titre under ITI, gene-therapy transgene) and the **Fokker–Planck** population density, which V9's dynamical appendices already scaffold.

### Chapters 8–9 — Discussion / Conclusions
From master §7. Mirror the **four contributions** (axiomatic foundation, psychometric firewall, policy architecture, translational versatility) and reuse the falsification criteria as the closing test. Chapter 9 already leads with the realisation model.

## One cleanup to watch
Chapter 2 (§2.14.1) and Appendix A (§A.6) already contain a policy-vector and a
relational-ontology passage. With Chapter 4 now the formal home for both, trim the
earlier passages to a short introduction that cross-references Chapter 4, so the
thesis states each once and in its central place (definition before use).
