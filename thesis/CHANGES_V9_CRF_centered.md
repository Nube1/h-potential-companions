# V9 update: the Capability Realisation Model made the central contribution

This revision keeps everything in the V9 project and changes what the thesis
is *about*, so the axiomatic Capability Realisation Model (CRF) and the
integrated policy calculation are the centre, not an appendix.

## What changed

**Promoted from appendix to a main chapter.** The CRF, previously Appendix E
("Axiomatic Capability Realisation Model"), is now **Chapter 4, "The Axiomatic
Capability Realisation Model and Integrated Policy Calculation"**, placed right
after the Formalism chapter and before the empirical chapters. The appendices
are now A–D. (`thesis.tex`)

**Reframed the CRF chapter opening.** Its "Purpose and Status" now states it is
the central formal contribution, replacing the heuristic `(L·Q)S−B` with a model
derived from axioms; the line that called the SEM score "primary" now calls it
the measurement backbone the CRF is tested against. (`Chapters/appendix_E.tex`)

**Abstract.** Objective and Contribution now lead with the axiomatic realisation
model and the equity-sensitive policy calculation; the measurement and prognostic
work is described as the empirical test of it. (`Chapters/abstract.tex`)

**Introduction.** The mathematical-proposition section is retitled and rewritten
to present the CRF as the principled successor to the heuristic and the formal
heart of the thesis (with Eq. `H = P·Sᵅ·R¹⁻ᵅ`); the research-gap/contribution
section, the overarching aim, a new objective, a new hypothesis (H6), and the
thesis-structure paragraph all now centre the model and frame the empirical
chapters as its head-to-head test. (`Chapters/intro.tex`)

**Conclusions.** The "Contribution" section now names the realisation model as
the central contribution, adds a subsection describing it, and the "Combined
Contribution" places the falsifiable formal model at the centre with the
measurement, prognostic and translational work built to test it.
(`Chapters/implementation.tex`)

## Build notes

- Restored four content figures and the Wits logo that V9 shipped without
  (`Figures/`), so the project compiles.
- Added `\label{ch:literature}` and `\label{chap:crf}` for the new
  cross-references.
- Builds clean: `pdflatex → bibtex → pdflatex ×2`, 182 pages, no undefined
  references or citations.

## Not done (flag)

- The empirical head-to-head that the CRF's claim depends on (Appendix-E
  protocol) still needs the locked NHATS analysis; results remain placeholders.
- The CRF uses `R = 1 − B` and `P = L·Q`; the thesis body still also carries the
  SEM factor score as the measurement metric. The two are now explicitly framed
  as model-versus-comparator, but confirm that framing reads as you intend
  before submission.
- The References heading sits under Appendix D's numbering in the ToC; this is a
  pre-existing V9 layout quirk, not introduced here.
