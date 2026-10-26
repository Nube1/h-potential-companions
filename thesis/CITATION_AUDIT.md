# Citation audit: sourcing every equation and named method

Goal: the thesis should read as a textbook, where every equation and every
substantive method or claim traces to a source. This records what was found, what
was done, and what a careful reader should still check.

## What the audit found (before)

A mechanical pass over all chapters and appendices:

- **34 displayed equations; 32 had no citation within eight lines.**
- Named methods used without a source at the point of use, including: Cox
  proportional hazards, Harrell's and Uno's concordance, Fine–Gray / cause-specific
  hazards, Kaplan–Meier, the Brier score, calibration slope, decision-curve net
  benefit, bootstrap optimism, Rubin's rules and MICE, Schoenfeld residuals; CES and
  Cobb–Douglas forms, the AM–GM inequality, the Atkinson equally-distributed
  equivalent; the Fokker–Planck equation, Gronwall, Lipschitz/Picard–Lindelöf,
  Euler–Maruyama, Runge–Kutta, Michaelis–Menten and Hill kinetics; measurement
  invariance (configural/metric/scalar), factor determinacy, bifactor models,
  reflective vs formative indicators, WLSMV and polychoric correlations, HTMT,
  Fornell–Larcker, AVE, composite reliability, Hu–Bentler cut-offs; survey
  linearisation and BRR; INMB, budget impact, distributional CEA; Boorse,
  Nordenfelt, Sen, Venkatapuram, Rockwood–Mitnitski, WHO intrinsic capacity,
  ATHLOS, ICECAP, social vulnerability, intersectionality; TRIPOD, PROBAST, STROBE,
  COSMIN; Delphi; HJHS, emicizumab, gene therapy, immune tolerance induction.

## What was done

**1. Reference base.** `refs.bib` grew from 92 to 135 entries. The 43 additions are
canonical, real sources (no invented references): Atkinson 1970; Cobb & Douglas
1928; Hardy–Littlewood–Pólya 1934; Fine & Gray 1999; Kaplan & Meier 1958; Brier
1950; Vickers & Elkin 2006; Efron & Tibshirani 1993; Meredith 1993; Vandenberg &
Lance 2000; Putnick & Bornstein 2016; Grice 2001; Reise 2012; Diamantopoulos &
Winklhofer 2001; Edwards & Bagozzi 2000; Bollen 1989; Muthén 1984; Rhemtulla et al.
2012; Olsson 1979; Binder 1983; Lumley 2010; Wolter 2007; Stinnett & Mullahy 1998;
Sullivan et al. 2014; Michaelis & Menten 1913; Hill 1910; Risken 1996; Gronwall
1919; Øksendal 2003; Kloeden & Platen 1992; Butcher 2008; Coddington & Levinson
1955; Boorse 1977; Sen 1999; Searle et al. 2008; Cesari et al. 2018; Coast et al.
2008; Kuklys 2005; Hilliard et al. 2006; Oldenburg et al. 2017; Nathwani et al.
2011; DiMichele et al. 2007; Hasson, Keeney & McKenna 2000.

**2. PubMed verification.** The 15 biomedical/statistical additions indexed by
PubMed were matched by journal, year, volume, first page and first author, and each
resolved to a real record. PMIDs are stamped into the `.bib` `note` fields:
Vickers 2006 (17099194); Grice 2001 (11778682); Putnick 2016 (27942093); Searle 2008
(18826625); Cesari 2018 (29408961); Coast 2008 (18572295); Hilliard 2006 (16919083);
Oldenburg 2017 (28691557); Nathwani 2011 (22149959); Rhemtulla 2012 (22799625);
Stinnett 1998 (9566468); Sullivan 2014 (24438712); Reise 2012 (24049214);
DiMichele 2007 (17593277); Hasson 2000 (11095242).

**3. Wiring.** 99 citations inserted across 13 chapter and appendix files (94 by
the systematic pass, 5 by hand for the last externally-sourced equations). Rule
applied: each displayed equation is cited at the sentence that introduces it, to the
source of its form; each named method or tradition is cited at its first mention in
each chapter. Insertions skipped math mode, headings, captions and labels.

**4. Result.** Clean build (192 pages), 0 undefined citations, 0 undefined
references, 109 sources cited, 249 `\cite` commands. **Of the 34 displayed
equations, 30 now cite an external origin; the remaining 4 are the thesis's own
definitions** ($H_P=f(BP,SM,SB)$; the conceptual heuristic $(L\cdot Q)S-B$; its
time-varying form), for which the thesis itself is the source, consistent with the
stated convention that a definition introduces a symbol and is neither true nor
false.

## What a careful reader should still check

- The **non-PubMed sources** (mathematics, economics, statistics books and classic
  papers) carry standard bibliographic details from memory. They are real works;
  confirm volume and page numbers against Crossref or the publisher before
  submission. Highest priority: Gronwall 1919, Michaelis & Menten 1913, Hill 1910,
  Cobb & Douglas 1928, Binder 1983, Olsson 1979.
- Where a method has several canonical sources (for example Fine–Gray for
  competing risks, or Meredith and Vandenberg & Lance for invariance), the chapter
  cites the one whose result is used; add the second where an examiner in that
  field would expect it.
- Citations were placed at *first* mention per file. A textbook reader may want the
  source repeated at a second, distant use; add by hand where the gap is large.
- Re-run the audit after any new equation is added:

      python3 scripts/audit_citations.py   # or the inline script in the session log

## What not to do

- Do not add a citation that does not support the exact statement it follows; a
  wrong citation is worse than none.
- Do not cite the thesis's own definitions to an outside source; they are the
  source.
- Do not remove a `note = {PMID: ...}` field; it is the verification trail.
