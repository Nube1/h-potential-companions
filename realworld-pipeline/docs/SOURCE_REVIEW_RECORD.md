# Source review record

Source reviewed: `H_Potential_Thesis_V12_source_2.zip`.

The archive was expanded and every line of every editable text source was traversed before implementation: LaTeX chapters and appendices, bibliography, Markdown audits/memos/protocols, Python verification modules/tests, R analysis scaffolds, CSV dictionaries, class/style files and the master thesis file. Compiled PDFs and figures were inventoried as binary artifacts; their editable source counterparts governed the implementation.

## Findings that changed the build

- The approved NHATS analytic extract is absent; the package therefore cannot claim or fabricate real results.
- The existing `analysis/03_empirical_comparison.R` is explicitly a scaffold and omits multiple imputation, survey design, competing risks, paired full-pipeline bootstrap, censoring-adjusted performance, calibration, decision curves and subgroup evaluation.
- Measurement must be fixed without future outcomes; the pipeline saves a frozen measurement artifact before prognostic fitting.
- All candidate models must use the same analytic sample, outcomes, horizon and split/resample.
- The geometric CRF must be compared against additive, conventional comparator and sensitivity forms, with predeclared alpha/rho and normalisation.
- A significant coefficient is insufficient; discrimination, calibration, Brier error, net benefit and subgroup behavior are separate questions.
- The thesis's rejection rules must be operational, and failure narrows or refutes the model rather than being hidden.
- Survey structure, missingness, censoring and competing risk remain publication-critical controls.
- Haemophilia is a translation roadmap, not an empirically validated disease-specific instrument.

## Implementation decision

The repository supplies a safe executable baseline and clear extension boundary. It does not mislabel median imputation as final multiple imputation, known-status horizon metrics as IPCW Uno estimates, a same-cohort split as external validation, or synthetic outputs as empirical findings. Those safeguards are intentional consequences of the thesis text.

