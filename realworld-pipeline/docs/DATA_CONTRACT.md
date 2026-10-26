# Canonical data contract

One row per participant at the frozen index date.

| Class | Canonical fields | Rule |
|---|---|---|
| Identity/design | `participant_id`, `survey_weight`, `psu`, `stratum` | Unique ID; positive weight; preserve design IDs |
| Covariates | `age`, `sex` | Age in years; declared sex reference |
| Outcomes | mortality/hospital time and event | Non-negative days; event exactly 0/1 |
| Domains | three indicators per domain | Numeric after missing recode; direction declared |
| Equity | `sex`, `race_group` | Predeclared categories; disclosure control required |
| Comparators | `frailty_index`, `comorbidity_index` | Computed independently of outcomes |

Never include names, addresses, dates of birth, record numbers or free text.

