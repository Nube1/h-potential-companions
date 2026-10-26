# Markdown recommendation implementation trace

| Recommendation | Thesis implementation | Code implementation |
|---|---|---|
| Mark WHO relation as conceptual, not estimable | Conceptual FA relation and explicit caveat | No WHO formula is fitted |
| Define H-Potential before selecting an equation | General model class with all symbols defined | Explicit biomedical, enablement and burden inputs |
| Separate framework from heuristic | Original heuristic retained and interrogated as a hypothesis | It is not the default estimator |
| Compare functional forms | Additive, interaction, multiplicative and CES equations | Same family in R and Python; scaffold compares all four plus SEM |
| Explain complementarity | First and cross-partial derivatives interpreted | Interaction is explicit and separately testable |
| Connect construct to measurement | Reflective and formative equations distinguished | Outcomes remain downstream |
| Clarify concepts | Concept-role table covers clinical, social, outcome and economic concepts | Data roles remain explicit |
| Make the argument falsifiable | Functional-form rejection is separated from construct rejection | No validation result is invented without NHATS data |

The package still requires the approved de-identified NHATS extract and frozen
source-variable mapping before empirical estimates can be produced.
