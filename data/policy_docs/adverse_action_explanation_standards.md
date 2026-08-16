# Adverse Action Explanation Standards (Reference Summary)

Original summary document for this project's retrieval corpus, describing what makes a credit-denial explanation adequate versus inadequate. Written for this project, not copied from any external source.

## Adequate explanation checklist

An adverse action explanation should:
- Name the **top 2-4 specific factors** driving the decision, ranked by influence where possible
- Use **plain language** an applicant without financial training can understand
- Avoid **generic boilerplate** ("did not meet our lending criteria") as the sole reason
- Be **traceable** to the underlying model's actual decision logic (i.e., the stated reasons should match what the model's feature attribution — such as SHAP values — actually shows drove the decision)

## Common failure modes to watch for (relevant to fairness auditing)

- **Reason-output mismatch**: the LLM states a reason that doesn't correspond to the top SHAP-attributed features for that applicant — a fidelity failure.
- **Tone or specificity disparity across groups**: explanations for one gender cohort are systematically more detailed, more hedged, or more sympathetic in tone than for another, even when the underlying decision factors are equivalent — a fairness failure distinct from the decision itself being fair.
- **Readability disparity**: explanations vary significantly in reading difficulty across cohorts, which can create unequal practical access to understanding (and contesting) a decision.

## Why this matters for the RAG extension

If the RAG system retrieves and grounds explanations in documents like this one, the hypothesis is that explanations should become more consistent (same standard applied per applicant) and more specific (grounded in named factors rather than generic language) — both of which are testable against the dissertation's original fidelity and fairness metrics in Week 5.
