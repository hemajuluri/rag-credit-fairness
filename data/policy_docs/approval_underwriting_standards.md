# Approval Notice & Positive Feature Standards (Reference Summary)

Original reference document for this project's retrieval corpus, describing standards and requirements for compliant credit approval notices and positive factor explanations.

## Approval notice requirements

When an application for credit is approved, the notice provided to the applicant should:
- Highlight the **top positive financial factors** that supported the approval decision (e.g., low debt-to-income ratio, long established credit history, zero recent delinquencies)
- Ensure **plain-language transparency** so applicants understand what financial behaviors contributed positively to their creditworthiness
- Maintain **fidelity to the model decision**, ensuring that positive factors cited reflect the top positive feature attributions (SHAP values) from the underwriting classifier
- Avoid **vague or generic statements** (such as "applicant met minimum qualifications") in favor of specific, verifiable financial metrics

## Fair lending considerations for approval notifications

- **Consistency across cohorts**: Approval notices must maintain consistent detail, professional tone, and plain language clarity across demographic groups.
- **Prohibition of proxy or protected factors**: Approval notifications must never attribute positive decisions to demographic variables, location proxies, or non-financial attributes.
- **Traceability**: All positive factors cited must be traceable directly to the underlying model's SHAP feature attributions.
