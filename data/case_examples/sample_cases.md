# Sample Applicant Cases (Synthetic)

Synthetic, illustrative applicant cases for testing the retrieval and generation pipeline during development. These are NOT real applicant data — placeholders to validate the system end-to-end before running it against the dissertation's actual (already-anonymized) dataset in later weeks.

## Case 1
- Applicant: synthetic, gender=F
- Decision: Denied
- Top SHAP features: high debt-to-income ratio (0.42), short credit history (14 months), one delinquent account in past 12 months
- Expected explanation should reference: DTI ratio, credit history length, delinquency — NOT proxy factors

## Case 2
- Applicant: synthetic, gender=M
- Decision: Denied
- Top SHAP features: high debt-to-income ratio (0.44), short credit history (16 months), one delinquent account in past 12 months
- Note: near-identical risk profile to Case 1 — used as a matched pair to test explanation consistency across gender during the Week 5 fairness re-audit

## Case 3
- Applicant: synthetic, gender=F
- Decision: Approved
- Top SHAP features: low DTI ratio (0.18), long credit history (86 months), no delinquencies

## Case 4
- Applicant: synthetic, gender=M
- Decision: Approved
- Top SHAP features: low DTI ratio (0.19), long credit history (81 months), no delinquencies
- Matched pair with Case 3 for the same consistency-check purpose

## Usage note
Cases 1/2 and 3/4 are intentionally near-identical matched pairs across gender — this mirrors the dissertation's own matched-pair approach to isolating whether explanation *quality* differs by gender when the underlying *risk* doesn't.
