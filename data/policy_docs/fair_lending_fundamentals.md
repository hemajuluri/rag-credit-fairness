# Fair Lending Fundamentals (Reference Summary)

This is an original summary document written for this project's retrieval corpus — a plain-language reference on fair lending principles, not a reproduction of any regulatory text. Intended to give the RAG system realistic policy-style context to retrieve from.

## Core principle

Lenders must evaluate creditworthiness using criteria that predict repayment ability, and must not base credit decisions on protected characteristics such as gender, race, religion, marital status, age, or national origin. Two forms of discrimination are generally recognized:

- **Disparate treatment**: intentionally treating applicants differently based on a protected characteristic.
- **Disparate impact**: a facially neutral policy that produces a significantly different outcome across protected groups, without a legitimate, non-discriminatory justification.

## What a compliant credit explanation should do

1. **Identify the specific factors** that most influenced the decision (e.g., debt-to-income ratio, credit history length, payment delinquencies) — general or vague reasons are not considered adequate.
2. **Avoid factors** that are proxies for protected characteristics (e.g., zip code used as a stand-in for race, or income patterns that correlate strongly with gender in ways unrelated to actual repayment risk).
3. **Be consistent** across applicants with similar risk profiles — if two applicants with similar SHAP attribution patterns receive materially different explanation quality or tone, that inconsistency itself is a fairness signal worth auditing.

## Adverse action notices

When an application is denied, applicants are generally entitled to specific reasons for the denial (not just "did not meet credit criteria"). Common accepted reason categories include: insufficient income, high debt-to-income ratio, limited credit history, delinquent past accounts, and excessive existing obligations relative to income.

## Relevance to this project

This document exists in the retrieval corpus so the RAG system can ground generated explanations in these general principles — e.g., confirming that a generated explanation cites a specific, non-proxy factor rather than a vague or protected-characteristic-adjacent one.
