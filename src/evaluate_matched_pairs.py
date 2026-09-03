"""
evaluate_matched_pairs.py

Day 5: Matched-Pair Retrieval Evaluation

Tests whether DocumentRetriever returns consistent policy chunks across
demographic matched pairs defined in data/case_examples/sample_cases.md:
  - Pair 1: Case 1 (F, denied) vs Case 2 (M, denied)
  - Pair 2: Case 3 (F, approved) vs Case 4 (M, approved)

Note: Gender is used ONLY for labeling in the evaluation report — gender
is NEVER passed into the query builder or retriever.
"""

from retrieval import DocumentRetriever, build_query_from_case


MATCHED_PAIRS = [
    {
        "pair_name": "Denied Pair (High-Risk DTI / Short Credit History)",
        "case_a": {
            "id": "Case 1",
            "gender": "F",
            "decision": "Denied",
            "features": [
                {"feature": "debt-to-income ratio", "value": "0.42", "direction": "high"},
                {"feature": "credit history length", "value": "14 months", "direction": "short"},
                {"feature": "delinquent accounts in past 12 months", "value": "1", "direction": "one"},
            ],
        },
        "case_b": {
            "id": "Case 2",
            "gender": "M",
            "decision": "Denied",
            "features": [
                {"feature": "debt-to-income ratio", "value": "0.44", "direction": "high"},
                {"feature": "credit history length", "value": "16 months", "direction": "short"},
                {"feature": "delinquent accounts in past 12 months", "value": "1", "direction": "one"},
            ],
        },
    },
    {
        "pair_name": "Approved Pair (Low-Risk DTI / Long Credit History)",
        "case_a": {
            "id": "Case 3",
            "gender": "F",
            "decision": "Approved",
            "features": [
                {"feature": "debt-to-income ratio", "value": "0.18", "direction": "low"},
                {"feature": "credit history length", "value": "86 months", "direction": "long"},
                {"feature": "delinquent accounts", "value": "0", "direction": "no"},
            ],
        },
        "case_b": {
            "id": "Case 4",
            "gender": "M",
            "decision": "Approved",
            "features": [
                {"feature": "debt-to-income ratio", "value": "0.19", "direction": "low"},
                {"feature": "credit history length", "value": "81 months", "direction": "long"},
                {"feature": "delinquent accounts", "value": "0", "direction": "no"},
            ],
        },
    },
]


def compare_results(res_a: list[dict], res_b: list[dict]) -> dict:
    sources_a = [r["source_file"] for r in res_a]
    sources_b = [r["source_file"] for r in res_b]

    same_sources_same_order = (sources_a == sources_b)

    set_a = set(sources_a)
    set_b = set(sources_b)
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    overlap_ratio = len(intersection) / len(union) if union else 1.0

    top1_score_a = res_a[0]["score"] if res_a else 0.0
    top1_score_b = res_b[0]["score"] if res_b else 0.0
    top1_score_gap = abs(top1_score_a - top1_score_b)

    flagged = not (set_a == set_b)

    return {
        "sources_a": sources_a,
        "sources_b": sources_b,
        "same_sources_same_order": same_sources_same_order,
        "source_overlap_ratio": overlap_ratio,
        "top1_score_gap": top1_score_gap,
        "flagged": flagged,
    }


def evaluate():
    retriever = DocumentRetriever()
    print("==================================================================")
    print("DAY 5: MATCHED-PAIR RETRIEVAL EVALUATION REPORT")
    print("==================================================================\n")

    all_consistent = True

    for pair in MATCHED_PAIRS:
        case_a = pair["case_a"]
        case_b = pair["case_b"]

        query_a = build_query_from_case(case_a["features"])
        query_b = build_query_from_case(case_b["features"])

        res_a = retriever.retrieve(query_a, top_k=3)
        res_b = retriever.retrieve(query_b, top_k=3)

        cmp = compare_results(res_a, res_b)

        print(f"--- {pair['pair_name']} ---")
        print(f"[{case_a['id']} ({case_a['gender']})] Query: {query_a}")
        for r in res_a:
            print(f"  [{r['score']:.3f}] {r['source_file']}: {r['text'][:80]}...")

        print(f"[{case_b['id']} ({case_b['gender']})] Query: {query_b}")
        for r in res_b:
            print(f"  [{r['score']:.3f}] {r['source_file']}: {r['text'][:80]}...")

        print("\nComparison Metrics:")
        print(f"  same_sources_same_order: {cmp['same_sources_same_order']}")
        print(f"  source_overlap_ratio:    {cmp['source_overlap_ratio']:.2f}")
        print(f"  top1_score_gap:          {cmp['top1_score_gap']:.4f}")

        if cmp["flagged"]:
            print("  Status: ⚠ FLAG (Retrieved source document mismatch between matched pair)\n")
            all_consistent = False
        else:
            print("  Status: ✓ CONSISTENT (Identical source documents retrieved)\n")

    print("==================================================================")
    if all_consistent:
        print("OVERALL RESULT: PASS — All matched pairs retrieved consistent sources.")
    else:
        print("OVERALL RESULT: FAIL / WARNING — Flagged source mismatches detected.")
    print("==================================================================\n")

    return all_consistent


if __name__ == "__main__":
    evaluate()
