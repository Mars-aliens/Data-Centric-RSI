"""Compare model-simulated judgments with independent human judgments."""

from __future__ import annotations

from collections import Counter
from typing import Any

from experiments.feedback.annotation import ISSUES, JUDGMENTS


def compare_feedback(model_rows: list[dict[str, Any]], human_rows: list[dict[str, Any]]) -> dict[str, Any]:
    model_by_id = {str(row.get("query_id") or ""): row for row in model_rows}
    human_by_id = {str(row.get("query_id") or ""): row for row in human_rows}
    if "" in model_by_id or "" in human_by_id:
        raise ValueError("feedback rows require non-empty query IDs")
    if len(model_by_id) != len(model_rows) or len(human_by_id) != len(human_rows):
        raise ValueError("feedback rows contain duplicate query IDs")
    if set(model_by_id) != set(human_by_id):
        raise ValueError("model and human feedback must cover exactly the same query IDs")

    labels = list(JUDGMENTS.values())
    confusion = {human: {model: 0 for model in labels} for human in labels}
    issue_matches = []
    actionable_pairs = []
    for query_id in sorted(model_by_id):
        model = model_by_id[query_id]
        human = human_by_id[query_id]
        if model.get("feedback_source") != "model_simulated":
            raise ValueError("model-side rows must use feedback_source=model_simulated")
        if human.get("feedback_source") != "live_human":
            raise ValueError("human-side rows must use feedback_source=live_human")
        if model.get("judgment") not in labels or human.get("judgment") not in labels:
            raise ValueError(f"invalid judgment for query {query_id}")
        confusion[human["judgment"]][model["judgment"]] += 1
        if model.get("issue_code") in ISSUES.values() and human.get("issue_code") in ISSUES.values():
            issue_matches.append(model["issue_code"] == human["issue_code"])
        if isinstance(model.get("actionable"), bool) and isinstance(human.get("actionable"), bool):
            actionable_pairs.append((model["actionable"], human["actionable"]))

    per_label_f1 = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in labels if actual != label)
        fn = sum(confusion[label][predicted] for predicted in labels if predicted != label)
        denominator = 2 * tp + fp + fn
        per_label_f1[label] = 2 * tp / denominator if denominator else 0.0
    count = len(model_by_id)
    exact = sum(confusion[label][label] for label in labels)
    actionable_tp = sum(model and human for model, human in actionable_pairs)
    actionable_fp = sum(model and not human for model, human in actionable_pairs)
    actionable_fn = sum(not model and human for model, human in actionable_pairs)
    return {
        "status": "analyzed",
        "paired_queries": count,
        "judgment_labels": labels,
        "judgment_confusion_matrix_human_rows_model_columns": confusion,
        "judgment_exact_agreement": exact / count if count else None,
        "judgment_macro_f1": sum(per_label_f1.values()) / len(labels),
        "judgment_per_label_f1": per_label_f1,
        "issue_code_exact_agreement": sum(issue_matches) / len(issue_matches) if issue_matches else None,
        "issue_code_paired_queries": len(issue_matches),
        "actionable_precision": actionable_tp / (actionable_tp + actionable_fp) if actionable_tp + actionable_fp else None,
        "actionable_recall": actionable_tp / (actionable_tp + actionable_fn) if actionable_tp + actionable_fn else None,
        "actionable_paired_queries": len(actionable_pairs),
    }
