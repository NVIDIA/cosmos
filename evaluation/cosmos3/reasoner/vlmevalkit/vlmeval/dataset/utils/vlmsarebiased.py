from collections import defaultdict
from typing import Any

import pandas as pd
from tqdm import tqdm

from vlmeval.smp import load


def vlms_are_biased_process_results(org_file: str) -> list[dict[str, Any]]:
    meta = load(org_file)

    # TSV / CSV → DataFrame → list[dict]
    if isinstance(meta, pd.DataFrame):
        meta = meta.to_dict(orient="records")

    for item in tqdm(meta, desc="Processing results for VLM Are Biased benchmark"):
        pred = str(item.get("prediction", "")).strip()
        ground_truth = str(item.get("ground_truth", "")).strip()
        expected_bias = str(item.get("expected_bias", "")).strip()
        topic = item.get("topic", "unknown")

        pred_normalized = pred.lower().strip("{}").strip()
        gt_normalized = ground_truth.lower().strip("{}").strip()
        bias_normalized = expected_bias.lower().strip("{}").strip()

        is_correct = pred_normalized == gt_normalized
        matches_bias = pred_normalized == bias_normalized

        if not is_correct and not matches_bias:
            pred_numbers = "".join(c for c in pred_normalized if c.isdigit())
            gt_numbers = "".join(c for c in gt_normalized if c.isdigit())
            bias_numbers = "".join(c for c in bias_normalized if c.isdigit())

            if pred_numbers and gt_numbers:
                is_correct = pred_numbers == gt_numbers
            if pred_numbers and bias_numbers:
                matches_bias = pred_numbers == bias_numbers

        item["is_correct"] = is_correct
        item["matches_bias"] = matches_bias
        item["topic"] = topic

    return meta


def vlms_are_biased_aggregate_by_topic(
    detail_result: list[dict[str, Any]],
) -> dict[str, float]:

    if not detail_result:
        return {
            "overall_acc": 0.0,
            "macro_acc": 0.0,
            "bias_ratio": 0.0,
        }

    topic_correct = defaultdict(int)
    topic_bias = defaultdict(int)
    topic_total = defaultdict(int)

    for r in detail_result:
        topic = r.get("topic", "unknown")
        topic_total[topic] += 1
        if r.get("is_correct", False):
            topic_correct[topic] += 1
        if r.get("matches_bias", False):
            topic_bias[topic] += 1

    per_topic_acc = {
        t: topic_correct[t] / topic_total[t]
        for t in topic_total
    }

    per_topic_bias = {
        t: topic_bias[t] / topic_total[t]
        for t in topic_total
    }

    overall_acc = sum(topic_correct.values()) / sum(topic_total.values())
    macro_acc = sum(per_topic_acc.values()) / len(per_topic_acc)

    bias_ratio = sum(
        bool(r.get("matches_bias", False))
        for r in detail_result
    ) / len(detail_result)

    return {
        "overall_acc": overall_acc,
        "macro_acc": macro_acc,
        "overall_bias_ratio": bias_ratio,
        "macro_bias_ratio": sum(per_topic_bias.values()) / len(per_topic_bias),
        'bias': per_topic_bias,
        'acc': per_topic_acc,
    }
