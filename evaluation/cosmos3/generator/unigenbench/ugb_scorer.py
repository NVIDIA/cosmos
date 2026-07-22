# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import argparse
import ast
import difflib
import json
import os.path as osp
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml
from tabulate import tabulate

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import GatewayVLM

EXPLANATION_DICT = {
    "Relationship - Comparison": "Comparison of attributes between two entities",
    "Relationship - Composition": "An entity is composed of one or more other entities",
    "Relationship - Inclusion": "A container contains an entity; the container can also be a plane, e.g., a snake in a painting on a wall",
    "Relationship - Similarity": "Existence of similarities between different entities",
    "Compound - Imagination": "Things that are impossible in real life",
    "Compound - Feature Matching": "Different entities possess different types of attribute features",
    "Attribute - Size": "Assessment of the subject's size, height, length, thickness, width, or tallness/shortness",
    "Attribute - Expression": "Distinguishing expressions from facial actions; expressions must convey a clear emotion",
    "Attribute - Quantity": "Focuses on the challenge of depicting three or more items accurately",
    "Attribute - Material": "Evaluation of different material types and textures",
    "Attribute - Color": "Assessment of different colors",
    "Attribute - Shape": "Assessment of different shapes",
    "Entity Layout - Two-Dimensional Space": "Arrangement and positioning of entities in two-dimensional space",
    "Entity Layout - Three-Dimensional Space": "Arrangement and positioning of entities in three-dimensional space",
    "Action - Full-body (Character/Anthropomorphic)": "Full-body actions by characters or anthropomorphized entities, such as running, diving, breakdancing, swinging, or hanging upside down",
    "Action - Hand (Character/Anthropomorphic)": "Focuses on hand structure—checking if fingers are missing, broken, or distorted",
    "Action - Animal": "Actions performed by animals",
    "Action - Contact Interaction": "Physical interactions between entities",
    "Action - Non-contact Interaction": "For example, two people making eye contact—testing if the model can accurately depict such interactions",
    "Action - State": "A sustained state of an entity, typically expressed with a verb",
    "Grammar - Negation": "Tests the model's understanding of negation grammar",
    "Grammar - Pronoun Reference": "Tests if the model can resolve ambiguous pronoun references correctly",
    "Grammar - Consistency": "Evaluation of shared attributes among entities",
    "World Knowledge": "Covers knowledge of celebrities, architecture, basic domain knowledge, and internet slang. Celebrities with modern copyright risk should be avoided",
    "Style": "Art, painting, photography, design styles, and corresponding artist names",
    "Text Generation": "The text content model needed to accurately generate without any omissions or extra words",
    "Logical Reasoning": "Requires the model to deeply understand the intent and perform reasoning",
}


def load_prompt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


SYSTEM_PROMPT_FILE = osp.join(osp.dirname(osp.abspath(__file__)), "assets", "unigenbench_scorer_prompt.md")
SYSTEM_PROMPT = load_prompt(SYSTEM_PROMPT_FILE)

USER_PROMPT = """{try_verbose}Prompt: 「{prompt}」
Testpoints: {testpoint}

{explanation}

{test_explanation}

Produce the YAML output list using these exact testpoint identifiers, in this exact order:
{output_template}
"""

NED_MATCH_THRESHOLD = 0.1


def compute_ned(a: str, b: str) -> float:
    return 1.0 - difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def yaml_dq(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return f'"{escaped}"'


_FENCE_RE = re.compile(r"```\s*([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)


def strip_yaml_fence(raw: str) -> str:
    s = raw.strip()
    matches = _FENCE_RE.findall(s)
    if matches:
        yaml_blocks = [content for tag, content in matches if tag.lower() in ("yaml", "yml")]
        if yaml_blocks:
            return max(yaml_blocks, key=len).strip()
        return max((content for _, content in matches), key=len).strip()
    list_start = re.search(r"(?m)^\s*- ", s)
    if list_start:
        return s[list_start.start():].strip()
    return s


def load_yaml_list(raw: str) -> Optional[list]:
    try:
        data = yaml.safe_load(strip_yaml_fence(raw))
    except yaml.YAMLError:
        return None
    if not isinstance(data, list):
        return None
    return data


def coerce_verdict(v) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v == 1:
            return True
        if v == 0:
            return False
        return None
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "yes", "y", "1", "pass", "passed", "satisfied"):
            return True
        if low in ("false", "no", "n", "0", "fail", "failed", "unsatisfied"):
            return False
    return None


def normalize_item_keys(item: dict) -> dict:
    return {k.strip().lower(): v for k, v in item.items() if isinstance(k, str)}


def build_output_template(testpoints: list[str]) -> str:
    lines = []
    for p in testpoints:
        lines.append(f"- testpoint: {yaml_dq(p)}")
        lines.append('  analysis: "<your analysis>"')
        lines.append("  verdict: <true or false>")
    return "\n".join(lines)


def build_request(vlm: GatewayVLM, sample: dict, try_cnt: int = 0) -> Optional[dict[str, Any]]:
    metadata = sample["metadata"]
    testpoints = metadata["Testpoints"]
    test_descs = metadata["Testpoint Description"]

    for point in testpoints:
        if point not in EXPLANATION_DICT:
            print(f"Unknown testpoint: {point}")
            return None

    explanation = "Testpoint Definitions:「\n" + "\n".join(f"{p}: {EXPLANATION_DICT[p]}" for p in testpoints) + "\n」"
    test_explanation = (
        "Testpoint Descriptions:「\n" + "\n".join(f"{p}: {test_descs[i]}" for i, p in enumerate(testpoints)) + "\n」"
    )

    user_text = USER_PROMPT.format(
        try_verbose="" if try_cnt == 0 else f"[This is try {try_cnt}] ",
        prompt=sample["prompt"],
        testpoint=testpoints,
        explanation=explanation,
        test_explanation=test_explanation,
        output_template=build_output_template(testpoints),
    )

    return vlm.build_request(
        system_prompt=SYSTEM_PROMPT,
        content_prompt=user_text,
        image_bytes=sample["image_bytes"],
        image_fmt=sample.get("extension", "png"),
        temperature=0.1,
        max_tokens=8192,
    )


def parse_vlm_response(text: str, testpoints: list[str]) -> Optional[tuple[list[str], list[bool]]]:
    items = load_yaml_list(text)
    if items is None:
        print("parse_vlm_response: response is not a valid YAML list.")
        return None
    if len(items) != len(testpoints):
        print(f"parse_vlm_response: expected {len(testpoints)} items, got {len(items)}.")
        return None

    analyses: list[str] = []
    verdicts: list[bool] = []
    for expected, raw_item in zip(testpoints, items):
        if not isinstance(raw_item, dict):
            print("parse_vlm_response: item is not a YAML mapping.")
            return None
        item = normalize_item_keys(raw_item)
        tp = item.get("testpoint")
        analysis = item.get("analysis")
        verdict = coerce_verdict(item.get("verdict"))

        if verdict is None:
            print(f"parse_vlm_response: 'verdict' must be true or false, got {item.get('verdict')!r}.")
            return None

        if tp is None:
            tp = expected
        elif not isinstance(tp, str):
            print(f"parse_vlm_response: 'testpoint' must be a string, got {tp!r}.")
            return None

        if analysis is None:
            analysis = ""
        elif not isinstance(analysis, str):
            analysis = str(analysis)

        ned = compute_ned(tp, expected)
        if ned > NED_MATCH_THRESHOLD:
            print(f"parse_vlm_response: testpoint mismatch '{expected}' vs '{tp}' (NED={ned:.2f}).")
            return None

        analyses.append(analysis.strip())
        verdicts.append(verdict)
    return analyses, verdicts


def compute_final_metrics(score_breakdown: dict, verbose: bool = True) -> dict:
    big_class_stats = defaultdict(lambda: [0, 0])
    small_class_stats = defaultdict(lambda: [0, 0])

    for name, result in score_breakdown.items():
        if result is None:
            if verbose:
                print(f"Skipping {name}: no result")
            continue
        testpoints = ast.literal_eval(result["testpoints"])
        verdicts = ast.literal_eval(result["verdicts"])

        for testpoint, verdict in zip(testpoints, verdicts):
            big_class = testpoint.split("-", 1)[0].strip() if "-" in testpoint else testpoint.strip()
            small_class = testpoint.strip()
            big_class_stats[big_class][1] += 1
            small_class_stats[small_class][1] += 1
            if verdict is True:
                big_class_stats[big_class][0] += 1
                small_class_stats[small_class][0] += 1

    total_correct = sum(s[0] for s in small_class_stats.values())
    total_count = sum(s[1] for s in small_class_stats.values())

    def _to_stat(v):
        return {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1] if v[1] > 0 else 0}

    return {
        "overall_accuracy": total_correct / total_count if total_count > 0 else 0,
        "total_correct": total_correct,
        "total_count": total_count,
        "big_class_stats": {k: _to_stat(v) for k, v in big_class_stats.items()},
        "small_class_stats": {k: _to_stat(v) for k, v in small_class_stats.items()},
    }


def format_statistics(label: str, stats: dict) -> str:
    lines = [
        f"[{label}] Overall Statistics:",
        f"  - Overall Accuracy: {stats['overall_accuracy']:.2%} ({stats['total_correct']}/{stats['total_count']})",
        f"  - Primary Dimensions: {len(stats['big_class_stats'])}",
        f"  - Sub-Dimensions: {len(stats['small_class_stats'])}",
        "",
    ]

    big_data = [
        [k, v["correct"], v["total"], f"{v['accuracy']:.2%}"]
        for k, v in sorted(stats["big_class_stats"].items())
    ]
    lines += ["Primary Dimensions:", tabulate(big_data, headers=["Primary Dimension", "Correct", "Total", "Accuracy"], tablefmt="github"), ""]

    small_data = [
        [k, v["correct"], v["total"], f"{v['accuracy']:.2%}"]
        for k, v in sorted(stats["small_class_stats"].items())
    ]
    lines += ["Sub-Dimensions:", tabulate(small_data, headers=["Sub-Dimension", "Correct", "Total", "Accuracy"], tablefmt="github"), ""]

    return "\n".join(lines)


def split_score_breakdown_by_prefix(score_breakdown: dict) -> tuple[dict, dict]:
    orig, phi = {}, {}
    for key, value in score_breakdown.items():
        if key.startswith("orig"):
            orig[key] = value
        elif key.startswith("phi"):
            phi[key] = value
        else:
            print(f"Unknown prefix for image key: {key}, skipping")
    return orig, phi


def print_stats_from_json(score_final: dict, input_folder: str, result_path: str) -> None:
    stats = score_final["stats"]
    success_count = score_final.get("success_count", "N/A")
    score_text = "\n".join(
        [
            "=" * 60,
            "Summary:",
            f"Success: {success_count}",
            f"all: {stats['all']['overall_accuracy']:.2%} orig: {stats['orig']['overall_accuracy']:.2%} phi: {stats['phi']['overall_accuracy']:.2%}",
            f"Input folder:  {input_folder}",
            f"Result file:   {result_path}",
            "=" * 60,
        ]
    )
    score_breakdown_text = "\n".join(
        [
            "=" * 60,
            format_statistics("orig", stats["orig"]),
            "=" * 60,
            format_statistics("phi", stats["phi"]),
            "=" * 60,
            format_statistics("all", stats["all"]),
            "=" * 60,
        ]
    )
    print(f"\n{score_breakdown_text}")
    print(f"\n{score_text}")


def main() -> None:
    args = parse_arguments()

    image_folder = Path(args.image_folder).expanduser().resolve()
    prompt_file = Path(args.benchmark_prompt_json).expanduser().resolve()
    result_path = Path(args.output_score_json).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    params_disp = tabulate(
        [
            ("IMAGE_FOLDER", image_folder),
            ("BENCHMARK_PROMPT_JSON", prompt_file),
            ("MODELSTR", args.modelstr),
            ("OUTPUT_SCORE_JSON", result_path),
        ],
        tablefmt="simple",
    )
    print(f"\nParams:\n{params_disp}")

    if result_path.exists():
        print(f"Found existing results at {result_path}, skipping scoring")
        score_final = json.loads(result_path.read_text())
        print_stats_from_json(score_final, str(image_folder), str(result_path))
        return

    benchmark_data = json.loads(prompt_file.read_text())
    if isinstance(benchmark_data, dict) and "benchmark" in benchmark_data:
        benchmark_rows = benchmark_data["benchmark"]
    else:
        benchmark_rows = benchmark_data

    samples_todo = []
    missing_images = 0
    for row in benchmark_rows:
        index = row.get("index") or row.get("id")
        prompt = row.get("prompt_en") or row.get("prompt")
        sub_dims = row.get("sub_dims_en") or row.get("sub_dims")
        metadata = ast.literal_eval(sub_dims) if isinstance(sub_dims, str) else sub_dims
        image_path = image_folder / f"{index}_0.{args.extension}"
        if not image_path.exists():
            missing_images += 1
            continue
        samples_todo.append(
            {
                "image_path": image_path,
                "prompt": prompt,
                "metadata": metadata,
                "extension": args.extension,
                "try_num": 0,
            }
        )

    samples_total = len(samples_todo)
    if missing_images:
        print(f"Missing {missing_images} expected image(s) under {image_folder}")
    print(f"Total samples: {samples_total}")

    vlm = GatewayVLM(
        gateway_url=args.gateway_url,
        gateway_api=args.gateway_api,
        model_str=args.modelstr,
        num_concurrency=args.num_concurrency,
    )

    score_breakdown: dict[str, Any] = {}

    while samples_todo:
        samples_todo = [s for s in samples_todo if s["try_num"] < args.max_retry]
        if not samples_todo:
            break

        batch = samples_todo[: args.batch_size]
        samples_todo = samples_todo[args.batch_size :]
        print(f"Processing {len(batch)} samples...")

        for si in batch:
            try:
                si["image_bytes"] = Path(si["image_path"]).read_bytes()
            except OSError:
                si["image_bytes"] = None

        samples_error = []
        valid_batch = []
        for si in batch:
            if si["image_bytes"] is not None:
                valid_batch.append(si)
            else:
                si["try_num"] += 1
                samples_error.append(si)

        request_list = [build_request(vlm, s, s["try_num"]) for s in valid_batch]
        response_list = vlm.query(request_list, pbar_desc="VLM scoring")

        for si, response in zip(valid_batch, response_list):
            name = osp.basename(si["image_path"])
            testpoints = si["metadata"]["Testpoints"]
            parsed = parse_vlm_response(response, testpoints) if response else None
            if parsed is None:
                si.pop("image_bytes", None)
                si["try_num"] += 1
                samples_error.append(si)
            else:
                analyses, verdicts = parsed
                score_breakdown[name] = {
                    "testpoints": str(testpoints),
                    "analyses": str(analyses),
                    "verdicts": str(verdicts),
                    "raw_output": response,
                }

        samples_todo = samples_todo + samples_error

    vlm.close()

    score_breakdown = {k: score_breakdown[k] for k in sorted(score_breakdown.keys())}
    orig_breakdown, phi_breakdown = split_score_breakdown_by_prefix(score_breakdown)
    score_final = {
        "stats": {
            "orig": compute_final_metrics(orig_breakdown),
            "phi": compute_final_metrics(phi_breakdown),
            "all": compute_final_metrics(score_breakdown, verbose=False),
        },
        "judge_model": args.modelstr,
        "success_count": f"{len(score_breakdown)}/{samples_total}",
        "breakdown": score_breakdown,
    }

    result_path.write_text(json.dumps(score_final, indent=4))
    print_stats_from_json(score_final, str(image_folder), str(result_path))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute unigenbench metric using VLM judge")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--benchmark_prompt_json", type=str, required=True)
    parser.add_argument("--output_score_json", type=str, required=True)
    parser.add_argument("--gateway_url", type=str, required=True)
    parser.add_argument("--gateway_api", type=str, required=True)
    parser.add_argument("--modelstr", type=str, required=True)
    parser.add_argument("--extension", type=str, default="png")
    parser.add_argument("--num_concurrency", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1170)
    parser.add_argument("--max_retry", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    main()
