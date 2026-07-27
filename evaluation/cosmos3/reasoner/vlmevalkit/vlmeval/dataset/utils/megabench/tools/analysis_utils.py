import ast
import json
from collections import defaultdict
from typing import Any, Dict, List

_DATASET_CACHE = {}
_SCORING_FUNCTIONS_CACHE = {}

def _load_hf(subset_name: str) -> List[Dict[str, Any]]:
    """
    Load the HF dataset for the given subset name.
    """
    if subset_name in _DATASET_CACHE:
        return _DATASET_CACHE[subset_name]
    
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/MEGA-Bench", subset_name)["test"]
    task_dict = {}
    for sample in dataset:
        task_name = sample["task_name"]
        if task_name not in task_dict:
            task_dict[task_name] = []
        task_dict[task_name].append(sample)

    _DATASET_CACHE[subset_name] = task_dict
    return task_dict


def _get_scoring_functions():
    if _SCORING_FUNCTIONS_CACHE:
        return _SCORING_FUNCTIONS_CACHE
    
    core_data = _load_hf("core")
    open_data = _load_hf("open")
    
    core_scoring_functions = {}
    open_scoring_functions = {}
    
    for task_name, task_samples in core_data.items():
        core_scoring_functions[task_name] = ast.literal_eval(
            task_samples[0]["metric_info"]
        )
    
    for task_name, task_samples in open_data.items():
        open_scoring_functions[task_name] = ast.literal_eval(
            task_samples[0]["metric_info"]
        )
    
    _SCORING_FUNCTIONS_CACHE["core"] = core_scoring_functions
    _SCORING_FUNCTIONS_CACHE["open"] = open_scoring_functions
    
    return _SCORING_FUNCTIONS_CACHE


def _determine_eval_style(task):
    """
    Determine the evaluation style (rule or llm) for a task.
    """
    scoring_functions = _get_scoring_functions()
    core_scoring_functions = scoring_functions["core"]
    open_scoring_functions = scoring_functions["open"]
    
    task_name = task["task_name"]
    if task_name in core_scoring_functions:
        metric_info = core_scoring_functions[task_name]
    elif task_name in open_scoring_functions:
        metric_info = open_scoring_functions[task_name]
    else:
        raise ValueError(f"Task '{task_name}' not found in either core or open datasets")
    
    all_task_metrics = list(metric_info["field_score_function"].values())
    eval_type = (
        "rule"
        if (
            "gpt_4o_as_judge" not in all_task_metrics
            and "ascii_art_gpt4o_judge" not in all_task_metrics
        )
        else "llm"
    )
    return eval_type


def clear_cache():
    """
    Clear the cache and force re-loading the dataset.
    """
    global _DATASET_CACHE, _SCORING_FUNCTIONS_CACHE
    _DATASET_CACHE.clear()
    _SCORING_FUNCTIONS_CACHE.clear()


def task_list_refine(task_list):
    task_results = []
    for task in task_list:
        if "mean_task_score" in task and task["mean_task_score"] != -1:
            num_demo = 1 if len(task["example_contents"]) > 0 else 0
            task_results.append(
                {
                    "name": task["task_name"],
                    "score": task["mean_task_score"],
                    "eval_type": task.get("eval_type", _determine_eval_style(task)),
                    "num_demo": num_demo,
                    "num_query": len(task["query_response"]),
                }
            )
    return task_results


def derive_keyword_stats(task_results_with_meta, include_per_task_info=False):
    """
    Calculate keyword-based statistics for skills, input_format, and output_format.
    """
    skills_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0, "num_samples": 0, "tasks": []})
    input_format_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0, "num_samples": 0, "tasks": []})
    output_format_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0, "num_samples": 0, "tasks": []})
    input_num_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0, "num_samples": 0, "tasks": []})
    app_stats = defaultdict(lambda: {"count": 0, "total_score": 0.0, "num_samples": 0, "tasks": []})

    for task_name, task in task_results_with_meta.items():
        task_name = task.get("original_task_name", "Unknown Task")
        score = task.get("score", 0.0)
        num_samples = task.get("num_query", 0) + task.get("num_demo", 0)

        if score == -1:
            continue

        for skill in task.get("skills", []):
            skills_stats[skill]["count"] += 1
            skills_stats[skill]["total_score"] += score
            skills_stats[skill]["num_samples"] += num_samples
            if include_per_task_info:
                skills_stats[skill]["tasks"].append((task_name, score))

        for stat_dict, key in [
            (input_format_stats, "input_format"),
            (output_format_stats, "output_format"),
            (input_num_stats, "num_input"),
            (app_stats, "app")
        ]:
            if value := task.get(key):
                stat_dict[value]["count"] += 1
                stat_dict[value]["total_score"] += score
                stat_dict[value]["num_samples"] += num_samples
                if include_per_task_info:
                    stat_dict[value]["tasks"].append((task_name, score))

    all_stats = {
        "skills": skills_stats,
        "input_format": input_format_stats,
        "output_format": output_format_stats,
        "input_num": input_num_stats,
        "app": app_stats,
    }

    for stats_dict in all_stats.values():
        for keyword, data in stats_dict.items():
            data["average_score"] = data["total_score"] / data["count"] if data["count"] > 0 else 0.0
            del data["total_score"]

    return dict(all_stats)


def collect_task_metadata(model_results, all_task_meta_path):
    """
    Collect task metadata for a model's results using the all_task_meta.json file
    """
    # Load the complete task metadata
    with open(all_task_meta_path, "r") as f:
        all_meta = json.load(f)
    
    # Create result dictionary
    all_task_meta = {}
    
    # Match results with metadata
    for task_result in model_results:
        task_name = task_result["name"]
        if task_name in all_meta:
            meta = all_meta[task_name].copy()  # Create a copy to avoid modifying original
            meta.update(task_result)
            all_task_meta[task_name] = meta
    
    return all_task_meta
