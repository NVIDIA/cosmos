"""
For propietary models that naturally suport multi-image or video inputs, we don't run the single-image setting, 
instead, we directly compute the SI results by extracting the subset results from the full task set to compute the stats.
"""


import argparse
import json
from pathlib import Path

from analysis_utils import collect_task_metadata, derive_keyword_stats
from derive_breakdown_results import calculate_model_summary


def process_subset_results(input_dir, eval_type):
    """Read results from the full results directory structure"""
    task_results_path = input_dir / "analysis" / "task_results.json"
    
    # Load task results
    with open(task_results_path, "r") as f:
        task_results = json.load(f)
    
    results_with_meta = collect_task_metadata(task_results, all_task_meta_path="all_task_meta.json")
    
    # Filter tasks by eval_type
    filtered_results_with_meta = {task_name:task for task_name, task in results_with_meta.items() if task["eval_type"] == eval_type and task["num_input"] == "1-image"}
    filtered_results = [task for task in task_results if task["name"] in filtered_results_with_meta]
    
    if not filtered_results:
        print(f"Warning: No tasks found in {input_dir} with eval_type {eval_type}")
        return None, None, None
    
    # Calculate summary statistics
    num_tasks = len(filtered_results)
    total_queries = sum(task["num_query"] for task in filtered_results)
    total_correct = sum(round(task["score"] * task["num_query"]) for task in filtered_results)
    
    summary = {
        "num_eval_tasks": num_tasks,
        "num_eval_samples": total_queries,
        "macro_mean_score": sum(task["score"] for task in filtered_results) / num_tasks,
    }
    
    return filtered_results, filtered_results_with_meta, summary


def main(input_dir, output_dir):
    # Process core and open set results
    filtered_tasks_core, filtered_tasks_core_with_meta, _ = process_subset_results(input_dir, "rule")
    filtered_tasks_open, filtered_tasks_open_with_meta, _ = process_subset_results(input_dir, "llm")

    if filtered_tasks_core and filtered_tasks_open:
        task_results = filtered_tasks_core + filtered_tasks_open
        task_results_with_meta = {**filtered_tasks_core_with_meta, **filtered_tasks_open_with_meta}

        # Save task results
        with open(output_dir / "task_results.json", "w") as f:
            json.dump(task_results, f, indent=4)

        # Collect metadata and derive keyword stats
        keyword_stats = derive_keyword_stats(task_results_with_meta)

        # Calculate model summary
        model_summary = calculate_model_summary(task_results_with_meta)

        summary_results = {
            "model_summary": model_summary,
            "keyword_stats": keyword_stats
        }
    
        # Save keyword stats
        stats_output = output_dir / "summary_and_keyword_stats.json"
        with open(stats_output, "w") as f:
            json.dump(summary_results, f, indent=4)
    
        print(f"\nResults saved in {output_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing full results")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(input_dir, output_dir)