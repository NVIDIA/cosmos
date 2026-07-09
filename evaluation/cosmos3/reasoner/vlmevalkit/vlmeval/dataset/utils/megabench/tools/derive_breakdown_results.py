import argparse
import json
from pathlib import Path

from analysis_utils import collect_task_metadata, derive_keyword_stats, task_list_refine


def calculate_model_summary(task_results_with_meta):
    """
    Re-calculate model performance summary statistics across core and open tasks.
    
    Args:
        task_results: List of task results with scores
        task_metadata: Dictionary containing task metadata including task types
    
    Returns:
        Dictionary containing summary statistics for core and open tasks
    """
    core_tasks = []
    open_tasks = []

    # Separate core and open tasks
    for task in task_results_with_meta.values():
        if task['eval_type'] == 'llm':
            open_tasks.append(task)
        else:
            core_tasks.append(task)
    
    def calculate_stats(tasks):
        if not tasks:
            return None
        
        total_samples = sum(task.get('num_query', 0) for task in tasks)
        macro_scores = [task.get('score', 0) for task in tasks]
        
        return {
            "num_eval_tasks": len(tasks),
            "num_eval_samples": total_samples,
            "macro_mean_score": sum(macro_scores) / len(tasks) if tasks else 0,
        }
    
    core_stats = calculate_stats(core_tasks)
    open_stats = calculate_stats(open_tasks)
    
    # Calculate overall score (weighted average based on number of tasks)
    total_tasks = (core_stats["num_eval_tasks"] + open_stats["num_eval_tasks"])
    overall_score = (
        (core_stats["macro_mean_score"] * core_stats["num_eval_tasks"] + 
         open_stats["macro_mean_score"] * open_stats["num_eval_tasks"]) / total_tasks
        if core_stats and open_stats
        else 0
    )
    
    return {
        "core": core_stats,
        "open": open_stats,
        "overall_score": overall_score
    }

def merge_json_files(input_dir, output_path, key="name"):
    """
    Merge multiple JSON files containing evaluation results from a directory.
    Looks for all files matching pattern 'data_with_scores*.json'.
    Prioritizes LLM evaluations over rule-based ones when duplicates exist.
    """
    data_dict = {}  # Using name as key for easy lookup and updates
    
    # Find all matching JSON files in the directory
    json_paths = list(Path(input_dir).glob("megabench_score*.json"))
    print(f"Found {len(json_paths)} files to merge")
    
    # Load and merge all JSON files
    for path in json_paths:
        print(f"Processing {path}")
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "data" in data:
                data = task_list_refine(data["data"])
            
            # Update or add entries
            for item in data:
                item_key = item[key]
                # If new item or if new item is LLM-evaluated (prioritize LLM eval)
                if item_key not in data_dict or (
                    item.get("eval_type") == "llm" and data_dict[item_key].get("eval_type") != "llm"
                ):
                    data_dict[item_key] = item

    # Convert back to list
    merged_data = list(data_dict.values())
    
    # Save the merged result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"Merged file with {len(merged_data)} tasks saved to {output_path}")
    return merged_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge and process evaluation score files.')
    parser.add_argument('--input_dir', type=str, help='Directory containing score files')
    args = parser.parse_args()

    # Convert path to Path object
    input_dir = Path(args.input_dir)
    
    # Create analysis directory under input directory
    output_dir = input_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge files
    output_path = output_dir / "task_results.json"
    task_results = merge_json_files(input_dir, output_path)
    
    # Collect metadata and derive keyword stats
    task_results_with_meta = collect_task_metadata(task_results, all_task_meta_path="all_task_meta.json")
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
    
    print(f"\nResults saved in {output_dir}:")
    print(f"- Merged data: {output_path}")
    print(f"- Multi-dimensional keywords stats: {stats_output}")

if __name__ == "__main__":
    main()
