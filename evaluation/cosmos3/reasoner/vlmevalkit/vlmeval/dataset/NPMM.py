import ast
from collections import defaultdict
from typing import Any

import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils.NPMM.dominating_set import validation as dominating_set_validation
from vlmeval.dataset.utils.NPMM.feedback_vertex import validation as feedback_vertex_validation
from vlmeval.dataset.utils.NPMM.gcp import validation as gcp_validation
from vlmeval.dataset.utils.NPMM.hamiltonian_cycle import validation as hamiltonian_cycle_validation
from vlmeval.dataset.utils.NPMM.maximum_cut import validation as maximum_cut_validation
from vlmeval.dataset.utils.NPMM.maximum_set import validation as maximum_set_validation
from vlmeval.dataset.utils.NPMM.mcp import validation as mcp_validation
from vlmeval.dataset.utils.NPMM.minimum_cut import validation as minimum_cut_validation
from vlmeval.dataset.utils.NPMM.tsp import validation as tsp_validation
from vlmeval.dataset.utils.NPMM.vertex_cover import validation as vertex_cover_validation
from vlmeval.smp import dump, get_intermediate_file_path, load


class NPMM(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'npmm': 'https://opencompass.openxlab.space/utils/VLMEval/npmm.tsv',
    }
    DATASET_MD5 = {
        'npmm': 'e161f597458786805cbb78e1a6f7e5f7',
    }
    GROUP_LIST = {
        "constraint": ["NpGcpD", "NpFeedbackVertexSet"],
        "covering": ["NpVertexCover", "NpDominatingSet"],
        "partitioning": ["NpMinimumCut", "NpMaximumCut"],
        "subgraph": ["NpMaximumCliqueProblem", "NpMaximumSet"],
        "path": ["NpTsp", "NpHamiltonianCycle"]
    }

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        stats = defaultdict[Any, dict[str, int | float]](lambda: {'total': 0, 'valid': 0, 'ar_accum': 0.0})
        target_tasks = ["NpTsp", "NpHamiltonianCycle", "NpMaximumCliqueProblem", "NpMinimumCut", "NpMaximumSet", "NpGcpD", "NpFeedbackVertexSet", "NpDominatingSet", "NpVertexCover", "NpMaximumCut"]  # noqa: E501
        for index, data_item in data.iterrows():
            task = data_item.get("data_source").split("/")[1]
            if task not in target_tasks:
                continue
            prediction = data_item.get('prediction')
            ground_truth = data_item.get('ground_truth')
            graph = data_item.get('graph')

            # Parse graph from string to dict if needed
            # graph should be string of a dict
            if isinstance(graph, str):
                try:
                    graph = ast.literal_eval(graph)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"Failed to parse graph for task {task}, index {index}: {e}")

            stats[task]['total'] += 1
            is_invalid = True
            value = 0
            try:
                if task == "NpTsp":
                    is_invalid, value, msg = tsp_validation(graph, prediction)
                elif task == "NpHamiltonianCycle":
                    is_invalid, value, msg = hamiltonian_cycle_validation(graph, prediction)
                elif task == "NpMaximumCliqueProblem":
                    is_invalid, value, msg = mcp_validation(graph, prediction)
                elif task == "NpMinimumCut":
                    is_invalid, value, msg = minimum_cut_validation(graph, prediction)
                elif task == "NpMaximumSet":
                    is_invalid, value, msg = maximum_set_validation(graph, prediction)
                elif task == "NpGcpD":
                    is_invalid, value, msg = gcp_validation(graph, prediction)
                elif task == "NpFeedbackVertexSet":
                    is_invalid, value, msg = feedback_vertex_validation(graph, prediction)
                elif task == "NpDominatingSet":
                    is_invalid, value, msg = dominating_set_validation(graph, prediction)
                elif task == "NpVertexCover":
                    is_invalid, value, msg = vertex_cover_validation(graph, prediction)
                elif task == "NpMaximumCut":
                    is_invalid, value, msg = maximum_cut_validation(graph, prediction)
            except Exception as e:
                print(f"Validation error for task {task}, index {index}: {e}")
                is_invalid = True
            if not is_invalid:
                stats[task]['valid'] += 1
                if task in ["NpGcpD", "NpMinimumCut", "NpTsp", "NpVertexCover",
                            "NpDominatingSet", "NpFeedbackVertexSet"]:
                    stats[task]['ar_accum'] += float(ground_truth) / value
                else:
                    ratio = value / float(ground_truth)
                    stats[task]['ar_accum'] += ratio
        # cal each task's result
        subtask_results = {}
        for task in target_tasks:
            t_stat = stats[task]
            total = t_stat['total']
            if total > 0:
                sr = t_stat['valid'] / total
                ar = t_stat['ar_accum'] / total
            else:
                sr = 0.0
                ar = 0.0

            subtask_results[task] = {'SR': sr, 'AR': ar, 'count': total}
        subtask_file = get_intermediate_file_path(eval_file, '_subtask_stats', 'json')
        dump(subtask_results, subtask_file)
        group_stats = []

        total_sr = 0
        total_ar = 0
        valid_tasks_count = 0

        for group, tasks in self.GROUP_LIST.items():
            relevant_tasks = [t for t in tasks if t in subtask_results and subtask_results[t]['count'] > 0]

            if not relevant_tasks:
                continue
            avg_sr = sum(subtask_results[t]['SR'] for t in relevant_tasks) / len(relevant_tasks)
            avg_ar = sum(subtask_results[t]['AR'] for t in relevant_tasks) / len(relevant_tasks)

            group_stats.append({
                'Task': group,
                'SR': avg_sr,
                'AR': avg_ar,
                'num_subtasks': len(relevant_tasks)
            })

            total_sr += avg_sr
            total_ar += avg_ar
            valid_tasks_count += 1

        if valid_tasks_count > 0:
            group_stats.append({
                'Task': 'Overall',
                'SR': total_sr / valid_tasks_count,
                'AR': total_ar / valid_tasks_count,
                'num_subtasks': valid_tasks_count
            })
        print(group_stats)
        res = {}
        for g in group_stats:
            if g['Task'] == 'Overall':
                res.update({'SR': round(g['SR'] * 100, 1), 'AR': round(g['AR'] * 100, 1)})
            else:
                res.update({f"{g['Task']}_success_rate": round(g['SR'] * 100, 1),
                            f"{g['Task']}_avg_ratio": round(g['AR'] * 100, 1)})
        accuracy_df = pd.DataFrame([res])
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(accuracy_df, score_file)
        return accuracy_df
