import re
from ast import literal_eval
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import dump, get_intermediate_file_path, load


def format_model_answer_tolist(model_answer, task_gt):
    """
    Extract 0/1 list from model answer

    Args:
        model_answer: Model's prediction answer (string)
        task_gt: Ground truth list, used to determine expected length

    Returns:
        list: 0/1 list
    """
    numbers = re.findall(r'\d+', str(model_answer))

    result = [int(num) for num in numbers]

    # Convert non-0/1 numbers to 1
    result = [num if num == 0 or num == 1 else 1 for num in result]

    # Adjust length to match task_gt
    if len(result) >= len(task_gt):
        return result[:len(task_gt)]
    else:
        return result + [0] * (len(task_gt) - len(result))


def format_ms_model_answer(model_answer):
    """
    Extract two scores from multi_solution model answer

    Args:
        model_answer: Model's prediction answer (string)
        Expected format: "[7, 8]" or "7, 8" or "Score 1: 7, Score 2: 8"

    Returns:
        list: Two scores [score1, score2]
    """
    numbers = re.findall(r'\d+', str(model_answer))
    result = [int(num) for num in numbers]

    # Return last two numbers (most likely the actual scores)
    if len(result) >= 2:
        return result[-2:]
    else:
        # If less than 2 numbers found, pad with 0
        return result + [0] * (2 - len(result))


def get_F1Score(gathered_model_answer, gathered_task_gt):
    """
    Calculate F1 score

    Args:
        gathered_model_answer: List of all model answers
        gathered_task_gt: List of all ground truth

    Returns:
        tuple: (F1_pos, F1_neg, F1_w) - positive class F1, negative class F1, weighted F1
    """
    model_answer = np.array(gathered_model_answer)
    task_gt = np.array(gathered_task_gt)

    pos_count = np.sum(task_gt == 1)
    neg_count = np.sum(task_gt == 0)

    F1_pos = f1_score(task_gt, model_answer, pos_label=1, zero_division=0)
    F1_neg = f1_score(task_gt, model_answer, pos_label=0, zero_division=0)

    w_pos = neg_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0
    w_neg = pos_count / (pos_count + neg_count) if (pos_count + neg_count) > 0 else 0

    F1_w = w_neg * F1_neg + w_pos * F1_pos

    return F1_pos, F1_neg, F1_w


class VLRMBench(ImageBaseDataset):
    """
    VLRMBench Dataset - Visual Language Reasoning Model Benchmark

    A comprehensive benchmark for evaluating visual reasoning capabilities including:
    - step_correctness: Step correctness detection
    - redundant_det: Redundancy detection
    - most_confidence: Highest confidence judgment
    - attribute_hallucination: Attribute hallucination detection
    - existence_hallucination: Existence hallucination detection
    - detail_error: Detail error detection
    - image_ref_error: Image reference error detection
    - location_error: Location error detection
    - multi_solution: Position bias resistance evaluation
    - foresight: Reasoning foresight capability evaluation

    Note: Currently only supports Outcome-based tasks and Step-based tasks.
    Criticism-based tasks are not supported in this implementation.
    """

    TYPE = 'VQA'
    DATASET_URL = {
        'VLRMBench': 'https://huggingface.co/datasets/Winston-Yuan/VLRMBench/resolve/main/VLRMBench.tsv',
        'VLRMBench_MultiSolution': (
            'https://huggingface.co/datasets/Winston-Yuan/VLRMBench/resolve/main/VLRMBench_MultiSolution.tsv'
        ),
        'VLRMBench_Foresight': (
            'https://huggingface.co/datasets/Winston-Yuan/VLRMBench/resolve/main/VLRMBench_Foresight.tsv'
        )
    }
    DATASET_MD5 = {
        'VLRMBench': 'f1dedeac74fc1112545390d6e2ecf4a2',
        'VLRMBench_MultiSolution': 'e8c15ab7c24568ba4d72375530389387',
        'VLRMBench_Foresight': '1e22f1b94afbd6f4f3a4028c91749311'
    }

    def __init__(self, **kwargs):
        """
        Initialize VLRMBench dataset with warning about supported task types.
        """
        import warnings
        warnings.warn(
            "VLRMBench currently only supports Outcome-based tasks and Step-based tasks. "
            "Criticism-based tasks are not supported in this implementation.",
            UserWarning,
            stacklevel=2
        )
        super().__init__(**kwargs)

    def build_prompt(self, line):
        """
        Build prompt information

        Args:
            line: Data row, can be int index or pd.Series

        Returns:
            list: Multimodal message list, format is [dict(type='image', value=path), dict(type='text', value=text),]
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Create a copy of line to avoid SettingWithCopyWarning
        line = line.copy()

        # Use parent class method to save image (decode from base64 and save locally)
        tgt_path = self.dump_image(line)

        # Build messages
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]

        # Add question text
        question = line.get('question', '')
        if question:
            msgs.append(dict(type='text', value=question))

        return msgs

    @classmethod
    def evaluate_multi_solution(cls, data):
        """
        Evaluate multi_solution type data (position bias resistance)

        Args:
            data: DataFrame containing multi_solution predictions

        Returns:
            dict: Evaluation results with accuracy metric
        """
        acc_sample = 0
        overall_sample = 0
        skipped = 0

        # Group by index pairs (front: even, back: odd)
        indices = sorted(data['index'].unique())

        for i in range(0, len(indices), 2):
            if i + 1 >= len(indices):
                skipped += 1
                continue

            front_idx = indices[i]
            back_idx = indices[i + 1]

            # Get front and back rows
            front_rows = data[data['index'] == front_idx]
            back_rows = data[data['index'] == back_idx]

            if len(front_rows) == 0 or len(back_rows) == 0:
                skipped += 1
                continue

            front_row = front_rows.iloc[0]
            back_row = back_rows.iloc[0]

            # Verify order field if exists
            if 'order' in data.columns:
                if front_row.get('order') != 'front' or back_row.get('order') != 'back':
                    print(f"Warning: Order mismatch at index {front_idx}, {back_idx}")
                    skipped += 1
                    continue

            try:
                # Parse model predictions
                front_scores = format_ms_model_answer(front_row.get('prediction', ''))
                back_scores = format_ms_model_answer(back_row.get('prediction', ''))

                # Apply evaluation formula: front[0] + back[1] vs front[1] + back[0]
                # This checks if model consistently prefers the better response regardless of position
                if front_scores[0] + back_scores[1] > front_scores[1] + back_scores[0]:
                    acc_sample += 1

                overall_sample += 1
            except Exception as e:
                print(f"Failed to process multi_solution pair ({front_idx}, {back_idx}): {e}")
                skipped += 1

        results = {
            'multi_solution_accuracy': acc_sample / overall_sample if overall_sample > 0 else 0.0,
            'multi_solution_count': overall_sample,
            'multi_solution_skipped': skipped
        }

        return results

    @classmethod
    def evaluate_foresight(cls, data):
        """
        Evaluate foresight type data (reasoning foresight capability)

        Args:
            data: DataFrame containing foresight predictions

        Returns:
            dict: Evaluation results with accuracy metric
        """
        acc_sample = 0
        overall_sample = 0
        skipped = 0

        for idx in range(len(data)):
            item = data.iloc[idx]

            try:
                task_gt = item['task_gt']  # True/False
                model_answer = item.get('prediction', '')

                # 关键词匹配逻辑（与get_fores_eval_res.py一致）
                if task_gt is True:
                    if re.search(r'\b(yes|true)\b', model_answer, re.IGNORECASE):
                        acc_sample += 1
                elif task_gt is False:
                    if re.search(r'\b(no|false)\b', model_answer, re.IGNORECASE):
                        acc_sample += 1

                overall_sample += 1
            except Exception as e:
                print(f"Failed to process foresight sample (idx={idx}): {e}")
                skipped += 1

        results = {
            'foresight_accuracy': acc_sample / overall_sample if overall_sample > 0 else 0.0,
            'foresight_count': overall_sample,
            'foresight_skipped': skipped
        }

        return results

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate model prediction results
        Automatically detects and handles step-based, multi_solution, and foresight data

        Args:
            eval_file: Path to model prediction results file
            **judge_kwargs: Other evaluation parameters

        Returns:
            pd.DataFrame: Evaluation results, including F1 scores and/or accuracy
        """
        # Load prediction data
        data = load(eval_file)

        # Ensure necessary fields exist
        assert 'prediction' in data.columns, "Evaluation file missing 'prediction' field"
        assert 'category' in data.columns, "Evaluation file missing 'category' field"

        # Detect data types
        categories = data['category'].unique()
        has_multi_solution = 'multi_solution' in categories
        has_foresight = 'foresight' in categories
        has_step_based = any(c not in ['multi_solution', 'foresight'] for c in categories)

        results = {}

        # Process step-based categories
        if has_step_based:
            # Filter step-based data
            step_data = data[~data['category'].isin(['multi_solution', 'foresight'])]

            # Ensure answer field exists for step-based data
            if 'answer' not in step_data.columns:
                print("Warning: Step-based data missing 'answer' field, skipping step-based evaluation")
            else:
                # Collect model answers and ground truth by category
                category_model_answers = defaultdict(list)
                category_task_gts = defaultdict(list)
                category_total = defaultdict(int)

                for idx in range(len(step_data)):
                    item = step_data.iloc[idx]
                    category = item['category']

                    try:
                        # Parse task_gt (answer field)
                        task_gt = item['answer']
                        if isinstance(task_gt, str):
                            # Try to parse string as list
                            task_gt = literal_eval(task_gt)

                        # Get model answer (prediction field)
                        model_answer = item.get('prediction', '')

                        # Format model answer using format_model_answer_tolist
                        formatted_model_answer = format_model_answer_tolist(model_answer, task_gt)

                        # Collect answers for each category
                        category_task_gts[category].extend(task_gt)
                        category_model_answers[category].extend(formatted_model_answer)
                        category_total[category] += 1
                    except Exception as e:
                        # If parsing fails, log and skip the sample
                        print(f"Failed to process sample (idx={idx}, category={category}): {e}")
                        continue

                # Calculate F1 scores for each category
                for category in category_task_gts:
                    gathered_task_gt = category_task_gts[category]
                    gathered_model_answer = category_model_answers[category]

                    if len(gathered_task_gt) > 0:
                        F1_pos, F1_neg, F1_w = get_F1Score(gathered_model_answer, gathered_task_gt)

                        results[f'{category}_F1_pos'] = F1_pos
                        results[f'{category}_F1_neg'] = F1_neg
                        results[f'{category}_F1_weighted'] = F1_w
                        results[f'{category}_count'] = category_total[category]
                    else:
                        results[f'{category}_F1_pos'] = 0.0
                        results[f'{category}_F1_neg'] = 0.0
                        results[f'{category}_F1_weighted'] = 0.0
                        results[f'{category}_count'] = 0

                # Calculate overall F1 score (all step-based categories combined)
                all_task_gts = []
                all_model_answers = []
                for category in category_task_gts:
                    all_task_gts.extend(category_task_gts[category])
                    all_model_answers.extend(category_model_answers[category])

                if len(all_task_gts) > 0:
                    F1_pos_overall, F1_neg_overall, F1_w_overall = get_F1Score(all_model_answers, all_task_gts)
                    results['Overall_F1_pos'] = F1_pos_overall
                    results['Overall_F1_neg'] = F1_neg_overall
                    results['Overall_F1_weighted'] = F1_w_overall
                    results['Overall_count'] = sum(category_total.values())
                else:
                    results['Overall_F1_pos'] = 0.0
                    results['Overall_F1_neg'] = 0.0
                    results['Overall_F1_weighted'] = 0.0
                    results['Overall_count'] = 0

        # Process multi_solution category
        if has_multi_solution:
            ms_data = data[data['category'] == 'multi_solution']
            ms_results = cls.evaluate_multi_solution(ms_data)
            results.update(ms_results)

        # Process foresight category
        if has_foresight:
            foresight_data = data[data['category'] == 'foresight']
            foresight_results = cls.evaluate_foresight(foresight_data)
            results.update(foresight_results)

        # Convert to DataFrame format
        results_df = pd.DataFrame([results])

        # Save results
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(results_df, score_file)

        return results_df
