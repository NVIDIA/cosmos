import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, model_validator
from tqdm import tqdm

from vlmeval.smp import load, load_env

current_script_path = Path(__file__).resolve()
current_dir = current_script_path.parent
template_path = current_dir / "mathcanvas_evaluate_template.txt"

if not template_path.exists():
    raise FileNotFoundError(f"Evaluation template not found at: {template_path}")

with open(template_path, 'r', encoding='utf-8') as f:
    EVAL_PROMPT_TEMPLATE = f.read()

SUB_QUESTION_WEIGHTS = {
    2: [0.4348, 0.5652],
    3: [0.2506, 0.3258, 0.4236],
    4: [0.1616, 0.2101, 0.2732, 0.3551]
}


def extract_reasoning_steps(input_text: str):
    steps = []
    iteration = 1

    pattern = re.compile(r'!\[.*?\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=\n\r]+)\)', re.DOTALL)

    last_end = 0
    for match in pattern.finditer(input_text):
        text_segment = input_text[last_end:match.start()].strip()
        if text_segment:
            steps.append({
                "type": "text",
                "content": text_segment,
                "iteration": iteration
            })

        base64_str = match.group(1).replace("\n", "").replace("\r", "")
        try:
            img_bytes = base64.b64decode(base64_str)
            steps.append({
                "type": "image",
                "content": img_bytes,
                "iteration": iteration
            })
        except Exception as e:
            steps.append({
                "type": "error",
                "content": f"Invalid base64 image data: {e}",
                "iteration": iteration
            })
        last_end = match.end()

    if last_end < len(input_text):
        tail_text = input_text[last_end:].strip()
        if tail_text:
            steps.append({
                "type": "text",
                "content": tail_text,
                "iteration": iteration
            })

    return {"reasoning_steps": steps}


class EvaluationResult(BaseModel):
    analysis: str = Field(description="An explanation of the extraction and judgment process for the prediction.")
    gt_answers: List[str] = Field(description="A list of parsed answers from the ground truth.")
    pred_answers: List[Optional[str]] = Field(description="A list of parsed/extracted answers from the prediction.")
    correctness: List[bool] = Field(description="A boolean list indicating if each predicted answer part is correct.")

    @model_validator(mode='after')
    def check_list_lengths(self) -> 'EvaluationResult':
        len_gt = len(self.gt_answers)
        len_pred = len(self.pred_answers)
        len_corr = len(self.correctness)
        if not (len_gt > 0 and len_gt == len_pred == len_corr):
            raise ValueError(
                f"List lengths must be equal and non-empty. Got: "
                f"gt_answers={self.gt_answers} with length {len_gt}, "
                f"pred_answers={self.pred_answers} with length {len_pred}, "
                f"correctness={self.correctness} with length {len_corr}."
            )
        return self


def _call_openai_with_pydantic(client, prompt, pydantic_class, model_name, **kwargs):
    messages = [{"role": "user", "content": prompt}]

    params = {
        'model': model_name,
        'messages': messages,
        'response_format': pydantic_class,
        'temperature': kwargs.get('temperature', 0.0),
        'max_completion_tokens': kwargs.get('max_tokens', 2048),
    }

    response = client.beta.chat.completions.parse(**params)
    api_result = response.choices[0].message.parsed
    return api_result.model_dump()


def _process_single_item(item_data):
    client, gt_row, pred_row, judge_kwargs = item_data

    try:
        gt_answer = gt_row.get("answer", "")
        prediction = pred_row.get("prediction", "")

        reasoning_steps = []
        if '![image]' in prediction:
            reasoning_steps = extract_reasoning_steps(prediction)['reasoning_steps']

            prediction_text = "\n\n".join(
                item["content"].strip()
                for item in reasoning_steps
                if item.get("type") == "text" and item.get("content")
            )
        else:
            prediction_text = prediction

        input_data_str = json.dumps({
            "question_text": gt_row.get("question", ""), "ground_truth_answer": gt_answer,
            "prediction_solution": prediction_text,
        }, ensure_ascii=False, indent=2)

        prompt = EVAL_PROMPT_TEMPLATE.format(input_data=input_data_str)

        eval_data = _call_openai_with_pydantic(client, prompt, EvaluationResult, judge_kwargs['model'], **judge_kwargs)

        final_result = {
            **gt_row,
            "prediction": prediction,
            "status": "success",
            "evaluation": eval_data,
        }
        if reasoning_steps:
            final_result["reasoning_steps"] = reasoning_steps

        return final_result

    except Exception as e:
        final_result = {
            **gt_row,
            "prediction": pred_row.get("prediction", ""),
            "status": "error",
            "reason": str(e),
        }
        return final_result


def evaluate_with_judge(eval_file, ground_truth_data, **judge_kwargs):
    if OpenAI is None:
        raise ImportError("OpenAI library is required for MathCanvas evaluation. Please install it.")

    predictions = load(eval_file)

    load_env()
    api_key = os.environ.get("OPENAI_API_KEY", None)
    base_url = os.environ.get("OPENAI_API_BASE", None)
    client = OpenAI(api_key=api_key, base_url=base_url)

    tasks = []
    gt_map = {row['index']: row for _, row in ground_truth_data.iterrows()}

    for _, pred_row in predictions.iterrows():
        gt_row = gt_map.get(pred_row['index'])
        if gt_row is not None:
            tasks.append((client, gt_row, pred_row.to_dict(), judge_kwargs))

    detailed_results = []
    max_workers = judge_kwargs.get('nproc', 16)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(_process_single_item, tasks)
        for result in tqdm(results_iterator, total=len(tasks), desc="Evaluating with Judge LLM"):
            detailed_results.append(result)

    return detailed_results


def _update_stats(stats_dict: Dict, key: str, score: float):
    # Helper function to update statistics for a given category key.
    if key not in stats_dict:
        stats_dict[key] = {'count': 0, 'total_score': 0.0}
    stats_dict[key]['count'] += 1
    stats_dict[key]['total_score'] += score


def summarize_mathcanvas_results(all_results: List[Dict]):
    stats = {
        'overall': {'count': 0, 'total_score': 0.0, 'completely_correct_count': 0},
        'by_knowledge': {},
        'by_question_image_count': {},
    }
    error_count = 0
    too_many_parts_count = 0

    for item in tqdm(all_results, desc="Summarizing results"):
        if item.get("status") != "success" or "evaluation" not in item:
            error_count += 1
            continue

        correctness = item["evaluation"]["correctness"]
        num_parts = len(correctness)
        score = 0.0

        if num_parts == 1:
            score = 1.0 if correctness[0] else 0.0
        elif num_parts in SUB_QUESTION_WEIGHTS:
            weights = SUB_QUESTION_WEIGHTS[num_parts]
            score = sum(w for c, w in zip(correctness, weights) if c)
        else:
            # Skip items with more answer parts than defined weights
            too_many_parts_count += 1
            continue

        # --- Update Statistics ---

        # 1. Overall Stats
        stats['overall']['count'] += 1
        stats['overall']['total_score'] += score
        if all(correctness):
            stats['overall']['completely_correct_count'] += 1

        # 2. Image Presence Stats
        q_images = [p for p in item.get("question_interleave", []) if p['type'] == 'image']
        image_presence_key = "Has Image" if len(q_images) > 0 else "No Image"
        _update_stats(stats['by_question_image_count'], image_presence_key, score)

        # 3. Knowledge Stats
        # Use the primary knowledge area for categorization.
        knowledges = item.get("knowledges", [])
        knowledge_key = knowledges[0] if knowledges else "Unknown"
        _update_stats(stats['by_knowledge'], knowledge_key, score)

    # --- Final Report Generation ---
    def calculate_accuracy(data_dict):
        report = {}
        for key, value in sorted(data_dict.items()):
            count = value.get('count', 0)
            total_score = value.get('total_score', 0.0)
            accuracy = round((total_score / count) * 100, 1) if count > 0 else 0.0
            report[key] = {'count': count, 'accuracy': accuracy}
        return report

    total_valid = stats['overall']['count']

    summary_report = {
        'overall_summary': {
            'total_evaluated': len(all_results),
            'valid_evaluations': total_valid,
            'processing_errors': error_count,
            'too_many_parts_skipped': too_many_parts_count,
            'weighted_accuracy': (
                round((stats['overall']['total_score'] / total_valid) * 100, 1)
                if total_valid > 0 else 0.0
            ),
            'complete_accuracy': (
                round((stats['overall']['completely_correct_count'] / total_valid) * 100, 1)
                if total_valid > 0 else 0.0
            ),
        },
        'knowledge_summary': calculate_accuracy(stats['by_knowledge']),
        'accuracy_by_question_image_count': calculate_accuracy(stats['by_question_image_count']),
    }

    return summary_report
