# flake8: noqa
import argparse
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def prompt_context(question, answer, q_type, vqa_type):
    assert vqa_type in ["Direct", "CoT", "PoT"]
    assert q_type in ["Factoid", "Multi Choice", "Conversational", "Fact Checking", "Hypothetical"]
    if vqa_type == "Direct":
        if q_type == "Factoid":
            question_context = f'''
            You are given a factoid question that you need to answer based on the provided image.
            Your answer should be a single word, number, or phrase. If the question is unanswerable based on
            the information in the provided image, your answer should be unanswerable. Do not generate units.
            But if numerical units such as million, m, billion, B, or K are required, use the exact notation
            shown in the chart.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
        elif q_type == "Multi Choice":
            question_context = f'''
            You are given a question along with different possible answers. You need to select the correct answer
            from them based on the provided image.
            Your answer should be one of the options letters only: a, b, c or d (just the letter itself without any
            additional text). If the question is unanswerable based on the information in the provided image, your
            answer should be unanswerable.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
        elif q_type == "Conversational":
            question_context = f'''
            You are given a multi-turn conversation, and your job is to answer the final question based on the
            conversation history and the information in the provided image.
            Your answer should be a single word, number, or phrase. If the question is unanswerable based on
            the information in the provided image, your answer should be unanswerable. Do not generate units.
            But if numerical units such as million, m, billion, B, or K are required, use the exact notation
            shown in the chart.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Conversation: {[x for qa in zip(question[:-1], answer[:-1]) for x in qa]} Question: {question[-1]}
            '''
        elif q_type == "Fact Checking":
            question_context = f'''
            You are given a fact statement that you need to assess based on the provided image.
            Your answer should be either true or false (without any additional text). If the question is
            unanswerable based on the information in the provided image, your answer should be unanswerable.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
        elif q_type == "Hypothetical":
            question_context = f'''
            You are given a hypothetical question that you need to answer based on the provided image.
            Your answer should be a single word, number, or phrase. If the question is unanswerable based on
            the information in the provided image, your answer should be unanswerable. Do not generate units.
            But if numerical units such as million, m, billion, B, or K are required, use the exact notation
            shown in the chart.
            If there are multiple answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to generate the final answer only without any additional text!
            Question: {question[0]}
            '''
    elif vqa_type == "CoT":
        if q_type == "Factoid":
            question_context = f'''
            You are given a factoid question that you need to answer based on the provided image.
            You need to think step-by-step, but your final answer should be a single word, number, or phrase. If
            the question is unanswerable based on the information in the provided image, your answer should be
            unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are
            required, use the exact notation shown in the chart.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’]. .
            Remember to think step-by-step and format the final answer in a separate sentence like "The answer is
            X"
            Question: {question[0]}
            '''
        elif q_type == "Multi Choice":
            question_context = f'''
            You are given a question along with different possible answers. You need to select the correct answer
            from them based on the provided image.
            You need to think step-by-step, but your final answer should be one of the options letters only: a, b, c
            or d (just the letter itself without any additional text). If the question is unanswerable based on the
            information in the provided image, your answer should be unanswerable.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’]. .
            Remember to think step-by-step and format the final answer in a separate sentence like "The answer is
            X"
            Question: {question[0]}
            '''
        elif q_type == "Conversational":
            question_context = f'''
            You are given a multi-turn conversation, and your job is to answer the final question based on the
            conversation history and the information in the provided image.
            You need to think step-by-step, but your final answer should be a single word, number, or phrase. If
            the question is unanswerable based on the information in the provided image, your answer should be
            unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are
            required, use the exact notation shown in the chart.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’]. .
            Remember to think step-by-step and format the final answer in a separate sentence like "The answer is
            X"
            Conversation: {[x for qa in zip(question[:-1], answer[:-1]) for x in qa]} Question: {question[-1]}
            '''
        elif q_type == "Fact Checking":
            question_context = f'''
            You are given a fact statement that you need to assess based on the information in the provided image.
            You need to think step-by-step, but your final answer should be either true or false (without any
            additional text). If the question is unanswerable based on the information in the provided image, your
            answer should be unanswerable.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’]. .
            Remember to think step-by-step and format the final answer in a separate sentence like "The answer is
            X"
            Question: {question[0]}
            '''
        elif q_type == "Hypothetical":
            question_context = f'''
            You are given a hypothetical question that you need to answer based on the provided image.
            You need to think step-by-step, but your final answer should be a single word, number, or phrase. If
            the question is unanswerable based on the information in the provided image, your answer should be
            unanswerable. Do not generate units. But if numerical units such as million, m, billion, B, or K are
            required, use the exact notation shown in the chart.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to think step-by-step and format the final answer in a separate sentence like "The answer is
            X"
            Question: {question[0]}
            '''
    elif vqa_type == "PoT":
        if q_type == "Factoid":
            question_context = f'''
            You are given a factoid question that you need to answer based on the provided image.
            You need to write an executable python code that calculates and prints the final answer, but your final
            answer should be a single word, number, or phrase. If the question is unanswerable based on the
            information in the provided image, your answer should be unanswerable. Do not generate units. But
            if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the
            chart.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to return a python code only without any additional text.
            Question: {question[0]}
            '''
        elif q_type == "Multi Choice":
            question_context = f'''
            You are given a question along with different possible answers. You need to select the correct answer
            from them based on the provided image.
            You need to write an executable python code that calculates and prints the final answer, but your final
            answer should be one of the options letters only: a, b, c or d (just the letter itself without any additional
            text). If the question is unanswerable based on the information in the provided image, your answer
            should be unanswerable.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to return a python code only without any additional text.
            Question: {question[0]}
            '''
        elif q_type == "Conversational":
            question_context = f'''
            You are given a multi-turn conversation, and your job is to answer the final question based on the
            conversation history and the information in the provided image.
            You need to write an executable python code that calculates and prints the final answer, but your final
            answer should be a single word, number, or phrase. If the question is unanswerable based on the
            information in the provided image, your answer should be unanswerable. Do not generate units. But
            if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the
            chart.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to return a python code only without any additional text.
            Conversation: {[x for qa in zip(question[:-1], answer[:-1]) for x in qa]} Question: {question[-1]}
            '''
        elif q_type == "Fact Checking":
            question_context = f'''
            You are given a fact statement that you need to assess based on the information in the provided image.
            You need to write an executable python code that calculates and prints the final answer, but your final
            answer should be either true or false (without any additional text). If the question is unanswerable
            based on the information in the provided image, your answer should be unanswerable.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to return a python code only without any additional text.
            Question: {question[0]}
            '''
        elif q_type == "Hypothetical":
            question_context = f'''
            You are given a hypothetical question that you need to answer based on the provided image.
            You need to write an executable python code that calculates and prints the final answer, but your final
            answer should be a single word, number, or phrase. If the question is unanswerable based on the
            information in the provided image, your answer should be unanswerable. Do not generate units. But
            if numerical units such as million, m, billion, B, or K are required, use the exact notation shown in the
            chart.
            If there are multiple final answers, put them in brackets using this format [’Answer1’, ’Answer2’].
            Remember to return a python code only without any additional text.
            Question: {question[0]}
            '''
    return question_context


def load_predictions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    return predictions


def fix_list_format(item: str) -> Any:
    """
    Standardize string representations of lists, adding quotes around elements if missing,
    and safely evaluate to Python list. Returns original item if parsing fails.
    """
    if not isinstance(item, str):
        return item
    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item
    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)
    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def parse_to_list(text: str) -> Optional[List[str]]:
    """
    Parses text to a list of strings if possible; strips quotes and whitespace.
    """
    if not isinstance(text, str):
        return None
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]
    return None


def to_float(text: str) -> Optional[float]:
    """
    Converts text to float, stripping percent signs. Returns None on failure.
    """
    try:
        return float(text.strip().strip('%'))
    except ValueError:
        return None


def evaluate_single_answer(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05
) -> float:
    """
    Evaluates a single target-prediction pair:
    - Numeric within tolerance or exact year match inside this helper.
    - Falls back to ANLS for text.
    """
    from anls import anls_score
    t = target.strip().strip('%').strip()
    p = prediction.strip().strip('%').strip()
    # print("Stripped", t, p)
    # Attempt numeric
    t_f = to_float(t)
    p_f = to_float(p)
    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0
        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0
    # Fallback text
    # print("P:", p, "T: ", t)
    return anls_score(prediction=p.lower(), gold_labels=[t.lower()], threshold=0.5)


def relaxed_correctness_chartqapro(
    target: str,
    prediction: str,
    max_relative_change: float = 0.05,
    year_flags: Optional[List[bool]] = None,
    always_use_exact_match: bool = False,
) -> float:
    """
    Calculates relaxed correctness between target and prediction.
    Supports list inputs; uses year_flags to override year handling.
    """
    fixed_t = fix_list_format(target)
    t_list = parse_to_list(str(fixed_t)) or [str(target)]
    p_list = parse_to_list(str(prediction)) or [str(prediction)]
    n = len(t_list)
    # Expand year_flags for questions with multiple answers.
    if year_flags is not None and len(year_flags) < n:
        year_flags = year_flags * n

    # Evaluate elements
    scores: List[float] = []
    # print(t_list, p_list)
    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            # Model predicted more or less elements that necessary.
            scores.append(0.0)
            continue
        t_item, p_item, flag = t_list[idx], p_list[idx], year_flags[idx]
        flag_cond = True if flag.upper() == 'YES' else False
        if flag_cond or always_use_exact_match:
            # Exact integer match for years, fact checking, or multichoice
            try:
                scores.append(1.0 if t_item.strip().lower() == p_item.strip().lower() else 0.0)
            except ValueError:
                scores.append(0.0)
        else:
            scores.append(
                evaluate_single_answer(t_item, p_item, max_relative_change)
            )
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_predictions_chartqapro(predictions, pred_key='prediction'):
    gts = [x['Answer'][-1].strip(".").strip("\n") for x in predictions]
    preds = [str(x[pred_key]).strip(".").strip("\n") for x in predictions]
    splits = [x['Question Type'] for x in predictions]
    year_flags = [x['Year'] for x in predictions]
    # Calculate accuracy by splits
    match_nums_per_split = {}
    match_nums = []
    for gt, pred, split, year_flags_per_row in zip(gts, preds, splits, year_flags):
        # check split and calculate
        if split == 'Conversational':
            year_flags_per_row = year_flags_per_row[-1:]
        if split not in match_nums_per_split:
            match_nums_per_split[split] = []

        always_use_exact_match = True if split in ['Fact Checking', 'Multi Choice'] else False
        score = relaxed_correctness_chartqapro(gt, pred, year_flags=year_flags_per_row)
        # print(gt, pred, year_flags_per_row, score)
        match_nums_per_split[split].append(score)
        match_nums.append(score)

    final_numbers = {}
    for split in match_nums_per_split:
        final_numbers[split] = sum(match_nums_per_split[split]) / len(match_nums_per_split[split])
    final_numbers['Overall'] = sum(match_nums) / len(match_nums)
    return final_numbers


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChartQAPro predictions.")
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to the JSON file containing model predictions."
    )
    args = parser.parse_args()
    predictions = load_predictions(args.predictions_file)
    scores = evaluate_predictions_chartqapro(predictions)
    print("📊 Evaluation Results:")
    for k, v in scores.items():
        print(f"  • {k:<15}: {v * 100:.2f}%")


if __name__ == "__main__":
    main()
