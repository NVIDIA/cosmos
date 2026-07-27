"""
Evaluation utilities for CoreCognition benchmark.
Implements Hybrid Matching: template matching first, then LLM matching as fallback.
Supports both MCQ (multiple choice) and YORN (yes or no) question types.
"""

import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from vlmeval.smp import get_logger

logger = get_logger(__name__)

OPTIONS_MCQ = ['A', 'B', 'C', 'D', 'E', 'F']
OPTIONS_YORN = ['YES', 'NO']

# ============================================================================
# Template Matching (MCQ and YORN)
# ============================================================================


def template_match(pred, question_type):
    """Template matching for answer extraction (MCQ and YORN)."""
    pred = str(pred).strip()
    valid_options = OPTIONS_YORN if question_type == 'YORN' else OPTIONS_MCQ

    if len(pred.split()) >= 2:
        # Multi-word response - use regex patterns
        patterns = [
            r'^(yes|no|\w)(,|\.|\;| |\n|\*)+',
            r'[\n\*\{]+(yes|no|\w)(,|\.|\;| |\n|\*|\})+',
            r'(yes|no|\w) is the correct answer',
            r'answer is[\:\;\*\n ]*(yes|no|\w)',
            r'answer[\:\;\*\n ]*(yes|no|\w)',
            r'choice is[\:\;\*\n ]*(yes|no|\w)',
            r'choice[\:\;\*\n ]*(yes|no|\w)',
            r'option is[\:\;\*\n ]*(yes|no|\w)',
            r'Assistant[\:\;\*\n ]*(yes|no|\w)',
        ]
        for pattern in patterns:
            match = re.search(pattern, pred, re.IGNORECASE)
            if match:
                res = match.group(1).upper()
                if res in valid_options:
                    return res
    else:
        # Short response - direct extraction
        res = re.split(r',|\.| |\:|\;|\n', pred)[0].upper() if pred else ''
        if res in valid_options:
            return res

    return 'Fail'


# ============================================================================
# LLM Prompt Builder
# ============================================================================

def build_prompt_mcq(question, prediction):
    """Build prompt for multiple choice questions."""
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with several options of a single-choice question. '
        'You are provided with a question, several options, and an answer, '
        'and you need to find which option is most similar to the answer. '
        'If the answer is already a single uppercase or lowercase character in the given options, '
        'output the answer directly.\n'
        'If the meaning of all options are significantly different from the answer, output Z.\n'
        'If the answer is random words, noise or gibberish, also output Z.\n'
        'You should output a single uppercase character in the given options (if they are valid options), '
        'or Z (if the answer is invalid). \n'
        'You should output ONLY a single uppercase character WITHOUT ANYTHING ELSE.\n'
        'Example 1: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear, B. rabbit, C. cat, D. dog\n'
        'Answer: a cute teddy bear\nYour output: A\n'
        'Example 2: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear, B. rabbit, C. cat, D. dog\n'
        'Answer: Spider\nYour output: Z\n'
        'Example 3: \n'
        'Question: To hang a framed photo on a wall, which tool should you use?\nOptions: A, B, C, D, E\n'
        'Answer: d\nYour output: D\n'
        'Example 4: \n'
        'Question: To hang a framed photo on a wall, which tool should you use?\nOptions: A, B, C, D, E\n'
        'Answer: (empty space)\nYour output: Z\n'
        'Example 5: \n'
        'Question: To hang a framed photo on a wall, which tool should you use?\nOptions: A, B, C, D, E\n'
        'Answer: 0\\<<<<<<>\nYour output: Z\n'
        'Example 6: \n'
        'Question: What is the main object in image?\nOptions: A. teddy bear, B. rabbit, C. cat, D. dog\n'
        'Answer: B\nYour output: B\n'
        'Question: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, prediction)


def build_prompt_yorn(question, prediction):
    """Build prompt for yes or no questions."""
    tmpl = (
        'You are an AI assistant who will help me to match '
        'an answer with "Yes" or "No" in a true-false question. '
        'You are provided with a question and an answer, '
        'and you need to find whether "Yes" or "No" matches the most with the answer. '
        'If the meaning of the answer fail to indicate "Yes" or "No", output Z. '
        'If the answer is random words, noise or gibberish, also output Z.\n'
        'Your should output ONLY Yes or No, or Z (if the answer is invalid) WITHOUT ANYTHING ELSE. \n'
        'Example 1: \n'
        'Question: In the image, are the lines completely separate from each other? '
        'Please answer with Yes or No. Do not show the reasoning trace, just output the final result.\n'
        'Answer: no\nYour output: No\n'
        'Example 2: \n'
        'Question: Are there circles that do not overlap other circles? Please answer with Yes or No.\n'
        'Answer: All the circles in the image overlap with at least one other circle\nYour output: No\n'
        'Example 3: \n'
        'Question: Are there grids on the book? Please answer with Yes or No.\n'
        'Answer: I do not know.\nYour output: Z\n'
        'Example 4: \n'
        'Question: Are there grids on the book? Please answer with Yes or No.\n'
        'Answer: True\nYour output: Yes\n'
        'Example 5: \n'
        'Question: Are there grids on the book? Please answer with Yes or No.\n'
        'Answer: aaa<><>>>>>>>\nYour output: Z\n'
        'Example 6: \n'
        'Question: Does the light blue circle overlap with the yellow circle? Please answer with Yes or No.\n'
        'Answer: No, the light blue circle does not overlap with the yellow circle\nYour output: No\n'
        'Example 7: \n'
        'Question: {}\nAnswer: {}\nYour output: '
    )
    return tmpl.format(question, prediction)


# ============================================================================
# LLM Matching Functions
# ============================================================================

def can_infer(answer):
    """Check if the answer can be inferred."""
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return False
    return True


def llm_match(model, question, prediction, question_type):
    """LLM-based answer matching for MCQ and YORN."""

    if model is None:
        return dict(opt='Fail', log='No model available for LLM matching.')

    # Select prompt builder and valid options based on question_type
    if question_type == 'YORN':
        prompt = build_prompt_yorn(question, prediction)
        valid_options = OPTIONS_YORN
    else:
        prompt = build_prompt_mcq(question, prediction)
        valid_options = OPTIONS_MCQ

    retry = 3
    while retry:
        ans = model.generate(prompt)
        if can_infer(ans):
            ret = str(ans).strip().upper()
            if ret in valid_options:
                return dict(opt=ret, log=ans)
            else:
                logger.warning(f'Invalid option extracted: {ret}')
        else:
            logger.warning(f'Cannot infer from response: {ans}')
        retry -= 1

    return dict(opt='Fail', log='Failed to extract answer via LLM.')


# ============================================================================
# Hybrid Matching Evaluation
# ============================================================================

def rm_model_special(pred):
    """Remove model special tokens from the prediction."""
    if '>\n\n' in pred:
        pred = pred.split('>\n\n')[-1]
    if '**\n\n' in pred:
        pred = pred.split('**\n\n')[-1]
    pred = pred.replace("\\[ \\boxed{", "")
    pred = pred.replace("} \\]", "")
    pred = pred.replace("<\uff5cend\u2581of\u2581sentence\uff5c>", "")
    pred = pred.replace("<|end_of_sentence|>", "")
    pred = pred.replace("</s>", "")
    pred = pred.replace("<CONCLUSION>", "")
    pred = pred.replace("</CONCLUSION>", "")
    pred = pred.replace("Falcon: ", "")
    pred = pred.strip()
    return pred


def hybrid_match(model, prediction, question, question_type):
    """Hybrid matching: template first, then LLM fallback."""
    pred = rm_model_special(prediction)

    # Step 1: Try template matching
    template_result = template_match(pred, question_type)
    if template_result != 'Fail':
        return template_result

    # Step 2: Fall back to LLM matching
    result = llm_match(model, question, prediction, question_type)
    return result['opt']


def _eval_single_row(args):
    """Process a single row for parallel execution."""
    model, row_dict = args
    prediction = str(row_dict.get('prediction', ''))
    answer = str(row_dict.get('answer', '')).strip().upper()
    question = str(row_dict.get('question', ''))
    question_type = str(row_dict.get('question_type', 'MCQ')).upper()

    matched = hybrid_match(model, prediction, question, question_type)
    return 1 if matched.upper() == answer else 0


def CoreCognition_eval(model, data, nproc=4):
    """Evaluate all predictions using hybrid matching with parallel processing."""
    tasks = [(model, row.to_dict()) for _, row in data.iterrows()]

    results = []
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        results_iterator = executor.map(_eval_single_row, tasks)
        for result in tqdm(results_iterator, total=len(tasks), desc="Evaluating CoreCognition"):
            results.append(result)

    return results


def CoreCognition_acc(data):
    """Calculate accuracy for CoreCognition benchmark."""
    results = {}

    # Overall accuracy
    results['Overall'] = data['correct'].mean() * 100

    # Accuracy by category (stage)
    if 'category' in data.columns:
        for cat in data['category'].unique():
            if pd.notna(cat):
                cat_data = data[data['category'] == cat]
                results[cat.replace(' ', '_')] = cat_data['correct'].mean() * 100

    # Accuracy by l2-category (concept)
    if 'l2-category' in data.columns:
        for l2cat in data['l2-category'].unique():
            if pd.notna(l2cat):
                l2cat_data = data[data['l2-category'] == l2cat]
                results[f'Concept_{l2cat}'] = l2cat_data['correct'].mean() * 100

    # Create DataFrame with rounded values
    acc_df = pd.DataFrame([{k: round(v, 2) for k, v in results.items()}])

    return acc_df
