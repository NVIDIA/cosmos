# flake8: noqa
import json

import numpy as np
import pandas as pd

from vlmeval.smp import load

FAIL_MSG = 'Failed to obtain answer via API.'


CAL_SCORE_PROMPT = """Please evaluate the following video-based question-answer pair:

Question: {question}
Correct Answer: {answer}
Predicted Answer: {pred_response}

Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.
Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.
DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string.
For example, your response should look like this: \{'pred': 'yes', 'score': 4.8\}.
"""

MOVIECHAT1K_DIMENSIONS = {
    'global': ['global'],
    'breakpoint': ['breakpoint'],
    'overall': []
}

L3_DIMS = []
for k, v in MOVIECHAT1K_DIMENSIONS.items():
    if k != 'overall':
        L3_DIMS.extend(v)
        MOVIECHAT1K_DIMENSIONS['overall'].extend(v)


def get_dimension_rating(data_path):
    data = load(data_path)
    coarse_rating = {k: [] for k in MOVIECHAT1K_DIMENSIONS}
    coarse_acc = {k: [] for k in MOVIECHAT1K_DIMENSIONS}

    def parse_score_dict(score_dict):
        """Helper function to parse score dictionary string"""
        if isinstance(score_dict, dict):
            return score_dict

        if isinstance(score_dict, str):
            try:
                # First try standard json loading
                return json.loads(score_dict)
            except json.JSONDecodeError:
                try:
                    # If that fails, try eval (safer than literal_eval for this case)
                    return eval(score_dict)
                except:
                    print(f"Failed to parse score_dict: {score_dict}")
                    return None
        return None

    for i in range(len(data)):
        mode = data.iloc[i]['mode'].lower()  # Convert to lowercase
        score_dict = parse_score_dict(data.iloc[i]['score'])
        if score_dict and isinstance(score_dict, dict) and 'pred' in score_dict and 'score' in score_dict:
            score = score_dict['score']
            is_correct = 1 if score_dict['pred'].lower() == 'yes' else 0
        else:
            score = -1
            is_correct = -1

        # Map caption types to their lowercase versions
        if mode in ['global', 'breakpoint']:
            coarse_rating[mode].append(score)
            coarse_rating['overall'].append(score)

            if is_correct != -1:
                coarse_acc[mode].append(is_correct)
                coarse_acc['overall'].append(is_correct)


    coarse_valid = {k: f'{np.mean([x for x in v if x >= 0]):.2f}' for k, v in coarse_rating.items()}
    coarse_accuracy = {k: f'{np.mean(v):.2f}' if v else '0.00' for k, v in coarse_acc.items()}

    return dict(
        coarse_valid=coarse_valid,
        coarse_accuracy=coarse_accuracy
    )



def prepare_score_prompt(item):
    """
    Prepare messages for score evaluation

    Args:
        item: DataFrame row containing question, answer, and prediction

    Returns:
        list: List of message dictionaries for the model
    """
    # Convert Series to dictionary if needed
    if isinstance(item, pd.Series):
        item = item.to_dict()


    prompt = f"""Please evaluate the following video-based question-answer pair:\n\n
            Question: {item['question']}\n
            Correct Answer: {item['answer']}\n
            Predicted Answer: {item['prediction']}\n\n
            Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.
            Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.
            DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string.
                For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."""

    return prompt
