import json

import numpy as np
import pandas as pd

from vlmeval.smp import load

FAIL_MSG = 'Failed to obtain answer via API.'

SYSTEM_CAL_SCORE_PROMPT = """You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.
Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:
------
##INSTRUCTIONS:
- Focus on the meaningful match between the predicted answer and the correct answer.
- Consider synonyms or paraphrases as valid matches.
- Evaluate the correctness of the prediction compared to the answer.
"""  # noqa: E501

USER_CAL_SCORE_PROMPT = """Please evaluate the following video-based question-answer pair:

Question: {question}
Correct Answer: {answer}
Predicted Answer: {pred_response}

Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.
Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.
DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string.
For example, your response should look like this: \{'pred': 'yes', 'score': 4.8\}.
"""  # noqa: W605, E501

SYSTEM_GENER_PRED_PROMPT = """You are an intelligent chatbot designed for providing accurate answers to questions related to the content based on a detailed description of a video or image.
Here's how you can accomplish the task:
------
##INSTRUCTIONS:
- Read the detailed description carefully.
- Answer the question only based on the detailed description.
- The answer should be a short sentence or phrase.
"""  # noqa: E501

USER_GENER_PRED_PROMPT = """Please provide accurate answers to questions related to the content based on a detailed description of a video or image:

detailed description: {pred_cap}
question: {q}

DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide short but accurate answer."""  # noqa: E501

VDC_DIMENSIONS = {
    'short': ['short'],
    'detailed': ['detailed'],
    'background': ['background'],
    'main_object': ['main_object'],
    'camera': ['camera'],
    'overall': []
}

L3_DIMS = []
for k, v in VDC_DIMENSIONS.items():
    if k != 'overall':
        L3_DIMS.extend(v)
        VDC_DIMENSIONS['overall'].extend(v)


def get_dimension_rating(data_path):
    data = load(data_path)
    coarse_rating = {k: [] for k in VDC_DIMENSIONS}
    coarse_acc = {k: [] for k in VDC_DIMENSIONS}

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
                except Exception:
                    print(f"Failed to parse score_dict: {score_dict}")
                    return None
        return None

    for i in range(len(data)):
        caption_type = data.iloc[i]['caption_type'].lower()  # Convert to lowercase
        score_dict = parse_score_dict(data.iloc[i]['score'])

        if score_dict and isinstance(score_dict,
                                     dict) and 'pred' in score_dict and 'score' in score_dict:
            score = score_dict['score']
            is_correct = 1 if score_dict['pred'].lower() == 'yes' else 0
        else:
            score = -1
            is_correct = -1

        # Map caption types to their lowercase versions
        if caption_type in ['short', 'detailed', 'background', 'main_object', 'camera']:
            coarse_rating[caption_type].append(score)
            coarse_rating['overall'].append(score)

            if is_correct != -1:
                coarse_acc[caption_type].append(is_correct)
                coarse_acc['overall'].append(is_correct)

    coarse_valid = {
        k: f'{np.mean([x for x in v if x >= 0]):.2f}'
        for k, v in coarse_rating.items()
    }
    coarse_accuracy = {k: f'{np.mean(v):.2f}' if v else '0.00' for k, v in coarse_acc.items()}

    return dict(coarse_valid=coarse_valid, coarse_accuracy=coarse_accuracy)


def prepare_response_prompt(item):
    """
    Prepare messages for response generation

    Args:
        item: DataFrame row containing pred_cap and question

    Returns:
        list: List of message dictionaries for the model
    """
    return USER_GENER_PRED_PROMPT.format(pred_cap=item['prediction'], q=item['question'])


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

    # prompt = USER_CAL_SCORE_PROMPT.format(
    #     question=item['question'],
    #     answer=item['answer'],
    #     pred_response=item['pred_response']
    # )

    prompt = f"""Please evaluate the following video-based question-answer pair:\n\n
            Question: {item['question']}\n
            Correct Answer: {item['answer']}\n
            Predicted Answer: {item['pred_response']}\n\n
            Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.
            Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING.
            DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string.
                For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."""  # noqa: E501

    return prompt
