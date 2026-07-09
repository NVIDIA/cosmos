import logging
import math
import re

logger = logging.getLogger(__name__)


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].lower().strip().replace("\n", " ")
            except Exception as e:
                logger.warning(f"Failed to process answer {j}: {type(e).__name__}: {e}")
                continue
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n", " ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def cn_vqa_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].lower().strip().replace("\n", " ").replace(" ", "")
            except Exception as e:
                logger.warning(f"Failed to process cn answer {j}: {type(e).__name__}: {e}")
                continue
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n", " ").replace(" ", "")
            if len(answer.split(",")) < 4:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n", " ").replace(" ", "")
        predict = predict.lower().strip().replace("\n", " ").replace(" ", "")
        if len(answer.split(",")) < 4:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def vqa_evaluation_case_sensitive(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            try:
                answer = answers[j].strip().replace("\n", " ")
            except Exception as e:
                logger.warning(f"Failed to process case-sensitive answer {j}: {type(e).__name__}: {e}")
                continue
            predict = predict.strip().replace("\n", " ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.strip().replace("\n", " ")
        predict = predict.strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def extract_first_number(string):
    match = re.search(r'\d+', string)
    if match:
        return int(match.group())
    return None


def counting_evaluation(predict, answers, eval_method):
    score = 0

    if isinstance(predict, str):
        predict_processed = predict.lower().strip().replace("\n", " ")
    elif math.isnan(predict):
        return 0
    else:
        predict_processed = int(predict)
    if isinstance(answers, list):
        temp_score = 0
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n", " ")
            if eval_method == "exact match":
                if answer in predict:
                    score = 1
                else:
                    score = 0
            elif eval_method == "regression":
                predict_number = extract_first_number(predict_processed)
                if predict_number:

                    answer = int(answer)

                    if predict_number <= 0 or predict_number >= 2 * answer:
                        score = 0
                    else:
                        iou = 1 - abs(predict_number - answer) / answer
                        if iou > 0.5:
                            score = iou
                        else:
                            score = 0
                else:
                    score = 0
            if score > temp_score:
                temp_score = score
        score = temp_score

    else:
        answers = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if eval_method == "exact match":
            if answer in predict:
                score = 1
            else:
                score = 0
        elif eval_method == "regression":
            predict = extract_first_number(predict)
            if predict:
                answer = int(answer)
                if predict <= 0 or predict >= 2 * answer:
                    score = 0
                else:
                    iou = 1 - abs(predict - answer) / answer

                    if iou > 0.5:
                        score = iou
                    else:
                        score = 0
            else:
                score = 0
    return score


def math_expression_evaluation(predict, answers):
    score = 0
    if isinstance(answers, list):
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n", " ").replace(" ", "")
            predict = predict.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace("\n", " ").replace(" ", "")
        predict = predict.strip().replace("\n", " ").replace(" ", "")
        if answers in predict:
            score = 1
    return score


def remove_text_tags(latex_str):
    """
    Removes LaTeX \text{...} tags while keeping their content.

    :param latex_str: A string containing LaTeX expressions
    :return: The processed string with \text{...} tags removed
    """

    pattern = r'\\text\{([^{}]*)\}'

    processed_str = re.sub(pattern, r'\1', latex_str)

    return processed_str


def cn_math_expression_evaluation(predict, answers):
    score = 0

    assert len(answers) == 1
    answers = [remove_text_tags(answers[0])]
    predict = remove_text_tags(predict)

    if isinstance(answers, list):
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n", " ").replace(" ", "")
            predict = predict.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace("\n", " ").replace(" ", "")
        predict = predict.strip().replace("\n", " ").replace(" ", "")
        if answers in predict:
            score = 1
    return score


if __name__ == "__main__":
    test_predict = "apple pie and banana"
    test_answers = ["apple", "banana pie", "apple pie and orange"]

    vqa_score = vqa_evaluation(test_predict, test_answers)
    print(f"VQA evaluation score for predict '{test_predict}' and answers {test_answers}: {vqa_score}")
