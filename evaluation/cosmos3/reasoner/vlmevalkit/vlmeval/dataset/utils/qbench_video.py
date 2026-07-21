from vlmeval.smp import load

FAIL_MSG = 'Failed to obtain answer via API.'

VQA_JUDGE_SYS_PROMPT = """
You are a helpful assistant that grades answers related to visual video quality.
There are a lot of special terms or keywords related to video processing and photography.
You will pay attention to the context of `quality evaluation' when grading.
"""

VQA_JUDGE_USER_PROMPT = """
Given the question {}, evaluate whether the response {} completely matches the correct answer {}.
First, check the response and please rate score 0 if the response is not a valid answer.
Please rate score 2 if the response completely or almost completely matches the correct answer on completeness, accuracy, and relevance.
Please rate score 1 if the response partly matches the correct answer on completeness, accuracy, and relevance.
Please rate score 0 if the response doesn't match the correct answer on completeness, accuracy, and relevance at all.
Please only provide the result in the following format: Score:'
"""  # noqa: E501


def check_ans_mcq(pred, ans, correct_choice, correct_answer):
    flag = False

    if correct_choice == pred or correct_choice + "." in pred or correct_answer == pred:
        flag = True
    elif correct_choice in pred.split("\n"):
        flag = True

    return flag


def check_ans_vqa(model, line):
    score = model.generate(VQA_JUDGE_USER_PROMPT.format(line['question'], line['prediction'], line['answer'])).strip()
    return score


def get_dimension_rating(score_file):
    score = load(score_file)
    result_dict = {}
    for idx, item in score.iterrows():
        question_type = eval(item['dimensions'])[0].split(',')[0]
        if question_type not in result_dict:
            result_dict[question_type] = [0, 0]
        result_dict[question_type][0] += int(item['score'])
        result_dict[question_type][1] += 1
    return result_dict
