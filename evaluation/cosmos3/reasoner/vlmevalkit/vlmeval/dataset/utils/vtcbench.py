import time
from typing import Literal

from rouge_score import rouge_scorer

from vlmeval.smp import istype
from .verifier import QUESTION_QUALITY_PROMPT_EN_NO_COT


def calc_vtc_metrics(
    response: str,
    gold_answers: list[str],
    metric: list[
        Literal[
            "EM",
            "contains",
            "contains_all",
            "lastline_EM",
            "lastline_contains",
            "ROUGE-L",
        ]
    ]
    | None = None,
) -> dict[str, float | int]:
    assert gold_answers is not None and len(gold_answers) > 0, (
        "gold_answers is None or empty"
    )
    # make sure gold answers are stripped strings, not int/float/etc.,
    # otherwise the 'contains' metric may fail
    gold_answers = [str(ans).strip() for ans in gold_answers]
    response = str(response).strip()
    if metric is None:
        metric = [
            "EM",
            "contains",
            "contains_all",
            "lastline_EM",
            "lastline_contains",
            "ROUGE-L",
        ]

    scores = {}
    for each_metric in metric:
        match each_metric:
            case "EM":
                scores[each_metric] = int(response.strip() in gold_answers)
            case "contains":
                scores[each_metric] = int(
                    any([f"{gold_answer}" in response for gold_answer in gold_answers])
                )
            case "contains_all":
                # all gold answers should be contained in the response
                # if so metric==1, can be fractional
                scores[each_metric] = float(
                    sum([f"{gold_answer}" in response for gold_answer in gold_answers])
                    / len(gold_answers)
                )
            case "lastline_EM":
                scores[each_metric] = int(
                    response.strip().split("\n")[-1] in gold_answers
                )
            case "lastline_contains":
                scores[each_metric] = int(
                    any(
                        [
                            gold_answer in response.strip().split("\n")[-1]
                            for gold_answer in gold_answers
                        ]
                    )
                )
            case "ROUGE-L":
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                if len(gold_answers) == 0:
                    scores[each_metric] = 0.0
                else:
                    scores[each_metric] = max(
                        s["rougeL"].fmeasure
                        for s in [scorer.score(response, ref) for ref in gold_answers]
                    )
            case _:
                raise ValueError(f"Invalid metric: {each_metric}")
    return scores


def process_vtc_line(line):
    ret = {}
    if istype(line['answer'], list):
        answers = eval(line['answer'])
    else:
        answers = [line['answer']]
    response = line['prediction']
    category = line['category']

    if category == 'Memory':
        metric = ['ROUGE-L']
    elif category == "Reasoning":
        metric = ['contains_all']
    elif category == "Retrieval":
        metric = ['contains']
    else:
        raise AssertionError

    result = calc_vtc_metrics(response, answers, metric)
    ret['score'] = result[metric[0]]
    ret['category'] = category
    ret['calc_metric'] = metric[0]

    return ret


def gpt_eval_vtcbemch(model, line):
    retry = 5
    ret = {}

    if istype(line['answer'], list):
        answers = eval(line['answer'])
    else:
        answers = [line['answer']]
    response = line['prediction']
    category = line['category']

    if category == 'Reasoning':
        metric = ['contains_all']
        result = calc_vtc_metrics(response, answers, metric)
        ret['score'] = result[metric[0]]
        ret['category'] = category
        ret['calc_metric'] = metric[0]
        return ret
    elif category == "Retrieval":
        metric = ['contains']
        result = calc_vtc_metrics(response, answers, metric)
        ret['score'] = result[metric[0]]
        ret['category'] = category
        ret['calc_metric'] = metric[0]
        return ret
    elif category == 'Memory':  # use gpt to judge the quality of the response in memory category
        prompt = QUESTION_QUALITY_PROMPT_EN_NO_COT.format(
            question=line['question'], gold_answer=answers, llm_response=response)

        retry = 5
        ret['calc_metric'] = 'gpt_judge'
        for i in range(retry):
            try:
                judge_response = model.generate(prompt, temperature=1).strip()

                if judge_response == 'A':
                    ret['score'] = 1
                    ret['category'] = category
                elif judge_response == 'B':
                    ret['score'] = 0
                    ret['category'] = category
                elif judge_response == 'C':
                    ret['score'] = 0
                    ret['category'] = category
                else:
                    raise RuntimeError(f'Invalid judge response: {judge_response}')
                break
            except Exception:
                time.sleep(1)

        if i == retry - 1:
            ret['score'] = ""
            ret['category'] = category
            raise RuntimeError('GPT evsal failed after retries')

        return ret
