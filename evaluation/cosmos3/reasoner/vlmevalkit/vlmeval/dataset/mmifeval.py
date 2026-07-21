import json
import os
import os.path as osp
import re

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, get_logger, load, toliststr
from vlmeval.utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

logger = get_logger(__name__)

aux_data_dict = {}
judge_model = None

# img_dict = {}
# <<< prompt >>>


def generate_eval_pt_c_level(constraints, prediction):
    constraints_str = "\n".join(
        [f"Constraint_{i + 1}: {constraint['value']}" for i, constraint in enumerate(constraints)]
    )
    pt = f"""\
Your task is to evaluate whether the response from an AI assistant adheres to all of the given constraints. \
Please follow the requirements below to make the judgment:
1. Be strict and consistent in your assessment.
2. You should refer to the content of image to make the judgment.
3. For each constraint, if the response fails to fully meet the constraint, \
give it a score of 0. Otherwise, give it a score of 1.

<start of response>
{prediction}
<end of response>

<start of constraint list>
{constraints_str}
<end of constraint list>

You must evaluate and provide an explanation for each constraint listed, ensuring no constraint is omitted. \
At the end, summarize the scores for all constraints in one sentence.

Your output should strictly follow the format below:
Judgement: ...
Summary: Score of constraint_1: x/1, Score of constraint_2: x/1, Score of constraint_3: x/1, ..., Score of \
constraint_n: x/1.
"""
    return pt


def generate_eval_pt_p_level(question, prediction, ground_truth):
    pt = f"""\
You are an expert evaluator. Your task is to extract the answer from the model output and \
compare it with the ground truth list \
to determine whether the model answer covers all the points in the ground truth list. \
The ground truth list is provided as a JSON array of strings, and the model answer is a text string. \
An answer is considered correct if every element from the ground truth list appears in the model \
answer (substring matching is acceptable). \
The order does not matter. \

Your response should only be 'right' if the model answer fully covers the ground truth, or 'wrong' if it does not. \
Do not provide any additional commentary.

Question: {question}
Response from the model: {prediction}
Ground Truth List: {ground_truth}
"""
    return pt


def generate_cmp_pt(constraint, pred_with_constraint, pred_without_constraint):
    pt = f"""\
You are an expert in judging whether the respone follow the given constraint. \
Your task is to assess whether the model's response satisfies \
the given constraint and return True or False. I will provide you \
with the constraint and the model's response under this constraint. \
To assist with your evaluation, I will also provide you with the model's response \
to the same question without the constraint.

<start of constraint>
{constraint}
<end of constraint>

<start of response under the constraint>
{pred_with_constraint}
<end of response under the constraint>

<start of response without the constraint>
{pred_without_constraint}
<end of response without the constraint>

**Please follow the steps below to evaluate**:
Step 1. Compare the model's response under the constraint with its response without the constraint. \
If you believe these two answers \
are very similar, it means the model has not fully considered the impact of the constraint on the answer. \
Please return False.
Step 2. Compare the model's response under the constraint with the content of the constraint. If you believe the model's response \
does not meet the requirements specified in the constraint, return False. Otherwise, \
if the response effectively satisfies the constraint, return True.

Start by briefly explaining your reasoning based on the above steps. At the end, provide a one-sentence \
summary of your evaluation.

Your output must strictly follow this format:
Reasoning: ...
Summary: "True" / "False".
"""  # noqa: E501
    return pt


# <<< re >>>
# extract score from gpt_resp
# format: Score of instruction: x/1, Score of constraint_1: y/1, ..., Score of constraint_n: w/1.
# return: score_dict {'instruction': x/1, 'constraint_1': y/1,
# 'constraint_2': z/1, ..., 'constraint_n': w/1}


def extract_score_from_direct_gpt_resp(raw_score):
    # Define regular expression patterns (updated to handle underscores in
    # constraint names)
    score_pattern = re.compile(r"Score\s+of\s+([a-zA-Z0-9_\-]+):\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)

    # Clean the raw score to remove unnecessary symbols (e.g., newlines,
    # multiple spaces)
    # Normalize whitespace
    cleaned_score = re.sub(r"\s+", " ", raw_score).strip()
    # delete all the '*'
    cleaned_score = re.sub(r"\*", "", cleaned_score)

    # Find all individual component scores
    score_matches = score_pattern.findall(cleaned_score)

    # If no valid score matches found, print and raise an exception
    if not score_matches:
        raise ValueError("raw_score format is incorrect, cannot parse scores")

    score_dict = {}

    # Parse each component score
    for match in score_matches:
        component_name = match[0].strip().lower()  # Component name, converted to lowercase
        component_name = component_name.replace(" ", "_")
        numerator = int(match[1])  # Numerator
        denominator = int(match[2])  # Denominator
        score = numerator / denominator  # Calculate the score
        score_dict[component_name] = score  # Store it in the dictionary

    return score_dict


# extract score from gpt_resp
# format: right or wrong
# return: score


def extract_score_from_p_level_gpt_resp(raw_score):
    if raw_score == "right":
        return 1
    elif raw_score == "wrong":
        return 0
    else:
        # try to find "right" or "wrong" in the raw_score
        if re.search(r"right", raw_score, re.IGNORECASE):
            return 1
        elif re.search(r"wrong", raw_score, re.IGNORECASE):
            return 0
        else:
            raise ValueError("raw_score format is incorrect, cannot parse scores")


# extract score from gpt_resp
# format: True or False
# return: score


def extract_score_from_cmp_gpt_resp(response_text):
    # Step 1: Find the last occurrence of 'summary:'
    summary_idx = response_text.lower().rfind("summary")
    if summary_idx == -1:
        raise ValueError("No 'summary' found in response.")

    # Step 2: Slice the string after 'summary:' and extract value
    after_summary = response_text[summary_idx + len("summary"):]

    # Match true/false ignoring markdown and formatting
    match = re.search(r"\b(true|false)\b", after_summary, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        return 1 if value == "true" else 0

    raise ValueError("No valid 'True' or 'False' found after 'summary'.")


# <<< gpt >>>


def run_once_with_image(pt, image, retry=4):
    global judge_model
    prefix = "data:image/jpeg;base64,"
    img = prefix + image
    messages = [dict(type="text", value=pt), dict(type="image", value=img)]
    while retry:
        try:
            ans = judge_model.generate(messages)
            return ans
        except Exception as e:
            logger.info(f"Error in run_once_with_image: {e}")
            retry -= 1
    return ans


def run_once_without_image(pt, retry=3):
    global judge_model
    messages = [
        dict(type="text", value=pt),
    ]
    while retry:
        try:
            ans = judge_model.generate(messages)
            return ans
        except Exception as e:
            logger.info(f"Error in run_once_without_image: {e}")
            retry -= 1
    return ans


# <<< score >>>


def judge_one_item(item, retry=3):
    global aux_data_dict
    item = json.loads(item)
    num_retry = 0
    while num_retry < retry:
        if item.get("tag", None) == "P-Level":
            # in tsv file, answer is a string, need to be converted to list
            pt = generate_eval_pt_p_level(item["question"], item["prediction"], json.loads(item["answer"]))
            gpt_resp = run_once_without_image(pt)
            try:
                score = extract_score_from_p_level_gpt_resp(gpt_resp)
                return (
                    0,
                    "success",
                    {
                        "total_score": score,
                        "gpt_resp": gpt_resp,
                    },
                )
            except Exception as e:
                logger.error(f"\nError:\n{e}\nItem:\n{item}\ngpt_resp:\n{gpt_resp}\n")
                num_retry += 1
                continue
        else:  # process C-Level data
            # split into direct_gpt and other
            # direct_gpt can be processed in batch
            # other needs to be processed one by one
            constraint_direct_gpt = []
            constraint_other = []
            for constraint in json.loads(item["constraints"]):
                method = constraint["judge"]["method"]
                if method == "direct_gpt":
                    constraint_direct_gpt.append(constraint)
                else:
                    constraint_other.append(constraint)
            score_dict = {}
            # 1. process direct_gpt: if there is no direct_gpt, instruction is also
            # needed
            if len(constraint_direct_gpt) > 0:
                pt_direct_gpt = generate_eval_pt_c_level(constraint_direct_gpt, item["prediction"])
                gpt_resp = run_once_with_image(pt_direct_gpt, item["image"])
                try:
                    direct_gpt_score_dict = extract_score_from_direct_gpt_resp(gpt_resp)
                    score_dict["gpt_resp_direct_gpt"] = gpt_resp
                    for i, constraint in enumerate(constraint_direct_gpt):
                        score_dict[constraint["key"]] = direct_gpt_score_dict[f"constraint_{i + 1}"]
                except Exception as e:
                    logger.error(
                        f"\nError:\n{e}\nItem:\n{item}\npt_direct_gpt:\n{pt_direct_gpt}\ngpt_resp:\n{gpt_resp}"
                    )
                    num_retry += 1
                    continue
            # 2. process rule_based
            for constraint in constraint_other:
                if constraint["judge"]["method"] == "rule_based":
                    # call function according to constraint["judge"]["verify_funcs"]
                    # maybe a list of function names (str)
                    # func in function_and_compare.py
                    # example: {"method": "rule_based", "verify_funcs": [{"func":
                    # "check_whether_response_paragraph_number_in_range", "params":
                    # [3, 3]}]}}
                    score = 1.0
                    # breakpoint()
                    for func_dict in constraint["judge"]["verify_funcs"]:
                        func = globals()[func_dict["func"]]
                        # use * to unpack the list, ** is used for dict
                        judge_result = func(str(item["prediction"]), *func_dict["params"])
                        # breakpoint()
                        if not judge_result:  # False -> score = 0
                            score = 0.0
                            break
                    # breakpoint()
                    score_dict[constraint["key"]] = score
            # 3. process cmp_gpt
            for constraint in constraint_other:
                if constraint["judge"]["method"] == "cmp_gpt":
                    del_cons_prediction = aux_data_dict[item["id"]][constraint["key"]]
                    pt = generate_cmp_pt(constraint["value"], item["prediction"], del_cons_prediction)
                    gpt_resp = run_once_without_image(pt)
                    try:
                        score = extract_score_from_cmp_gpt_resp(gpt_resp)
                        score_dict[constraint["key"]] = score
                        score_dict[f"gpt_resp_cmp_gpt_{constraint['key']}"] = gpt_resp
                    except Exception as e:
                        logger.error(f"\nError:\n{e}\nItem:\n{item}\ngpt_resp:\n{gpt_resp}")
                        num_retry += 1
                        continue
            # add total_score
            total_score = 0.0
            cnt = 0
            for key, value in score_dict.items():
                if key.startswith("gpt_resp_"):
                    continue
                total_score += value
                cnt += 1
            score_dict["total_score"] = total_score / cnt
            logger.debug(f"score_dict:\n{score_dict}")
            return 0, "success", score_dict
    return 1, "C-Level, fail in judge", {}


class MMIFEval(ImageBaseDataset):
    TYPE = "VQA"

    # TODO: add dataset url and md5
    DATASET_URL = {"MM-IFEval": 'https://opencompass.openxlab.space/utils/VLMEval/MM-IFEval.tsv'}
    DATASET_MD5 = {
        "MM-IFEval": '973bb839961a449565073a5ee70ae7a6'
    }

    # Given one data record, return the built prompt (a multi-modal message), can override
    # Actually, all lines have single image
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_path"])
        else:
            tgt_path = self.dump_image(line)

        question = line["question"]

        # save images for evaluation
        # global img_dict
        # img_dict[line["index"]] = line["image"]

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]

        # WildVision adopts text first
        msgs = [dict(type="text", value=question)] + msgs

        return msgs

    # score for the infer file
    # @classmethod

    def evaluate(self, eval_file, **judge_kwargs):
        raw_bench_data = MMIFEval("MM-IFEval").data
        global aux_data_dict
        model = judge_kwargs["model"]
        storage = get_intermediate_file_path(eval_file, f"_{model}", "jsonl")
        score_file = get_intermediate_file_path(eval_file, f"_{model}_score", "csv")
        tmp_file = get_intermediate_file_path(eval_file, f"_{model}_tmp", "pkl")
        nproc = judge_kwargs.pop("nproc", 4)

        data_all = load(eval_file).to_dict(orient="records")

        main_data = []
        aux_data = []
        for i, line in enumerate(data_all):
            if line.get("infer_type", None) == "main":
                main_data.append(line)
            else:
                aux_data.append(line)

            line["image"] = raw_bench_data.iloc[i]["image"]

        aux_data_dict = {}
        for line in aux_data:
            assert line["infer_type"] == "aux_cmp_gpt"
            del_cons = line["del_cons"]
            if line["id"] not in aux_data_dict:
                aux_data_dict[line["id"]] = {}
            aux_data_dict[line["id"]][del_cons] = line["prediction"]

        # params
        params_all = [json.dumps(item) for item in main_data]
        indices_all = [line["id"] for line in main_data]

        ans = {}
        if os.path.exists(tmp_file):
            ans_tuples = load(tmp_file)
            for k, v in ans_tuples.items():
                if v[0] == 0:
                    ans[k] = {"eval_ret_code": v[0], "eval_msg": v[1], "eval_score_dict": v[2]}
            # ans is a dict
            logger.info(f"Tmp file exists, loaded {len(ans)} data from {tmp_file}")

        tups = [x for x, i in zip(params_all, indices_all) if i not in ans]
        indices = [i for i in indices_all if i not in ans]

        # judge
        if not osp.exists(storage):
            # judge_kwargs['system_prompt'] = SYSTEM_PROMPT
            judge_kwargs["temperature"] = 0
            judge_kwargs["img_detail"] = "high"
            judge_kwargs["timeout"] = 300
            global judge_model
            judge_model = build_judge(max_tokens=4096, **judge_kwargs)

            assert judge_model.working(), "MMIFEval evaluation requires a working OPENAI API\n" + DEBUG_MESSAGE

            if len(indices):
                new_results = track_progress_rich(
                    judge_one_item,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                for k, v in zip(indices, new_results):
                    ans[k] = {"eval_ret_code": v[0], "eval_msg": v[1], "eval_score_dict": v[2]}
            else:
                for k, v in ans.items():
                    if isinstance(v, tuple):
                        ans[k] = {"eval_ret_code": v[0], "eval_msg": v[1], "eval_score_dict": v[2]}
            for item in main_data:
                item.pop("image")

            for item in main_data:
                item["eval_ret_code"] = ans[item["id"]]["eval_ret_code"]
                item["eval_msg"] = ans[item["id"]]["eval_msg"]
                item["eval_score_dict"] = ans[item["id"]]["eval_score_dict"]
            # storage is a jsonl file
            with open(storage, "w") as f:
                for item in main_data:
                    f.write(json.dumps(item) + "\n")

        eval_data = load(storage)
        # eval_data = [json.loads(line) for line in eval_data]
        # calculate P-Level scores
        p_level_score_sum = 0
        c_level_score_sum = 0
        p_level_cnt = 0
        c_level_cnt = 0
        for line in eval_data:
            if line["tag"] == "P-Level":
                p_level_score_sum += line["eval_score_dict"]["total_score"]
                p_level_cnt += 1
            elif line["tag"] == "C-Level":
                c_level_score_sum += line["eval_score_dict"].get("total_score", 0.)
                c_level_cnt += 1
        p_level_accuracy = p_level_score_sum / p_level_cnt
        c_level_accuracy = c_level_score_sum / c_level_cnt
        # save to score_file
        score_dict = {
            "p_level_accuracy": [p_level_accuracy],
            "c_level_accuracy": [c_level_accuracy],
            "p_level_cnt": [p_level_cnt],
            "c_level_cnt": [c_level_cnt],
            "overall_accuracy": [
                (p_level_accuracy * p_level_cnt + c_level_accuracy * c_level_cnt) / (p_level_cnt + c_level_cnt)
            ],
        }
        score_df = pd.DataFrame(score_dict)
        dump(score_df, score_file)

        return score_df
