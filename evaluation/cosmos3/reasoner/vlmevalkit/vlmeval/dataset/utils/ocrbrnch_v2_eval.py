# flake8: noqa
import argparse
import ast
import json
import os
import re

import ipdb
import numpy as np
from tqdm import tqdm

from .Ocrbench_v2.IoUscore_metric import (calculate_iou, extract_coordinates,
                                          vqa_with_position_evaluation)
from .Ocrbench_v2.page_ocr_metric import cal_per_metrics
from .Ocrbench_v2.spotting_metric import extract_bounding_boxes_robust, spotting_evaluation
from .Ocrbench_v2.TEDS_metric import (TEDS, compute_f1_score, convert_markdown_table_to_html,
                                      convert_str_to_dict, convert_str_to_multi_dict, dict_to_html,
                                      doc_parsing_evaluation, generate_combinations,
                                      wrap_html_table)
from .Ocrbench_v2.vqa_metric import (cn_math_expression_evaluation, cn_vqa_evaluation,
                                     counting_evaluation, math_expression_evaluation,
                                     vqa_evaluation, vqa_evaluation_case_sensitive)


def is_nan_value(value):
    if value is None:
        return True
    if isinstance(value, str) and value.lower() == 'nan':
        return True
    try:
        import pandas as pd
        if pd.isna(value):
            return True
    except:
        pass
    return False


def get_value_or_zero(value):
    return 0.0 if value is None else value


def calculate_average(scores_dict):
    averages = {key: sum(values) / len(values) for key, values in scores_dict.items() if len(values) > 0}
    return averages


def process_predictions(predict_file):
    teds = TEDS(n_jobs=32)

    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file)):
        if data_item["type"] == "APP agent en" or data_item["type"] == "ASCII art classification en" or data_item["type"] == "math QA en" \
            or data_item["type"] == "reasoning VQA en" or data_item["type"] == "science QA en" \
            or data_item["type"] == "text recognition en" or data_item["type"] == "document classification en" \
            or data_item["type"] == "cognition VQA en" or data_item["type"] == "diagram QA en":
            if "eval" in data_item.keys():
                if data_item["eval"] == "multiple choice":
                    if not isinstance(data_item["answers"], list):
                        data_item["answers"] = [data_item["answers"]]
                    assert len(data_item["answers"]) == 1

                    if not isinstance(data_item["predict"], str):
                        data_item["score"] = 0
                    else:
                        predict = ''.join(c for c in data_item["predict"] if c.isalpha())

                        if predict == data_item["answers"][0]:
                            data_item["score"] = 1
                        else:
                            data_item["score"] = 0
                elif data_item["eval"] == "case sensitive":
                    data_item["score"] = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
                else:
                    raise ValueError("No such evaluation method")
            else:
                data_item["score"] = vqa_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "cognition VQA cn" or data_item["type"] == "reasoning VQA cn":
            if "eval" in data_item.keys():
                if data_item["eval"] == "multiple choice":
                    assert len(data_item["answers"]) == 1
                    predict = ''.join(c for c in data_item["predict"] if c.isalpha())

                    if predict == data_item["answers"][0]:
                        data_item["score"] = 1
                    else:
                        data_item["score"] = 0
                elif data_item["eval"] == "case sensitive":
                    data_item["score"] = vqa_evaluation_case_sensitive(data_item["predict"], data_item["answers"])
                else:
                    raise ValueError("No such evaluation method")
            else:
                data_item["score"] = cn_vqa_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "handwritten answer extraction cn":
            if "简答" in data_item["question"]:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) +
                    get_value_or_zero(ocr_metric["meteor"]) +
                    get_value_or_zero(ocr_metric["f_measure"]) +
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
            else:
                assert len(data_item["answers"]) == 1
                answer = data_item["answers"][0]
                chars = list(answer)
                if len(answer) > 1:

                    answer_list = [
                        "".join(chars),
                        ".".join(chars),
                        ". ".join(chars),
                        ",".join(chars),
                        ", ".join(chars),
                        "、".join(chars),
                        ";".join(chars),
                        "; ".join(chars),
                        " ".join(chars),
                        "和".join(chars)
                    ]
                    max_score = 0
                    for answer in answer_list:
                        if answer in data_item["predict"]:
                            temp_score = 1
                        else:
                            temp_score = 0
                        if temp_score > max_score:
                            max_score = temp_score
                    data_item["score"] = max_score

                else:
                    if data_item["answers"][0] in data_item["predict"]:
                        data_item["score"] = 1
                    else:
                        data_item["score"] = 0

        elif data_item["type"] == "formula recognition cn":
            if is_nan_value(data_item["predict"]):
                data_item["score"] = 0
            else:
                data_item["score"] = cn_math_expression_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "text counting en":
            data_item["score"] = counting_evaluation(data_item["predict"], data_item["answers"], data_item["eval"])

        elif data_item["type"] == "formula recognition en":
            data_item["score"] = math_expression_evaluation(data_item["predict"], data_item["answers"])

        elif data_item["type"] == "table parsing en":
            if type(data_item["answers"])==list and len(data_item["answers"]) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                elif not isinstance(data_item["question"], str):
                    data_item["ignore"] = "True"
                    data_item["score"] = 0

                elif "html" in data_item["question"].lower():
                    no_find = False
                    predict_table = data_item["predict"].replace('\n','')
                    if "<body" in predict_table:
                        predict_table = re.findall('<body.*', predict_table)[0]
                    elif "<table" in predict_table:
                        predict_table = re.findall('<table.*', predict_table)[0]
                    else:
                        no_find = True

                    if no_find:
                        data_item["score"] = 0
                    else:
                        pred_table_html = wrap_html_table(predict_table)
                        gold_table_html = wrap_html_table(data_item["answers"][0])
                        try:
                            data_item["score"] = teds.evaluate(pred_table_html, gold_table_html)
                        except:
                            data_item["score"] = 0

                elif "markdown" in data_item["question"].lower():
                    if not isinstance(data_item["predict"], str):

                        prediction = str(data_item["predict"])
                        pred_table_html = convert_markdown_table_to_html(prediction)
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)

                    else:
                        pred_table_html = convert_markdown_table_to_html(data_item["predict"])
                        gt_table_html = convert_markdown_table_to_html(data_item["answers"][0])
                        data_item["score"] = teds.evaluate(pred_table_html, gt_table_html)
            else:
                raise ValueError

        elif data_item["type"] == "table parsing cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                no_find = False
                predict_table = data_item["predict"].replace('\n','')
                if "<body" in predict_table:
                    predict_table = re.findall('<body.*', predict_table)[0]
                elif "<table" in predict_table:
                    predict_table = re.findall('<table.*', predict_table)[0]
                else:
                    no_find = True

                if no_find:
                    data_item["score"] = 0
                else:
                    pred_table_html = wrap_html_table(predict_table)
                    gold_table_html = wrap_html_table(data_item["answers"][0])
                    try:
                        data_item["score"] = teds.evaluate(pred_table_html, gold_table_html)
                    except:
                        data_item["score"] = 0
                        print("error")

        elif data_item["type"] == "chart parsing en":
            answer = data_item["answers"][0]
            if data_item["predict"]:

                pred_chart_dict = convert_str_to_multi_dict(data_item["predict"])
                if len(pred_chart_dict) == 0:
                    data_item["score"] = 0
                else:
                    pred_chart_html = dict_to_html(pred_chart_dict)
                    gt_chart_html = dict_to_html(answer)
                    data_item["score"] = teds.evaluate(pred_chart_html, gt_chart_html)
            else:
                data_item["score"] = 0

        elif data_item["type"] == "document parsing en":
            assert type(data_item["answers"])==list and len(data_item["answers"]) == 1
            data_item["score"] = doc_parsing_evaluation(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "document parsing cn":
            assert type(data_item["answers"])==list and len(data_item["answers"]) == 1
            data_item["score"] = doc_parsing_evaluation(data_item["predict"], data_item["answers"][0])

        elif data_item["type"] == "key information extraction en" or data_item["type"] == "key information mapping en":
            assert len(data_item["answers"]) == 1
            answers = generate_combinations(data_item["answers"][0])

            if type(answers)==list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "key information extraction cn":
            assert len(data_item["answers"]) == 1
            answers = ast.literal_eval(data_item["answers"][0])
            answers = {k: v if isinstance(v, list) else [v] for k, v in answers.items()}
            answers = generate_combinations(answers)
            if type(answers)==list and len(answers) == 1:
                if not isinstance(data_item["predict"], str):
                    data_item["score"] = 0
                else:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answers[0])
            else:
                max_score = 0
                for answer in answers:
                    pred_kie_dict = convert_str_to_dict(data_item["predict"])
                    data_item["score"] = compute_f1_score(pred_kie_dict, answer)

                    if data_item["score"] > max_score:
                        max_score = data_item["score"]
                data_item["score"] = max_score

        elif data_item["type"] == "VQA with position en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                pred_dict = convert_str_to_dict(data_item["predict"])
                data_item["score"] = vqa_with_position_evaluation(pred_dict, data_item)

        elif data_item["type"] == "text translation cn":
            if len(data_item["predict"]) == 0:
                data_item["score"] = 0
            elif len(data_item["answers"][0]) == 0:
                data_item["score"] = 0
                data_item["ignore"] = "True"
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4

        elif data_item["type"] == "fine-grained text recognition en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            elif len(data_item["predict"]) == 0:
                data_item["score"] = 0
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) +
                    get_value_or_zero(ocr_metric["meteor"]) +
                    get_value_or_zero(ocr_metric["f_measure"]) +
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4
        elif data_item["type"] == "full-page OCR en":
            if not data_item["predict"]:
                data_item["score"] = 0
            else:
                ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                data_item["score"] = (
                    get_value_or_zero(ocr_metric["bleu"]) +
                    get_value_or_zero(ocr_metric["meteor"]) +
                    get_value_or_zero(ocr_metric["f_measure"]) +
                    (1 - get_value_or_zero(ocr_metric["edit_dist"]))
                ) / 4

        elif data_item["type"] == "full-page OCR cn":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                if len(data_item["predict"]) == 0:
                    data_item["score"] = 0
                else:
                    ocr_metric = cal_per_metrics(data_item["predict"], data_item["answers"][0])
                    data_item["score"] = (ocr_metric["bleu"] + ocr_metric["meteor"] + ocr_metric["f_measure"] + (1 - ocr_metric["edit_dist"])) / 4

        elif data_item["type"] == "text grounding en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_coordinates(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = calculate_iou(predict_bbox, data_item["answers"])

        elif data_item["type"] == "text spotting en":
            if not isinstance(data_item["predict"], str):
                data_item["score"] = 0
            else:
                predict_bbox = extract_bounding_boxes_robust(data_item["predict"])
                if not predict_bbox:
                    data_item["score"] = 0
                else:
                    data_item["score"] = spotting_evaluation(predict_bbox, data_item)

        else:
            raise ValueError("Unknown task type!")

        res_data_list.append(data_item)


    return res_data_list



def ocrbench_v2_aggregate_accuracy(data_list):
    en_text_recognition_list, en_text_detection_list, en_text_spotting_list, en_relationship_extraction_list = [], [], [], []
    en_element_parsing_list, en_mathematical_calculation_list, en_visual_text_understanding_list = [], [], []
    en_knowledge_reasoning_list = []

    cn_text_recognition_list, cn_relationship_extraction_list = [], []
    cn_element_parsing_list, cn_visual_text_understanding_list = [], []
    cn_knowledge_reasoning_list = []

    res_list = []
    for item in data_list:
        if "ignore" in item.keys():
            assert item["ignore"] == "True"

        elif item["type"] == "text recognition en" or item["type"] == "fine-grained text recognition en" or item["type"] == "full-page OCR en":
            en_text_recognition_list.append(item["score"])

        elif item["type"] == "text grounding en" or item["type"] == "VQA with position en":
            en_text_detection_list.append(item["score"])

        elif item["type"] == "text spotting en":
            en_text_spotting_list.append(item["score"])

        elif item["type"] == "key information extraction en" or item["type"] == "key information mapping en":
            en_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing en" or item["type"] == "chart parsing en" \
        or item["type"] == "table parsing en" or item["type"] == "formula recognition en":
            en_element_parsing_list.append(item["score"])

        elif item["type"] == "math QA en" or item["type"] == "text counting en":
            en_mathematical_calculation_list.append(item["score"])

        elif item["type"] == "document classification en" \
        or item["type"] == "cognition VQA en" or item["type"] == "diagram QA en":
            en_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA en" or item["type"] == "science QA en" \
        or item["type"] == "APP agent en" or item["type"] == "ASCII art classification en":
            en_knowledge_reasoning_list.append(item["score"])

        elif item["type"] == "full-page OCR cn":
            cn_text_recognition_list.append(item["score"])

        elif item["type"] == "key information extraction cn" or item["type"] == "handwritten answer extraction cn":
            cn_relationship_extraction_list.append(item["score"])

        elif item["type"] == "document parsing cn" or item["type"] == "table parsing cn" or item["type"] == "formula recognition cn":
            cn_element_parsing_list.append(item["score"])

        elif item["type"] == "cognition VQA cn":
            cn_visual_text_understanding_list.append(item["score"])

        elif item["type"] == "reasoning VQA cn" or item["type"] == "text translation cn":
            cn_knowledge_reasoning_list.append(item["score"])

        else:
            raise ValueError("Unknown task type!")

    en_scores = {
        "en_text_recognition": en_text_recognition_list,
        "en_text_detection": en_text_detection_list,
        "en_text_spotting": en_text_spotting_list,
        "en_relationship_extraction": en_relationship_extraction_list,
        "en_element_parsing": en_element_parsing_list,
        "en_mathematical_calculation": en_mathematical_calculation_list,
        "en_visual_text_understanding": en_visual_text_understanding_list,
        "en_knowledge_reasoning": en_knowledge_reasoning_list
    }

    cn_scores = {
        "cn_text_recognition": cn_text_recognition_list,
        "cn_relationship_extraction": cn_relationship_extraction_list,
        "cn_element_parsing": cn_element_parsing_list,
        "cn_visual_text_understanding": cn_visual_text_understanding_list,
        "cn_knowledge_reasoning": cn_knowledge_reasoning_list
    }

    en_averages = calculate_average(en_scores)
    cn_averages = calculate_average(cn_scores)

    return en_averages,cn_averages
