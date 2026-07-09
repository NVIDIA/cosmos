import random

import numpy as np
import pandas as pd


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    response = str(response)
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response or f'{choice}. ' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def get_mc_score(row, use_parse=True):
    if use_parse:
        if pd.isna(row["A"]):
            return False
        response = row["prediction"]
        all_choices = []
        for i in range(9):
            if chr(65 + i) in row and not pd.isna(row[chr(65 + i)]):
                all_choices.append(chr(65 + i))
        index2ans = {index: row[index] for index in all_choices}
        pred_index = parse_multi_choice_response(response, all_choices, index2ans)
    else:
        pred_index = row["output"]
    return int(pred_index == row["answer"])


def report_vmc_acc(data):
    general_datasets = ["SEEDBench", "MMStar", "A-OKVQA", "VizWiz", "MMVet", "VQAv2", "OKVQA"]
    reason_datasets = ["MMMU", "MathVista", "ScienceQA", "RealWorldQA", "GQA", "MathVision"]
    ocr_datasets = ["TextVQA", "OCRVQA"]
    doc_datasets = ["AI2D", "ChartQA", "DocVQA", "InfoVQA", "TableVQABench"]
    results = {}
    for category in data['category'].unique():
        results[category] = data[data['category'] == category]['hit'].mean()
    results = pd.DataFrame(results, index=[0])
    results["Overall"] = data['hit'].mean()
    results['General'] = results[general_datasets].mean(axis=1)
    results['Reasoning'] = results[reason_datasets].mean(axis=1)
    results['OCR'] = results[ocr_datasets].mean(axis=1)
    results['Doc & Chart'] = results[doc_datasets].mean(axis=1)
    for key in results:
        results[key] = round(results[key] * 100, 2)
    results = results[['Overall', 'General', 'Reasoning', 'OCR', 'Doc & Chart']
                      + general_datasets + reason_datasets + ocr_datasets + doc_datasets]
    return results
