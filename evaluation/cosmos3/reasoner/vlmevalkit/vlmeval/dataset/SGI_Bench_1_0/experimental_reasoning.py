import ast
import base64
import io
import os
import os.path as osp
import re
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from vlmeval.smp import (decode_base64_to_image_file, dump, encode_image_to_base64,
                         get_intermediate_file_path, load, read_ok, toliststr)
from vlmeval.utils.mp_util import track_progress_rich
from ..image_base import ImageBaseDataset
from ..utils.judge_util import build_judge


def extract_answer_from_response(response):
    match = re.search(r"\\boxed\{([A-Za-z])\}", response)
    if match:
        return match.group(1)
    else:
        return None


def mm_reasoning_is_correct(pred, gold):
    try:
        ans = extract_answer_from_response(pred).strip()
    except Exception:
        return False
    return ans.lower() == gold.lower()


def b64_encode_image(img) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def judge_aux(judge, row):
    reference_steps = row["steps"]
    reference_steps = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(reference_steps)])

    judge_prompt = (
        f"You are a strict evaluator assessing the "
        f"**validity of the model prediction's reasoning "
        f"process**. You must score this reasoning validity "
        f"on a scale from 0 to 10, where 0 means the "
        f"reasoning is completely invalid and 10 means the "
        f"reasoning is fully rigorous.\n"
        f"# Input\n"
        f"Question:\n"
        f"```\n"
        f"{row['question']}\n"
        f"```\n"
        f"Reference Reasoning:\n"
        f"```\n"
        f"{reference_steps}\n"
        f"```\n"
        f"Model Prediction:\n"
        f"```\n"
        f"{row['prediction']}\n"
        f"```\n"
        f"# Evaluation Rules\n"
        f"1. First, identify the **complete reasoning "
        f"process** from the model prediction (ignore only "
        f"the final answer if it is not accompanied by "
        f"reasoning).\n"
        f"2. Evaluate reasoning validity against two core "
        f"criteria:\n"
        f"   - **Logical Coherence**: Check if the reasoning "
        f"steps are sequential, self-consistent, and free of "
        f"contradictions (e.g., no conflicting premises or "
        f"illogical deductions).\n"
        f"   - **Alignment with Reference Reasoning**: Check "
        f"if the reasoning direction, key premises, and "
        f"deduction logic match the reference reasoning "
        f"(partial alignment counts for partial credit).\n"
        f"3. Deduct points for:\n"
        f"   - Irrelevant content (reasoning that does not "
        f"address the question or key conditions).\n"
        f"   - Missing key reasoning steps (even if the "
        f"final answer is correct).\n"
        f"   - Flawed logic (e.g., circular reasoning, "
        f"false premises leading to conclusions).\n"
        f"4. Do not prioritize the correctness of the "
        f"**final answer**\u2014a correct answer with invalid "
        f"reasoning still scores low, while an incorrect "
        f"answer with partially valid reasoning may score "
        f"higher.\n"
        f"# Scoring Guide\n"
        f"- **10**: Reasoning is fully rigorous, logically "
        f"coherent (no contradictions), and perfectly "
        f"aligned with the reference reasoning (all key "
        f"steps and logic match).\n"
        f"- **7-9**: Reasoning is mostly coherent, with "
        f"minor logical gaps or partial misalignment with "
        f"the reference reasoning (no major "
        f"contradictions).\n"
        f"- **4-6**: Reasoning has obvious logical flaws "
        f"(e.g., one missing key step, minor "
        f"contradictions) or limited alignment with the "
        f"reference reasoning (only some core logic "
        f"matches).\n"
        f"- **1-3**: Reasoning is barely valid, with severe "
        f"logical flaws (e.g., multiple contradictions) or "
        f"almost no alignment with the reference reasoning "
        f"(only tangentially related to the question).\n"
        f"- **0**: Reasoning is completely invalid, "
        f"contradictory (self-conflicting logic), or "
        f"irrelevant (no connection to the question or key "
        f"conditions).\n"
        f"# Strict Output format example\n"
        f"6"
    )
    try:
        msgs = []
        msgs.append({'role': 'system', 'value': 'You are a helpful assistant.'})
        msgs.append({'role': 'user', 'type': 'text', 'value': judge_prompt})

        images = ast.literal_eval(row['step_images'])
        for image in images:
            msgs.append({'role': 'user', 'value': image, 'type': 'image'})
        llm_judge = judge.generate(msgs).strip()
        pattern = r"(\d+)"
        match = re.search(pattern, llm_judge)
        rv_score = float(match.group(1)) if match else 0.0
    except Exception:
        rv_score = 0.0

    mcc_score = mm_reasoning_is_correct(row['prediction'], chr(ord('A') + int(row['answer'])))
    return dict(mcc_score=mcc_score, rv_score=rv_score)


class SGI_Bench_Experimental_Reasoning(ImageBaseDataset):
    TYPE = 'MCQ '

    @classmethod
    def supported_datasets(cls):
        return ["SGI-Experimental-Reasoning"]

    def dump_images(self, line):
        step_dir = osp.join(self.img_root, 'step_images')
        os.makedirs(self.img_root, exist_ok=True)
        os.makedirs(step_dir, exist_ok=True)

        results = {}

        def _process_field(key_name, path_key_name, save_root):
            tgt_paths = []
            if key_name in line:
                content = line[key_name]
                if path_key_name in line and isinstance(line[path_key_name], list):
                    fnames = line[path_key_name]
                else:
                    count = len(content) if isinstance(content, list) else 1
                    fnames = [f"{line['index']}_{i}.png" for i in range(count)]
                imgs = content if isinstance(content, list) else [content]
                for img, fname in zip(imgs, fnames):
                    full_path = osp.join(save_root, fname)
                    if not read_ok(full_path):
                        decode_base64_to_image_file(img, full_path)
                    tgt_paths.append(full_path)

            elif path_key_name in line:
                paths = toliststr(line[path_key_name])
                read_ok_flag = [read_ok(x) for x in paths]

                if not all(read_ok_flag):
                    paths_abs = [osp.join(save_root, x) for x in paths]
                    read_ok_flag = [read_ok(x) for x in paths_abs]
                    assert read_ok_flag, f"Field `{key_name}` missing and files not found: {paths}"
                    tgt_paths = paths_abs
                else:
                    tgt_paths = paths

            return tgt_paths

        if 'image' in line or 'image_path' in line:
            results['image'] = _process_field('image', 'image_path', self.img_root)
        if 'step_images' in line or 'step_image_path' in line:
            results['step_images'] = _process_field('step_images', 'step_image_path', step_dir)

        return results

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-Reasoning", split="test")

        rows: List[Dict[str, Any]] = []
        idx = 0

        for prob in hf:
            current_row = {
                "index": idx,  #
                "id": prob["idx"],
                "question": prob["question"],
                "image": [encode_image_to_base64(img) for img in prob["images"]],
                "options": prob["options"],
                "steps": prob["steps"],
                "step_images": [encode_image_to_base64(img) for img in prob["step_images"]],
                "answer": prob["answer"],
                "image_type": prob["image_type"],
                "discipline": prob["discipline"],
                "direction": prob["direction"],
                "type": prob["type"]
            }
            saved_paths = self.dump_images(current_row)
            if 'image' in saved_paths:
                current_row['image'] = saved_paths['image']

            if 'step_images' in saved_paths:
                current_row['step_images'] = saved_paths['step_images']
            rows.append(current_row)
            idx += 1

        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = (
            "Please solve the following multiple-choice "
            "question step-by-step. Each question is "
            "provided with several options labeled A, B, "
            "C, D, E, etc. Carefully analyze the question "
            "and each option, reason step-by-step, then "
            "select the single most correct option.\n\n"
            "Your final output **must** include both "
            "**the reasoning** and **the final answer**. "
            "The final answer must meet two core "
            "requirements:\n"
            "1. It consists solely of the corresponding "
            "letter of the correct option (e.g., A, B, C, "
            "D, E, etc.);\n"
            "2. This letter is enclosed in the \\boxed{} "
            "format. Example: \\boxed{A}"
            "\n\nQuestion:\n" + line['question']
            + "\n\nOptions:\n"
        )
        for i, option in enumerate(line['options']):
            option_label = chr(ord('A') + i)
            question += f"{option_label}. {option}\n"

        msgs = []
        if isinstance(line['image'], list):
            for p in line['image']:
                msgs.append({'type': 'image', 'value': p})
        elif isinstance(line['image'], str):
            msgs.append({'type': 'image', 'value': line['image']})
        msgs.append({'type': 'text', 'value': question})
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = pd.DataFrame(data)

        data['mcc'] = 0
        data['rv'] = 0

        all_mcc, all_rv = [], []
        if judge_kwargs.get('model') is None:
            judge_kwargs['model'] = 'o4-mini'
        if judge_kwargs.get('max_tokens') is None:
            judge_kwargs['max_tokens'] = None
        judge = build_judge(**judge_kwargs)

        tups = []
        indices = []
        tmp_file = get_intermediate_file_path(eval_file, '_judge_tmp', 'pkl')
        if osp.exists(tmp_file):
            ans = load(tmp_file)
        else:
            ans = {}

        for index, row in data.iterrows():
            if index in ans:
                continue
            tups.append(dict(judge=judge, row=row))
            indices.append(index)

        if len(indices) > 0:
            track_progress_rich(
                judge_aux,
                tasks=tups,
                nproc=judge_kwargs.get('nproc', 32),
                save=tmp_file,
                keys=indices
            )
            ans = load(tmp_file)

        for index, res in ans.items():
            rv_score = res['rv_score']
            mcc_score = res['mcc_score']
            all_mcc.append(mcc_score)
            data.at[index, 'mcc'] = 1 if mcc_score else 0
            all_rv.append(rv_score)
            data['rv'] = data['rv'].astype(float)
            data.at[index, 'rv'] = rv_score / 10.0

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        result = {"MCC": sum(all_mcc) / len(all_mcc), "RV": sum(all_rv) / (10.0 * len(all_rv))}
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(data, score_file)
        dump(result, result_file)
        return result
