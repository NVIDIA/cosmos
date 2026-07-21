import ast
import json
import os
import os.path as osp
import re
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import pandas as pd

from ..smp import decode_base64_to_image_file, dump, load, read_ok
from .image_base import ImageBaseDataset


def _find_first_bracketed_list(text: str) -> Optional[str]:
    if not text:
        return None
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _iter_bracketed_spans(text: str):
    if not text:
        return
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _coerce_int_list(value) -> Optional[List[int]]:
    if isinstance(value, (np.ndarray, pd.Series)):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    out: List[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            out.append(item)
        elif isinstance(item, str) and re.fullmatch(r"\s*-?\d+\s*", item):
            out.append(int(item))
        else:
            return None
    return out


def _parse_python_list_from_text(text) -> Optional[List[int]]:
    if text is None:
        return None
    try:
        text = str(text)
    except Exception:
        return None

    for snippet in _iter_bracketed_spans(text):
        try:
            parsed = ast.literal_eval(snippet)
        except Exception:
            continue
        coerced = _coerce_int_list(parsed)
        if coerced is not None:
            return coerced

    return None


def _parse_all_python_int_lists_from_text(text) -> List[List[int]]:
    if text is None:
        return []
    try:
        text = str(text)
    except Exception:
        return []

    out: List[List[int]] = []
    for snippet in _iter_bracketed_spans(text):
        try:
            parsed = ast.literal_eval(snippet)
        except Exception:
            continue
        coerced = _coerce_int_list(parsed)
        if coerced is not None:
            out.append(coerced)
    return out


def _load_taxonomy() -> dict:
    taxonomy_path = osp.join(osp.dirname(__file__), "utils", "ssi_bench", "taxonomy.json")
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pairwise_accuracy(gt: List[int], pred: List[int]) -> float:
    if gt is None or pred is None:
        return 0.0
    if len(gt) != len(pred):
        return 0.0
    if len(gt) < 2:
        return 1.0

    pos_gt = {v: i for i, v in enumerate(gt)}
    pos_pred = {v: i for i, v in enumerate(pred)}
    if len(pos_gt) != len(gt) or len(pos_pred) != len(pred):
        return 0.0
    if set(pos_gt.keys()) != set(pos_pred.keys()):
        return 0.0

    keys = list(pos_gt.keys())
    correct = 0
    total = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            gt_order = pos_gt[a] < pos_gt[b]
            pred_order = pos_pred[a] < pos_pred[b]
            if gt_order == pred_order:
                correct += 1
            total += 1
    return float(correct / total) if total else 0.0


class SSIBenchDataset(ImageBaseDataset):
    TYPE = "VQA"

    DATASET_URL = {
        "SSI_Bench": "https://huggingface.co/datasets/cyang203912/SSI-Bench/resolve/main/SSI_Bench.tsv",
    }

    DATASET_MD5 = {
        "SSI_Bench": "d2d364795c917a27c674d44f62599427",
    }

    @classmethod
    def supported_datasets(cls):
        return ["SSI_Bench"]

    def load_data(self, dataset):
        url = self.DATASET_URL.get(dataset, None)
        if url is None or url == "":
            url = dataset + ".tsv"
        file_md5 = self.DATASET_MD5.get(dataset, None)
        return self.prepare_tsv(url, file_md5)

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        if "image_path" in line and isinstance(line["image_path"], str):
            return [line["image_path"]]

        img_field = line.get("image", None)
        if img_field is None:
            raise KeyError("SSI_Bench requires an `image` column with base64 data.")

        if isinstance(img_field, (pd.Series, np.ndarray)):
            img_field = img_field.tolist()

        if isinstance(img_field, list):
            img_list = img_field
        elif isinstance(img_field, str) and img_field.startswith("[") and img_field.endswith("]"):
            try:
                img_list = json.loads(img_field)
            except json.JSONDecodeError:
                img_list = [img_field]
        else:
            img_list = [img_field]

        tgt_paths: list[str] = []
        for i, img_base64 in enumerate(img_list):
            path = osp.join(self.img_root, f"{line['index']}_{i}.jpg")
            if not read_ok(path):
                decode_base64_to_image_file(img_base64, path)
            tgt_paths.append(path)
        return tgt_paths

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_paths = self.dump_image(line)

        template_name = str(line.get("question", "")).strip()
        color = str(line.get("annotation_color", "")).strip() or "cyan"

        prompt_text = template_name
        try:
            from .utils.ssi_bench import prompts as ssi_prompts

            if hasattr(ssi_prompts, template_name):
                template = getattr(ssi_prompts, template_name)
                try:
                    prompt_text = template.format(color=color)
                except Exception:
                    prompt_text = template
        except Exception:
            pass

        msgs = [dict(type="image", value=p) for p in tgt_paths]
        msgs.append(dict(type="text", value=prompt_text))
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        data = load(eval_file)
        taxonomy = _load_taxonomy()
        category_order = [c.get("slug") for c in taxonomy.get("categories", []) if c.get("slug")]
        task_order = [t.get("slug") for t in taxonomy.get("tasks", []) if t.get("slug")]
        task_to_nopts = {t["slug"]: int(t.get("optionCount", 0)) for t in taxonomy.get("tasks", [])}

        data["extracted_pred"] = None
        data["score"] = 0.0
        data["pairwise_score"] = 0.0

        for idx, row in data.iterrows():
            task_slug = row.get("task", None)
            n_opts = task_to_nopts.get(task_slug, None)

            gt = _parse_python_list_from_text(row.get("answer", None))

            pred = None
            pred_candidates = _parse_all_python_int_lists_from_text(row.get("prediction", None))
            if pred_candidates:
                if n_opts is None:
                    pred = pred_candidates[-1]
                else:
                    for cand in reversed(pred_candidates):
                        if len(cand) != n_opts:
                            continue
                        if len(set(cand)) != len(cand):
                            continue
                        if any((x < 1 or x > n_opts) for x in cand):
                            continue
                        pred = cand
                        break

            data.at[idx, "extracted_pred"] = str(pred) if pred is not None else None
            if gt is not None and pred is not None and gt == pred:
                data.at[idx, "score"] = 1.0
            if gt is not None and pred is not None:
                data.at[idx, "pairwise_score"] = _pairwise_accuracy(gt, pred)

        def _calc_res(score_col: str) -> OrderedDict:
            overall = float(np.mean(data[score_col])) if len(data) else 0.0

            res = OrderedDict()
            res["overall"] = overall

            if "category" in data.columns:
                categories = [c for c in data["category"].astype(str).tolist() if c and c != "nan"]
                present = set(categories)
                for cat in category_order:
                    if cat not in present:
                        continue
                    sub = data[data["category"].astype(str) == cat]
                    res[f"category/{cat}"] = float(np.mean(sub[score_col])) if len(sub) else 0.0
                for cat in sorted(present.difference(category_order)):
                    sub = data[data["category"].astype(str) == cat]
                    res[f"category/{cat}"] = float(np.mean(sub[score_col])) if len(sub) else 0.0

            if "task" in data.columns:
                tasks = [t for t in data["task"].astype(str).tolist() if t and t != "nan"]
                present = set(tasks)
                for task in task_order:
                    if task not in present:
                        continue
                    sub = data[data["task"].astype(str) == task]
                    res[task] = float(np.mean(sub[score_col])) if len(sub) else 0.0
                for task in sorted(present.difference(task_order)):
                    sub = data[data["task"].astype(str) == task]
                    res[task] = float(np.mean(sub[score_col])) if len(sub) else 0.0
            return res

        res_task = _calc_res("score")
        res_pairwise = _calc_res("pairwise_score")

        base, ext = osp.splitext(eval_file)
        score_file = base + "_score.xlsx"
        acc_file = base + "_acc.csv"
        data.to_excel(score_file, index=False)
        out = pd.DataFrame(
            [
                dict(acc_type="task_acc", **res_task),
                dict(acc_type="pairwise_acc", **res_pairwise),
            ]
        )
        metric_cols = [c for c in out.columns if c != "acc_type"]
        out[metric_cols] = out[metric_cols].astype(float) * 100.0
        dump(out, acc_file)
        return out
