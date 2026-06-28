# ReasonMap Plus Dataset is an extension of the original ReasonMap dataset,
# designed for providing more dense rewards in visual understanding and reasoning
# tasks.
# The reference paper is:
#   1. Can MLLMs Guide Me Home? A Benchmark Study on Fine-Grained Visual
#      Reasoning from Transit Maps: https://arxiv.org/abs/2505.18675
#   2. RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via
#      Multi-Stage Reinforcement Learning: https://arxiv.org/abs/2510.02240
#
# If any problem occurs, please open an issue on GitHub
# (https://github.com/fscdc/RewardMap or https://github.com/fscdc/ReasonMap).

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import load

_BOXED_PAT = re.compile(r'(?:\\boxed|boxed)\{([^}]*)\}', re.IGNORECASE)
_TEXT_PAT = re.compile(r'\\text\{([^}]*)\}', re.IGNORECASE)

_YES = {"yes", "y", "true", "t", "1"}
_NO = {"no", "n", "false", "f", "0"}


def _strip(s: Any) -> str:
    return ("" if s is None else str(s)).strip()


def _lower(s: Any) -> str:
    return _strip(s).lower()


def _extract_boxed(s: str) -> str | None:
    m = list(_BOXED_PAT.finditer(s))
    if not m:
        return None
    raw = m[-1].group(1).strip()
    texts = _TEXT_PAT.findall(raw)
    return " ".join(t.strip() for t in texts) if texts else raw


def _extract_after_phrases(s: str) -> str:
    phrases = [
        "the final answer is", "final answer is",
        "the answer is", "answer is",
        "the correct answer is", "correct answer is",
        "final answer:", "final:", "answer:", "ans:"
    ]
    lo = s.lower()
    for ph in phrases:
        if ph in lo:
            part = s[lo.rfind(ph) + len(ph):].strip()
            cand = re.split(r'(?:\n|\. |\.$)', part, maxsplit=1)[0]
            return cand.strip()
    return s.strip()


def _normalize_yesno(s: str) -> str | None:
    t = _lower(s)
    if t in _YES:
        return "yes"
    if t in _NO:
        return "no"
    return None


def _normalize_abcd(s: str) -> str | None:
    m = re.search(r'\b([ABCD])\b', s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _extract_int(s: str) -> int | None:
    m = re.search(r'[-+]?\d+', s)
    return int(m.group(0)) if m else None


def normalize_prediction(pred_raw: Any, typ: str) -> str:
    s = _strip(pred_raw)
    if not s:
        return ""

    boxed = _extract_boxed(s)
    cand = boxed if boxed else _extract_after_phrases(s)

    t = (typ or "").lower()
    if "torf" in t:
        yn = _normalize_yesno(cand)
        if yn is None:
            yn = _normalize_yesno(s)
        return yn or cand

    if t == "counting1" or "counting1" in t:
        abcd = _normalize_abcd(cand)
        if abcd is None:
            abcd = _normalize_abcd(s)
        return abcd or cand

    if t in {"counting2", "counting3"} or t.startswith("counting"):
        num = _extract_int(cand)
        if num is None:
            num = _extract_int(s)
        return str(num) if num is not None else cand

    return cand


class ReasonMap_Plus(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "ReasonMap-Plus": "https://opencompass.openxlab.space/utils/VLMEval/ReasonMap-Plus.tsv"
    }

    DATASET_MD5 = {
        "ReasonMap-Plus": "205d3ac1c3af07d3e4930f25e01008be"
    }

    @classmethod
    def supported_datasets(cls):
        return ['ReasonMap-Plus']

    def build_prompt(self, line):
        if not isinstance(line, pd.Series):
            line = self.data_df.iloc[line]

        img_val = line.get("image", None)
        if not img_val:
            img_val = line.get("image_path", "")
        prompt = line.get("question", "")

        return [
            dict(type="image", value=img_val),
            dict(type="text", value=prompt),
        ]

    def evaluate(self, eval_file, **judge_kwargs):
        df = load(eval_file)
        if len(df) == 0:
            return pd.DataFrame([dict(metric="accuracy", value=0.0, n=0)])

        df["_pred_norm"] = [
            normalize_prediction(p, t)
            for p, t in zip(df.get("prediction", ""), df.get("type", ""))
        ]

        def _score_one(a, p, t):
            tlo = (t or "").lower()
            try:
                if "torf" in tlo:
                    gt = "yes" if int(a) == 1 else "no"
                    pp = _normalize_yesno(p)
                    return 1 if (pp == gt) else 0

                if tlo == "counting1" or "counting1" in tlo:
                    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
                    pp = _normalize_abcd(p)
                    if pp is None:
                        return 0
                    return 1 if mapping[pp] == int(a) else 0

                if tlo in {"counting2", "counting3"} or tlo.startswith("counting"):
                    return 1 if int(str(p)) == int(a) else 0

                return 1 if _strip(a).lower() == _strip(p).lower() else 0
            except Exception:
                return 0

        difficulty_weights = {
            "easy": 1.0,
            "middle": 1.5,
            "hard": 2.0
        }

        def _score_weighted_one(a, p, t, difficulty):
            weighted_acc = difficulty_weights[difficulty]
            tlo = (t or "").lower()
            try:
                if "torf" in tlo:
                    gt = "yes" if int(a) == 1 else "no"
                    pp = _normalize_yesno(p)
                    return weighted_acc if (pp == gt) else 0

                if tlo == "counting1" or "counting1" in tlo:
                    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
                    pp = _normalize_abcd(p)
                    if pp is None:
                        return 0
                    return weighted_acc if mapping[pp] == int(a) else 0

                if tlo in {"counting2", "counting3"} or tlo.startswith("counting"):
                    return weighted_acc if int(str(p)) == int(a) else 0

                return (
                    weighted_acc if _strip(a).lower() == _strip(p).lower() else 0
                )
            except Exception:
                return 0

        df["_correct"] = [
            _score_one(a, p, t)
            for a, p, t in zip(df.get("answer", ""), df["_pred_norm"], df.get("type", ""))
        ]

        df["_weighted_correct"] = [
            _score_weighted_one(a, p, t, difficulty)
            for a, p, t, difficulty in zip(
                df.get("answer", ""),
                df["_pred_norm"],
                df.get("type", ""),
                df.get("difficulty_city", ""),
            )
        ]

        total = np.sum(difficulty_weights[a] for a in df.get("difficulty_city", ""))

        overall = float(np.mean(df["_correct"])) if len(df) else 0.0
        weighted_overall = (
            float(np.sum(df["_weighted_correct"]) / total) if len(df) else 0.0
        )

        out_rows = [
            dict(metric="accuracy", value=overall, n=len(df)),
            dict(metric="weighted_accuracy", value=weighted_overall, n=len(df)),
        ]

        for tname, sub in df.groupby(df.get("type", "")):
            total_sub = np.sum(
                difficulty_weights[a] for a in sub.get("difficulty_city", "")
            )
            if len(sub):
                out_rows.append(
                    dict(
                        metric=f"accuracy[{tname}]",
                        value=float(np.mean(sub["_correct"])),
                        n=len(sub),
                    )
                )
                out_rows.append(
                    dict(
                        metric=f"weighted_accuracy[{tname}]",
                        value=float(np.sum(sub["_weighted_correct"]) / total_sub),
                        n=len(sub),
                    )
                )
        out_df = pd.DataFrame(out_rows, columns=["metric", "value", "n"])
        try:
            eval_path = Path(eval_file)
            out_path = eval_path.with_name(f"{eval_path.stem}_metrics.tsv")
            out_df.to_csv(out_path, sep="\t", index=False)
        except TypeError:
            pass

        return pd.DataFrame(out_rows, columns=["metric", "value", "n"])
