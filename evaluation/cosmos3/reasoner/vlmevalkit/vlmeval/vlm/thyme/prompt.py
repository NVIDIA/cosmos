from __future__ import annotations

from vlmeval.smp import listinstr


class ThymePromptMixin:
    """
    Mixin class for Thyme to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(
            self,
            *args,
            use_custom_prompt: bool = True,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        return self._use_custom_prompt

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {"MMMU_DEV_VAL", "MMMU_TEST"}:
            return self._build_mmmu_prompt(line, dataset)
        if "OlympiadBench" in dataset:
            return self._build_olympiad_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type in ["MCQ", "MMERealWorld"]:
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == "Y/N":
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == "VQA":
            return self._build_vqa_prompt(line, dataset)
        # Fall back to default prompt
        return self._build_default_prompt(line, dataset)

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        hint = line["hint"] if (
            "hint" in line and not pd.isna(
                line["hint"])) else None
        prompt = ""
        if hint is not None:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        if len(options):
            prompt += options_prompt
            prompt += "Please select the correct answer from the options above. \n"
        prompt = prompt.rstrip()
        msgs = []

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = "请直接回答选项字母。"
        MCQ_EN_PROMPT = "Please select the correct answer from the options above."

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search("[\u4e00-\u9fff]", s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        hint = line["hint"] if (
            "hint" in line and not pd.isna(
                line["hint"])) else None
        prompt = ""
        if hint is not None:
            prompt += f"Hint: {hint}\n"
        prompt += f"Question: {question}\n"
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for YORN dataset:"""
        YORN_PROMPT = " Please answer yes or no."

        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        msgs = []

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=question))
        assert msgs[-1]["type"] == "text"
        if not listinstr(["MME"], dataset):
            msgs[-1]["value"] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = "\nPlease try to answer the question with short words or phrases if possible."

        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        msgs = []

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=question))
        assert msgs[-1]["type"] == "text"
        if not listinstr(["MMVet"], dataset):
            msgs[-1]["value"] += VQA_PROMPT
        return msgs

    def _build_olympiad_prompt(
            self, line, dataset: str) -> list[dict[str, str]]:

        from vlmeval.dataset.utils.olympiadbench import get_answer_type_text, make_input

        self.is_chinese = "zh" in line["source"]
        self.is_math = "maths" in line["source"]
        self.is_theorem_proving = "TP" in line["source"]

        if self.is_chinese:
            subject_content = "数学" if self.is_math else "物理"
            if self.is_theorem_proving:
                prompt = (
                    f"以下是中国{subject_content}竞赛中的证明题。请根据题目的要求，运用逻辑推理及常用定理证明题目中的命题。"
                    "证明过程中使用的变量和公式请使用LaTeX格式表示。")
            else:
                answer_type_text = get_answer_type_text(
                    line["answer_type"],
                    is_chinese=True,
                    multiple_answer=line["is_multiple_answer"],
                )
                if line["is_multiple_answer"]:
                    multiple_answer_text = "\\boxed{用英文逗号连接的多个答案}"
                else:
                    multiple_answer_text = "\\boxed{答案}"
                unit_text = ""
                if line["unit"]:
                    multiple_answer_text += "(单位)"
                    unit_text = "，注意答案的单位不要放在\\boxed{}中"
                prompt = (
                    f"以下是中国{subject_content}竞赛中的解答题{answer_type_text}。请根据题目的要求和所提供的信息计算得出答案。"
                    f"解答过程和结果中使用的变量和公式请使用LaTeX格式表示。请在最后以“所以最终答案是{multiple_answer_text}。”"
                    f"显式给出结果{unit_text}。")
        else:
            subject_content = "Math" if self.is_math else "Physics"
            if self.is_theorem_proving:
                prompt = (
                    f"The following is a theorem proving problem from an International {subject_content} competition. "
                    "Please use logical reasoning and common theorems to prove the proposition in the problem "
                    "according to the given requirements. "
                    "Please use LaTeX format to represent the variables and formulas used in the proof.")
            else:
                if line["is_multiple_answer"]:
                    multiple_answer_text = (
                        "\\boxed{multiple answers connected with commas}"
                    )
                else:
                    multiple_answer_text = "\\boxed{answer}"
                unit_text = ""
                if line["unit"]:
                    multiple_answer_text += "(unit)"
                    unit_text = ", note that the unit of the answer should not be included in \\boxed{}"
                answer_type_text = get_answer_type_text(
                    line["answer_type"],
                    is_chinese=False,
                    multiple_answer=line["is_multiple_answer"],
                )
                prompt = (
                    f"The following is an open-ended problem from an International {subject_content} competition. "
                    f"{answer_type_text}Please calculate the answer according to the given requirements and "
                    "the information provided. Please use LaTeX format to represent the variables and formulas "
                    'used in the solution process and results. Please end your solution with "So the final answer '
                    f'is {multiple_answer_text}." and give the result explicitly{unit_text}.')

        if self.is_math:
            input = make_input(prompt, line["question"])
        else:
            if (
                "context" in line.keys() and str(line["context"]) != "nan"
            ):  # cannot be null
                input = make_input(
                    prompt, line["context"] + "\n" + line["question"])
            else:
                input = make_input(prompt, line["question"])

        tgt_path = self.dump_image(line, dataset)

        msgs = []

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=input))

        return msgs

    def _build_default_prompt(
            self, line, dataset: str) -> list[dict[str, str]]:
        """For non-customized datasets, use a simple default prompt."""
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        msgs = []

        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=question))
        return msgs
