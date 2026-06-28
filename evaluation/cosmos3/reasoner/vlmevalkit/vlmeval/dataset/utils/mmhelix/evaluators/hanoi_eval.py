import ast
import re
from typing import Any, Dict


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        # Extract content within <answer></answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, model_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return model_output.strip()  # Fallback to full output if no tags found

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class TowerOfHanoiEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        from utils.constants import PROMPT_HANOI

        if isinstance(question, list):
            question_str = str(question)
        else:
            question_str = question

        return PROMPT_HANOI.format(question_str)

    def extract_answer(self, model_output: str) -> str:
        answer = model_output.strip()
        return answer

    def evaluate(self, predicted_answer: str, ground_truth: Any, initial_state: Any) -> bool:
        from vlmeval.dataset.utils.mmhelix.utils.validation import hanoi_check
        if not initial_state:
            return False
        lst = ast.literal_eval(initial_state) if isinstance(initial_state, str) else initial_state
        return hanoi_check(lst, predicted_answer)
