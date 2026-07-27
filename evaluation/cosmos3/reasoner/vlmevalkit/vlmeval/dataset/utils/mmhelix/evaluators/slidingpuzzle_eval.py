import re
from typing import Any, Dict, List


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(
        self, predicted_answer: Any, ground_truth: Any,
        initial_state: str, params: Dict[str, Any] = None
    ) -> bool:
        raise NotImplementedError


class SlidingPuzzleEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        from utils.constants import PROMPT_15PUZZLE

        if isinstance(question, list):
            question_str = str(question)
        else:
            question_str = question

        return PROMPT_15PUZZLE.format(question_str)

    def extract_answer(self, model_output: str) -> List[int]:
        answer = model_output.strip()

        try:
            numbers = re.findall(r'\d+', answer)
            moves = [int(num) for num in numbers if num.strip()]
            return moves
        except Exception:
            return []

    def evaluate(self, predicted_answer: List[int], ground_truth: Any, initial_state: Any) -> bool:
        from vlmeval.dataset.utils.mmhelix.utils.validation import puzzle_15_check
        return puzzle_15_check(initial_state, predicted_answer)
