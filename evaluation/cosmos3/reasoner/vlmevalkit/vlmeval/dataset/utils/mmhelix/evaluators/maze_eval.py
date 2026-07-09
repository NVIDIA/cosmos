from typing import Any, Dict


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


class MazeEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        from utils.constants import PROMPT_MAZE
        return PROMPT_MAZE.format(question)

    def extract_answer(self, model_output: str) -> str:
        answer = model_output.strip().lower()

        valid_directions = ['up', 'down', 'left', 'right']
        words = answer.split()
        directions = [word for word in words if word in valid_directions]

        return " ".join(directions)

    def evaluate(self, predicted_answer: str, ground_truth: Any, initial_state: Any) -> bool:
        from vlmeval.dataset.utils.mmhelix.utils.validation import maze_check
        return maze_check(initial_state, predicted_answer)
