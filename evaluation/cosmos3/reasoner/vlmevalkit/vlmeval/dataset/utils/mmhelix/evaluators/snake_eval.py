import re
from typing import Any, Dict, List


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


class SnakeEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        from utils.constants import PROMPT_SNAKE
        return PROMPT_SNAKE.format(question)

    def extract_answer(self, model_output: str) -> List[tuple]:
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, model_output, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            content = model_output.strip()  # Fallback to full output if no tags found

        # Extract all coordinates in the format (x,y)
        coords_pattern = r'\((\d+)\s*,\s*(\d+)\)'
        matches = re.findall(coords_pattern, content)

        # Convert to list of tuples with integers
        coords = [(int(x), int(y)) for x, y in matches]
        return coords

    def evaluate(self, predicted_answer: str, ground_truth: str, initial_state: Any) -> bool:
        # Pattern to match coordinates like (0,8)
        coords_pattern = r'\((\d+)\s*,\s*(\d+)\)'

        # Extract all coordinates from ground truth
        gt_matches = re.findall(coords_pattern, ground_truth)
        gt_coords = {(int(x), int(y)) for x, y in gt_matches}

        # Extract all coordinates from predicted answer
        pred_matches = re.findall(coords_pattern, predicted_answer)
        pred_coords = {(int(x), int(y)) for x, y in pred_matches}

        # Check if both sets contain the same coordinates
        if pred_coords == gt_coords:
            return True

        return False
