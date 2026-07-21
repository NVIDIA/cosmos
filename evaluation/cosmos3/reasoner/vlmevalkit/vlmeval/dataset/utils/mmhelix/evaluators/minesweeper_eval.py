import re
from typing import Any, Dict


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: str,
                 params: Dict[str, Any] = None) -> bool:
        raise NotImplementedError


class MinesweeperEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        from utils.constants import PROMPT_MINESWEEPER
        return PROMPT_MINESWEEPER.format(question)

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        """
        Evaluate minesweeper solution by comparing predicted mine coordinates with ground truth.
        Both predicted_answer and ground_truth should be lists of (row, col) tuples.
        """
        # Extract coordinates from predicted answer if it's a string
        if isinstance(predicted_answer, str):
            pred_coordinates = self._extract_coordinates(predicted_answer)
        elif isinstance(predicted_answer, list):
            pred_coordinates = set(predicted_answer)
        else:
            return False

        # Extract coordinates from ground truth
        if isinstance(ground_truth, str):
            truth_coordinates = self._extract_coordinates(ground_truth)
        elif isinstance(ground_truth, list):
            truth_coordinates = set(ground_truth)
        else:
            return False

        # Compare the coordinate sets
        return pred_coordinates == truth_coordinates

    def _extract_coordinates(self, coord_str: str) -> set:
        """
        Extract coordinates from string format like "(0,5),(0,7),(1,1),(1,2)"
        Returns a set of (row, col) tuples
        """
        coordinates = set()

        # Pattern to match coordinates like (0,5) or (0, 5)
        pattern = r'\((\d+)\s*,\s*(\d+)\)'
        matches = re.findall(pattern, coord_str)

        for match in matches:
            row, col = int(match[0]), int(match[1])
            coordinates.add((row, col))

        return coordinates

    def extract_answer(self, model_output: str) -> str:
        """
        Extract coordinate list from the model output.
        Look for content within <answer></answer> tags or coordinate patterns.
        """
        # First try to extract content within <answer></answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, model_output, re.DOTALL)

        if match:
            content = match.group(1).strip()
        else:
            # If no tags found, use the whole output
            content = model_output.strip()

        # Look for coordinate patterns like (0,5),(0,7),(1,1),(1,2)
        coord_pattern = r'(\(\d+\s*,\s*\d+\)(?:\s*,\s*\(\d+\s*,\s*\d+\))*)'
        coord_match = re.search(coord_pattern, content)

        if coord_match:
            return coord_match.group(1).strip()

        # If no clear coordinate pattern found, return the content as is
        return content
