import re
from typing import Any, Dict, List


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class NonogramsEvaluator(BaseEvaluator):

    def extract_answer(self, model_output: str) -> List[List[bool]]:
        """Extract the model's answer from its output"""
        matches = re.findall(r'\[answer\](.*?)\[/answer\]', model_output, re.DOTALL)
        if not matches:
            return None

        grid_str = matches[-1].strip()
        solution = []
        for line in grid_str.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Consider 'X' or 'x' as filled cells, anything else as empty
            solution.append([c.upper() == 'X' for c in line])
        return solution

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state=None) -> bool:
        if predicted_answer is None:
            return False

        solution = ground_truth

        if len(predicted_answer) != len(solution):
            return False

        for i, row in enumerate(predicted_answer):
            if len(row) != len(solution[i]):
                return False

            if row != solution[i]:
                return False

        return True

    @staticmethod
    def _get_clues(line):
        """Extract clues from a line (row or column)"""
        clues = []
        current = 0
        for cell in line:
            if cell:
                current += 1
            elif current > 0:
                clues.append(current)
                current = 0
        if current > 0:
            clues.append(current)
        return clues

    def _verify_clues(self, grid, params):
        """Verify that the grid matches the given row and column clues"""
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        # Check row clues
        for i, row in enumerate(grid):
            row_clues = self._get_clues(row)
            if row_clues != params['rows'][i]:
                return False

        # Check column clues
        for j in range(cols):
            col = [grid[i][j] for i in range(rows)]
            col_clues = self._get_clues(col)
            if col_clues != params['columns'][j]:
                return False

        return True
