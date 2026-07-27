import ast
from typing import Any, Dict, List, Optional, Union


def _parse_grid_like(obj: Union[str, List[List[int]]]) -> Optional[List[List[int]]]:
    if isinstance(obj, list):
        try:
            return [[int(v) for v in row] for row in obj]
        except Exception:
            return None
    if isinstance(obj, str):
        s = obj.strip()
        # Try list literal first
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return _parse_grid_like(parsed)
        except Exception:
            pass
        # Then try whitespace grid with '.' as blanks
        tokens = []
        for line in s.splitlines():
            for tok in line.strip().split():
                if tok == '.':
                    tokens.append(0)
                else:
                    try:
                        tokens.append(int(tok))
                    except Exception:
                        pass
        if len(tokens) == 81:
            grid = [tokens[i * 9:(i + 1) * 9] for i in range(9)]
            return grid
    return None


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


class SudokuEvaluator(BaseEvaluator):
    """Evaluator for classic 9x9 Sudoku using initial givens and Sudoku rules.
    - Parses model output into a 9x9 integer grid.
    - Validates that all givens from initial_state are preserved.
    - Checks rows, columns, and 3x3 blocks contain digits 1..9 exactly once.
    - Ignores ground_truth; correctness is rule-based.
    """

    def prepare_prompt(self, question: str) -> str:
        return str(question)

    def extract_answer(self, model_output: str) -> Optional[List[List[int]]]:
        return _parse_grid_like(model_output)

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        pred_grid = predicted_answer if isinstance(predicted_answer, list) else _parse_grid_like(predicted_answer)
        init_grid = initial_state if isinstance(initial_state, list) else _parse_grid_like(initial_state)
        if pred_grid is None or init_grid is None:
            return False
        # shape check
        if len(pred_grid) != 9 or any(len(r) != 9 for r in pred_grid):
            return False
        if len(init_grid) != 9 or any(len(r) != 9 for r in init_grid):
            return False

        # all entries in prediction must be 1..9
        for i in range(9):
            for j in range(9):
                v = pred_grid[i][j]
                try:
                    iv = int(v)
                except Exception:
                    return False
                if iv < 1 or iv > 9:
                    return False

        # givens preserved
        for i in range(9):
            for j in range(9):
                g = int(init_grid[i][j])
                if g != 0 and int(pred_grid[i][j]) != g:
                    return False

        # rows and cols unique 1..9
        full_set = set(range(1, 10))
        for i in range(9):
            if set(int(x) for x in pred_grid[i]) != full_set:
                return False
        for j in range(9):
            col = [int(pred_grid[i][j]) for i in range(9)]
            if set(col) != full_set:
                return False

        # 3x3 blocks
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                block = [int(pred_grid[r][c]) for r in range(br, br + 3) for c in range(bc, bc + 3)]
                if set(block) != full_set:
                    return False

        return True
