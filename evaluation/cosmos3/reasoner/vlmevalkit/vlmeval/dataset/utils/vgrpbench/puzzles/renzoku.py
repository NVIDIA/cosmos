import copy
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintRowNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:
            values = [x for x in row if x != 0]
            if len(set(values)) != len(values):
                return False
        return True

class ConstraintColNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for col in range(size):
            values = [board[row][col] for row in range(size) if board[row][col] != 0]
            if len(set(values)) != len(values):
                return False
        return True

class ConstraintAdjacency(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacency"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        # Get hints with proper default structure
        # default_hints = {
        #     "row": [['0' for _ in range(size - 1)] for _ in range(size)],
        #     "col": [['0' for _ in range(size)] for _ in range(size - 1)]
        # }
        # always use hints from the game state
        hints = game_state.get("hints") # , default_hints)
        # Ensure hints have proper dimensions
        if len(hints.get("row", [])) < size:
            hints["row"] = [['0' for _ in range(size - 1)] for _ in range(size)]
        if len(hints.get("col", [])) < size - 1:
            hints["col"] = [['0' for _ in range(size)] for _ in range(size - 1)]
        # convert board to int
        board_copy = copy.deepcopy(board)
        for i in range(size):
            for j in range(size):
                if board_copy[i][j] != 0:
                    board_copy[i][j] = int(board_copy[i][j])

        # Check row adjacency hints
        for row in range(size):
            for col in range(size - 1):
                if hints["row"][row][col] == "1":
                    if board_copy[row][col] == 0 or board_copy[row][col + 1] == 0:
                        continue
                    if abs(board_copy[row][col] - board_copy[row][col + 1]) != 1:
                        return False
        # Check column adjacency hints
        for row in range(size - 1):
            for col in range(size):
                if hints["col"][row][col] == "1":
                    if board_copy[row][col] == 0 or board_copy[row + 1][col] == 0:
                        continue
                    if abs(board_copy[row][col] - board_copy[row + 1][col]) != 1:
                        return False
        return True




class RenzokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4 or size > 12:
            raise ValueError("Grid size must be between 4 and 12")
        self.game_name = "renzoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintAdjacency()
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]
        self.num_solver_processes = max(os.cpu_count() // 2, 1)  # Limit to 4 processes or CPU count

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
