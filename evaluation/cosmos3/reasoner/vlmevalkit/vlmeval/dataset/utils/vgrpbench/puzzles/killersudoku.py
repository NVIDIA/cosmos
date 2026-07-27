import argparse
import copy
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import (Constraint, ConstraintColNoRepeat, ConstraintRowNoRepeat,
                                 ConstraintSubGridNoRepeat)
from .common_puzzle_factory import PuzzleFactory


class ConstraintCageSum(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_cage_sum"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        cages = game_state.get("cages", [])  # Default to empty list if no cages

        for cage in cages:
            cells = cage["cells"]
            target_sum = cage["sum"]
            current_sum = 0
            for row, col in cells:
                if board[row][col] == 0:  # Skip empty cells
                    continue
                current_sum += board[row][col]
            if current_sum > target_sum:  # Can't exceed target sum
                return False
            # Only check equality if all cells in cage are filled
            if all(board[row][col] != 0 for row, col in cells) and current_sum != target_sum:
                return False
        return True

class KillerSudokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "killersudoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintSubGridNoRepeat(),
            ConstraintCageSum()
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        # Ensure cages exist in game_state
        if "cages" not in game_state:
            game_state["cages"] = []

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
