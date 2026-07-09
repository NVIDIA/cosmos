import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintKakurasuSum(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_kakurasu_sum"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", {"row_clues": [], "col_clues": []})
        size = len(board)
        weights = [i + 1 for i in range(size)]
        # Check row sums
        for i in range(size):
            row_sum = sum(weights[j] for j in range(size) if board[i][j] == "s")
            # If row is complete (no 0s), sum must equal clue
            if 0 not in board[i]:
                if row_sum != clues["row_clues"][i]:
                    return False
            # If row is incomplete, check both conditions:
            else:
                # 1. Current sum must not exceed clue
                if row_sum > clues["row_clues"][i]:
                    return False
                # 2. Check if remaining undefined cells can potentially reach the clue
                undefined_cells = [j for j in range(size) if board[i][j] == 0]
                max_possible_sum = row_sum + sum(weights[j] for j in undefined_cells)
                if max_possible_sum < clues["row_clues"][i]:
                    return False
        # Check column sums
        for i in range(size):
            col_sum = sum(weights[j] for j in range(size) if board[j][i] == "s")
            # If column is complete (no 0s), sum must equal clue
            if all(board[j][i] != 0 for j in range(size)):
                if col_sum != clues["col_clues"][i]:
                    return False
            # If column is incomplete, check both conditions:
            else:
                # 1. Current sum must not exceed clue
                if col_sum > clues["col_clues"][i]:
                    return False
                # 2. Check if remaining undefined cells can potentially reach the clue
                undefined_cells = [j for j in range(size) if board[j][i] == 0]
                max_possible_sum = col_sum + sum(weights[j] for j in undefined_cells)
                if max_possible_sum < clues["col_clues"][i]:
                    return False
        return True

class KakurasuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 3:
            raise ValueError("Grid size must be at least 3")
        self.game_name = "kakurasu"
        self.size = size
        self.constraints = [
            ConstraintKakurasuSum()
        ]
        self.all_possible_values = ["e", "s"]
        self.weights = [i + 1 for i in range(size)]

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
