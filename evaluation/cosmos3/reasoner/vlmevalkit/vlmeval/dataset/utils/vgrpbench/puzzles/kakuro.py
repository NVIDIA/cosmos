import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import (Constraint, ConstraintColNoRepeat, ConstraintRowNoRepeat,
                                 ConstraintSubGridNoRepeat)
from .common_puzzle_factory import PuzzleFactory


class ConstraintKakuroSum(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_kakuro_sum"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        # if any of board is str, then convert to int
        if any(isinstance(cell, str) for row in board for cell in row):
            board = [[int(cell) for cell in row] for row in board]

        sums = game_state.get("sums", {"row": [], "col": []})
        # Check row sums
        for row in range(len(board)):
            if row < len(sums["row"]):
                target_sum = sums["row"][row]
                current_sum = sum(x for x in board[row] if x != 0)
                if current_sum > target_sum:
                    return False
                if all(x != 0 for x in board[row]) and current_sum != target_sum:
                    return False

        # Check column sums
        for col in range(len(board[0])):
            if col < len(sums["col"]):
                target_sum = sums["col"][col]
                current_sum = sum(board[row][col] for row in range(len(board)) if board[row][col] != 0)
                if current_sum > target_sum:
                    return False
                if all(board[row][col] != 0 for row in range(len(board))) and current_sum != target_sum:
                    return False

        return True

class ConstraintKakuroAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_kakuro_adjacent"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        # if any of board is str, then convert to int
        if any(isinstance(cell, str) for row in board for cell in row):
            board = [[int(cell) for cell in row] for row in board]

        size = len(board)

        for row in range(size):
            for col in range(size):
                if board[row][col] == 0:
                    continue
                # Check adjacent cells (up, down, left, right)
                if row > 0 and board[row-1][col] == board[row][col]:
                    return False
                if row < size-1 and board[row+1][col] == board[row][col]:
                    return False
                if col > 0 and board[row][col-1] == board[row][col]:
                    return False
                if col < size-1 and board[row][col+1] == board[row][col]:
                    return False
        return True

class KakuroPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 3 or size > 12:
            raise ValueError("Grid size must be between 3 and 12")
        self.game_name = "kakuro"
        self.size = size
        self.constraints = [
            ConstraintKakuroSum(),
            ConstraintKakuroAdjacent()
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]
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
