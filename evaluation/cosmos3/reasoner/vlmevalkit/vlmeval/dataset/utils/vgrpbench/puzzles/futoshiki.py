import argparse
import copy
import json
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

class ConstraintInequality(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_inequality"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        inequalities = game_state.get("inequalities", {"row": [], "col": []})
        # Check row inequalities
        row_ineq = inequalities.get("row", [['' for _ in range(size-1)] for _ in range(size)])
        for row in range(size):
            for col in range(size-1):
                if row_ineq[row][col] == '<':
                    if board[row][col] != 0 and board[row][col+1] != 0:
                        if board[row][col] >= board[row][col+1]:
                            return False
                elif row_ineq[row][col] == '>':
                    if board[row][col] != 0 and board[row][col+1] != 0:
                        if board[row][col] <= board[row][col+1]:
                            return False
        # Check column inequalities
        col_ineq = inequalities.get("col", [['' for _ in range(size)] for _ in range(size-1)])
        for row in range(size-1):
            for col in range(size):
                if col_ineq[row][col] == '^':
                    if board[row][col] != 0 and board[row+1][col] != 0:
                        if board[row][col] >= board[row+1][col]:
                            return False
                elif col_ineq[row][col] == 'v':
                    if board[row][col] != 0 and board[row+1][col] != 0:
                        if board[row][col] <= board[row+1][col]:
                            return False
        return True



class FutoshikiPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 3 or size > 12:
            raise ValueError("Grid size must be between 3 and 9")
        self.game_name = "futoshiki"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintInequality()
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
