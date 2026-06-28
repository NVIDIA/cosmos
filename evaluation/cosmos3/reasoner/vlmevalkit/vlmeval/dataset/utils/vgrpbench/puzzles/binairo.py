import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintRowBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_balance"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        assert all(all(cell != '*' for cell in row) for row in board), "'*' should be replaced by '0' in the initialization board"

        for row in board:
            if 0 not in row:  # Only check completed rows
                white_count = sum(1 for x in row if x == 'w')
                black_count = sum(1 for x in row if x == 'b')
                if white_count != black_count or white_count != expected_count:
                    return False
        return True

class ConstraintColBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_balance"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        for col in range(size):
            column = [board[row][col] for row in range(size)]
            if 0 not in column and '*' not in column:  # Only check completed columns
                white_count = sum(1 for x in column if x == 'w')
                black_count = sum(1 for x in column if x == 'b')
                if white_count != black_count or white_count != expected_count:
                    return False
        return True

class ConstraintNoTripleAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_no_triple_adjacent"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check rows
        for row in range(size):
            for col in range(size - 2):
                if (board[row][col] != 0 and
                    board[row][col] == board[row][col + 1] == board[row][col + 2]):
                    return False

        # Check columns
        for col in range(size):
            for row in range(size - 2):
                if (board[row][col] != 0 and
                    board[row][col] == board[row + 1][col] == board[row + 2][col]):
                    return False
        return True


class BinairoPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4 or size % 2 != 0:
            raise ValueError("Size must be an even number greater than or equal to 4")
        self.game_name = "binairo"
        self.size = size
        self.constraints = [
            ConstraintRowBalance(),
            ConstraintColBalance(),
            ConstraintNoTripleAdjacent(),
            # ConstraintUniqueLines()
        ]
        self.all_possible_values = ['w', 'b']  # 'w' for white, 'b' for black

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
