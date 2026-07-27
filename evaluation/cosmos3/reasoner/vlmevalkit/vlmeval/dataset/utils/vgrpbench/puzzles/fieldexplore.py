import argparse
import copy
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintAdjacentNumbers(Constraint):
    """Ensures revealed numbers match adjacent mine counts"""
    def check(self, game_state: List[List[Any]]) -> bool:

        board = game_state["board"]

        size = len(board)
        for i in range(size):
            for j in range(size):
                if isinstance(board[i][j], int) and board[i][j] != 0:  # If cell is a revealed number
                    # Count adjacent mines and undefined cells
                    i_start = max(0, i-1)
                    i_end = min(size, i+2)
                    j_start = max(0, j-1)
                    j_end = min(size, j+2)

                    adjacent_mines = sum(1 for r in range(i_start, i_end)
                                      for c in range(j_start, j_end)
                                      if board[r][c] == 's')

                    adjacent_undefined = sum(1 for r in range(i_start, i_end)
                                          for c in range(j_start, j_end)
                                          if board[r][c] == 0)

                    # Check if current mines <= number <= potential mines (current + undefined)
                    if adjacent_mines > board[i][j] or adjacent_mines + adjacent_undefined < board[i][j]:
                        return False
        return True

class FieldExplorePuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "fieldexplore"
        self.size = size
        self.constraints = [ConstraintAdjacentNumbers()]
        self.all_possible_values = ['s', 'e']  # True for 's', False for 'e'

    def check(self, board: List[List[Any]]) -> bool:
        for constraint in self.constraints:
            if not constraint.check(board):
                return False
        return True

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
