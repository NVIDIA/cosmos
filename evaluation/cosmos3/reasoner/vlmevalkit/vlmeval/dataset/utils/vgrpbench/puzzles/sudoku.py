import argparse
import copy
import json
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from .common_constriants import (Constraint, ConstraintColNoRepeat, ConstraintRowNoRepeat,
                                 ConstraintSubGridNoRepeat)
from .common_puzzle_factory import PuzzleFactory


class SudokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "sudoku"
        self.size = size

        self.constraints.append(ConstraintRowNoRepeat())
        self.constraints.append(ConstraintColNoRepeat())
        self.constraints.append(ConstraintSubGridNoRepeat())

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
