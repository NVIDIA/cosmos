import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple, Union

from .common_constriants import (Constraint, ConstraintColNoRepeat, ConstraintRowNoRepeat,
                                 ConstraintSubGridNoRepeat)
from .common_puzzle_factory import PuzzleFactory


class ConstraintColorNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_color_no_repeat"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        colors = game_state.get("colors", None)

        # If no colors are specified, skip this constraint
        if colors is None:
            return True

        color_groups = {}
        for i in range(len(board)):
            for j in range(len(board[0])):
                color = colors[i][j]
                if color not in color_groups:
                    color_groups[color] = []
                if board[i][j] != 0:
                    color_groups[color].append(board[i][j])
        for color_values in color_groups.values():
            if len(set(color_values)) != len(color_values):
                return False
        return True

class ColoredSudokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "coloredsudoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintColorNoRepeat()
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]
        self.colors = [chr(65 + i) for i in range(size)]

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
