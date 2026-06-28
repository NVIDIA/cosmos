import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import (Constraint, ConstraintColNoRepeat, ConstraintRowNoRepeat,
                                 ConstraintSubGridNoRepeat)
from .common_puzzle_factory import PuzzleFactory


class ConstraintRowNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
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
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for col in range(size):
            values = [board[row][col] for row in range(size) if board[row][col] != 0]
            if len(set(values)) != len(values):
                return False
        return True

class ConstraintVisibility(Constraint):
    def __init__(self) -> None:
        super().__init__()
    def calculate_visible_buildings(self, line: List[int]) -> int:
        visible = 0
        max_height = 0
        for height in line:
            if int(height) > max_height:
                visible += 1
                max_height = int(height)
        return visible
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state["clues"]
        size = len(board)

        # Check all directions
        for i in range(size):
            # Top clues
            if clues["top"][i] != 0:
                col = [board[row][i] for row in range(size)]
                if 0 not in col and self.calculate_visible_buildings(col) != clues["top"][i]:
                    return False
            # Bottom clues
            if clues["bottom"][i] != 0:
                col = [board[row][i] for row in range(size-1, -1, -1)]
                if 0 not in col and self.calculate_visible_buildings(col) != clues["bottom"][i]:
                    return False
            # Left clues
            if clues["left"][i] != 0:
                if 0 not in board[i] and self.calculate_visible_buildings(board[i]) != clues["left"][i]:
                    return False
            # Right clues
            if clues["right"][i] != 0:
                if 0 not in board[i] and self.calculate_visible_buildings(board[i][::-1]) != clues["right"][i]:
                    return False
        return True

class SkyscraperPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4 or size > 12:
            raise ValueError("Grid size must be between 4 and 12")
        self.game_name = "skyscraper"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintVisibility()
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]
        self.possible_hint_counts = [4, 5, 6, 7, 8, 9, 10, 11, 12]

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        board = game_state["board"]
        original_value = board[row][col]
        possible_values = []
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
