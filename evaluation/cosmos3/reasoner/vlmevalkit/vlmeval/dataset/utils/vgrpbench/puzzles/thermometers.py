import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintThermometerFill(Constraint):
    """Check if thermometers are filled correctly (from bulb to top, no gaps)"""
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_fill"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        thermometers = game_state.get("clues", {}).get("thermometers", [])  # Fixed: get thermometers from clues


        # Create a set of all thermometer positions for efficient lookup
        thermometer_positions = {(r, c) for therm in thermometers for r, c in therm}

        # Check non-thermometer cells are empty or undefined
        for i in range(len(board)):
            for j in range(len(board[i])):
                if (i, j) not in thermometer_positions and board[i][j] == "s":
                    return False
        # Check thermometer filling rules
        for thermometer in thermometers:
            # Find first empty cell in thermometer
            first_empty = -1
            for i, (r, c) in enumerate(thermometer):
                if board[r][c] == "e":  # if empty
                    first_empty = i
                    break
            # After first empty, all cells must be empty
            if first_empty != -1:
                for i, (r, c) in enumerate(thermometer):
                    if i > first_empty and board[r][c] == "s":  # if selected
                        return False
        return True

class ConstraintThermometerCount(Constraint):
    """Check if row and column counts match the clues"""
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_count"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True

        size = len(board)
        row_counts = clues["row_counts"]
        col_counts = clues["col_counts"]

        # Check rows
        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            if 0 not in board[i]:  # if row is complete
                if row_selected != row_counts[i]:
                    return False
            else:  # if row is incomplete
                if row_selected > row_counts[i]:  # too many selected
                    return False
                if row_selected + row_undefined < row_counts[i]:  # impossible to reach target
                    return False
        # Check columns
        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 for i in range(size)):  # if column is complete
                if col_selected != col_counts[j]:
                    return False
            else:  # if column is incomplete
                if col_selected > col_counts[j]:  # too many selected
                    return False
                if col_selected + col_undefined < col_counts[j]:  # impossible to reach target
                    return False
        return True

class ThermometersPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4:
            raise ValueError("Size must be at least 4")
        self.game_name = "thermometers"
        self.size = size
        self.constraints = [
            ConstraintThermometerFill(),
            ConstraintThermometerCount()
        ]

        self.all_possible_values = ["e", "s"]  # empty or selected
    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
