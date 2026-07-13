import argparse
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintBase:
    def _check_line_hints(self, line, hints) -> bool:
        # Convert line to runs of filled cells ('s' = filled, 'e' = empty, 0 = undefined)
        runs = []  # Will store lengths of consecutive filled cells
        count = 0  # Counter for current run length
        current_run = []  # Track cells in current run for debugging/future use

        if hints == [0]:
            # the line should not contain 's'
            return line.count('s') == 0

        # First pass: Calculate runs of filled cells
        for cell in line:
            if cell == "s":  # Found a filled cell
                count += 1
                current_run.append(cell)
            elif cell == "e":  # Found an empty cell
                if count > 0:  # If we were counting a run
                    runs.append(count)
                    count = 0
                current_run = []
            else:  # cell is 0 (undefined)
                if count > 0:
                    current_run.append(cell)
        # Don't forget to add the last run if it exists
        if count > 0:
            runs.append(count)
        # Calculate cell statistics
        filled_cells = line.count("s")     # Number of definitely filled cells
        undefined_cells = line.count(0)     # Number of cells yet to be determined
        required_cells = sum(hints)         # Total number of cells that should be filled according to hints

        # Early failure: Check if we have enough cells to satisfy hints
        if filled_cells + undefined_cells < required_cells:
            return False

        # For completely defined lines (no undefined cells)
        if undefined_cells == 0:
            # Simple comparison: runs must exactly match hints
            if runs != hints:
                return False
        else:
            # For partially defined lines, check if current definite runs are valid
            definite_runs = []
            count = 0
            # Calculate runs that are definitely complete (bounded by empty cells or edges)
            for cell in line:
                if cell == "s":
                    count += 1
                elif (cell == "e" or cell == 0) and count > 0:
                    definite_runs.append(count)
                    count = 0
                    if cell == 0:  # Stop at first undefined cell
                        break
            if count > 0:
                definite_runs.append(count)
            # Validate the definite runs we've found
            if definite_runs:
                # Can't have more runs than hints
                if len(definite_runs) > len(hints):
                    return False
                # FIXME: Additional validation commented out
                # Check if any run is longer than corresponding hint
                # if any(definite_runs[j] > hints[j] for j in range(len(definite_runs))):
                #     return False
        return True

class ConstraintRowHints(ConstraintBase):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state.get("hints", None)
        if not hints:
            raise ValueError("Hints are not provided")
        row_hints = hints["row_hints"]

        for i, row in enumerate(board):
            if not self._check_line_hints(row, row_hints[i]):
                return False
        return True

class ConstraintColHints(ConstraintBase):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state.get("hints", None)
        if not hints:
            raise ValueError("Hints are not provided")

        col_hints = hints["col_hints"]
        size = len(board)

        for j in range(size):
            col = [board[i][j] for i in range(size)]
            if not self._check_line_hints(col, col_hints[j]):
                return False
        return True

class NonogramPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "nonogram"
        self.size = size
        self.constraints = [
            ConstraintRowHints(),
            ConstraintColHints()
        ]
        self.all_possible_values = ["e", "s"]  # Consistent with paper

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[str]:
        board = game_state["board"]
        if board[row][col] != 0:  # If cell is already filled
            return []

        possible_values = []
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
