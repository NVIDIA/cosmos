import argparse
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintAquariumFill(Constraint):
    """Check aquarium conditions:
    1. If there's a highest water row in the aquarium, all cells from that row downward in the same aquarium must not be empty.
    2. For every row in the aquarium, if a cell is defined as empty or filled, all defined cells in that row must match.
    3. For every column in the aquarium, if a top cell is filled with water, all consecutive lower cells must not be empty.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_aquarium_fill"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        aquariums = game_state.get("clues", {}).get("aquariums", [])

        for aquarium in aquariums:
            # Find highest water cell in this aquarium
            highest_water_row = float('inf')
            for r, c in aquarium:
                if board[r][c] == "s":
                    highest_water_row = min(highest_water_row, r)

            if highest_water_row == float('inf'):
                continue  # No water in this aquarium

            # Check all cells at or below highest water level
            for r, c in aquarium:
                if r >= highest_water_row:  # if cell is at same height or lower than highest water
                    if board[r][c] == "e":  # if cell is empty
                        return False

        return True

class ConstraintAquariumCount(Constraint):
    """Check if row and column counts match the clues"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_aquarium_count"

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
            # If row is fully decided (no zeros), it must match exactly
            if 0 not in board[i]:
                if row_selected != row_counts[i]:
                    return False
            else:
                # If not fully decided, no more than the count should be selected
                if row_selected > row_counts[i]:
                    return False
                # Also must be possible to still reach the target
                if row_selected + row_undefined < row_counts[i]:
                    return False

        # Check columns
        for j in range(size):
            col_cells = [board[i][j] for i in range(size)]
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(cell != 0 for cell in col_cells):
                if col_selected != col_counts[j]:
                    return False
            else:
                if col_selected > col_counts[j]:
                    return False
                if col_selected + col_undefined < col_counts[j]:
                    return False

        return True


class AquariumPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4:
            raise ValueError("Size must be at least 4")

        self.game_name = "aquarium"
        self.size = size
        self.constraints = [ConstraintAquariumFill(), ConstraintAquariumCount()]
        self.all_possible_values = ["e", "s"]  # empty or selected (water)

    def get_possible_values(
        self, game_state: Dict[str, Any], row: int, col: int
    ) -> List[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values
