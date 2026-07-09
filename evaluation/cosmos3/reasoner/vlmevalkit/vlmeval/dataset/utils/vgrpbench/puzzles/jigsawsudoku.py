import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple, Union

from .common_constriants import (Constraint, ConstraintColNoRepeat, ConstraintRowNoRepeat,
                                 ConstraintSubGridNoRepeat)
from .common_puzzle_factory import PuzzleFactory


class ConstraintRegionNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_region_no_repeat"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        regions = game_state.get("regions", None)

        if regions is None:
            return True

        region_groups = {}
        for i in range(len(board)):
            for j in range(len(board[0])):
                region = regions[i][j]
                if region not in region_groups:
                    region_groups[region] = []
                if board[i][j] != 0:
                    region_groups[region].append(board[i][j])
        for region_values in region_groups.values():
            if len(set(region_values)) != len(region_values):
                return False
        return True

class JigsawSudokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "jigsawsudoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintRegionNoRepeat()
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]
        self.cached_region_splits = []

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
        """Get possible values for a cell based on row, column, and region constraints."""
        if game_state["board"][row][col] != 0:
            return []
        possible_values = []
        for value in self.all_possible_values:
            # Try the value
            original_value = game_state["board"][row][col]
            game_state["board"][row][col] = value
            # Check if it's valid according to all constraints
            valid = True
            for constraint in self.constraints:
                if not constraint.check(game_state):
                    valid = False
                    break

            # Restore original value
            game_state["board"][row][col] = original_value

            if valid:
                possible_values.append(value)

        return possible_values
