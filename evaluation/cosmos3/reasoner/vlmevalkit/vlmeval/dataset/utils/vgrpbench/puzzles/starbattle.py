import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory

DEBUG_CONSTRAINT_ERROR = False

class ConstraintRowStar(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        for row_idx, row in enumerate(board):
            if 0 not in row:
                star_count = sum(1 for cell in row if cell == 's')
                if star_count != 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(f"RowStar constraint failed: Row {row_idx} has {star_count} stars (expected 1)")
                    return False
            else:
                star_count = sum(1 for cell in row if cell == 's')
                if star_count > 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(f"RowStar constraint failed: Incomplete row {row_idx} has {star_count} stars (max 1)")
                    return False
        return True

class ConstraintColStar(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for col in range(size):
            col_values = [board[row][col] for row in range(size)]

            if 0 not in col_values:
                star_count = sum(1 for val in col_values if val == 's')
                if star_count != 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(f"ColStar constraint failed: Column {col} has {star_count} stars (expected 1)")
                    return False
            else:
                star_count = sum(1 for val in col_values if val == 's')
                if star_count > 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(f"ColStar constraint failed: Incomplete column {col} has {star_count} stars (max 1)")
                    return False
        return True

class ConstraintRegionStar(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        regions = game_state["regions"]
        size = len(board)
        region_counts = {}
        for i in range(size):
            for j in range(size):
                if board[i][j] == 's':
                    region = regions[i][j]
                    region_counts[region] = region_counts.get(region, 0) + 1
                    if region_counts[region] > 1:
                        if DEBUG_CONSTRAINT_ERROR:
                            print(f"RegionStar constraint failed: Region {region} has {region_counts[region]} stars (max 1)")
                        return False

        for region_num in set(cell for row in regions for cell in row):
            region_cells = [(i, j) for i in range(size) for j in range(size)
                          if regions[i][j] == region_num]
            if all(board[i][j] != 0 for i, j in region_cells):
                if region_counts.get(region_num, 0) != 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(f"RegionStar constraint failed: Completed region {region_num} has {region_counts.get(region_num, 0)} stars (expected 1)")
                    return False
        return True

class ConstraintAdjacentStar(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        for row in range(size):
            for col in range(size):
                if board[row][col] == 's':
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            new_row, new_col = row + dr, col + dc
                            if (0 <= new_row < size and
                                0 <= new_col < size and
                                board[new_row][new_col] == 's'):
                                if DEBUG_CONSTRAINT_ERROR:
                                    print(f"AdjacentStar constraint failed: Stars at ({row},{col}) and ({new_row},{new_col}) are adjacent")
                                return False
        return True

class StarBattlePuzzleFactory(PuzzleFactory):
    def __init__(self, size: int, num_stars: int = 1) -> None:
        super().__init__()
        self.game_name = "starbattle"
        self.size = size
        self.num_stars = num_stars
        self.colors = [chr(65 + i) for i in range(size)]
        # During generation, only use row, column, and adjacent constraints
        self.constraints = [
            ConstraintRowStar(),
            ConstraintColStar(),
            ConstraintAdjacentStar(),
            ConstraintRegionStar()
        ]

        self.all_possible_values = ['s', 'e']

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[str]:
        """Get possible values ('e' for empty or 's' for star) for a given cell."""
        board = game_state["board"]

        # If the cell is already filled with 'e' or 's', return empty list
        if board[row][col] in ['s', 'e']:
            return []

        # Try both values and return those that don't immediately violate constraints
        possible = []
        for val in ['s', 'e']:
            board[row][col] = val
            if self.check(game_state):
                possible.append(val)
            board[row][col] = 0  # Reset to initial state
        return possible
