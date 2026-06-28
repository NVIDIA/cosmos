import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintRowTents(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:

        board = game_state["board"]
        # if board[0][0] == 'e' and board[0][1] == 'e':
        #     import ipdb; ipdb.set_trace()
        clues = game_state.get("clues", None)
        if not clues:
            return True
        for i, row in enumerate(board):
            if 0 not in row:  # If row is complete
                tent_count = row.count("tt")
                if tent_count != clues["row_clues"][i]:
                    return False
            else:  # If row is incomplete
                tent_count = row.count("tt")
                if tent_count > clues["row_clues"][i]:
                    return False
        return True

class ConstraintColTents(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:

        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True
        size = len(board)
        for j in range(size):
            col = [board[i][j] for i in range(size)]
            if 0 not in col:  # If column is complete
                tent_count = col.count("tt")
                if tent_count != clues["col_clues"][j]:
                    return False
            else:  # If column is incomplete
                tent_count = col.count("tt")
                if tent_count > clues["col_clues"][j]:
                    return False
        return True

class ConstraintTentTree(Constraint):
    """
    Check if:
    1. Each tent has exactly one adjacent tree (horizontally or vertically)
    2. Each tree has exactly one adjacent tent (horizontally or vertically) when complete
    3. Each tree should have exactly one tent or potential tent spot (empty cell) adjacent
    """
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Keep track of which trees are paired with which tents
        tree_tent_pairs = {}  # tree position -> tent position

        # First, check each tent has exactly one adjacent tree
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    adjacent_trees = []
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:  # Only orthogonal
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tr":
                                adjacent_trees.append((ni, nj))
                    # Each tent must have exactly one adjacent tree
                    if len(adjacent_trees) != 1:
                        return False

                    tree_pos = adjacent_trees[0]

                    tree_tent_pairs[tree_pos] = (i, j)

        # Then, check each tree
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tr":
                    # Count adjacent tents and empty cells
                    adjacent_tents = 0
                    adjacent_non_allocated = 0
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tt":
                                adjacent_tents += 1
                            elif board[ni][nj] == 0:
                                adjacent_non_allocated += 1

                    if adjacent_tents > 1:
                        return False
                    if adjacent_tents == 1:
                        pass
                    if adjacent_tents == 0:
                        if adjacent_non_allocated == 0:
                            return False

        return True

class ConstraintAdjacentTents(Constraint):
    """
    Check if tents are not adjacent (including diagonally).
    """
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        # Check tents are not adjacent (including diagonally)
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                if board[ni][nj] == "tt":
                                    return False
        return True

class ConstraintTentTreeCount(Constraint):
    """
    Check if:
    1. Number of tents + unallocated cells >= number of trees (during solving)
    2. Number of tents == number of trees (for completed board)
    """
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        num_trees = sum(row.count("tr") for row in board)
        num_tents = sum(row.count("tt") for row in board)
        num_unallocated = sum(row.count(0) for row in board)

        # If board is complete (no unallocated cells)
        if num_unallocated == 0:
            return num_tents == num_trees

        # During solving, ensure we can still potentially place enough tents
        return (num_tents + num_unallocated) >= num_trees


class TreesAndTentsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "treesandtents"
        self.size = size
        assert size >= 3, "Size must be at least 3"
        self.constraints = [
            ConstraintRowTents(),
            ConstraintColTents(),
            ConstraintTentTree(),
            ConstraintAdjacentTents(),
            ConstraintTentTreeCount()
        ]
        self.all_possible_values = ["tt", 'e']
        self.num_generator_processes = max(os.cpu_count() // 2, 1)  # Limit to 4 processes or CPU count

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[str]:
        """Get possible values for a given cell."""
        board = game_state["board"]
        if board[row][col] != 0:  # If cell is already filled
            return []
        possible = []
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible.append(value)
        board[row][col] = original_value
        return possible
