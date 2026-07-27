import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintHitoriNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]  # This is the shading state
        numbers = game_state.get("numbers", [])  # Get the numbers from additional state
        size = len(board)
        # Check rows and columns for unshaded duplicates
        for i in range(size):
            row_values = [numbers[i][j] for j in range(size) if board[i][j] == "e"]  # 'e' means unshaded
            col_values = [numbers[j][i] for j in range(size) if board[j][i] == "e"]
            if len(row_values) != len(set(row_values)) or len(col_values) != len(set(col_values)):
                return False
        return True

class ConstraintHitoriAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_adjacent"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for row in range(size):
            for col in range(size):
                if board[row][col] == "s":  # shaded cell
                    # Check adjacent cells
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == "s":
                            return False
        return True

class ConstraintHitoriConnected(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_connected"

    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Find first unshaded or undecided cell
        start = None
        for r in range(size):
            for c in range(size):
                if board[r][c] in ["e", 0]:  # 'e' means unshaded, 0 means undecided
                    start = (r, c)
                    break
            if start:
                break

        if not start:
            return False

        # BFS to check connectivity
        visited = [[False] * size for _ in range(size)]
        queue = [start]
        visited[start[0]][start[1]] = True
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < size and 0 <= nc < size and
                    not visited[nr][nc] and board[nr][nc] in ["e", 0]):
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        # Check if all unshaded and undecided cells are visited
        for r in range(size):
            for c in range(size):
                if board[r][c] in ["e", 0] and not visited[r][c]:
                    return False
        return True

class HitoriPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "hitori"
        self.size = size
        self.constraints = [
            ConstraintHitoriNoRepeat(),
            ConstraintHitoriAdjacent(),
            ConstraintHitoriConnected()
        ]
        self.all_possible_values = ["e", "s"]  # 'e' for empty/unshaded, 's' for shaded

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
