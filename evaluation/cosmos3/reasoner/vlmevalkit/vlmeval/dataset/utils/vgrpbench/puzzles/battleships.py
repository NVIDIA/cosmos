import argparse
import os
import random
from typing import Any, Dict, List, Tuple

from .common_constriants import Constraint
from .common_puzzle_factory import PuzzleFactory


class ConstraintBattleships(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        # Check if ships touch diagonally or orthogonally
        for i in range(size):
            for j in range(size):
                if isinstance(board[i][j], tuple):  # Check if it's a revealed ship with direction
                    ship_cell, direction = board[i][j]
                    # Add direction-specific checks here
                    if direction in "<>-":  # Horizontal ship
                        # Check cells above and below
                        for di in [-1, 1]:
                            if 0 <= i + di < size and board[i + di][j] == "s":
                                return False
                    elif direction in "^V|":  # Vertical ship
                        # Check cells left and right
                        for dj in [-1, 1]:
                            if 0 <= j + dj < size and board[i][j + dj] == "s":
                                return False
                elif board[i][j] == "s":
                    # Regular ship cell checks
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (0 <= ni < size and 0 <= nj < size and
                                (board[ni][nj] == "s" or (isinstance(board[ni][nj], tuple) and board[ni][nj][0] == "s")) and
                                (di != 0 and dj != 0)):  # Diagonal check
                                return False
        return True

class ConstraintBattleshipsHints(Constraint):
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state["hints"]
        row_hints = hints["row_hints"]
        col_hints = hints["col_hints"]
        ships = hints["ships"]
        size = len(board)
        # Calculate total required ship cells from ships configuration
        total_ship_cells_required = sum(int(length) * int(count) for length, count in ships.items())
        total_ship_cells_selected = sum(1 for i in range(size) for j in range(size) if board[i][j] == "s")
        total_undefined_cells = sum(1 for i in range(size) for j in range(size) if board[i][j] == 0)

        # Check if we have enough cells (placed + potential) to fit all ships
        if total_ship_cells_selected + total_undefined_cells < total_ship_cells_required:
            return False

        # Check if we haven't exceeded the total required ship cells
        if total_ship_cells_selected > total_ship_cells_required:
            return False

        # Check row hints
        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            # Consider both undefined (0) and non-revealed water cells for potential ships
            if all(cell != 0 and cell != -1 for cell in board[i]):  # if row is complete
                if row_selected != row_hints[i]:
                    return False
            else:  # if row is incomplete
                if row_selected > row_hints[i]:  # too many selected
                    return False
                if row_selected + row_undefined < row_hints[i]:  # impossible to reach target
                    return False
        # Check column hints
        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 and board[i][j] != -1 for i in range(size)):  # if column is complete
                if col_selected != col_hints[j]:
                    return False
            else:  # if column is incomplete
                if col_selected > col_hints[j]:  # too many selected
                    return False
                if col_selected + col_undefined < col_hints[j]:  # impossible to reach target
                    return False
        # When all cells are filled, check ship shapes
        if total_undefined_cells == 0:
            # Find all ships by finding connected components
            visited = [[False] * size for _ in range(size)]
            ship_lengths = []

            def get_ship_length(i: int, j: int) -> int:
                if (i < 0 or i >= size or j < 0 or j >= size or
                    visited[i][j] or board[i][j] != "s"):
                    return 0

                visited[i][j] = True
                length = 1

                # Check if ship is horizontal
                if (j + 1 < size and board[i][j + 1] == "s"):
                    # Add all horizontal cells
                    for col in range(j + 1, size):
                        if board[i][col] != "s":
                            break
                        visited[i][col] = True
                        length += 1
                # Check if ship is vertical
                elif (i + 1 < size and board[i + 1][j] == "s"):
                    # Add all vertical cells
                    for row in range(i + 1, size):
                        if board[row][j] != "s":
                            break
                        visited[row][j] = True
                        length += 1

                return length

            # Find all ships
            for i in range(size):
                for j in range(size):
                    if not visited[i][j] and board[i][j] == "s":
                        ship_lengths.append(get_ship_length(i, j))
            # Count ships of each length
            ship_counts = {}
            for length in ship_lengths:
                ship_counts[length] = ship_counts.get(length, 0) + 1
            # Verify against required ships
            for length, count in ships.items():
                if ship_counts.get(int(length), 0) != int(count):
                    return False
        return True

class BattleshipsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "battleships"
        self.size = size
        self.constraints = [
            ConstraintBattleships(),
            ConstraintBattleshipsHints()
        ]
        self.all_possible_values = ["e", "s"]

    def get_possible_values(self, game_state: Dict[str, Any], row: int, col: int) -> List[int]:
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
