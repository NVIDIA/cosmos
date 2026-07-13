from typing import Any, Dict


class Constraint():
    def __init__(self) -> None:
        self.name = ""
    def check(self, game_state: Dict[str, Any]) -> bool:
        pass

class ConstraintRowNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:
            row_tmp = [cell for cell in row if cell != 0]
            if len(set(row_tmp)) != len(row_tmp):
                return False
        return True

class ConstraintColNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        for col in range(len(board[0])):
            col_tmp = [board[row][col] for row in range(len(board)) if board[row][col] != 0]
            if len(set(col_tmp)) != len(col_tmp):
                return False
        return True

class ConstraintSubGridNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_sub_grid_no_repeat"
    def check(self, game_state: Dict[str, Any]) -> bool:
        board = game_state["board"]
        assert len(board) == len(board[0]), "board is not square"
        assert len(board) in [4, 9], "board size is not 4 or 9"

        sub_grid_size = int(len(board) ** 0.5)
        for i in range(0, len(board), sub_grid_size):
            for j in range(0, len(board[0]), sub_grid_size):
                sub_grid = [
                    board[x][y] for x in range(i, i + sub_grid_size)
                    for y in range(j, j + sub_grid_size)
                    if board[x][y] != 0
                ]
                if len(set(sub_grid)) != len(sub_grid):
                    return False
        return True
