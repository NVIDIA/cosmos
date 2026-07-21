from typing import Any, Dict

import numpy as np


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: str,
                 params: Dict[str, Any] = None) -> bool:
        raise NotImplementedError


class NumbrixEvaluator(BaseEvaluator):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def prepare_prompt(self, question: str) -> str:
        from utils.constants import PROMPT_NUMBRIX
        return PROMPT_NUMBRIX.format(question)

    def evaluate(self, predicted_answer: str, ground_truth: Any, initial_state: Any) -> bool:
        if self.verbose:
            print("predicted_answer: ", predicted_answer[:255])
            print("ground_truth: ", ground_truth[:255])
            print("initial_state: ", initial_state)
        predicted = self._normalize_grid(predicted_answer)
        ground_truth = self._normalize_grid(ground_truth) if ground_truth else None

        try:
            predicted_grid = self._parse_grid(predicted)

            if not self._check_number_uniqueness(predicted_grid):
                return False

            if initial_state:
                initial_grid = self._parse_grid(self._normalize_grid(initial_state))

                if initial_grid.shape[0] > predicted_grid.shape[0] or initial_grid.shape[1] > predicted_grid.shape[1]:
                    return False

                rows, cols = initial_grid.shape
                for i in range(rows):
                    for j in range(cols):
                        if i < predicted_grid.shape[0] and j < predicted_grid.shape[1] and initial_grid[i, j] != 0:
                            if predicted_grid[i, j] != initial_grid[i, j]:
                                return False

            if ground_truth:
                ground_truth_grid = self._parse_grid(ground_truth)
                if predicted_grid.shape != ground_truth_grid.shape:
                    if (
                        ground_truth_grid.shape[0] <= predicted_grid.shape[0]
                        and ground_truth_grid.shape[1] <= predicted_grid.shape[1]
                    ):
                        predicted_grid = predicted_grid[:ground_truth_grid.shape[0], :ground_truth_grid.shape[1]]
                    else:
                        return False

                if np.array_equal(predicted_grid, ground_truth_grid):
                    return True

            return self._validate_numbrix_rules(predicted_grid)
        except Exception:
            import traceback
            traceback.print_exc()
            return False

    def _check_number_uniqueness(self, grid):
        import numpy as np

        numbers = grid[grid > 0]

        unique_numbers, counts = np.unique(numbers, return_counts=True)

        duplicates = [num for num, count in zip(unique_numbers, counts) if count > 1]

        if duplicates:
            return False

        return True

    def _normalize_grid(self, grid_str: str) -> str:
        """标准化网格字符串，移除多余空格并统一换行格式"""
        if not grid_str:
            return ""
        lines = [line.strip() for line in grid_str.strip().split("\n")]
        return "\n".join(lines)

    def _parse_grid(self, grid_str: str):
        """将文本表示解析为二维数组"""
        import numpy as np

        lines = [line for line in grid_str.strip().split('\n') if line.strip()]
        rows = []

        for line in lines:
            # 移除行首尾的'|'字符
            if line.startswith('|'):
                line = line[1:]
            if line.endswith('|'):
                line = line[:-1]

            # 按'|'分割并转换为整数
            row = []
            for cell in line.split('|'):
                cell = cell.strip()
                if cell and cell.isdigit():
                    row.append(int(cell))
                else:
                    row.append(0)  # 空单元格或非数字内容
            rows.append(row)

        # 确保所有行长度一致
        if self.verbose:
            print("rows: ", rows)
        max_length = max(len(row) for row in rows)
        padded_rows = [row + [0] * (max_length - len(row)) for row in rows]
        return np.array(padded_rows)

    def _validate_numbrix_rules(self, grid):
        """
        验证网格是否符合Numbrix规则:
        1. 每对连续数字必须相邻(水平或垂直)
        2. 从1到网格中的最大数字，所有数字必须存在且不重复
        3. 可能会有空格（不需要填满每个格子）
        """
        import numpy as np

        # 找出网格中的非零值
        non_zero_values = grid[grid > 0]
        if len(non_zero_values) == 0:
            return False

        max_num = np.max(non_zero_values)

        # 检查从1到max_num所有数字是否存在
        expected_nums = set(range(1, max_num + 1))
        actual_nums = set(non_zero_values.flatten())

        missing = expected_nums - actual_nums
        extra = actual_nums - expected_nums - {0}  # 排除0（空格）

        if missing:
            return False

        if extra:
            return False

        # 检查每对连续数字是否相邻
        for num in range(1, max_num):
            # 找到当前数字和下一个数字的位置
            current_pos = np.where(grid == num)
            next_pos = np.where(grid == num + 1)

            if len(current_pos[0]) == 0 or len(next_pos[0]) == 0:
                return False

            r1, c1 = current_pos[0][0], current_pos[1][0]
            r2, c2 = next_pos[0][0], next_pos[1][0]

            # 计算曼哈顿距离，相邻单元格距离为1
            manhattan_dist = abs(r2 - r1) + abs(c2 - c1)
            if manhattan_dist != 1:
                return False

        # 验证通过
        return True
