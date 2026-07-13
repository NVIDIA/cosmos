#!/usr/bin/env python3

import json
import re
from functools import reduce
from typing import Any, Dict, List


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class CalcudokuEvaluator(BaseEvaluator):
    """
    评估Calcudoku（计算数独）解答的评估器
    验证模型输出的解答是否：
    1. 符合Calcudoku的基本规则（每行每列包含1到n的数字且不重复）
    2. 符合每个区域的数学运算规则
    """

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """准备发送给模型的提示词"""
        size = params.get("size", 3)
        regions = params.get("regions", [])

        prompt = (
            f"This is a {size}x{size} Calcudoku puzzle. Each row and column must contain the numbers 1 to {size} "
            f"exactly once.\n"
            f"The grid is divided into regions, each with a target number and a specified operation.\n"
            f"The numbers within each region must be combined using the given operation to achieve the "
            f"target number.\n\n"
        )

        # 添加区域信息
        for i, region in enumerate(regions):
            cells = region.get('cells', [])
            operator = region.get('operator', '+')
            target = region.get('target', 0)

            # 将乘法*符号转换为×以便显示
            display_op = '×' if operator == '*' else operator

            cell_str = ', '.join([f"({r},{c})" for r, c in cells])
            prompt += f"Region {i+1}: Cells {cell_str}, Operation: {display_op}, Target: {target}\n"

        prompt += (
            "\nPlease solve the puzzle and provide the solution as a two-dimensional array.\n"
            "Example answer format: [[1, 2, 3], [3, 1, 2], [2, 3, 1]]"
        )

        return prompt

    def extract_answer(self, model_output: str) -> List[List[int]]:
        """从模型输出中提取Calcudoku解答"""
        if isinstance(model_output, dict) and "text" in model_output:
            model_output = model_output["text"]

        # 尝试查找完整的二维数组格式
        # 匹配 [[数字, 数字, ...], [数字, 数字, ...], ...]
        array_pattern = r'\[\s*\[(?:\s*\d+\s*,\s*)*\s*\d+\s*\](?:\s*,\s*\[\s*(?:\d+\s*,\s*)*\d+\s*\])*\s*\]'
        matches = re.findall(array_pattern, model_output)

        if matches:
            # 取最后一个匹配的数组（可能是最终答案）
            try:
                # 尝试解析匹配到的字符串为JSON格式的数组
                return json.loads(matches[-1])
            except json.JSONDecodeError:
                pass

        # 如果无法直接解析为JSON，尝试手动解析
        # 首先检查是否有明显的二维数组表示
        lines = model_output.split('\n')
        grid_lines = []

        for line in lines:
            # 查找包含多个数字的行
            if re.search(r'\[\s*\d+.*\d+\s*\]', line):
                grid_lines.append(line)

        if grid_lines:
            # 尝试构建一个有效的二维数组字符串
            grid_str = '[' + ','.join(grid_lines) + ']'
            grid_str = re.sub(r'[^\[\],\d\s]', '', grid_str)  # 移除不应出现在JSON数组中的字符
            try:
                return json.loads(grid_str)
            except json.JSONDecodeError:
                pass

        # 最后尝试提取所有数字序列，根据问题规模构建网格
        all_numbers = re.findall(r'\d+', model_output)

        # 猜测网格大小（假设网格是方形的）
        grid_size = int(len(all_numbers) ** 0.5) if all_numbers else 0

        if grid_size > 0 and grid_size ** 2 == len(all_numbers):
            grid = []
            for i in range(0, len(all_numbers), grid_size):
                row = [int(num) for num in all_numbers[i:i + grid_size]]
                grid.append(row)
            return grid

        return []

    def evaluate(self, model_output: str, ground_truth: Any, params: Dict[str, Any]) -> bool:
        """
        评估预测的Calcudoku解答是否正确，不直接比对ground_truth，而是验证解是否满足所有规则

        参数:
        model_output: 模型生成的文本输出
        ground_truth: 不再直接使用，但保留参数以保持接口一致性
        params: 包含谜题信息的参数

        返回:
        是否正确（布尔值）
        """
        # 从模型输出中提取答案
        extracted_answer = self.extract_answer(model_output)

        # 如果无法提取有效答案，直接返回False
        if not extracted_answer or not isinstance(extracted_answer, list):
            return False

        # 提取谜题信息
        size = params.get("size", len(extracted_answer))
        regions = params.get("regions", [])

        # 1. 验证网格尺寸
        if len(extracted_answer) != size:
            return False

        for row in extracted_answer:
            if len(row) != size or not isinstance(row, list):
                return False

        # 2. 验证每行每列包含1到n的数字且不重复
        expected_set = set(range(1, size + 1))

        # 检查每行
        for row in extracted_answer:
            if set(row) != expected_set:
                return False

        # 检查每列
        for col in range(size):
            column_values = [extracted_answer[row][col] for row in range(size)]
            if set(column_values) != expected_set:
                return False

        # 3. 验证每个区域的运算规则
        for region in regions:
            cells = region.get('cells', [])
            operator = region.get('operator', '+')
            target = region.get('target', 0)

            # 提取区域中的值
            region_values = []
            for r, c in cells:
                # 注意：cells坐标可能是1-indexed，需要转换为0-indexed
                row_idx = r - 1
                col_idx = c - 1

                # 确保索引在有效范围内
                if 0 <= row_idx < len(extracted_answer) and 0 <= col_idx < len(extracted_answer[row_idx]):
                    region_values.append(extracted_answer[row_idx][col_idx])
                else:
                    # 索引超出范围，说明解答有问题
                    return False

            # 确保提取了正确数量的值
            if len(region_values) != len(cells):
                return False

            # 根据运算符验证
            result = self._calculate_region(region_values, operator)
            if result != target:
                return False

        # 所有规则验证通过
        return True

    def _calculate_region(self, values: List[int], operator: str) -> int:
        """
        根据指定的操作符计算区域的结果值

        参数:
            values: 区域内的数值列表
            operator: 操作符（+, -, *, ÷）

        返回:
            计算结果
        """
        if not values:
            return 0

        if operator == '+':
            return sum(values)
        elif operator == '*':
            return reduce(lambda x, y: x * y, values)
        elif operator == '-':
            # 减法适用于两个数字的情况，取绝对值
            if len(values) == 2:
                return abs(values[0] - values[1])
            return 0
        elif operator == '÷':
            # 除法适用于两个数字的情况，取最大值除以最小值
            if len(values) == 2:
                return max(values) // min(values) if min(values) != 0 else 0
            return 0
        else:
            return 0
