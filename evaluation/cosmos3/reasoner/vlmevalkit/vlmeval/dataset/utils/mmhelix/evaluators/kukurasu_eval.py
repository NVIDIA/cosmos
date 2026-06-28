import json
import re
from typing import Any, Dict, List, Union


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class KukurasuEvaluator(BaseEvaluator):
    """
    评估Kukurasu谜题解答的评估器
    Kukurasu是一种填充黑白格子的谜题，要求：
    1. 每行黑格子的列位置之和等于给定的行约束
    2. 每列黑格子的行位置之和等于给定的列约束
    """

    def extract_answer(self, model_output: str) -> List[List[int]]:
        """
        从模型输出中提取Kukurasu解答矩阵

        Args:
            model_output: 模型生成的字符串输出

        Returns:
            提取的二维列表，表示Kukurasu的解答
        """
        # 使用正则表达式寻找类似 [[1, 1, 1], [1, 0, 0], [1, 1, 0]] 的模式
        pattern = r'\[\s*\[(?:\s*\d+\s*,\s*)*\s*\d+\s*\](?:\s*,\s*\[\s*(?:\d+\s*,\s*)*\d+\s*\])*\s*\]'
        matches = re.findall(pattern, model_output)

        if not matches:
            # 如果没有找到符合格式的答案，返回空列表
            return []

        try:
            # 尝试解析找到的第一个匹配项
            answer_matrix = json.loads(matches[0])
            # 确保它是一个二维列表，且每个元素都是整数
            if (
                isinstance(answer_matrix, list)
                and all(
                    isinstance(row, list)
                    and all(isinstance(item, int) for item in row)
                    for row in answer_matrix
                )
            ):
                return answer_matrix
            return []
        except json.JSONDecodeError:
            # 如果JSON解析失败，返回空列表
            return []

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """
        准备用于解决Kukurasu谜题的提示词

        Args:
            question: 问题描述
            params: 包含谜题信息的参数，如行列约束和尺寸

        Returns:
            格式化的提示词
        """
        row_sums = params.get("row_sums", [])
        col_sums = params.get("col_sums", [])
        size = params.get("size", len(row_sums))

        prompt = f"""请解决以下Kukurasu谜题:

    这是一个 {size}x{size} 的网格，需要填充黑色格子(1)和白色格子(0)。

    规则:
    1. 每行黑格子的列位置之和等于行约束
    2. 每列黑格子的行位置之和等于列约束

    列位置指的是从左到右的列索引(从1开始)，行位置指的是从上到下的行索引(从1开始)。

    行约束: {row_sums}
    列约束: {col_sums}

    例如，如果一行的约束是6，且该行在第1、2、3列有黑格子，那么 1+2+3=6，满足约束。
    同样，如果一列的约束是4，且该列在第1、3行有黑格子，那么 1+3=4，满足约束。

    请给出网格的填充方案，格式为二维数组，其中1表示黑格子，0表示白格子。
    例如: [[1, 1, 1], [1, 0, 0], [1, 1, 0]]

    解题过程:
    1. 分析行列约束
    2. 确定每个格子的颜色(黑/白)
    3. 验证所有约束是否满足
    4. 提供最终的网格填充方案
    """

        return prompt

    def evaluate(self, output: Union[str, List[List[int]]], ground_truth: List[List[int]],
                 params: Dict[str, Any]) -> bool:
        """
        评估模型的解答是否正确，基于Kukurasu规则验证

        Args:
            output: 模型的原始输出字符串或已提取的答案
            ground_truth: 正确的解答（仅用于参考，不直接比对）
            params: 包含行列约束和尺寸的参数

        Returns:
            解答是否正确的布尔值
        """
        # 检查output类型，如果是字符串，则提取答案；如果已经是列表，则直接使用
        if isinstance(output, str):
            predicted_answer = self.extract_answer(output)
        else:
            predicted_answer = output

        # 如果无法提取有效答案，直接返回False
        if not predicted_answer:
            return False

        # 获取行列约束和尺寸
        row_sums = params.get("row_sums", [])
        col_sums = params.get("col_sums", [])
        size = params.get("size", len(row_sums))

        # 检查答案维度是否正确
        if len(predicted_answer) != size:
            return False

        if any(len(row) != size for row in predicted_answer):
            return False

        # 检查格子值是否只包含0和1
        if any(cell not in [0, 1] for row in predicted_answer for cell in row):
            return False

        # 验证每行的约束
        for i, row in enumerate(predicted_answer):
            row_sum = sum((j + 1) for j, cell in enumerate(row) if cell == 1)
            if row_sum != row_sums[i]:
                return False

        # 验证每列的约束
        for j in range(size):
            col_sum = sum((i + 1) for i, row in enumerate(predicted_answer) if row[j] == 1)
            if col_sum != col_sums[j]:
                return False

        # 所有约束都满足，返回True
        return True
