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


class SkyscrapersEvaluator(BaseEvaluator):
    """
    评估摩天楼谜题解答的评估器
    摩天楼(Skyscrapers)是一种逻辑谜题，规则为：
    1. 每行每列必须包含1到n的每个数字恰好一次
    2. 四周的数字表示从该方向看过去可以看到的摩天楼数量
       (较高的摩天楼会挡住后面较矮的摩天楼)
    """

    def extract_answer(self, model_output: str) -> List[List[int]]:
        """
        从模型输出中提取摩天楼谜题的解答矩阵

        Args:
            model_output: 模型生成的字符串输出

        Returns:
            提取的二维列表，表示摩天楼谜题的解答
        """
        # 使用正则表达式寻找类似 [[3, 1, 2], [2, 3, 1], [1, 2, 3]] 的模式
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
        准备用于解决摩天楼谜题的提示词

        Args:
            question: 问题描述
            params: 包含谜题信息的参数，如尺寸和四个方向的约束

        Returns:
            格式化的提示词
        """
        n = params.get("n", 4)
        top = params.get("top", [])
        bottom = params.get("bottom", [])
        left = params.get("left", [])
        right = params.get("right", [])

        prompt = f"""请解决以下摩天楼(Skyscrapers)谜题:

    这是一个 {n}x{n} 的网格，需要放置高度从1到{n}的摩天楼。

    规则:
    1. 每行和每列必须包含从1到{n}的每个数字恰好一次
    2. 网格四周的数字表示从该方向看过去能看到的摩天楼数量
    3. 较高的摩天楼会挡住后面较矮的摩天楼

    四周的约束条件:
    - 上方(从上往下看): {top}
    - 下方(从下往上看): {bottom}
    - 左侧(从左往右看): {left}
    - 右侧(从右往左看): {right}

    例如，如果一行从左边看的约束是2，且该行的摩天楼高度依次为3,1,4,2，那么只能看到高度为3和4的两栋摩天楼(3挡住了后面的1，4挡住了后面的2)。

    请给出摩天楼的排列方案，格式为二维数组，每个数字表示对应位置摩天楼的高度。
    例如: [[3, 1, 2], [2, 3, 1], [1, 2, 3]]

    解题过程:
    1. 分析四个方向的约束
    2. 确定每个位置可能的摩天楼高度
    3. 使用逻辑推理填写整个网格
    4. 验证所有约束是否满足
    5. 提供最终的排列方案
    """

        return prompt

    def evaluate(self, output: Union[str, List[List[int]]], ground_truth: List[List[int]],
                 params: Dict[str, Any]) -> bool:
        """
        评估模型的解答是否正确，基于摩天楼谜题规则验证

        Args:
            output: 模型的原始输出字符串或已提取的答案
            ground_truth: 正确的解答（仅用于参考，不直接比对）
            params: 包含谜题信息的参数，如尺寸和四个方向的约束

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

        # 获取谜题参数
        n = params.get("n", 4)
        top = params.get("top", [])
        bottom = params.get("bottom", [])
        left = params.get("left", [])
        right = params.get("right", [])

        # 检查答案维度是否正确
        if len(predicted_answer) != n:
            return False
        if any(len(row) != n for row in predicted_answer):
            return False

        # 1. 验证每行每列包含1到n的每个数字恰好一次
        for row in predicted_answer:
            if set(row) != set(range(1, n + 1)):
                return False

        for col_idx in range(n):
            column = [predicted_answer[row_idx][col_idx] for row_idx in range(n)]
            if set(column) != set(range(1, n + 1)):
                return False

        # 2. 验证四个方向的可见性约束

        # 验证上方约束
        for col_idx, constraint in enumerate(top):
            if constraint > 0:  # 0表示没有约束
                visible_count = self._count_visible_from_top(predicted_answer, col_idx)
                if visible_count != constraint:
                    return False

        # 验证下方约束
        for col_idx, constraint in enumerate(bottom):
            if constraint > 0:
                visible_count = self._count_visible_from_bottom(predicted_answer, col_idx)
                if visible_count != constraint:
                    return False

        # 验证左侧约束
        for row_idx, constraint in enumerate(left):
            if constraint > 0:
                visible_count = self._count_visible_from_left(predicted_answer, row_idx)
                if visible_count != constraint:
                    return False

        # 验证右侧约束
        for row_idx, constraint in enumerate(right):
            if constraint > 0:
                visible_count = self._count_visible_from_right(predicted_answer, row_idx)
                if visible_count != constraint:
                    return False

        # 所有验证通过
        return True

    def _count_visible_from_top(self, grid: List[List[int]], col_idx: int) -> int:
        """计算从上方看某一列可见的摩天楼数量"""
        visible_count = 0
        max_height = 0
        for row_idx in range(len(grid)):
            current_height = grid[row_idx][col_idx]
            if current_height > max_height:
                visible_count += 1
                max_height = current_height
        return visible_count

    def _count_visible_from_bottom(self, grid: List[List[int]], col_idx: int) -> int:
        """计算从下方看某一列可见的摩天楼数量"""
        visible_count = 0
        max_height = 0
        for row_idx in range(len(grid) - 1, -1, -1):
            current_height = grid[row_idx][col_idx]
            if current_height > max_height:
                visible_count += 1
                max_height = current_height
        return visible_count

    def _count_visible_from_left(self, grid: List[List[int]], row_idx: int) -> int:
        """计算从左侧看某一行可见的摩天楼数量"""
        visible_count = 0
        max_height = 0
        for col_idx in range(len(grid[row_idx])):
            current_height = grid[row_idx][col_idx]
            if current_height > max_height:
                visible_count += 1
                max_height = current_height
        return visible_count

    def _count_visible_from_right(self, grid: List[List[int]], row_idx: int) -> int:
        """计算从右侧看某一行可见的摩天楼数量"""
        visible_count = 0
        max_height = 0
        for col_idx in range(len(grid[row_idx]) - 1, -1, -1):
            current_height = grid[row_idx][col_idx]
            if current_height > max_height:
                visible_count += 1
                max_height = current_height
        return visible_count
