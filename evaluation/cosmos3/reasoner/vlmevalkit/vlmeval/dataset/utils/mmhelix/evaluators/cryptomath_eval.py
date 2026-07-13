import ast
import json
import re
from typing import Any, Dict, Union


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class CryptoMathEvaluator(BaseEvaluator):
    """
    评估字母算术谜题(CryptoMath)解答的评估器
    验证模型输出的字母到数字映射是否：
    1. 每个字母对应唯一数字(0-9)
    2. 满足等式计算
    3. 确保首位字母不为零
    """

    def extract_answer(self, model_output: str) -> Dict[str, int]:
        """
        从模型输出中提取字母到数字的映射

        Args:
            model_output: 模型生成的字符串输出

        Returns:
            字母到数字的映射字典，如 {'A': 1, 'B': 2, ...}
        """
        # 方法1：尝试直接使用ast.literal_eval解析Python字典格式
        dict_pattern = r'\{[^{}]*\}'
        dict_matches = re.search(dict_pattern, model_output)

        if dict_matches:
            try:
                dict_str = dict_matches.group(0)
                mapping = ast.literal_eval(dict_str)
                # 确保是字典且值是整数类型
                if isinstance(mapping, dict):
                    mapping = {k: int(v) for k, v in mapping.items() if isinstance(k, str) and str(k).isalpha()}
                    return mapping
            except Exception:
                pass

        # 方法2：尝试找到JSON对象格式
        json_pattern = r'\{(?:\s*[\'\"]([A-Za-z])[\'\"]:\s*(\d+)\s*,?\s*)+\}'
        json_matches = re.search(json_pattern, model_output)

        if json_matches:
            # 尝试提取完整的JSON对象并解析
            try:
                json_str = json_matches.group(0)
                mapping = json.loads(json_str)
                # 确保值是整数类型
                mapping = {k: int(v) for k, v in mapping.items()}
                return mapping
            except Exception:
                pass

        # 方法3：查找形如 "S"=9,"E"=5,... 的格式
        bracket_pattern = r'\[\s*(?:[\'\"]([A-Za-z])[\'\"]=\"?(\d+)\"?\s*,?\s*)+\]'
        bracket_matches = re.search(bracket_pattern, model_output)

        if bracket_matches:
            mapping = {}
            for match in re.finditer(r'[\'\"]([A-Za-z])[\'\"]=\"?(\d+)\"?', model_output):
                letter, digit = match.groups()
                mapping[letter] = int(digit)
            return mapping

        # 方法4：查找形如 "A=1, B=2, ..." 的格式
        eq_pattern = r'([A-Za-z])\s*=\s*(\d+)'
        eq_matches = re.findall(eq_pattern, model_output)

        if eq_matches:
            mapping = {letter: int(digit) for letter, digit in eq_matches}
            return mapping

        # 方法5：尝试在文本中查找明确的赋值语句
        text_pattern = r'([A-Za-z])\s+is\s+(\d+)'
        text_matches = re.findall(text_pattern, model_output)

        if text_matches:
            mapping = {letter: int(digit) for letter, digit in text_matches}
            return mapping

        # 如果没有找到有效的映射，返回空字典
        return {}

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """
        准备用于解决字母算术谜题的提示词

        Args:
            question: 问题描述
            params: 包含谜题信息的参数，如等式

        Returns:
            格式化的提示词
        """
        equation = params.get("equation", "")
        if not equation:
            return question

        # 提取等式中的单词（不需要用到，只是为了文档说明）
        # words = re.findall(r'[A-Za-z]+', equation)

        prompt = f"""请解决以下字母算术谜题（也称为字谜算术或cryptarithmetic）:

    {equation}

    在这个谜题中:
    1. 每个字母代表0到9之间的一个唯一数字
    2. 没有两个字母可以代表相同的数字
    3. 等式必须在数学上成立
    4. 每个单词的第一个字母不能为0

    你的任务是找出每个字母对应的数字，使得等式成立。

    请给出字母到数字的映射，格式如: {{"A": 1, "B": 2, ...}}

    解题过程:
    1. 分析等式中的限制条件
    2. 确定每个字母可能的取值
    3. 列出你的推理步骤
    4. 提供最终的字母到数字映射
    """

        return prompt

    def evaluate(self, output: Union[str, Dict[str, int]], ground_truth: Dict[str, int],
                 params: Dict[str, Any]) -> bool:
        """
        评估模型的解答是否正确，仅基于规则验证

        Args:
            output: 模型的原始输出字符串或已提取的字母到数字映射
            ground_truth: 正确的字母到数字映射（仅用于参考，不直接比对）
            params: 其他参数，包括等式

        Returns:
            解答是否正确的布尔值
        """
        if isinstance(output, str):
            predicted_mapping = self.extract_answer(output)
        else:
            predicted_mapping = output

        # 提取等式
        equation = params.get("equation", "")
        if not equation:
            return False  # 如果没有提供等式，无法验证

        # 提取等式中的所有字母
        all_letters = set(re.findall(r'[A-Za-z]', equation))

        # 1. 检查是否包含等式中的所有字母
        if not all(letter in predicted_mapping for letter in all_letters):
            return False

        # 2. 检查每个字母对应唯一数字(0-9)
        if len(set(predicted_mapping.values())) != len(predicted_mapping):
            return False  # 有重复数字

        if any(not isinstance(digit, int) or digit < 0 or digit > 9 for digit in predicted_mapping.values()):
            return False  # 有非法数字

        # 3. 检查首位字母不为零
        words = re.findall(r'[A-Za-z]+', equation)
        leading_letters = [word[0] for word in words]

        for letter in leading_letters:
            if letter in predicted_mapping and predicted_mapping[letter] == 0:
                return False

        # 4. 验证等式是否成立
        eval_equation = equation
        for letter, digit in predicted_mapping.items():
            eval_equation = eval_equation.replace(letter, str(digit))

        try:
            left_side, right_side = eval_equation.split('=')
            if eval(left_side) != eval(right_side):
                return False
        except Exception:
            return False  # 等式计算出错

        # 所有条件均满足，返回True
        return True
