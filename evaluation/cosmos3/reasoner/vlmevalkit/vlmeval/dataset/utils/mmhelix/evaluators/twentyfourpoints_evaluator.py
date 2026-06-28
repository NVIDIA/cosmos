import re
from typing import Any, Dict, List


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(self, predicted_answer: Any, ground_truth: Any, params: Dict[str, Any]) -> bool:
        raise NotImplementedError


class TwentyFourPointsEvaluator(BaseEvaluator):
    """
    评估24点游戏解答的评估器
    验证模型输出的表达式是否：
    1. 正确使用了所有给定的数字（每个数字恰好使用一次）
    2. 计算结果是否等于24
    """

    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        """准备发送给模型的提示词"""
        prompt = (
            "Use these numbers exactly once, and combine them with +, -, ×, ÷, and parentheses to make 24.\n"
            "Please provide your answer as an expression that includes only numbers, operators, and parentheses.\n"
            "Example answer format: (9 - 3) × 8 ÷ 2."
        )
        return prompt

    def extract_answer(self, model_output: str) -> str:
        """从模型输出中提取表达式答案，优先提取最终答案"""
        if isinstance(model_output, dict) and "text" in model_output:
            model_output = model_output["text"]

        # 先预处理，将可能的LaTeX符号表示修复
        # 处理 \times 被解释为制表符的情况
        processed_output = model_output.replace('\times', r'\times')
        processed_output = processed_output.replace('\\div', r'\div')
        processed_output = processed_output.replace('\\cdot', r'\cdot')

        # 查找包含LaTeX符号的完整表达式
        # 匹配包含LaTeX符号的表达式，包括完整的符号和残余符号
        # 使用更宽松的匹配来获取完整表达式
        latex_patterns = [
            # 匹配包含完整\times或残余imes的表达式
            r'[\(\)\d\s\+\-×÷\*/\\timesa-z]*(?:\\times|imes)[\(\)\d\s\+\-×÷\*/\\timesa-z]*',
            # 匹配包含完整\div或单独div的表达式
            r'[\(\)\d\s\+\-×÷\*/\\diva-z]*(?:\\div|div)[\(\)\d\s\+\-×÷\*/\\diva-z]*',
            # 匹配包含完整\cdot或残余cdot的表达式
            r'[\(\)\d\s\+\-×÷\*/\\cdota-z]*(?:\\cdot|cdot)[\(\)\d\s\+\-×÷\*/\\cdota-z]*'
        ]

        for pattern in latex_patterns:
            matches = re.findall(pattern, model_output)
            if matches:
                # 找到最长的匹配项
                longest_match = max(matches, key=len)
                # 检查是否包含足够的数字
                numbers_in_match = re.findall(r'\d+', longest_match)
                if len(numbers_in_match) >= 3:  # 至少3个数字才可能是完整的24点表达式
                    return longest_match.strip()

        # 如果上面没有找到，尝试更通用的方法：查找整个输入中的完整表达式
        # 如果输入本身看起来就是一个表达式，直接使用
        if (re.search(r'\d+', model_output)
                and any(keyword in model_output for keyword in [
                    'imes', 'div', 'cdot', '+', '-', '*', '/', '×', '÷', '\\times', '\\div', '\\cdot'])):
            return model_output.strip()

        # 定义可接受的字符模式（不包含量词）
        # 包括：数字、括号、空格、各种运算符（标准符号、Unicode符号、反斜杠、字母）
        expression_chars = r'[\(\)\d\s\+\-×÷\*/\\a-zA-Z]'

        # 1. 查找显式标记的最终答案
        final_answer_patterns = [
            rf'(?:final answer|answer is|the answer is)[^\n]*?[=:]?\s*({expression_chars}+)',
            rf'(?:so|thus|hence)[^\n]*?[=:]?\s*({expression_chars}+)\s*=\s*24',
            rf'That\'s it!\s*(?:The)?\s*(?:answer|expression)\s*(?:is)?\s*:?\s*({expression_chars}+)'
        ]

        for pattern in final_answer_patterns:
            matches = re.findall(pattern, processed_output, re.IGNORECASE)
            if matches:
                return matches[-1].strip()  # 返回最后一个匹配（通常是最终答案）

        # 2. 查找等于24的表达式（优先选择文本最后出现的）
        expressions_with_24 = re.findall(rf'({expression_chars}+)\s*=\s*24', processed_output)
        if expressions_with_24:
            return expressions_with_24[-1].strip()

        # 3. 提取模型提供的最后一个完整表达式
        # 先按行分割文本
        lines = processed_output.split('\n')
        for line in reversed(lines):  # 从后往前检查
            # 查找包含数字和运算符的表达式（包括LaTeX格式和文字运算符）
            # 长度至少7个字符的表达式
            expr_matches = re.findall(rf'({expression_chars}{{7,}})', line)
            if expr_matches:
                # 过滤出格式有效的表达式
                valid_expressions = [expr for expr in expr_matches
                                     if self._is_valid_expression_format(expr)]
                if valid_expressions:
                    return valid_expressions[-1].strip()

        # 4. 如果上述方法都失败，提取整个文本中最可能的表达式
        all_expressions = re.findall(rf'({expression_chars}{{7,}})', processed_output)
        valid_expressions = [expr for expr in all_expressions
                             if self._is_valid_expression_format(expr)]

        if valid_expressions:
            # 按照启发式规则排序：优先选择包含括号、长度适中的表达式
            sorted_expressions = sorted(
                valid_expressions,
                key=lambda x: (
                    '(' in x and ')' in x,  # 优先有括号的
                    len(re.findall(r'\d+', x)),  # 优先包含更多数字的
                    -abs(len(x) - 15)  # 优先长度接近15个字符的（启发式值）
                ),
                reverse=True
            )
            return sorted_expressions[0].strip()

        # 5. 最后手段：如果输入本身就是一个表达式，直接返回
        if self._is_valid_expression_format(processed_output):
            return processed_output.strip()

        # 6. 返回最后一行非空文本
        for line in reversed(lines):
            if line.strip():
                return line.strip()

        return processed_output.strip()

    def _is_valid_expression_format(self, expr: str) -> bool:
        """检查表达式格式是否有效（包含数字和运算符）"""
        # 确保表达式包含至少一个数字和一个运算符
        has_number = bool(re.search(r'\d', expr))

        # 检查是否包含运算符（包括LaTeX符号和文字运算符）
        operator_patterns = [
            r'[\+\-\×\÷\*/]',  # 标准符号
            r'\\times|\\div|\\cdot',  # 完整的LaTeX符号
            r'\bimes\b|\bdiv\b|\bcdot\b',  # LaTeX符号的残余部分
            r'\bmul\b|\btimes\b|\bplus\b|\bminus\b'  # 文字运算符
        ]
        has_operator = any(re.search(pattern, expr, re.IGNORECASE) for pattern in operator_patterns)

        # 还要检查括号是否匹配
        open_brackets = expr.count('(')
        close_brackets = expr.count(')')
        brackets_match = open_brackets == close_brackets

        return has_number and has_operator and brackets_match

    def evaluate(self, output: str, ground_truth: Dict[str, Any], params: Dict[str, Any]) -> bool:
        """
        评估预测的答案是否正确

        参数:
            predicted_answer: 模型预测的表达式
            ground_truth: 包含正确答案和输入数字的字典
            params: 其他参数

        返回:
            是否正确（布尔值）
        """
        predicted_answer = self.extract_answer(output)
        if not predicted_answer:
            return False

        # 清理和标准化表达式
        expression = self._normalize_expression(predicted_answer)

        # 获取输入数字（从params的initial_state中获取）
        input_numbers = params.get("numbers", [])
        if not input_numbers and ground_truth:
            input_numbers = ground_truth.get("initial_state", {}).get("numbers", [])

        if not input_numbers:
            return False  # 如果无法获取输入数字，则认为答案不正确

        try:
            # 检查是否使用了所有给定的数字，每个数字恰好使用一次
            used_numbers = self._extract_numbers(expression)
            if sorted(used_numbers) != sorted(input_numbers):
                return False

            # 计算表达式的值，检查是否等于24
            value = self._evaluate_expression(expression)
            return abs(value - 24) < 1e-6

        except Exception:
            # 如果解析或计算过程出错，视为不正确
            return False

    def _normalize_expression(self, expression: str) -> str:
        """标准化表达式，统一运算符符号"""
        # 替换LaTeX格式的数学符号（注意：需要处理字面的反斜杠字符串）
        expression = expression.replace('\\times', '*')
        expression = expression.replace('\\div', '/')
        expression = expression.replace('\\cdot', '*')

        # 处理可能的转义序列残余
        expression = expression.replace('\times', '*')  # 制表符+imes -> *
        expression = expression.replace('imes', '*')     # 单独的imes -> *

        # 替换Unicode乘除符号为Python可以计算的符号
        expression = expression.replace('×', '*').replace('÷', '/')

        # 替换文字形式的运算符
        expression = re.sub(r'\bdiv\b', '/', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\bmul\b', '*', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\btimes\b', '*', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\bplus\b', '+', expression, flags=re.IGNORECASE)
        expression = re.sub(r'\bminus\b', '-', expression, flags=re.IGNORECASE)

        # 移除空格
        expression = expression.replace(' ', '')

        # 移除可能包含的"="和之后的内容
        expression = re.sub(r'=.*$', '', expression)

        return expression

    def _extract_numbers(self, expression: str) -> List[int]:
        """从表达式中提取所有使用的数字"""
        return [int(num) for num in re.findall(r'\d+', expression)]

    def _evaluate_expression(self, expression: str) -> float:
        """
        计算表达式的值

        注意：使用eval函数存在安全风险，但在这个受控的评估环境中是可以接受的
        """
        # 检查表达式中是否只包含允许的字符
        if not re.match(r'^[\d\+\-\*/\(\)\.]+$', expression):
            raise ValueError("Expression contains invalid characters")

        # 计算表达式值
        return eval(expression)
