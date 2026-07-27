import logging
import re
import sys
from decimal import getcontext

import timeout_decorator

try:
    import sympy as sp
    from sympy import Eq, Pow, simplify, sympify
    from sympy.parsing.latex import parse_latex
except ImportError:
    logging.warning('sympy or antlr4 is not installed, please install it for OlympiadBench evaluation.')

FAIL_MSG = 'Failed to obtain answer via API.'


def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer] : Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer] : Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer] : null
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4]


def build_olympiad_gpt4_extract_prompt(line):
    task_description = """
I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n
""" # noqa
    prediction = str(line['prediction'])
    demo_prompt = task_description
    examples = get_gpt4_extract_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"Model response: '{prediction}'\nExtracted Answer: "
    full_prompt = f'{demo_prompt}7.\n{test_prompt}'

    return full_prompt


def build_olympiad_gpt4_score_prompt(line):
    task_description = """
Below are two answers to a math or a physics question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n
""" # noqa
    question_for_eval = line['question']
    extract = line['extract']
    answer = line['final_answer']
    demo_prompt = task_description
    examples = get_gpt4_score_ICE()
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'

    return full_prompt


def post_check_score(line, prefetch=False):
    ans = str(line['final_answer']).strip()
    response = str(line['extract']).strip()

    if response == ans:
        return response if prefetch else True
    else:
        return False


def Olympiad_auxeval_extract(model, line):
    prompt = build_olympiad_gpt4_extract_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_extract=log, extract=res)
    log += 'All 5 retries failed.\n'
    return dict(log_extract=log, extract='')


def Olympiad_auxeval_score(model, line):
    prompt = build_olympiad_gpt4_score_prompt(line)
    log = ''
    retry = 5
    if post_check_score(line, prefetch=True):
        res = post_check_score(line, prefetch=True)
        return dict(log_score='Prefetch succeed', score=True)
    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res or res.strip() not in ['0', '1']:
            log += f'Try {i}: output is {prediction}, res is {res}, failed to parse.\n'
        else:
            log += 'Succeed'
            return dict(log_score=log, score=int(res) == 1)
    log += 'All 5 retries failed.\n'
    return dict(log_score=log, score=False)


chinese_answer_type_dict = {
    'Numerical': '数值',
    'Expression': '表达式',
    'Equation': '方程',
    'Interval': '区间'
}
english_answer_type_dict = {
    'Numerical': 'a numerical value',
    'Expression': 'an expression',
    'Equation': 'an equation',
    'Interval': 'an interval'
}


def get_single_answer_type_text(answer_type, is_chinese):
    if '-' in answer_type:  # No need now
        answer_type = answer_type[:answer_type.find('-')]
    for t in ['Numerical', 'Expression', 'Equation', 'Interval']:
        if t in answer_type:
            if is_chinese:
                return chinese_answer_type_dict[t]
            else:
                return english_answer_type_dict[t]
    exit(f'Error parsing answer type {answer_type}!')


def get_answer_type_text(answer_type, is_chinese, multiple_answer):
    # 'Tuple' has various meanings in different context, such as position or values of a series of variable,
    # so it may lead to confusion to directly use 'tuple' in the prompt.
    if ('Need_human_evaluate' in answer_type) or ('Tuple' in answer_type):
        full_answer_text = ''
    else:
        if not multiple_answer:
            answer_text = get_single_answer_type_text(answer_type, is_chinese)
            if is_chinese:
                full_answer_text = f'，答案类型为{answer_text}'
            else:
                full_answer_text = f"The answer of The problem should be {answer_text}. "
        else:
            if ',' not in answer_type:  # Same answer type for all answers
                answer_text = get_single_answer_type_text(answer_type, is_chinese)
                if is_chinese:
                    full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                else:
                    full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
            else:
                answer_types = answer_type.split(',')
                answer_types = [get_single_answer_type_text(t, is_chinese) for t in answer_types]
                if len(set(answer_types)) == 1:
                    answer_text = answer_types[0]
                    if is_chinese:
                        full_answer_text = f'，题目有多个答案，答案类型均为{answer_text}'
                    else:
                        full_answer_text = f'The problem has multiple answers, each of them should be {answer_text}. '
                else:
                    if is_chinese:
                        answer_text = '、'.join(answer_types)
                        full_answer_text = f'，题目有多个答案，答案类型分别为{answer_text}'
                    else:
                        answer_text = ', '.join(answer_types)
                        full_answer_text = (
                            f'The problem has multiple answers, with the answers in order being {answer_text}. '
                        )
    return full_answer_text


def make_input(prompt, question_content):
    # diversified based on the vllm, which is not implemented temporarily
    input = prompt + '\n' + question_content
    return input


sys.set_int_max_str_digits(1000000)
# 设置decimal的精度
getcontext().prec = 50


class MathJudger:
    def __init__(self):
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8

    def split_by_comma(self, expr: str):
        in_bracket_num = 0
        splitted_expr = []
        start_idx = 0
        for i, char in enumerate(expr):
            if char == "(" or char == "[":
                in_bracket_num += 1
            elif char == ")" or char == "]":
                in_bracket_num -= 1
            elif char == "," and in_bracket_num == 0:
                splitted_expr.append(expr[start_idx:i].strip())
                start_idx = i + 1

        if start_idx < len(expr):
            splitted_expr.append(expr[start_idx:].strip())

        return splitted_expr

    def trans_plus_minus_sign(self, expr_list: list):
        new_expr_list = []
        for expr in expr_list:
            if "\\pm" in expr:
                new_expr_list.append(expr.replace("\\pm", "+"))
                new_expr_list.append(expr.replace("\\pm", "-"))
            else:
                new_expr_list.append(expr)

        return new_expr_list

    def judge(self, expression1, expression2, precision=1e-8):
        # (默认 expression1 为 Ground_Truth)
        precision = precision if isinstance(precision, list) else [precision]

        try:
            expression1, expression2 = self.preprocess(expression1, expression2)
        except Exception:
            return False
        if expression1 == expression2:
            # print("原生相等")
            return True

        # 去除字符串中的中文字符，因为上面已经判断过了类似回答为"能"或"不能"的含有中文字符的回答情况
        expression1 = re.sub(r'[\u4e00-\u9fff]+', '', expression1)
        expression2 = re.sub(r'[\u4e00-\u9fff]+', '', expression2)

        expression1 = self.split_by_comma(expression1)
        expression2 = self.split_by_comma(expression2)

        temp_list1 = self.trans_plus_minus_sign(expression1)
        temp_list2 = self.trans_plus_minus_sign(expression2)

        # 设计误差值列表
        if len(precision) <= 1:
            precision = precision * len(temp_list1)

        if len(temp_list1) != len(temp_list2):
            return False

        # 判断两个列表中的元素是否可以两两配对，并且两两相等，由此支持多个回答的比较
        idx = -1
        while len(temp_list1) != 0:
            idx = (idx + 1) % len(temp_list1)

            item1 = temp_list1[idx]
            self.precision = precision[idx]
            # print(self.precision)

            for item2 in temp_list2:
                try:
                    if self.is_equal(item1, item2):
                        temp_list1.remove(item1)
                        temp_list2.remove(item2)
                        precision.remove(self.precision)
                        break
                except Exception as err:
                    logging.warning(f'{type(err)}: {err}')
                    continue
            else:
                # If we didn't break from the inner loop, it means no match was found
                return False

        # If all elements are matched and removed, the lists can be paired
        return True

    def is_interval(self, epr):
        return epr.startswith(("(", "[")) and epr.endswith((")", "]"))

    # 在进行数值计算前，需要将sympy中的pi符号替换为pi的近似数值
    # def sympy_sub_pi(self, expression_sympy):
    #     return expression_sympy.subs(self.pi, math.pi)

    # 默认第一个表达式是 ground_truth
    @timeout_decorator.timeout(30)
    def is_equal(self, expression1, expression2):
        if expression1 == expression2 and expression1 != "" and expression2 != "":
            # print("原生等价")
            return True

        # 先判断是否是两个区间，是的话进行判断相等，不相等则返回 False
        if self.is_interval(expression1) and self.is_interval(expression2):
            try:
                if self.interval_equal(expression1, expression2):
                    # print("区间等价")
                    return True
            except Exception:
                return False

        # 再判断是否在数值上相等
        try:
            if self.numerical_equal(expression1, expression2):
                # print("数值等价")
                return True
        except Exception:
            pass

        # 再判断是否是表达式相等
        try:
            if self.expression_equal(expression1, expression2) and not ("=" in expression1 and "=" in expression2):
                # print("表达式等价")
                return True
        except Exception:
            pass

        # 再判断是否是等式相等
        try:
            if self.equation_equal(expression1, expression2):
                # print("等式等价")
                return True
        except Exception:
            pass

        return False

    # 判断两个数值在误差允许范围内是否相等
    def numerical_equal(self, expression1: str, expression2: str, include_percentage: bool = True):
        """
        (默认 expression1 为 Ground_Truth)
        函数: 判读两个数值是否在误差允许范围内相等
        步骤1: 将可能出现的百分号的情况包含进来
        步骤2: 使用 math.isclose 函数判断是否相等
        """
        reference = float(expression1)
        prediction = float(expression2)

        if include_percentage:
            gt_result = [reference / 100, reference, reference * 100]
        else:
            gt_result = [reference]

        for item in gt_result:
            # if isclose(item, prediction, abs_tol=self.precision, rel_tol=0):
            if abs(item - prediction) <= self.precision * 1.01:
                return True
        return False

    def expression_equal(self, exp1, exp2):
        """
        (默认 expression1 为 Ground_Truth)
        函数: 判断两个表达式是否在数学意义上等价
        步骤1: 提取表达式, 防止有的模型会给出"x=1"而不是"1"
        步骤2: 使用 sympy 库进行等价判断
        """

        # 只提取等号右边的表达式，一般左边是所求的量
        def extract_expression(expression):
            if "=" in expression:
                expression = expression.split("=")[1]
            return expression.strip()

        exp1 = extract_expression(exp1)
        exp2 = extract_expression(exp2)

        exp_too_long = len(exp1) > 300 or len(exp2) > 300

        # 将表达式转换为 sympy 中能够进行处理的格式
        expr1_sym = sympify(parse_latex(exp1))
        expr2_sym = sympify(parse_latex(exp2))

        if expr1_sym == expr2_sym:
            return True
        else:
            expr1_sym = self.sympy_sub_pi(expr1_sym)
            expr2_sym = self.sympy_sub_pi(expr2_sym)
            # 如果输入的表达式可以计算出具体数值的话，则将其进行数值计算的比较

            if (expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol)) or (
                    not expr1_sym.has(sp.Symbol) and expr2_sym.has(sp.Symbol)):
                return False
            elif not expr1_sym.has(sp.Symbol) and not expr2_sym.has(sp.Symbol):
                try:
                    if not (self.can_compute_power(expr1_sym) and self.can_compute_power(expr2_sym)):
                        print(
                            "These two number can not be calculated by current computer for: "
                            f"\"{str(expr1_sym)}\" and \"{str(expr2_sym)}\""
                        )
                        return False
                    if exp_too_long:
                        print(f'Expression {exp1} or {exp2} is too long to compute. ')
                        return False

                    if abs(expr1_sym.evalf() - expr2_sym.evalf()) <= self.precision * 1.01:
                        return True
                    else:
                        return False
                except Exception:
                    return False
            elif exp_too_long:
                print(f'Expression {exp1} or {exp2} is too long to compute. ')
                return False
            else:
                try:
                    simplified_expr = simplify(expr1_sym - expr2_sym)

                    num_value = simplified_expr.evalf()

                    return abs(num_value) < 1e-3
                except Exception:
                    return False

    def equation_equal(self, expression1, expression2):
        """
        (默认 expression1 为 Ground_Truth)
        函数: 判断两个方程是否在数学意义上等价
        步骤1: 将一个方程/等式化简为标准方程, 即等式的右边严格等于0, 接下来只需要判断两个等式的左边是否"等价"
        步骤2: 使用 sympy 库计算两个等式左边的商, 如果这个商或者这个商的倒数为整数, 那么数学意义上我们可以推导出这两个方程等价👌
        """

        # 将等式的右边都移到左边，并返回一个 sympy 格式的表达式
        def simplify_equation(latex_eq):
            # 分割等式的左边和右边
            lhs, rhs = latex_eq.split('=')

            # 使用 parse_latex 解析 LaTeX 表达式
            lhs_expr = parse_latex(lhs)
            rhs_expr = parse_latex(rhs)

            # 创建等式对象
            equation = Eq(lhs_expr, rhs_expr)

            # 化简等式：将等式右边移到左边
            simplified_eq = simplify(equation.lhs - equation.rhs)

            return simplified_eq

        expr1_sym = simplify_equation(expression1)
        expr2_sym = simplify_equation(expression2)

        division_result_1 = simplify(expr1_sym / expr2_sym)
        division_result_2 = simplify(expr2_sym / expr1_sym)

        # 如果两个方程转换后的式子相除为整数 且非零，则根据推导可知这两个方程等价
        if (division_result_1.is_Integer and division_result_1 != 0) or (
                division_result_2.is_Integer and division_result_2 != 0):
            return True
        else:
            return False

    def interval_equal(self, expression1, expression2):
        # 函数: 判断两个区间是否在数学意义上等价
        # 步骤1: 简化区间的表达式, 去除无关的符号比如"\left", "\right", 同时将可能出现的"x \in"删去
        # 步骤2: 对比两个区间的左右符号、中间出现的数学表达式等是否一致

        def compare_two_interval(inter1, inter2):

            # 首先比较两边的括号是否一致，一致的话再进行下一步比较
            if inter1[0] != inter2[0] or inter1[-1] != inter2[-1]:
                return False

            inter1 = inter1.strip('[]()')
            inter2 = inter2.strip('[]()')

            # 分割区间的左右部分
            items_1 = inter1.split(',')
            items_2 = inter2.split(',')

            for item_1, item_2 in zip(items_1, items_2):
                if not self.expression_equal(item_1, item_2):
                    return False
            return True

        interval1 = expression1
        interval2 = expression2

        if interval1 == interval2:
            return True
        else:
            inter_list1 = interval1.split("\\cup")
            inter_list2 = interval2.split("\\cup")

            if len(inter_list1) != len(inter_list2):
                return False
            else:
                for inter1, inter2 in zip(inter_list1, inter_list2):
                    if not compare_two_interval(inter1, inter2):
                        return False
                return True

    def preprocess(self, expression1, expression2):

        # 尝试捕获box中的内容，如果有多个则以逗号相连返回，如果一个都没有，则报错
        def extract_boxed_content(latex_str):
            # 查找所有的 \boxed{...} 结构
            boxed_matches = re.finditer(r'\\boxed{', latex_str)
            results = ""

            for match in boxed_matches:
                start_index = match.end()
                end_index = start_index
                stack = 1

                # 从 \boxed{ 之后开始搜索，直到找到对应的闭合括号
                while stack > 0 and end_index < len(latex_str):
                    if latex_str[end_index] == '{':
                        stack += 1
                    elif latex_str[end_index] == '}':
                        stack -= 1
                    end_index += 1

                if stack == 0:
                    # 提取 \boxed{} 内部的内容
                    content = latex_str[start_index:end_index - 1]
                    results += content + ","
                else:
                    # 如果括号没有正确闭合，则返回错误信息
                    raise ValueError("Mismatched braces in LaTeX string.")

            # 如果没有匹配到'\boxed{}'字符，则默认提取有内容的文字最后一行中的所有公式部分
            if results == "":
                last_line_ans = latex_str.strip().split("\n")[-1]
                dollar_pattern = r"\$(.*?)\$"
                answers = re.findall(dollar_pattern, last_line_ans)

                if answers:
                    for ans in answers:
                        results += ans + ","
                else:
                    results = latex_str

            return results

        def sepcial_symbol_replace(expression):
            if "\\in " in expression:
                expression = expression.split("\\in ")[1]

            # 进行特殊字符的替换，这些字符都不影响latex的解析，属于美观/修饰性字符
            for signal in self.special_signal_map:
                expression = expression.replace(signal, self.special_signal_map[signal])

            expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")

            pattern = r'\\(?:mathrm|mathbf)\{~?([^}]*)\}'
            expression = re.sub(pattern, r'\1', expression)

            return expression

        exp1, exp2 = extract_boxed_content(expression1), extract_boxed_content(expression2)
        exp1, exp2 = sepcial_symbol_replace(exp1), sepcial_symbol_replace(exp2)

        return exp1, exp2

    def can_compute_power(self, expr):
        """
        Check if the power expression can be computed.

        Parameters:
        expr (sympy expression): The expression to check.

        Returns:
        bool: True if the expression can be computed, False otherwise.
        """
        # Check if the expression is a power expression
        if isinstance(expr, Pow):
            # Extract the base and the exponent
            base, exp = expr.as_base_exp()

            # Check if the base and the exponent are numbers
            if base.is_number and exp.is_number:
                # Set a threshold for the maximum size of the exponent
                MAX_EXP = 1000  # This threshold can be adjusted based on the computing environment

                # Check if the exponent is greater than the threshold
                if abs(exp.evalf()) > MAX_EXP:
                    return False
                else:
                    return True
            else:
                # If the base or the exponent is not a number, we cannot compute the power
                return False
        else:
            # If the expression is not a power expression, return True as it is not the case we are checking for
            return True


def extract_answer(is_chinese, model_output, is_deepseek=False):
    # deepseekmath has special answering format
    if str(model_output) == 'nan':
        model_output = 'nan'

    if is_deepseek:
        if is_chinese:
            matches = re.findall('## 解题答案(.*)', model_output)
        else:
            matches = re.findall('The answer is: (.*)', model_output)

        # 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
        if matches:
            # 如果找到多个匹配，取最后一个
            model_answer = matches[-1].strip()
            return model_answer
        else:
            return model_output

    if is_chinese:
        matches = re.findall('所以最终答案是(.*)', model_output)
    else:
        matches = re.findall('So the final answer is (.*)', model_output)

    # 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
    if matches:
        # 如果找到多个匹配，取最后一个
        model_answer = matches[-1].strip()
        return model_answer
    else:
        return model_output


def calculate_merged_accuracy(reference_dir, text_only):
    pass
