import math
import multiprocessing
import re
import signal

from sympy.parsing.latex import parse_latex

from .simple_str_match import SimpleStrMatch


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


E = 2.718

############## Begin
# Numerical comparison from https://github.com/TIGER-AI-Lab/MAmmoTH/blob/main/math_eval/number_utils.py


def run_eval(expression, output):
    try:
        # Safely evaluate the expression
        result = eval(expression)
        output.put(result)
    except Exception as e:
        output.put(e)


def eval_with_timeout(expression, timeout=5):
    # Create a multiprocessing.Queue to receive the output
    output = multiprocessing.Queue()

    # Define and start the process
    process = multiprocessing.Process(target=run_eval, args=(expression, output))
    process.start()

    # Wait for the process to complete or timeout
    process.join(timeout)

    if process.is_alive():
        # Terminate the process
        process.terminate()
        process.join()
        return "Timeout or error during evaluation"

    # Get result from the queue
    try:
        return output.get_nowait()
    except Exception as e:
        return "Error retrieving result"


def compare_two_list(pred, gt):
    if not isinstance(pred, list):
        return False
    elif len(pred) != len(gt):
        return False
    elif any([not isinstance(x, (int, float)) for x in pred]):
        return False
    else:
        pred = sorted(pred)
        gt = sorted(gt)
        return all([compare_two_numbers(p, g) for p, g in zip(pred, gt)])


def compare_two_numbers(p, gt):
    try:
        if math.isnan(p):
            return False
        else:
            return within_eps(pred=p, gt=gt)
    except Exception:
        return False


def within_eps(pred: float, gt: float):
    eps = abs(gt) * 0.01
    if pred >= gt - eps and pred <= gt + eps:
        return True
    else:
        return False


def clean_units(pred_str: str):
    """Clean the units in the number."""

    def convert_pi_to_number(code_string):
        code_string = code_string.replace("\\pi", "π")
        # Replace \pi or π not preceded by a digit or } with 3.14
        code_string = re.sub(r"(?<![\d}])\\?π", "3.14", code_string)
        # Replace instances where π is preceded by a digit but without a multiplication symbol, e.g., "3π" -> "3*3.14"
        code_string = re.sub(r"(\d)(\\?π)", r"\1*3.14", code_string)
        # Handle cases where π is within braces or followed by a multiplication symbol
        # This replaces "{π}" with "3.14" directly and "3*π" with "3*3.14"
        code_string = re.sub(r"\{(\\?π)\}", "3.14", code_string)
        code_string = re.sub(r"\*(\\?π)", "*3.14", code_string)
        return code_string

    pred_str = convert_pi_to_number(pred_str)
    pred_str = pred_str.replace("%", "/100")
    pred_str = pred_str.replace("$", "")
    pred_str = pred_str.replace("¥", "")
    pred_str = pred_str.replace("°C", "")
    pred_str = pred_str.replace(" C", "")
    pred_str = pred_str.replace("°", "")
    return pred_str


def number_it(num):
    if isinstance(num, (int, float)):
        return num

    num = clean_units(num)
    try:
        num = str(parse_latex(num))
    except Exception:
        pass

    if floatify(num) is not None:
        return floatify(num)
    else:
        try:
            num = eval_with_timeout(num)
            if isinstance(num, list) or isinstance(num, tuple):
                return num  # return num list
            if floatify(num) is not None:
                return floatify(num)
            else:
                return None
        except Exception:
            return None


def floatify(num: str):
    try:
        num = float(num)
        if num.is_integer():
            return round(num)
        else:
            return num
    except Exception:
        return None


def remove_latex_math_brackets(latex_str):
    """
    Removes LaTeX math mode delimiters (\( ... \) and \[ ... \]) from a string
    while preserving the contents inside the delimiters.
    If no such delimiters are found, the original string is returned.
    """
    # Regex pattern for inline math \( ... \)
    inline_pattern = re.compile(r"\\\((.*?)\\\)")
    # Regex pattern for TeX inline math $...$
    tex_inline_pattern = re.compile(r"$(.*?)$")
    # Regex pattern for display math \[ ... \]
    display_pattern = re.compile(r"\\\[(.*?)\\\]")

    latex_patterns = (inline_pattern, tex_inline_pattern, display_pattern)

    if any(pattern.search(latex_str) for pattern in latex_patterns):
        # Remove inline math mode brackets
        latex_str = inline_pattern.sub(r"\1", latex_str)
        # Remove display math mode brackets
        latex_str = display_pattern.sub(r"\1", latex_str)
    return latex_str


def parse_assignment(expression):
    # match the content after "=", "≈", or "\approx"
    pattern = r"(?:=|≈|\\approx)\s*(.+)"

    match = re.search(pattern, expression)
    if match:
        # Return the content after the sign
        return match.group(1).strip()
    else:
        return expression


############## End


class GeneralSingleNumericalMatch:
    """
    Extract the results from ```\\boxed{xxxx}``` and match with the anaswer
    """

    @classmethod
    def match(cls, responses, targets) -> float:
        if not isinstance(responses, str):
            responses = str(responses)
        responses = remove_latex_math_brackets(responses)
        responses = parse_assignment(responses)
        targets = remove_latex_math_brackets(targets)
        targets = parse_assignment(targets)
        res = number_it(responses)
        tgt = number_it(targets)

        if res is not None and tgt is not None:
            if (
                isinstance(res, list)
                and isinstance(tgt, list)
                or isinstance(res, tuple)
                and isinstance(tgt, tuple)
            ):
                score = float(compare_two_list(res, tgt))
            else:
                score = float(compare_two_numbers(res, tgt))
        else:
            score = SimpleStrMatch.match(responses, targets)

        return score


class BoxedSingleNumericalMatch:
    """
    Extract the results from ```\\boxed{xxxx}``` and match with the anaswer
    """

    @staticmethod
    def parse_boxed_content(text):
        ###
        #   Pattern: r'\\boxed\{((?:[^\{\}]+|\{[^\{\}]*\})*)\}':
        #    \\boxed\{: Matches the literal \boxed{.
        #    ((?:[^\{\}]+|\{[^\{\}]*\})*): This part matches the content inside the \boxed{}.
        #    (?:...): A non-capturing group that allows us to match both non-brace content and brace-enclosed content.
        #    [^\{\}]+: Matches any content that is not an opening { or closing } brace.
        #    \{[^\{\}]*\}: Matches balanced braces containing non-nested content (e.g., {5} or {3} in the LaTeX expression \frac{5}{3}).
        ###
        pattern = r"\\boxed\{((?:[^\{\}]+|\{[^\{\}]*\})*)\}"
        match = re.search(pattern, text)
        return match.group(1) if match else text

    @classmethod
    def match(cls, responses, targets, timeout_duration=10) -> float:
        if not isinstance(responses, str):
            responses = str(responses)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)  # Set the timeout duration in seconds
        try:
            parsed_res = cls.parse_boxed_content(responses)
            targets = cls.parse_boxed_content(targets)
            score = GeneralSingleNumericalMatch.match(parsed_res, targets)
            return score
        except TimeoutException:
            return SimpleStrMatch.match(responses, targets)
        finally:
            signal.alarm(0)
