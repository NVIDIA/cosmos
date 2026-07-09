import re
import signal

from sympy.core.sympify import SympifyError
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError

from .common.transformations import normalize_latex
from .simple_str_match import SimpleStrMatch


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


class LatexExprEquality:
    """Determines if two LaTeX expressions are equal."""

    @classmethod
    def match(cls, responses, targets, timeout_duration=15) -> int:
        """Whether two LaTeX expressions are equal."""
        if not isinstance(responses, str) or not isinstance(targets, str):
            return 0
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)  # Set the timeout duration in seconds
        try:
            # seems that this eval can get stuck when evaluating all tasks..
            responses = normalize_latex(responses)
            targets = normalize_latex(targets)
            responses_expr = parse_latex(responses)
            targets_expr = parse_latex(targets)
            result = 1 if responses_expr.equals(targets_expr) else 0
            return result
        except (
            LaTeXParsingError,
            SympifyError,
            TypeError,
            TimeoutException,
            NotImplementedError,
        ):
            return SimpleStrMatch.match(responses, targets)
        finally:
            signal.alarm(0)  # Cancel the alarm if it completes successfully


def separate_text_and_latex(text):
    # Regular expression to match LaTeX content between $ symbols
    pattern = r"(\$[^$]*\$)"

    # Split the text based on LaTeX parts
    parts = re.split(pattern, text)

    # Separate plain text and LaTeX
    latex_content = []
    plain_text = []

    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            latex_content.append(part)
        else:
            plain_text.append(part.strip())

    return plain_text, latex_content


def join_latex(latex_exps):
    result = []
    for exp in latex_exps:
        result.append(exp[1:-1].strip().replace(",", ""))
    result = f"{' '.join(result)}"
    return result


class TextLatexExprEquality:
    """Determines if two LaTeX expressions are equal."""

    @classmethod
    def match(cls, responses, targets) -> int:
        """Whether two LaTeX expressions are equal."""
        if not isinstance(responses, str) or not isinstance(targets, str):
            return 0

        tgt_texts, tgt_latex = separate_text_and_latex(targets)
        res_texts, res_latex = separate_text_and_latex(responses)

        res_text_join = "".join(res_texts).replace(",", "")
        tgt_text_join = "".join(tgt_texts).replace(",", "")
        text_match = SimpleStrMatch.match(res_text_join, tgt_text_join)

        res_latex_join = join_latex(res_latex)
        tgt_latex_join = join_latex(tgt_latex)
        latex_match = LatexExprEquality.match(res_latex_join, tgt_latex_join)

        return 1 if text_match and latex_match else 0
