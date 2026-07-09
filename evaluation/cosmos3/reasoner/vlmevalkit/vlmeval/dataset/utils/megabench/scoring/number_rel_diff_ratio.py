import ast
import math
from numbers import Number


class NumberRelDiffRatio:
    """Number relative difference ratio scoring = min(0, 1 - |pred - gt| / gt)"""

    @staticmethod
    def match(response: str | Number, correct_answer: str) -> int:
        """Return the relative difference ratio."""
        try:
            if isinstance(response, Number):
                pred = response
            else:
                pred = ast.literal_eval(response)
            if not isinstance(pred, Number):
                return 0
            gt = ast.literal_eval(correct_answer)
            return max(0, 1 - math.fabs((pred - gt) / gt))
        except (SyntaxError, ValueError):
            return 0
