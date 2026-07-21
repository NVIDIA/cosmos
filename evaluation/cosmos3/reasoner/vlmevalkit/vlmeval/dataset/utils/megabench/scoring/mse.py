import ast
import math

import numpy as np

from .common.conversions import str_to_list
from .common.metrics import mse


class MSE:
    """Mean Squared Error."""

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        """Return the mean squared error."""
        try:
            return mse(ast.literal_eval(response), ast.literal_eval(correct_answer))
        except (SyntaxError, ValueError):
            return 0


class NormalizedRMSE:
    """Mean Squared Error."""

    MIN = 0.0
    MAX = 0.1

    @classmethod
    def match(cls, response: str, correct_answer: str) -> int:
        """Return the mean squared error."""
        try:
            mse_val = mse(ast.literal_eval(response), ast.literal_eval(correct_answer))
            rmse = np.clip(np.sqrt(mse_val), cls.MIN, cls.MAX)
            norm_rmse = 1 - (rmse - cls.MIN) / (cls.MAX - cls.MIN)
            return norm_rmse
        except (SyntaxError, ValueError):
            return 0


class AngleSeqFloatRMSE:
    """Whether the sequence of numbers is close enough to the real answer."""

    MIN = 0.0
    MAX = 10.0

    @classmethod
    def match(cls, responses, targets) -> float:
        """Determines whether the sequence of floats are close enough to the real answer."""
        responses = str_to_list(responses)
        targets = str_to_list(targets)

        if len(responses) != len(targets):
            return 0

        try:
            res = np.array(responses)
            tgt = np.array(targets)
            rmse = np.sqrt(mse(res, tgt)).sum() / len(targets)
        except:  # cannot obtain the rmse from the response, return 0
            return 0

        rmse = np.clip(rmse, cls.MIN, cls.MAX)
        norm_rmse = 1 - (rmse - cls.MIN) / (cls.MAX - cls.MIN)
        if math.isnan(norm_rmse):
            return 0
        return norm_rmse
