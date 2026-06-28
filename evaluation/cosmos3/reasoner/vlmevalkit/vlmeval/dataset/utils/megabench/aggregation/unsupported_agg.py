from numbers import Number
from typing import Dict


class UnsupportedAggregation:
    @staticmethod
    def aggregate(scores: Dict[str, Number], weights: Dict[str, Number]) -> Number:
        return -1
