from numbers import Number
from typing import Dict


class MinAggregation:
    """Take the minimum of all valid scores."""

    @staticmethod
    def aggregate(scores: Dict[str, Number], weights: Dict[str, Number]) -> Number:
        """Exact match between targets and responses."""
        filtered_scores = [s for s in scores.values() if s >= 0]
        if not filtered_scores:
            return -1
        return min(filtered_scores)
