from numbers import Number
from typing import Dict

import numpy as np


class MeanAggregation:
    """Take the mean of all valid scores."""

    @staticmethod
    def aggregate(scores: Dict[str, Number], weights: Dict[str, Number]) -> Number:
        """Exact match between targets and responses."""
        filtered_scores = {f: s for f, s in scores.items() if s >= 0}
        if not filtered_scores:
            return -1

        # Align the key order
        flattened_scores = []
        flattened_weights = []
        for field in filtered_scores:
            flattened_scores.append(filtered_scores[field])
            flattened_weights.append(weights[field])
        return np.average(flattened_scores, weights=flattened_weights)
