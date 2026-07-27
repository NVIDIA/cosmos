from .common.conversions import cast_to_dict
from .exact_str_match import ExactStrMatch


class DictExactStrMatchAggRecall:
    """Calculates the exact string match across the dict.

    1. Calculates the exact match for all keys in the solution
    2. Calculates the total, then divides by the size of the solution
    """

    @classmethod
    def match(cls, responses, targets) -> float:
        """Return the aggregated Jaccard index between targets and responses."""
        responses = cast_to_dict(responses)
        targets = cast_to_dict(targets)

        if not isinstance(responses, dict):
            return 0

        num_keys = 0
        total_score = 0
        for key, answer in targets.items():
            total_score += ExactStrMatch.match(responses.get(key), answer)
            num_keys += 1

        return total_score / num_keys
