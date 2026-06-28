from .common.conversions import cast_to_dict
from .simple_str_match import ExactStrMatch


class DictEquality:
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

        return 1 if responses == targets else 0


class DictPrecision:

    @classmethod
    def match(cls, responses, targets) -> float:
        """Return the aggregated Jaccard index between targets and responses."""
        responses = cast_to_dict(responses)
        targets = cast_to_dict(targets)

        if not isinstance(responses, dict):
            return 0

        if len(responses) == 0:
            return 0

        matched = 0
        for key, val in responses.items():
            if key in targets:
                if ExactStrMatch.match(val, targets[key]):
                    matched += 1

        return matched / len(responses)
