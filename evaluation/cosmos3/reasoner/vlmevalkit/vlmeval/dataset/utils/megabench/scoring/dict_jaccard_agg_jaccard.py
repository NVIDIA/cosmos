from .common.conversions import cast_to_dict
from .jaccard import Jaccard


class DictJaccardAggJaccard:
    """Calculates the Jaccard index, dividing by the union of the predictions.

    1. Calculates the Jaccard index for all sets with the same key,
    if it appears in either pred or targets
    2. Calculates the total, then divides by the size of the union
    """

    @classmethod
    def match(cls, responses, targets) -> float:
        """Return the aggregated Jaccard index between targets and responses."""
        responses = cast_to_dict(responses)
        if not isinstance(responses, dict):
            return 0

        all_keys = set(responses) | set(targets)

        num_keys = 0
        total_score = 0
        for key in all_keys:
            total_score += Jaccard.match(responses.get(key, []), targets.get(key, []))
            num_keys += 1

        return total_score / num_keys
