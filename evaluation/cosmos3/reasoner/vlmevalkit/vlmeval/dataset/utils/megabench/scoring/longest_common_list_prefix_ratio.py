from .common.conversions import str_to_list
from .common.metrics import longest_common_prefix


class LongestCommonListPrefixRatio:
    """Determines how much of the first part of the list
    was predicted correctly.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        responses = str_to_list(responses)
        targets = str_to_list(targets)
        return len(longest_common_prefix(responses, targets)) / len(targets)
