from .common.conversions import cast_to_set, str_to_set


def _convert_to_hashable(item):
    """将不可哈希的类型转换为可哈希类型"""
    if isinstance(item, (list, tuple)):
        return tuple(item)  # 将列表转换为元组
    return item


class SetEquality:
    """Determines whether two sets are equal."""

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        if isinstance(responses, (list, tuple)):
            responses = {_convert_to_hashable(item) for item in responses}
        if isinstance(targets, (list, tuple)):
            targets = {_convert_to_hashable(item) for item in targets}
        return 1 if responses == targets else 0


class SetEqualityCaseInsensitive:
    """Determines whether two sets are equal, ignoring string case."""

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        try:
            responses: set[str] = {text.upper() for text in cast_to_set(responses)}
            targets: set[str] = {text.upper() for text in cast_to_set(targets)}
        except AttributeError:
            return 0
        return 1 if responses == targets else 0


class StringSetEqualityLineSplit:
    """Determines whether two sets are equal, for string inputs, separated by line breaks"""

    @classmethod
    def match(cls, responses, targets) -> int:
        if "\\n" in targets:
            targets = targets.replace("\\n", "\n")
        if "\\n" in responses:
            responses = responses.replace("\\n", "\n")
        responses_set = set(responses.split("\n"))
        targets_set = set(targets.split("\n"))
        responses_set = {
            item.lower() if isinstance(item, str) else item for item in responses_set
        }
        targets_set = {
            item.lower() if isinstance(item, str) else item for item in targets_set
        }
        return 1 if responses_set == targets_set else 0


class StringSetEqualityCommaSplit:
    """Determines whether two sets are equal, for string inputs, separated by commas
    Handles some corner cases that would fail the general SetEquality metric, like the string
    with "None", which fails the eval. Also do case-insensitive eval.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        responses_set = str_to_set(responses)
        targets_set = str_to_set(targets)
        responses_set = {
            item.lower() if isinstance(item, str) else item for item in responses_set
        }
        targets_set = {
            item.lower() if isinstance(item, str) else item for item in targets_set
        }
        return 1 if responses_set == targets_set else 0
