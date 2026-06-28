from numbers import Number

from .common.conversions import str_to_list


class SequenceEquality:
    """Determines how much of the first part of the list
    was predicted correctly.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        if not isinstance(responses, str):
            responses = str(responses)
        responses = str_to_list(responses)
        targets = str_to_list(targets)
        return 1 if responses == targets else 0


class SequenceEqualityCaseInsensitive:
    """Determines how much of the first part of the list
    was predicted correctly.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        if not isinstance(responses, str):
            responses = str(responses)
        responses = str_to_list(responses)
        targets = str_to_list(targets)

        responses = [
            item.lower() if isinstance(item, str) else str(item) for item in responses
        ]
        targets = [item.lower() for item in targets]
        return 1 if responses == targets else 0


class SequenceAccuracyCaseInsensitive:
    """Determines how much of the first part of the list
    was predicted correctly.
    """

    @classmethod
    def match(cls, responses, targets) -> int:
        """Exact match between targets and responses."""
        responses = str_to_list(responses)
        targets = str_to_list(targets)
        if len(targets) != len(responses):
            return 0
        correct = 0
        for res, tgt in zip(responses, targets):
            if isinstance(tgt, str):
                if res.lower() == tgt.lower():
                    correct += 1
            elif isinstance(tgt, Number) and isinstance(res, Number):
                if res == tgt:
                    correct += 1
            else:
                pass
        return correct / len(targets)
