from .exact_str_match import ExactStrMatch


class ExactStrMatchCaseInsensitive:
    """Case-insensitive exact string matching."""

    @staticmethod
    def match(response, correct_answer) -> int:
        """Case-insensitive exact match between targets and responses."""
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return ExactStrMatch.match(response.lower(), correct_answer.lower())
