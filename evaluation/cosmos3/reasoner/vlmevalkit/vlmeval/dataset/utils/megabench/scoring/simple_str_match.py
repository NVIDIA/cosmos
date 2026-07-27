from .exact_str_match import ExactStrMatch


class SimpleStrMatch:
    """Basic string matching, without spaces or hyphens."""

    @staticmethod
    def match(response, correct_answer: str) -> int:
        """Simple string match between response and correct_answer."""
        if not isinstance(response, str):
            response = str(response)  # If it is JSON-like
        response = (
            response.replace(" ", "")
            .replace("-", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace(".", "")
            .lower()
        )
        correct_answer = (
            correct_answer.replace(" ", "")
            .replace("-", "")
            .replace("\n", "")
            .replace("\t", "")
            .replace(".", "")
            .lower()
        )

        return ExactStrMatch.match(response, correct_answer)
