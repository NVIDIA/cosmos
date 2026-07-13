import re

from ..parsing.common.utils import extract_code_block_content


def parse_single_letter(s):
    # Regular expression to match (A)XXXXX, A . XXXXXXX, or A.XXXXXX
    match = re.match(r"^\(?([A-Za-z])\)?(?:\s*\.\s*|\.)?(.*)", s)

    if match:
        # Extract and return the single letter
        return match.group(1)
    else:
        # Return the original string if no match is found
        return s


class ExactStrMatch:
    """Exact string matching."""

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        """Exact match between targets and responses."""
        if not isinstance(response, str):
            response = str(response)
        if not isinstance(correct_answer, str):
            correct_answer = str(correct_answer)

        if len(correct_answer) == 1 and correct_answer.isalpha() and len(response) > 1:
            # handle special case of choice letter,
            # drop the potential parenthesis
            response = parse_single_letter(response)

        return 1 if response == correct_answer else 0


class CodeResultExactStrMatch:
    """Exact string matching, with the results from a results code block."""

    @staticmethod
    def match(response: str, correct_answer: str) -> int:
        """Exact match between targets and responses."""
        correct_answer, is_code = extract_code_block_content(
            correct_answer,
            is_ascii_art=True,
            should_remove_surrounding_whitespace=False,
        )
        # assert is_code
        return ExactStrMatch.match(response, correct_answer)
