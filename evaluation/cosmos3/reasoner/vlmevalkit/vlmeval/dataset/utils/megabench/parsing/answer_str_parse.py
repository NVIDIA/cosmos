from .common.parsers import parse_json
from .common.utils import (drop_additional_text, evaluate_as_string, extract_answer_content,
                           extract_code_block_content)


class AnswerStrParse:
    """Parse the response for the single answer field."""

    @classmethod
    def _parse(
        cls,
        response: str,
        *,
        is_ascii_art: bool = False,
        should_remove_surrounding_whitespace=True,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        """Try to parse a single answer."""
        if response is None:
            response = ""

        # Extract the answer content based on "Answer: ..." format
        answer_content = extract_answer_content(
            response,
            is_ascii_art=is_ascii_art,
            should_remove_surrounding_whitespace=should_remove_surrounding_whitespace,
        )

        # Extract things from the code block if response is wrapped by a code block
        answer_content, is_code = extract_code_block_content(
            answer_content,
            is_ascii_art=is_ascii_art,
            should_remove_surrounding_whitespace=should_remove_surrounding_whitespace,
        )

        if not is_code and is_single_line_ans and not is_ascii_art:
            answer_content = drop_additional_text(answer_content)

        # Check if the content is a potential dict or list.
        if answer_content.startswith("{") or answer_content.startswith("["):
            # Attempt to parse the content as JSON
            response_obj = parse_json(answer_content)
            if response_obj == {}:
                if "{}" not in answer_content:
                    return answer_content
            elif response_obj == []:
                # logger.error(
                #    f"Unexpected answer parsing error:\n{response=}\n{global_description=}\n{query_question=}\n{is_ascii_art=}"
                # )
                if "[]" not in answer_content:
                    return answer_content
            return str(response_obj) # make sure the response to the metric is always a string
        else:
            # drop the redundant string quotes
            answer_content = evaluate_as_string(answer_content)
            return answer_content

    @classmethod
    def parse(
        cls,
        response: str,
        answer_key: str,
        *,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        """Try to parse a single answer."""
        response_parsed = cls._parse(
            response,
            is_ascii_art=False,
            global_description=global_description,
            query_question=query_question,
            is_single_line_ans=is_single_line_ans,
        )
        results = {answer_key: response_parsed}
        return results


class AsciiAnswerStrParse(AnswerStrParse):
    """Parse the response for the single ASCII answer field."""

    @classmethod
    def parse(
        cls,
        response: str,
        answer_key: str,
        *,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        """Try to parse a single answer."""
        response_parsed = cls._parse(
            response,
            is_ascii_art=True,
            global_description=global_description,
            query_question=query_question,
            is_single_line_ans=is_single_line_ans,
        )
        results = {answer_key: response_parsed}
        return results


class VerbatimAnswerStrParse(AnswerStrParse):
    """Parse the response for a single answer field that should not have preceding or trailing whitespace removed."""

    @classmethod
    def parse(
        cls,
        response: str,
        answer_key: str,
        *,
        global_description: str = "",
        query_question: str = "",
        is_single_line_ans: bool = None,
    ) -> dict:
        """Try to parse a single answer."""
        response_parsed = cls._parse(
            response,
            is_ascii_art=True,
            should_remove_surrounding_whitespace=False,
            global_description=global_description,
            query_question=query_question,
            is_single_line_ans=is_single_line_ans,
        )
        results = {answer_key: response_parsed}
        return results
