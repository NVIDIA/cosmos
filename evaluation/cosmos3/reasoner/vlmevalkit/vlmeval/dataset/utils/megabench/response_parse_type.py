from enum import Enum
from functools import cached_property

from vlmeval.dataset.utils.megabench.parsing.dummy_parse import DummyParse
from .parsing.answer_str_parse import AnswerStrParse, AsciiAnswerStrParse, VerbatimAnswerStrParse
from .parsing.json_parse import JsonParse


class ResponseParseType(Enum):
    """Parse the response."""

    JSON = "json"
    ANSWER_STR = "answer_string"
    ASCII_ANSWER_STR = "ascii_answer_string"
    VERBATIM_ANSWER_STR = "verbatim_answer_string"
    DUMMY = "dummy"
    UNSUPPORTED = "unsupported"

    @cached_property
    def class_impl(self):
        if self == ResponseParseType.ANSWER_STR:
            return AnswerStrParse
        elif self == ResponseParseType.ASCII_ANSWER_STR:
            return AsciiAnswerStrParse
        elif self == ResponseParseType.VERBATIM_ANSWER_STR:
            return VerbatimAnswerStrParse
        elif self == ResponseParseType.DUMMY:
            return DummyParse
        else:
            return JsonParse

    def is_single_field_parser(self):
        return self in [
            ResponseParseType.ANSWER_STR,
            ResponseParseType.ASCII_ANSWER_STR,
            ResponseParseType.VERBATIM_ANSWER_STR,
        ]

    def parse(self, response: str, *args, **kwargs):
        """Parse the response."""
        return self.class_impl.parse(response, *args, **kwargs)

    @staticmethod
    def from_string(s):
        """Initialize the response parsing type from a string."""
        try:
            if s is None:
                return ResponseParseType("unsupported")
            return ResponseParseType(s.lower())
        except KeyError as exc:
            raise ValueError(f"Invalid metric type: {s}") from exc
