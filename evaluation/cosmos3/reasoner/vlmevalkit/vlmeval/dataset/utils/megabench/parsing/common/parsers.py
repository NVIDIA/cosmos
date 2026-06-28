import ast
import json
import re
from typing import List

import regex  # Supports the non-standard ?R regex operator

from .utils import extract_answer_at_beginning_of_line, extract_code_block_content

PARSING_TIMEOUT = 0.1


def parse_json(response: str):
    """Parse the JSON object, including nested JSON strings."""

    response_ = extract_answer_at_beginning_of_line(response)

    # If it's wrapped in code block like json, drop it
    response_, _ = extract_code_block_content(response_, "json")

    # Regular expression to match JSON-like structures, including nested quotes
    json_pattern = r"(\{(?:[^{}]|(?R))*\}|\[(?:[^{}]|(?R))*\])"
    string_pattern = r'"(?:\\.|[^"\\])*"'

    # Find all potential JSON objects
    try:
        potential_jsons = regex.findall(
            json_pattern, response_, timeout=PARSING_TIMEOUT
        )
    except TimeoutError:
        if response_.startswith("["):
            return []
        return {}

    valid_jsons = []

    for potential_json in potential_jsons:
        # Replace escaped quotes with a placeholder
        potential_json = potential_json.replace('\\"', "__DOUBLE_QUOTE__")
        potential_json = potential_json.replace("\\'", "__SINGLE_QUOTE__")

        # Find all string literals
        strings = regex.findall(string_pattern, potential_json)

        # Process each string literal
        for s in strings:
            # Unescape the string content
            unescaped = (
                s[1:-1]
                .replace("__DOUBLE_QUOTE__", '"')
                .replace("__SINGLE_QUOTE__", "'")
            )
            # Try to parse it as JSON
            try:
                parsed = json.loads(unescaped)
                if isinstance(parsed, (dict, list)):
                    # If it's a valid JSON object or array, replace it in the original string
                    potential_json = potential_json.replace(s, json.dumps(parsed))
            except json.JSONDecodeError:
                pass

        # Restore escaped quotes
        potential_json = potential_json.replace("__DOUBLE_QUOTE__", '\\"')
        potential_json = potential_json.replace("__SINGLE_QUOTE__", "\\'")

        try:
            # Attempt to parse the potential JSON
            json_object = json.loads(potential_json)
            valid_jsons.append(json_object)
        except json.JSONDecodeError:
            # try to update single quote to double quote for some special failure case
            # caused by quote's type
            potential_json_ = re.sub(r"(?<!\w)\'|\'(?!\w)", '"', potential_json)
            try:
                json_object = json.loads(potential_json_)
                valid_jsons.append(json_object)
            except json.JSONDecodeError:
                # If parsing still fails, it's not a valid JSON object
                pass

        try:
            # Attempt to parse the Python structure
            valid_jsons.append(ast.literal_eval(potential_json))
            continue
        except (SyntaxError, ValueError):
            pass
        # Escape the backslashes.
        potential_json = potential_json.replace('\\"', '\\\\"')
        potential_json = potential_json.replace("\\'", "\\\\'")
        try:
            # Attempt to parse the Python structure
            valid_jsons.append(ast.literal_eval(potential_json))
        except (SyntaxError, ValueError):
            pass

    # Return the last valid JSON if any, otherwise an empty dict or list
    if valid_jsons:
        return valid_jsons[-1]
    if response_.startswith("["):
        return []
    return {}


def parse_nested_str_list(input_string):
    # Add quotation marks around the words in the string
    quoted_string = re.sub(r"(\w+)", r'"\1"', input_string)

    # Safely evaluate the string as a Python object
    try:
        python_object = ast.literal_eval(quoted_string)
        return python_object
    except (ValueError, SyntaxError) as e:
        print(f"Failed to convert string to Python object: {e}")
        return input_string


def parse_syllable_ranges(input_str: str) -> List[List[int]]:
    """Convert a bunch of syllable ranges into a list of intervals.

    Examples:
        parse_syllable_ranges('[7,10][7, 10][5,7][5,7][7,10]')
        >>> [[7, 10], [7, 10], [5, 7], [5, 7], [7, 10]]
        parse_syllable_ranges('575 575')
        >>> [[5, 5], [7, 7], [5, 5], [0, 0], [5, 5], [7, 7], [5, 5]]
        parse_syllable_ranges('[11]5')
        >>> [[11, 11], [5, 5]]
    """

    def convert_to_range(match):
        match = match.strip("[]")
        if "," in match:
            start, end = map(int, match.split(","))
            return [start, end]
        elif match == " ":
            return [0, 0]
        else:
            num = int(match)
            return [num, num]

    # Split the input string into chunks
    chunks = re.findall(r"(?:\[\d+(?:,\s*\d+)?\]|\d| )", input_str.strip())

    # Convert each chunk to a range and create the result list
    result = [convert_to_range(chunk) for chunk in chunks]

    return result
