import ast
import re


def extract_code_block_content(
    response,
    code_type=None,
    is_ascii_art: bool = False,
    should_remove_surrounding_whitespace=True,
):
    # If code_type is specified, construct the pattern to match that specific code block
    if code_type:
        pattern = rf"```{code_type}\s*\n*(.*?)\s*```"
    elif is_ascii_art:
        if not response.strip() or len(response) > 10000:
            # handle the special case of pure whitespace or super long empty string
            response = response.rstrip()
        if should_remove_surrounding_whitespace:
            pattern = r"```\w*(?:\s*\n+)?(.*?)\s*```"
        else:
            pattern = r"```\w*(?:\s*\n+)?(.*?)(?:\n+\s*)?```"
    else:
        # If code_type is None, match any code block
        pattern = r"```\w*\s*\n*(.*?)\s*```"

    # Search for the code block in the response
    match = re.search(pattern, response, flags=re.DOTALL)

    if match:
        # If a match is found, return the content inside the code block
        if is_ascii_art:
            return match.group(1), True
        else:
            return match.group(1).strip(), True
    else:
        # If no code block is found, return the original string
        return response, False


def extract_answer_content(
    response, is_ascii_art=False, should_remove_surrounding_whitespace=True
):
    if is_ascii_art:
        match = re.search(r"\*\*?Answer:(.*?)\*\*?|\bAnswer:(.*)", response, re.DOTALL)
    else:
        match = re.search(
            r"\*\*?Answer:\s*(.*?)\*\*?|\bAnswer:\s*(.*)", response, re.DOTALL
        )
    if match:
        # Extract the content after "Answer:"
        response = match.group(1) or match.group(
            2
        )  # Return the first capturing group or second if the first is None
        if response is None:
            response = ""
    if is_ascii_art:
        # Reduce anything that is more than one blank line to a single blank line.
        response = re.sub(r"^\s*$(\n^\s*$)+", "", response, flags=re.MULTILINE)

        if should_remove_surrounding_whitespace:
            # Remove trailing whitespace
            response = response.rstrip()
        else:
            # Remove trailing blank lines
            response = re.sub(r"(\n\s*)+$", "", response)
        # Remove leading blank lines
        response = re.sub(r"^(\s*\n)+", "", response)
    else:
        response = response.strip()

    return response


def extract_answer_at_beginning_of_line(response):
    # Regular expression to match either "Answer:" or "**Answer:**" at the beginning of a new line
    match = re.search(r"^(?:\*\*Answer:|Answer:)\s*(.+)", response, re.MULTILINE)

    if match:
        # Return the content after "Answer:" or "**Answer: **"
        return match.group(1).strip()
    else:
        # Return None if no match is found
        return response.strip()


def drop_additional_text(result):
    # Heuristic to catch multiple-choice queries. Does not use metadata.json.
    result_first_paragraph = result.split("\n\n")[0].strip()
    potential_ans_in_single_line = re.search(
        r"^(?:(?:[a-zA-Z0-9_-]+)(?:,\s*[a-zA-Z0-9_-]+)*|(?:[a-zA-Z0-9_-]+)\.|\((?:[a-zA-Z0-9_-]+)\)$)",
        result_first_paragraph,
    )

    only_return_first_paragraph = (
        potential_ans_in_single_line
        and result_first_paragraph.strip() != ""
        and not _is_multiline_answer(result)
    )

    if only_return_first_paragraph:
        return result_first_paragraph
    else:
        return result


def _is_multiline_answer(text):
    # Split the text into lines
    lines = text.splitlines()

    # Find the "Answer:" line
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line != "":
            # Check if the next line (second line after "Answer:") is blank
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                return False  # Second line is blank, single-line answer,
                # remaining parts are additional
            return True  # Second line is not blank, multi-line answer

    return False  # empty result found, treat as single-line


def evaluate_as_string(s):
    try:
        # Try to evaluate the string using ast.literal_eval
        evaluated = ast.literal_eval(s)
        # If it's a valid Python string, return it
        if isinstance(evaluated, str):
            return evaluated
        else:
            # If it's not a string, return the original input
            return s
    except (ValueError, SyntaxError):
        # If it's not valid, return the original input
        return s
    except MemoryError:
        # the result overflows, simply return an empty string
        return ""
