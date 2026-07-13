# flake8: noqa
import re
from typing import List

import nltk

# from dotenv import load_dotenv

# load_dotenv()

# # nltk.download("punkt")
# nltk.data.path.append(
#     os.environ["NLTK_DATA_PATH"]
# )

# HumanCheck: True


def check_whether_response_paragraph_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    actual_count = len(paragraphs)
    # print(actual_count)

    return lower_bound <= actual_count <= upper_bound

# HumanCheck: True


def check_whether_response_sentence_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    # use nltk to split the response into sentences
    sentences = nltk.sent_tokenize(response)
    actual_count = len(sentences)
    # print(actual_count)

    return lower_bound <= actual_count <= upper_bound

# HumanCheck: True


def check_whether_each_paragraph_sentence_number_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        # use nltk to split the paragraph into sentences
        sentences = nltk.sent_tokenize(paragraph)
        actual_count = len(sentences)
        # print(f"paragraph {i}: {actual_count}")
        if actual_count < lower_bound or actual_count > upper_bound:
            return False

    return True

# HumanCheck: True


def check_whether_each_paragraph_sentence_number_in_range_list(
    response: str, ranges: List[List[int]]
) -> bool:
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    if len(paragraphs) != len(ranges):
        return False

    for i, (paragraph, range_pair) in enumerate(zip(paragraphs, ranges)):
        lower_bound, upper_bound = range_pair
        sentences = nltk.sent_tokenize(paragraph)
        actual_count = len(sentences)
        # print(f"paragraph {i}: {actual_count}")
        if not (lower_bound <= actual_count <= upper_bound):
            return False

    return True

# HumanCheck: True


def check_whether_response_word_count_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    # this line is used to filter out all non-word characters
    response_clean = re.sub(r"[^\w\s.-]", "", response)
    word_list = response_clean.split()
    word_count = len(word_list)
    # print(word_count)
    return lower_bound <= word_count <= upper_bound

# HumanCheck: True


def check_whether_each_paragraph_word_count_in_range(
    response: str, lower_bound: int, upper_bound: int
) -> bool:
    # Check whether the number of words in each paragraph of the response is greater than or equal to lower_bound and less than or equal to upper_bound.
    # Here are some examples of calling this function based on constraints:
    # If the constraint requires that the number of words in each paragraph
    # should be between 50 and 80, then lower_bound = 50 and upper_bound = 80.
    def clean_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines()).strip()

    cleaned_response = clean_text(response)

    # use re to check the number of paragraphs
    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    for i, paragraph in enumerate(paragraphs):
        paragraph_clean = re.sub(r"[^\w\s.-]", "", paragraph)
        word_count = len(paragraph_clean.split())
        # print(f"paragraph {i} word count: {word_count}")
        if not (lower_bound <= word_count <= upper_bound):
            return False

    return True

# HumanCheck: True


def check_whether_whole_response_not_contain_certain_substrings(
    response: str, substrings: List[str]
) -> bool:
    # Check whether the entire response does not contain any of the specified substrings.
    # Here are some examples of calling this function based on constraints:
    # If the constraint requires that the response should not contain the
    # words "apple" and "banana", then substrings = ["apple", "banana"].
    return all(substring not in response for substring in substrings)

# HumanCheck: True


def check_whether_whole_response_not_contain_certain_substring(
    response: str, substring: str
) -> bool:
    return substring not in response

# HumanCheck: True


def check_whether_each_sentence_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    # Check whether each sentence in the response starts with the specified substring.
    # Here are some examples of calling this function based on constraints:
    # If the constraint requires that each sentence should start with
    # exclamation point, then substring = "!".
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    sentences = nltk.sent_tokenize(response)

    return all(sentence.startswith(substring) for sentence in sentences)

# HumanCheck: True


def check_whether_each_paragraph_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    cleaned_response = clean_text(response)

    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    return all(paragraph.startswith(substring) for paragraph in paragraphs)

# HumanCheck: True


def check_whether_each_paragraph_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    cleaned_response = clean_text(response)

    paragraphs = [
        p for p in re.split(
            r"\n\s*\n",
            cleaned_response) if p.strip()]

    return all(paragraph.endswith(substring) for paragraph in paragraphs)

# HumanCheck: True


def check_whether_each_sentence_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)

    sentences = nltk.sent_tokenize(response)

    return all(sentence.endswith(substring) for sentence in sentences)

# HumanCheck: True


def check_whether_whole_response_begin_with_certain_substring(
    response: str, substring: str
) -> bool:
    return response.strip().startswith(substring)

# HumanCheck: True


def check_whether_whole_response_end_with_certain_substring(
    response: str, substring: str
) -> bool:
    return response.strip().endswith(substring)

# HumanCheck: True


def check_whether_each_keyword_in_list_metioned_in_range(
        response: str,
        keywords: List[str],
        lower_bound_times: int,
        upper_bound_times: int) -> bool:
    # should notice case like "Reddit" is counted as "Redditor"
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)
    response_lower = response.lower()

    for keyword in keywords:
        # use \b to match the whole word
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, response_lower)
        if len(matches) < lower_bound_times or len(
                matches) > upper_bound_times:
            return False

    return True

# HumanCheck: True


def check_whether_total_keyword_in_list_metioned_in_range(
        response: str,
        keywords: List[str],
        lower_bound_times: int,
        upper_bound_times: int) -> bool:
    # should notice case like "Reddit" is counted as "Redditor"
    def clean_text(response: str) -> str:
        return "\n".join(line.strip()
                         for line in response.splitlines()).strip()

    response = clean_text(response)
    response_lower = response.lower()

    count = 0
    for keyword in keywords:
        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
        matches = re.findall(pattern, response_lower)
        count += len(matches)

    return lower_bound_times <= count <= upper_bound_times

# HumanCheck: True


def check_percentage_number_precision_in_response(
        response: str, precision: int) -> bool:
    # All numeric values that appear before a percentage sign (%) must be
    # rounded and retained to two decimal places.
    pattern = r'(\d+\.\d+|\d+)\s*%'  # allow numbers and % to have spaces

    matches = re.findall(pattern, response)

    for num_str in matches:
        if '.' not in num_str:
            # no decimal point, not a float number
            return False
        decimal_part = num_str.split('.')[1]
        if len(decimal_part) != precision:
            return False

    return True

# HumanCheck: True


def check_number_precision_in_response(response: str, precision: int) -> bool:
    # Regex pattern to extract numbers, including scientific notation and
    # percentages
    number_pattern = r'''
        (?<!\w)                     # Not preceded by a word character
        [+-]?                      # Optional sign
        (?:                        # Number formats:
            \d{1,3}(?:,\d{3})*(?:\.\d+)?   # e.g., 1,234.56
            | \d+\.\d+             # e.g., 123.456
            | \.\d+                # e.g., .456
            | \d+                  # e.g., 123
        )
        (?:[eE][+-]?\d+)?          # Optional scientific notation
        %?                         # Optional percentage
        (?!\w)                     # Not followed by a word character
    '''

    matches = re.finditer(number_pattern, response, flags=re.VERBOSE)

    for match in matches:
        num_str = match.group()
        clean_num = num_str.replace(',', '').rstrip('%')

        # Split out mantissa if scientific notation
        if 'e' in clean_num.lower():
            mantissa = re.split('[eE]', clean_num)[0]
        else:
            mantissa = clean_num

        # Check digits after decimal in mantissa
        if '.' in mantissa:
            decimal_part = mantissa.split('.')[-1]
            if len(decimal_part) != precision:
                return False
        else:
            if precision != 0:
                return False

    return True

# HumanCheck: True


def check_whether_has_no_arabic_number_in_response(response: str) -> bool:
    number_pattern = r"""
        (?<![.\w])                            # Ensure no preceding . or word char
        (?:                                   # Start of number pattern
            \d{1,3}(?:,\d{3})+(?:\.\d+)?%?    |  # 1,000 or 1,000.00 or 1,000%
            \d+\.\d+%?                        |  # decimals: 3.14, 0.5%
            \d+%?                             |  # integers: 100, 100%
            \d+(?:\.\d+)?(?:[eE][+-]?\d+)        # scientific: 5e-10, 5.09e-10
        )
        (?![.\w])                             # Ensure no trailing . or word char
    """
    numbers = re.findall(
        number_pattern,
        response,
        flags=re.IGNORECASE | re.VERBOSE)
    # print(numbers)
    return len(numbers) == 0

# HumanCheck: True
# def check_scientific_notation_precision_in_response(
#     response: str, significant_digits: int
# ) -> bool:
#     scientific_pattern = r"(?<!\w)(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)(?!\w)"

#     numbers = re.findall(scientific_pattern, response)

#     for number in numbers:
#         # Split into base and exponent
#         parts = re.split(r"[eE]", number.lower())
#         if len(parts) != 2:
#             continue  # Skip invalid scientific notation

#         base, exponent = parts

#         # Handle cases like "0.000" or "0"
#         if all(c == "0" for c in base.replace(".", "")):
#             base_digits = "0"  # Treat as 0 with 1 significant digit
#         else:
#             # Remove leading and trailing zeros (but keep significant zeros)
#             base_digits = base.replace(".", "").lstrip("0") or "0"

#         significant_count = len(base_digits)
#         print(f"Number: {number}, Significant digits: {significant_count}")

#         if significant_count != significant_digits:
#             return False

#     return True
