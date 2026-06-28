import re


def text_to_int(text):
    """
    Convert a string or number to an integer between 0 and 20.
    If the input is a string, it can be a digit or a word representing a number (e.g., "one", "two", etc.).
    If the input is a number, it must be an integer or a float between 0 and 20.
    """

    if isinstance(text, (int, float)):
        if isinstance(text, float) and (text != text):
            return None
        return int(text) if 0 <= text <= 20 else None

    text = str(text).lower()

    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }

    word_pattern = r"\b(" + "|".join(number_words.keys()) + r")\b"
    digit_pattern = r"\b([0-9]|1[0-9]|20)\b"

    combined_pattern = f"({digit_pattern}|{word_pattern})"

    match = re.search(combined_pattern, text)

    if match:
        found = match.group()
        if found.isdigit():
            return int(found)
        return number_words.get(found, None)

    return None


def extract_count_from_prediction(prediction):
    """
    Extract an integer count (0-20) from a prediction string or number.
    Returns None if extraction fails.
    """
    try:
        return text_to_int(prediction)
    except Exception:
        return None
