import collections
import itertools
import logging
import re
import signal
from numbers import Number

import pronouncing
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from ..parsing.common.parsers import parse_nested_str_list, parse_syllable_ranges
from .common.conversions import str_to_iterable


def custom_lemmatize(word, lemmatizer):
    """
    Custom lemmatization to handle special cases like 'puppies' -> 'puppy'.
    """
    lemma = lemmatizer.lemmatize(word, wordnet.NOUN)

    # Handle irregular plural forms manually
    if word.endswith("ies") and lemma.endswith("y"):
        lemma = lemma[:-1] + "y"
    elif word.endswith("ves") and lemma.endswith("f"):
        lemma = lemma[:-1] + "f"

    return lemma


def lemmatize_phrase(phrase, lemmatizer):
    """
    Lemmatizes a phrase (multiple words).
    """
    words = word_tokenize(phrase.lower())
    lemmatized_words = [custom_lemmatize(word, lemmatizer) for word in words]
    return " ".join(lemmatized_words)


custom_phones_for_word = {
    "'gainst": ["G EH1 N S T", "G EY1 N S T"],
    "'midst": ["M IH1 D S T"],
    "'mongst": ["M AH1 NG S T"],
    "'neath": ["N IY1 TH"],
    "beguiles": ["B IH0 G AY1 L Z"],
    "cerulean": ["S ER0 UW1 L IY0 AH0 N"],
    "doggo": ["D AO1 G OW0"],
    "downtorn": ["D AW1 N T AO2 R N"],
    "enthrall": ["EH0 N TH R AO1 L"],
    "fam'ly": ["F AE1 M L IY0"],
    "fiery": ["F AY1 ER0 IY0", "F AY1 R IY0"],
    "flits": ["F L IH1 T S"],
    "furred": ["F ER1 D"],
    "kneels": ["N IY1 L Z"],
    "o'er": ["OW1 ER0"],
    "orbs": ["AO1 R B Z"],
    "quenched": ["K W EH1 N CH D"],
    "quietude": ["K W AY1 AH0 T UW0 D"],
    "retold": ["R IY0 T OW1 L D"],
    "scurries": ["S K ER1 IY0 Z"],
    "sunbeams": ["S AH1 N B IY2 M Z"],
    "syncs": ["S IH1 NG K S"],
    "'twixt": ["T W IH1 K S T"],
}


file_logger = logging.getLogger("errorLogger")


def phones_for_word(text: str) -> list[str]:
    """A wrapper for pronouncingpy's phones_for_word to handle out-of-vocab issues."""
    text = text.replace("’", "'").lower()

    suffixes = [""]
    prefixes = [""]
    prefixes_to_remove = [""]
    if text.endswith("'s"):
        suffixes = [" Z"]
        text = text.removesuffix("'s")

    if text in custom_phones_for_word:
        return [
            pr + suffix
            for pr, suffix in itertools.product(custom_phones_for_word[text], suffixes)
        ]

    pronunciations = pronouncing.phones_for_word(text)

    # Guess pronunciation from word stem
    if not pronunciations:
        if suffixes[0] != "":  # "'s doesn't really work with the rest."
            file_logger.error(f"OOV: {text}")
            return []

        if text.endswith("ed"):
            suffixes = [" D", " T", " AH0 D", " IH0 D"]
            text = text.removesuffix("ed")
        elif text.endswith("s"):
            # Some words, like bustles, end with es but the plural suffix is s.
            if pronouncing.phones_for_word(text.removesuffix("s")):
                # On the other hand, pierces is two syllables but pierce is one.
                if text.endswith("es"):
                    suffixes = [" S", " Z", " AH0 Z", " IH0 Z"]
                    text = text.removesuffix("s")
                else:
                    suffixes = [" S", " Z"]
                    text = text.removesuffix("s")
            elif text.endswith("es"):
                suffixes = [" AH0 Z", " IH0 Z"]
                text = text.removesuffix("es")
        if text.startswith("un"):
            prefixes = ["AH0 N "]
            text = text.removeprefix("un")
        elif text.startswith("'"):
            if pronouncing.phones_for_word("a" + text.removeprefix("'")):
                prefixes_to_remove = ["AH0 "]
                text = "a" + text.removeprefix("'")
        pronunciations = pronouncing.phones_for_word(text)
    pronunciations = [
        (prefix + pr + suffix).removeprefix(prefix_to_remove)
        for prefix, pr, suffix, prefix_to_remove in itertools.product(
            prefixes, pronunciations, suffixes, prefixes_to_remove
        )
    ]

    if not pronunciations:
        file_logger.error(f"OOV: {text}")
    return pronunciations


def rhyming_part_include_unstressed(phones: str) -> str:
    """Get the "rhyming part" of a string with CMUdict phones.

    "Rhyming part" here means everything from the vowel in the
    last syllable up to the end of the word.

    Example:
        >>> import pronouncing
        >>> phones = pronouncing.phones_for_word("purple")
        >>> rhyming_part_include_unstressed(phones[0])
        'AH0 L'

    Args:
        phones: a string containing space-separated CMUdict phones

    Returns:
        a string with just the "rhyming part" of those phones
    """
    phones_list = phones.split()
    for i in range(len(phones_list) - 1, 0, -1):
        if phones_list[i][-1] in "012":
            phones = " ".join(phones_list[i:])
            break
    return re.sub(r"\d", "", phones)


def count_syllables(text: str) -> list[int]:
    """Count the number of syllables in a piece of text."""
    pronunciations = [phones_for_word(p) for p in text.split()]
    syllable_counts = []
    for pronun_possibility in itertools.product(*pronunciations):
        syllable_counts.append(
            sum([pronouncing.syllable_count(p) for p in pronun_possibility])
        )
    return syllable_counts


def find_string_occurrences_with_variations(text, search_string):
    lemmatizer = WordNetLemmatizer()

    # Lemmatize the entire search phrase
    search_lemma = lemmatize_phrase(search_string, lemmatizer)

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    occurrences = []
    total_count = 0

    # Iterate over each sentence
    for i, sentence in enumerate(sentences, 1):  # Sentence numbers start from 1
        # Lemmatize the entire sentence
        lemmatized_sentence = lemmatize_phrase(sentence, lemmatizer)

        # Count occurrences of the lemmatized search phrase in the lemmatized sentence
        count_in_sentence = lemmatized_sentence.count(search_lemma)
        if count_in_sentence > 0:
            occurrences.append((i, count_in_sentence))
            total_count += count_in_sentence

    return total_count, occurrences


def word_to_stresses(word: str) -> list[list[int]]:
    """Convert a word to a list of stresses, for each valid pronunciation."""
    pronunciations = phones_for_word(word)
    stresses = {
        tuple(int(stress) for stress in pronouncing.stresses(pronunc))
        for pronunc in pronunciations
    }
    return [list(pronunc_stresses) for pronunc_stresses in stresses]


def is_iambic_pair(stress1: int, stress2: int) -> bool:
    """Whether the pair of stresses is a valid iambic pair."""
    valid_pairs = {(2, 1), (0, 2), (0, 1), (0, 0), (1, 1), (2, 2)}
    return (stress1, stress2) in valid_pairs


def grouper_ignore_last(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF
    args = [iter(iterable)] * n
    return zip(*args)


def is_line_iambic(line: str) -> bool:
    """Determine if a line is iambic."""
    words = line.split()
    stress_options = [word_to_stresses(word) for word in words]

    def backtrack(word_index: int, syllable_index: int, prev_stress: int) -> bool:
        """Using backtracking, determine if there is a sequence of pronunciations that is in iambic pentameter."""
        if word_index == len(words):
            # At this point, syllable_index is the number of syllables
            return syllable_index % 2 == 0

        for stress_pattern in stress_options[word_index]:
            word_syllable_index = 0
            if syllable_index % 2 != 0:
                current_stress = stress_pattern[word_syllable_index]
                if not is_iambic_pair(prev_stress, current_stress):
                    continue
                word_syllable_index += 1

            word_valid_iambic_pairs = True
            for stress1, stress2 in grouper_ignore_last(
                stress_pattern[word_syllable_index:], 2
            ):
                if not is_iambic_pair(stress1, stress2):
                    word_valid_iambic_pairs = False
                    break
                word_syllable_index += 2
            if not word_valid_iambic_pairs:
                continue

            if word_syllable_index < len(stress_pattern):
                assert word_syllable_index + 1 == len(stress_pattern)
                next_stress = stress_pattern[word_syllable_index]
                if backtrack(
                    word_index + 1,
                    syllable_index + word_syllable_index + 1,
                    next_stress,
                ):
                    return True
            else:
                assert word_syllable_index == len(stress_pattern)
                if backtrack(word_index + 1, syllable_index + word_syllable_index, -1):
                    return True

        return False

    return backtrack(
        0, 0, -1
    )  # Start with -1 as prev_stress as a placeholder for the first syllable


def parse_constraints(key_string, value_string):
    key_components = key_string.strip().split("##")
    # Remove trailing numbers from each key
    key_components = [re.sub(r"\d+$", "", key) for key in key_components]
    # Extract value components by splitting on ##
    value_components = value_string.strip().split("##")
    # Clean value components by removing brackets and spaces
    value_components = [comp.strip().strip('"').strip() for comp in value_components]

    # Handle cases where we expect integers
    for i, value in enumerate(value_components):
        if value.isdigit():
            value_components[i] = int(value)

    # Combine keys and values into a dictionary
    if len(key_components) == len(value_components):
        result = {
            key.lower(): value for key, value in zip(key_components, value_components)
        }
    elif len(key_components) == 1 and len(value_components) == 1:
        result = {key_components[0].lower(): value_components[0]}
    else:
        raise ValueError("Mismatch between number of keys and values.")

    return result


def check_constraint(response, constraint, constraint_val):
    if constraint_val.strip() == "":
        # empty contraint (placeholder), directly return 1
        return 1
    elif "contain" in constraint:
        occurs_records = {}
        parsed_constraint = parse_constraints(constraint, constraint_val)
        response = response.replace("**", "")  # Remove markdown around bolded letters
        if "contain_only" in parsed_constraint:
            num_satisfied = 0
            conditions = parse_nested_str_list(parsed_constraint["contain_only"])
            for cond in conditions:
                count, occurs = 0, []
                for item in cond:  # check one condition
                    count_, occurs_ = find_string_occurrences_with_variations(
                        response, item
                    )
                    if count_ > 0:
                        count += count_
                        occurs.extend(occurs_)
                if count > 0:
                    num_satisfied += 1
                    occurs_records[tuple(cond)] = occurs
            score = 1 if num_satisfied == 1 else 0
        else:  # the vanilla "contain" constraint
            items = str_to_iterable(list, parsed_constraint["contain"])
            count, occurs = 0, []
            for item in items:
                count_, occurs_ = find_string_occurrences_with_variations(
                    response, item
                )
                if count_ > 0:
                    count += count_
                    occurs.extend(occurs_)
            if count > 0:
                occurs_records[tuple(items)] = occurs
            score = 0 if count == 0 else 1

        ## Other logics like position or repeat, only check when
        ## previous "contain" consraint passes
        if score > 0:
            occurs = list(occurs_records.values())[0]
            if "position_only" in parsed_constraint:
                pos = parsed_constraint["position_only"]
                score = 1 if len(occurs) == 1 and occurs[0][0] == pos else 0
                return score
            elif "position" in parsed_constraint:
                pos = parsed_constraint["position"]
                occurs_sent_ids = [item[0] for item in occurs]
                score = 1 if pos in occurs_sent_ids else 0

            # check occurance times
            if "times" in parsed_constraint:
                repeat_times = parsed_constraint["times"]
                total_occurs = sum([item[1] for item in occurs])
                score = 1 if total_occurs == repeat_times else 0

    elif "length" in constraint:
        try:
            len_constraint = int(constraint_val[1:])
            words = re.findall(r"\b\w+\b", response)
            if constraint_val.strip() == "":
                score = 1  ## dummy placeholder constraint, score is 1
            elif constraint_val[0] == "<":
                score = 1 if len(words) < len_constraint else 0
            elif constraint_val[0] == ">":
                score = 1 if len(words) > len_constraint else 0
            else:
                file_logger.warning(f"Unknown length info {constraint_val}")
        except ValueError:
            file_logger.warning(f"Wrong length info {constraint_val}")
            score = 0
    elif "acrostic" in constraint:
        response = response.replace("**", "")

        lines = response.strip().lower().split("\n")
        if len(lines) != len(constraint_val):
            return 0
        all_match = True
        if "acrostic_alliteration" in constraint:
            for line, letter in zip(lines, constraint_val.lower()):
                line = line.strip()
                if letter == " ":
                    if line != "":
                        all_match = False
                        break
                elif not line or not all(word[0] == letter for word in line.split(" ")):
                    all_match = False
                    break
        else:
            for line, letter in zip(lines, constraint_val.lower()):
                line = line.strip()
                if letter == " ":
                    if line != "":
                        all_match = False
                        break
                elif not line or not line[0] == letter:
                    all_match = False
                    break
        score = 1 if all_match else 0
    else:
        response = response.strip()
        response = response.replace(".", "")
        response = response.replace(",", "")
        response = response.replace("!", "")
        response = response.replace("?", "")
        response = response.replace(":", "")
        response = response.replace(";", "")
        response = response.replace('"', "")
        response = response.replace("-", " ")
        response = response.replace("—", " ")
        response = re.sub(
            " *\(\w\) *(?=\n|$)", "", response
        )  # The parenthesized letter in the rhyming scheme

        lines = response.lower().split("\n")
        match constraint:
            case "syllables":
                syllable_count_intervals = parse_syllable_ranges(constraint_val)
                if len(lines) != len(syllable_count_intervals):
                    return 0
                try:
                    all_match = all(
                        any(
                            min_count <= syll_count <= max_count
                            for syll_count in count_syllables(line)
                        )
                        for line, (min_count, max_count) in zip(
                            lines, syllable_count_intervals
                        )
                    )
                except IndexError:
                    all_match = None
                score = 1 if all_match else 0
            case "rhyming_scheme":
                # Ensure that the number of lines is the same as the number in the rhyming scheme
                if len(lines) != len(constraint_val):
                    return 0
                last_words = []
                for line in lines:
                    if line.strip():  # Check if line has non-whitespace content
                        words = line.split()
                        last_words.append(words[-1] if words else "")
                    else:
                        last_words.append("")

                # Map each rhyming scheme letter to the last word of a line
                letter_to_words = collections.defaultdict(set)
                for rhyme_letter, word in zip(constraint_val, last_words):
                    if rhyme_letter == " ":
                        if word != "":
                            return 0
                    else:
                        letter_to_words[rhyme_letter].add(word)

                # Check that 1. The words for the same letter all rhyme
                letter_to_rhyming_parts = {}
                for letter, words in letter_to_words.items():
                    rhyming_parts: list[set[str]] = [
                        {
                            rhyming_part_include_unstressed(pronunciations)
                            for pronunciations in phones_for_word(word)
                        }
                        for word in words
                    ]
                    common_rhyming_parts = set.intersection(*rhyming_parts)
                    if not common_rhyming_parts:
                        return 0
                    letter_to_rhyming_parts[letter] = common_rhyming_parts
                # Check that 2. The words for different letters do not rhyme
                for a, b in itertools.combinations(letter_to_rhyming_parts, 2):
                    # To simplify things, if there are any shared pronunciations between two different letters, we reject it
                    if letter_to_rhyming_parts[a] & letter_to_rhyming_parts[b]:
                        return 0
                score = 1
            case "poetry_meter":
                all_match = all(is_line_iambic(line) for line in lines)
                score = 1 if all_match else 0
            case _:
                file_logger.warning(f"Unknown constraint type {constraint}")
                score = 0

    return score


class ConstrainedGenerationEval:
    """
    Constrained generation metric
    """

    timeout = 10

    @classmethod
    def match(cls, response, constraints) -> Number:
        scores = []
        eval_results = {}

        def handler(signum, frame):
            raise TimeoutError()

        def check_with_timeout(constraint, constraint_val):
            # Set the signal handler and a timeout
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(cls.timeout)  # Set the timeout

            try:
                # Try to check the constraint
                score = check_constraint(response, constraint, constraint_val)
            except TimeoutError:
                print(f"Timeout reached for constraint: {constraint}")
                score = 0  # Set score to 0 if timeout occurs
            finally:
                signal.alarm(0)  # Reset the alarm

            return score

        for constraint, constraint_val in constraints.items():
            score = check_with_timeout(constraint, constraint_val)
            scores.append(score)
            eval_results[constraint] = score

        final_score = min(scores)
        eval_info = "\t".join([f"{key}: {val}" for key, val in eval_results.items()])

        return final_score, eval_info
