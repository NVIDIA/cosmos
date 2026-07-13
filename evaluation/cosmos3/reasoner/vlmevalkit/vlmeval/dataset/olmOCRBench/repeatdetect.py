import random
import re
import string
import time
import unittest


class RepeatDetector:
    def __init__(self, max_ngram_size: int = 10):
        self.max_ngram_size = max_ngram_size
        self.data = ""

    def add_letters(self, new_str: str):
        self.data += new_str

    def ngram_repeats(self) -> list[int]:
        result = [0] * self.max_ngram_size

        if not self.data:
            return result

        # Normalize all whitespace to single spaces
        text = re.sub(r"\s+", " ", self.data)

        # For each n-gram size
        for size in range(1, self.max_ngram_size + 1):
            if len(text) < size:
                continue

            # Get the last n-gram
            target = text[-size:]

            # Count backwards from the end to find repeats
            count = 0
            pos = len(text) - size  # Start position for previous n-gram

            while pos >= 0:
                if text[pos: pos + size] == target:
                    count += 1
                    pos -= size  # Move back by the size of the n-gram
                else:
                    break

            result[size - 1] = count

        return result


class RepeatDetectorTest(unittest.TestCase):
    def test_basicTest1(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("a")
        self.assertEqual(d.ngram_repeats(), [1, 0, 0])

    def test_basicTest2(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abab")
        self.assertEqual(d.ngram_repeats(), [1, 2, 1])

    def test_longer_sequence(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("aabaabaa")
        self.assertEqual(d.ngram_repeats(), [2, 1, 2])

    def test_no_repeats(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1])

    def test_empty_data(self):
        d = RepeatDetector(max_ngram_size=3)
        self.assertEqual(d.ngram_repeats(), [0, 0, 0])

    def test_max_ngram_greater_than_data_length(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 0, 0])

    def test_large_single_char(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("a" * 10000)
        self.assertEqual(d.ngram_repeats(), [10000, 5000, 3333, 2500, 2000])

    def test_repeating_pattern(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abcabcabcabc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 4, 1, 1])

    def test_mixed_characters(self):
        d = RepeatDetector(max_ngram_size=4)
        d.add_letters("abcdabcabcdabc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 1])

    def test_palindrome(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("racecar")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 1, 1])

    def test_repeats_not_at_end(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abcabcxyz")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1])

    def test_long_repeat_at_end(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abcabcabcabcabcabcabcabcabcabc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 10, 1, 1])

    def test_large_repeating_pattern(self):
        d = RepeatDetector(max_ngram_size=4)
        pattern = "abcd"
        repeat_count = 1000
        d.add_letters(pattern * repeat_count)
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, repeat_count])

    def test_unicode_characters(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("αβγαβγ")
        self.assertEqual(d.ngram_repeats(), [1, 1, 2])

    def test_random_data(self):
        random.seed(42)
        d = RepeatDetector(max_ngram_size=5)
        data = "".join(random.choices(string.ascii_letters, k=10000))
        d.add_letters(data)
        counts = d.ngram_repeats()
        for count in counts:
            self.assertTrue(0 <= count <= len(data))

    def test_special_characters(self):
        d = RepeatDetector(max_ngram_size=4)
        d.add_letters("@@##@@##")
        self.assertEqual(d.ngram_repeats(), [2, 1, 1, 2])

    def test_incremental_addition(self):
        d = RepeatDetector(max_ngram_size=3)
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1])
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 2])
        d.add_letters("abc")
        self.assertEqual(d.ngram_repeats(), [1, 1, 3])

    def test_long_non_repeating_sequence(self):
        d = RepeatDetector(max_ngram_size=5)
        d.add_letters("abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(d.ngram_repeats(), [1, 1, 1, 1, 1])

    def test_alternating_characters(self):
        d = RepeatDetector(max_ngram_size=4)
        d.add_letters("ababababab")
        self.assertEqual(d.ngram_repeats(), [1, 5, 1, 2])


class BenchmarkRepeatDetect(unittest.TestCase):
    def testLargeRandom(self):
        all_data = []

        for iter in range(1000):
            all_data.append("".join(random.choices("a", k=10000)))

        start = time.perf_counter()

        for data in all_data:
            d = RepeatDetector(max_ngram_size=20)
            d.add_letters(data)
            print(d.ngram_repeats())

        end = time.perf_counter()

        print(f"testLargeRandom took {end-start:0.0001f} seconds")


if __name__ == "__main__":
    unittest.main()
