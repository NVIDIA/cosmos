import rapidfuzz


class NormalizedSimilarityDamerauLevenshtein:
    """Normalized Damerau-Levenshtein Similarity."""

    @staticmethod
    def match(response, correct_answer) -> int:
        """Normalized indel similarityuiio do between targets and responses."""
        if not isinstance(response, str) and isinstance(correct_answer, str):
            return 0
        return rapidfuzz.distance.DamerauLevenshtein.normalized_similarity(
            response, correct_answer
        )
