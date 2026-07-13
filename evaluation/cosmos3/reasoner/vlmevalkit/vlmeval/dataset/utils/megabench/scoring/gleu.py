from numbers import Number

import jieba
from nltk.translate.gleu_score import sentence_gleu


class GLEUChinese:
    """Compute GLEU score for Chinese text."""

    @staticmethod
    def match(response, correct_answer) -> Number:
        """Compute the BLEU scores between two strings."""
        if isinstance(response, str) and isinstance(correct_answer, str):
            reference_tokens = list(jieba.cut_for_search(response))
            translation_tokens = list(jieba.cut_for_search(correct_answer))
        else:
            return 0
        return sentence_gleu([reference_tokens], translation_tokens)
