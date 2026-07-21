import re
from typing import Any, Dict


class BaseEvaluator:
    def prepare_prompt(self, question: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        raise NotImplementedError

    def evaluate(
        self, predicted_answer: Any, ground_truth: Any,
        initial_state: str, params: Dict[str, Any] = None
    ) -> bool:
        raise NotImplementedError


class WordSearchEvaluator(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        from vlmeval.dataset.utils.mmhelix.utils.constants import PROMPT_WORDSEARCH
        return PROMPT_WORDSEARCH.format(question, "")

    def extract_answer(self, model_output: str) -> Dict[str, tuple]:
        if not isinstance(model_output, str):
            return {}

        text = model_output.strip()

        # Prefer the last \boxed{...} content if present (LaTeX-style final answer)
        boxed_blocks = re.findall(r'\\boxed\{(.*?)\}', text, re.DOTALL)
        if boxed_blocks:
            text = boxed_blocks[-1].strip()
        else:
            # Fallback: Prefer the last <answer>...</answer> block if present
            answer_blocks = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
            if answer_blocks:
                text = answer_blocks[-1].strip()

        # Clean escaped characters that might appear in model output
        # Handle cases like "BEE\\ S\\ @\\ (1,2)" -> "BEE S @ (1,2)"
        text = re.sub(r'\\(.)', r'\1', text)

        # Strict format: WORD DIRECTION @ (x, y)
        # - WORD: letters only
        # - DIRECTION: one of N,S,E,W,NE,NW,SE,SW (case-insensitive)
        # - x,y: positive integers
        # Note: do not use a trailing word boundary after ')' because ')' is a non-word char,
        # which prevents matches at end-of-string. This pattern tolerates extra spaces.
        pattern = r'([A-Za-z]+)\s+(N|S|E|W|NE|NW|SE|SW)\s+@\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        word_locations: Dict[str, tuple] = {}
        for word, direction, x_str, y_str in matches:
            normalized_word = word.lower()
            normalized_direction = direction.upper()
            x = int(x_str)
            y = int(y_str)
            word_locations[normalized_word] = (normalized_direction, (x, y))

        return word_locations

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        # Normalize both predicted and ground truth to the strict dict format
        if isinstance(predicted_answer, str):
            predicted_answer = self.extract_answer(predicted_answer)
        if isinstance(ground_truth, str):
            ground_truth = self.extract_answer(ground_truth)

        if not isinstance(predicted_answer, dict) or not isinstance(ground_truth, dict):
            return False

        if not predicted_answer or not ground_truth:
            return False

        # Directions must match exactly among allowed codes; coordinates must match exactly
        # Compare keys and values strictly
        if set(predicted_answer.keys()) != set(ground_truth.keys()):
            return False

        for word in ground_truth.keys():
            gt_direction, gt_coords = ground_truth[word]
            pred_direction, pred_coords = predicted_answer.get(word, (None, None))
            if pred_direction != gt_direction or pred_coords != gt_coords:
                return False

        return True
