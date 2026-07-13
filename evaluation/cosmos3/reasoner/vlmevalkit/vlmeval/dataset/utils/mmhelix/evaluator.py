import re
from typing import Any


class BaseEvaluator:
    def prepare_prompt(self, question: str) -> str:
        raise NotImplementedError

    def extract_answer(self, model_output: str) -> Any:
        # Extract content within <answer></answer> tags
        answer_pattern = r'<answer>(.*?)</answer>'
        match = re.search(answer_pattern, model_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return model_output.strip()  # Fallback to full output if no tags found

    def evaluate(self, predicted_answer: Any, ground_truth: Any, initial_state: Any) -> bool:
        raise NotImplementedError


class SimpleStrMatch(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        return question

    def evaluate(self, predicted_answer: str, ground_truth: str, initial_state: Any) -> bool:
        clean_answer = re.sub(r'\s+', '', str(ground_truth).lower())
        clean_response = re.sub(r'\s+', '', str(predicted_answer).lower())
        return clean_answer == clean_response


class MatchFromList(BaseEvaluator):
    def prepare_prompt(self, question: str) -> str:
        return question

    def evaluate(self, predicted_answer: str, ground_truth: list, initial_state: Any) -> bool:
        """
        Check if the response matches any of the multiple possible answers
        """
        if not ground_truth:
            return False

        # Clean the response text
        clean_response = re.sub(r'\s+', '', str(predicted_answer).lower())

        # Check if it matches any of the answers
        for answer in ground_truth:
            clean_answer = re.sub(r'\s+', '', str(answer).lower())
            if clean_answer == clean_response:
                return True

        return False
