import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = pipeline(
    "text-classification", model="microsoft/deberta-large-mnli", device=device
)


class NliEntailment:
    """NLI entailment, where the correct answer is used as the premise."""

    @staticmethod
    def match(response, correct_answer) -> int:
        """Return whether the response and correct answer agree with each other."""
        if not isinstance(response, str) or isinstance(correct_answer, str):
            return 0
        resp = pipe(f"[CLS] {correct_answer.strip()} [SEP] {response.strip()} [SEP]")
        return 1 if resp[0]["label"] == "ENTAILMENT" else 0
