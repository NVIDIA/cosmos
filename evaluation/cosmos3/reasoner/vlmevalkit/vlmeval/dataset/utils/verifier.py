import os
from abc import ABC, abstractmethod

from vlmeval.smp.log import get_logger

logger = get_logger(__name__)

QUESTION_QUALITY_PROMPT_EN_NO_COT = (
    "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with "
    "the standard answers, that is, whether the candidates answered correctly. "
    "Here are some evaluation criteria: "
    "1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because "
    "the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the "
    "standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS "
    "PERFECTLY VALID. NEVER QUESTION THEM. "
    "2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES. "
    "3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some "
    "answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, "
    "please understand the question and the standard answer first, and then judge whether the candidate's answer is "
    "correct.If the standard answer does not specify a unit, but the candidate's answer includes a unit that is "
    "correct for the value given, consider it correct. "
    "4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, "
    "fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct "
    "as long as it matches the standard answer, regardless of whether the reasoning process is correct. For "
    "multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must "
    "be answered correctly and match the standard answer exactly to be deemed correct. "
    "5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the "
    "candidate's answer is consistent with the standard answer. "
    "6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive "
    "content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, "
    "like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following "
    "answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this "
    "new question as one of: "
    "A: CORRECT "
    "B: INCORRECT "
    "C: INVALID "
    "Just return the letters \"A\", \"B\", or \"C\", with no text around it. "
    "Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself "
    "if there was a mistake; we are just trying to grade the answer. "
    "<Original Question Begin>: "
    "{question} "
    "<Original Question End> "
    "<Standard Answer Begin>: "
    "{gold_answer} "
    "<Standard Answer End> "
    "<Candidate's Answer Begin>: "
    "{llm_response} "
    "<Candidate's Answer End> "
    "Judging the correctness of the candidate's answer:")

QUESTION_QUALITY_PROMPT_EN_COT = (
    "As a grading expert, your task is to determine whether the candidate's final answer matches the provided "
    "standard answer. Follow these evaluation guidelines precisely: "
    "Evaluation Protocol: "
    "1. Reference Standard: "
    "- The standard answer is definitive and always correct "
    "- The question is perfectly valid - never question them "
    "- Do not regenerate answers; only compare with the given standard "
    "2. Comparison Method: "
    "- Carefully analyze the question's requirements and the standard answer's structure "
    "* Determine whether the question expects exact matching of the entire standard answer or allows partial "
    "matching of its components. "
    "* This determination must be made based on the question's phrasing and the nature of the standard answer. "
    "- Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors) "
    "- Disregard any differences in formatting or presentation style "
    "- For mathematical expressions: calculate step by step whether the two formulas are equivalent "
    "- For multiple-choice questions: compare only the final choice and corresponding option content "
    "3. Multi-part Answers: "
    "- For questions requiring multiple responses (e.g., multi-select): "
    "- All parts must match the standard answer exactly. "
    "- Compare each sub-answer step by step. Partial matches are considered incorrect. "
    "4. Validity Check: "
    "- Reject answers that are: "
    "* Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE "
    "* Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE "
    "* Explicit refusals (e.g., directly return \"I cannot answer/provide/access ...\") → Label as REFUSAL "
    "- For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE). "
    "Grading Scale: "
    "\\boxed{{A}} - CORRECT: "
    "- Answer matches standard exactly (including equivalent expressions) "
    "- For numerical answers: consider as equivalent if values match when rounded appropriately "
    "- Semantically equivalent responses "
    "\\boxed{{B}} - INCORRECT: "
    "- Any deviation from standard answer "
    "- Partial matches for multi-part questions "
    "\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL: "
    "- Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL) "
    "Execution Steps and Output Formats: "
    "Analysis step by step: [ "
    "Thoroughly evaluate the candidate's answer including: "
    "(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a "
    "REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type. "
    "(2) Analyze the question's core requirements and the standard answer's structure, for example: "
    "- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part "
    "completeness) "
    "- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but "
    "unformatted expressions) "
    "- Required answer type, precision level, etc. "
    "(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example: "
    "- Content equivalence "
    "- Permitted variations in numerical precision "
    "- Allowed expression formats] "
    "Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL> "
    "Here is your task. "
    "<Original Question Begin> "
    "{question} "
    "<Original Question End> "
    "<Standard Answer Begin> "
    "{gold_answer} "
    "<Standard Answer End> "
    "<Candidate's Answer Begin> "
    "{llm_response} "
    "<Candidate's Answer End> "
    "Analysis step by step and Final Judgment:"
)


class BaseVerifierModel(ABC):

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def evaluate(self, question, prediction, groundtruth):
        pass


class Verifier:
    """Main Verifier Class"""

    def __init__(self, model_path, use_vllm=False, use_cot=False, **kwargs):
        if 'VERIFIER_PATH' in os.environ and os.environ['VERIFIER_PATH'] != '':
            logger.info('Environment variable VERIFIER_PATH is set. Will use it as model_path. ')
            model_path = os.environ['VERIFIER_PATH']
        else:
            raise ValueError('VERIFIER_PATH is not set. Please set it in the .env file.')
        self.model_path = model_path
        self.model = self._create_model(model_path, use_vllm=use_vllm, use_cot=use_cot, **kwargs)

    def _create_model(self, model_path, **kwargs):
        return Verifier_Model(model_path=model_path, **kwargs)

    def evaluate(self, question, prediction, groundtruth):
        return self.model.evaluate(question, prediction, groundtruth)

    @staticmethod
    def clear_model_cache():
        Verifier_Model._model_cache.clear()
        Verifier_Model._tokenizer_cache.clear()
        print("All model caches cleared.")

    @staticmethod
    def get_cache_info():
        verifier_count = len(Verifier_Model._model_cache)
        tokenizer_count = len(Verifier_Model._tokenizer_cache)

        print("Cache status:")
        print(f"  Verifier_Model cache: {verifier_count} items")
        print(f"  Tokenizer cache: {tokenizer_count} items")

        return {
            "verifier_cache_size": verifier_count,
            "tokenizer_cache_size": tokenizer_count,
        }


class Verifier_Model(BaseVerifierModel):

    _model_cache = {}
    _tokenizer_cache = {}

    def __init__(self, model_path, use_vllm=False, use_cot=False, **kwargs):
        self.model = None
        self.tokenizer = None
        self.use_vllm = use_vllm
        self.use_cot = use_cot
        self.model_path = model_path
        self.load_model(model_path=self.model_path)
        print(
            f"Using Verifier_Model with use_vllm: {self.use_vllm}, use_cot: {self.use_cot}"
        )

    def _get_cache_key(self, model_path):
        if self.use_vllm:
            return f"vllm_{model_path}_16384_4_eager_chunked_131072"
        else:
            return f"transformers_{model_path}_auto_auto"

    def load_model(self, model_path):
        if self.use_vllm:
            self.load_model_vllm(model_path)
        else:
            self.load_model_transformers(model_path)

    def load_model_vllm(self, model_path):
        cache_key = self._get_cache_key(model_path)

        if cache_key in Verifier_Model._model_cache:
            print(f"Using cached vLLM model: {cache_key}")
            self.model = Verifier_Model._model_cache[cache_key]
            self.tokenizer = Verifier_Model._tokenizer_cache[cache_key]
        else:
            from transformers import AutoTokenizer
            from vllm import LLM

            print("Loading verifier model with vLLM...")
            import os

            import torch

            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            gpu_count = torch.cuda.device_count()
            if gpu_count >= 8:
                tp_size = 8
            elif gpu_count >= 4:
                tp_size = 4
            elif gpu_count >= 2:
                tp_size = 2
            else:
                tp_size = 1

            self.model = LLM(
                model=model_path,
                max_model_len=16384,
                max_num_seqs=3,
                tensor_parallel_size=tp_size,
                enable_chunked_prefill=True,
                max_num_batched_tokens=131072,
                gpu_memory_utilization=0.9,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # 缓存模型实例
            Verifier_Model._model_cache[cache_key] = self.model
            Verifier_Model._tokenizer_cache[cache_key] = self.tokenizer
            print(f"Cached vLLM model: {cache_key}")

    def load_model_transformers(self, model_path):
        cache_key = self._get_cache_key(model_path)

        if cache_key in Verifier_Model._model_cache:
            print(f"Using cached transformers model: {cache_key}")
            cached_data = Verifier_Model._model_cache[cache_key]
            self.model = cached_data["model"]
            self.tokenizer = cached_data["tokenizer"]
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print("Loading verifier model with transformers...")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto"
            )

            Verifier_Model._model_cache[cache_key] = {
                "model": self.model,
                "tokenizer": self.tokenizer,
            }
            print(f"Cached transformers model: {cache_key}")

    def evaluate_vllm(self, question, prediction, groundtruth):
        if self.use_cot:
            prompt = QUESTION_QUALITY_PROMPT_EN_COT.format(
                question=question, gold_answer=groundtruth, llm_response=prediction
            )
        else:
            prompt = QUESTION_QUALITY_PROMPT_EN_NO_COT.format(
                question=question, gold_answer=groundtruth, llm_response=prediction
            )
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            detokenize=True,
            max_tokens=4096,
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        outputs = self.model.generate([text], sampling_params=sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
        response = self._process_judgment(generated_text)
        if "CORRECT" in response or "A" in response:
            return True
        else:
            return False

    def _process_judgment(self, judgment_str: str) -> str:
        import re

        # First try to find the exact \boxed{letter} pattern
        boxed_matches = re.findall(r"boxed{([A-C])}", judgment_str)
        if boxed_matches:
            return boxed_matches[-1]  # Return the last boxed judgment

        # Directly return the judgment if it is A, B, or C
        if judgment_str == "A":
            return "A"
        elif judgment_str == "B":
            return "B"
        elif judgment_str == "C":
            return "C"
        else:
            judgment_str = judgment_str.split("Final Judgment:")[-1]
            matches = re.findall(r"\(([A-C])\)*", judgment_str)
            if matches:
                return matches[-1]
            matches = re.findall(r"([A-C])", judgment_str)
            if matches:
                return matches[-1]
            return ""

    def evaluate_transformers(self, question, prediction, groundtruth):
        # prepare the model input
        if self.use_cot:
            prompt = QUESTION_QUALITY_PROMPT_EN_COT.format(
                question=question, gold_answer=groundtruth, llm_response=prediction
            )
        else:
            prompt = QUESTION_QUALITY_PROMPT_EN_NO_COT.format(
                question=question, gold_answer=groundtruth, llm_response=prediction
            )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # the result will begin with thinking content in <think></think> tags, followed by the actual response
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        response = self._process_judgment(response)

        # Parse the response to determine if it's correct
        response = response.strip().upper()
        if "CORRECT" in response or "A" in response:
            return True
        else:
            return False

    def evaluate(self, question, prediction, groundtruth):
        if self.use_vllm:
            return self.evaluate_vllm(question, prediction, groundtruth)
        else:
            return self.evaluate_transformers(question, prediction, groundtruth)
