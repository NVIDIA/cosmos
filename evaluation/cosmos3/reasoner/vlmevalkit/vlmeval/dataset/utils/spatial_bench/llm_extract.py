import os
import re
from typing import Any, Dict, List

import pandas as pd

from vlmeval.smp.file import load
from vlmeval.smp.log import get_logger
from vlmeval.utils.mp_util import track_progress_rich
from .tools.utils import build_choices

logger = get_logger(__name__)


GENERIC_EXTRACT_JUDGE_PROMPT = (
    "You are an expert grading assistant.\n"
    "Your job has TWO tasks:\n"
    "1) From the candidate's full response, EXTRACT the final answer in a concise, normalized form.\n"
    "2) Compare this extracted answer with the STANDARD ANSWER and grade it as:\n"
    "   - A: CORRECT\n"
    "   - B: INCORRECT\n"
    "   - C: INVALID\n"
    "\n"
    "Here are the detailed evaluation criteria:\n"

    "1. ALWAYS refer to the given STANDARD ANSWER. You do NOT need to re-solve the question; the standard answer has "
    "already been provided and is always correct. Your only job is to judge whether the candidate's final answer is "
    "consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT "
    "AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.\n"

    "2. ONLY compare the FINAL ANSWER — COMPLETELY IGNORE any potential errors or issues in the REASONING PROCESS. "
    "Even if the reasoning is wrong, as long as the final answer matches the standard answer, grade it as CORRECT.\n"

    "3. Answers may be expressed in different ways (e.g., mathematical expressions, textual descriptions). As long as "
    "the meaning is the same as the standard answer, treat them as equivalent. If the standard answer does not specify "
    "a unit but the candidate's answer includes a correct unit for the given value, consider it CORRECT.\n"

    "4. Some answers may consist of multiple items, such as multiple-choice questions with multiple correct options, "
    "multi-select questions, or multi-blank fill-in-the-blank questions. Regardless of the question type, the final "
    "answer is considered CORRECT only if it matches the standard answer exactly at the level of all required items. "
    "For multi-select or multi-blank questions, ALL parts must be answered correctly and match the standard answer "
    "exactly to be deemed CORRECT.\n"

    "5. If the candidate's answer is wrapped in LaTeX-style markers like \\boxed{{...}}, IGNORE the \\boxed and only "
    "use the inner content as the candidate's final answer when comparing with the standard answer.\n"

    "6. If the candidate's answer is INVALID — for example, incomplete (cut off mid-response), containing a large "
    "amount of abnormal repetitive content, clearly irrelevant to the question, or explicitly refusing to answer due "
    "to ethical concerns, lack of information, or other external factors — then you MUST grade it as C: INVALID.\n"

    "7. This instruction applies to all problem types, including single-choice MCQ, multi-select MCQ, numeric "
    "problems, short-answer questions, and general VQA-style questions. In all cases, only the FINAL ANSWER and its "
    "consistency with the standard answer matter.\n"

    "8. The question or options may contain image placeholders such as '<image>', '<image id=1>', or similar tokens. "
    "You CANNOT see these images. Treat these placeholders as unknown content and DO NOT hallucinate or infer any "
    "specific visual details from them. If the standard answer or candidate's answer refers to an option associated "
    "with an image (e.g., 'choose A'), judge correctness only based on the stated answer, not by imagining the image.\n"

    "\n"
    "IMPORTANT – OUTPUT FORMAT:\n"
    "• You MUST return EXACTLY ONE line in the following format:\n"
    "  <GRADE>\\t<EXTRACTED_ANSWER>\n"
    "  where <GRADE> is one of A, B, or C.\n"
    "• <EXTRACTED_ANSWER> should be the final answer you extracted from the candidate's response, in a normalized, "
    "concise form (e.g., a number, a letter option, or a short phrase).\n"
    "• If you cannot extract any meaningful answer, or the response is INVALID, output:\n"
    "  C\\tN/A\n"
    "• Do NOT add any extra text, explanation, or additional lines.\n"
    "\n"
    "Now, judge the following question.\n"
    "<Original Question Begin>\n"
    "{question}\n"
    "{options_block}"
    "<Original Question End>\n"
    "<Standard Answer Begin>\n"
    "{gold_answer}\n"
    "<Standard Answer End>\n"
    "<Candidate's Answer Begin>\n"
    "{llm_response}\n"
    "<Candidate's Answer End>\n"
    "Your output:"
)


def build_option_str(option_dict):
    s = ''
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def call_llm_extract(
    model,
    max_retry: int,
    question: str,
    prediction: str,
    gold_answer: str,
    options_block: str = ''
):
    """
    Generic LLM call + parsing helper.

    Returns:
        (grade, extracted_answer)
          - grade: 'A' / 'B' / 'C'
          - extracted_answer: the final answer extracted by the LLM as a string,
            or 'N/A' if none can be extracted.
    """
    prompt = GENERIC_EXTRACT_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        llm_response=prediction,
        options_block=options_block,
    )

    for _ in range(max_retry):
        ans = model.generate(prompt).strip()
        if 'Failed to obtain answer via API' in ans:
            logger.warning('GPT API failed to answer. ')
            continue

        # Use only the first non-empty line to avoid verbose responses
        line = ans.splitlines()[0].strip()

        # Case 1. Grade + extracted
        m = re.match(r'^\s*([ABC])\b(.*)$', line)
        if m:
            grade = m.group(1)

            # Clean grade: uppercase and clamp to {A, B, C}
            grade = str(grade).strip().upper()
            if grade not in ('A', 'B', 'C'):
                grade = 'C'

            # Clean extracted answer
            rest = m.group(2)  # get the raw remainder
            rest = re.sub(r'^[\s\|,:]+', '', rest)  # strip leading whitespace + common separators
            rest = re.sub(r'^(?:\\[tnr])+', '', rest)  # turn "\t", "\n", "\r" to spaces

            extracted = rest.strip() or 'N/A'
            return grade, extracted

        # Case 2. Grade only
        m2 = re.match(r'^\s*([ABC])\s*$', ans)
        if m2:
            grade = str(m2.group(1)).strip().upper()
            if grade not in ('A', 'B', 'C'):
                grade = 'C'
            return grade, 'N/A'

        logger.warning(f'Unparsable LLM output: {ans}')

    logger.warning('LLM extract failed after max_retry, fallback to INVALID.')
    return 'C', 'N/A'


def extract_ans_by_llm(
    model,
    row: pd.Series,
    mode: str = 'mcq',
    max_retry: int = 3
):
    """
    Generic LLM-based extraction + grading entry point.

    Returns:
        (grade, extracted_answer)
        - grade in {'A', 'B', 'C'}
        - extracted_answer: the final answer string extracted by the LLM
    """
    valid_mode = ['mcq', 'vqa']
    assert mode in valid_mode, f'Extract llm func mode must be in {valid_mode}, but got {mode}!'

    question = str(row.get('question', ''))
    prediction = str(row.get('prediction', ''))
    gold_raw = row.get('answer', '')

    # Mode mcq
    if mode == 'mcq':
        # Build choices
        choices = build_choices(row)
        option_str = build_option_str(choices) if choices else ''

        # Build options block for llm to know if there are options
        options_block = ''
        if option_str:
            options_block = 'Options:\n' + option_str + '\n'
        else:
            options_block = ''

        # Standard answer: prefer "letter + text" form if possible
        answer_letter = str(gold_raw).strip().upper()
        if choices and answer_letter in choices:
            gold_answer = f'{answer_letter}. {choices[answer_letter]}'
        else:
            # Fallback: use raw answer field
            gold_answer = str(gold_raw)

    # Mode vqa
    else:
        options_block = ''
        gold_answer = str(gold_raw)

    grade, extracted = call_llm_extract(
        model=model,
        max_retry=max_retry,
        question=question,
        prediction=prediction,
        gold_answer=gold_answer,
        options_block=options_block,
    )

    return grade, extracted


def parallel_llm_extract(
    df: pd.DataFrame,
    model,
    *,
    mode: str,
    max_retry: int,
    nproc: int,
    cache_file: str | None = None,
    key_col: str = 'index',
) -> tuple[list, list]:
    """
    Run LLM-based answer extraction with optional cache.

    Returns:
        grades: list of 'A' / 'B' / 'C' (or None)
        extracted_list: list of extracted answer strings (or None)
    """
    valid_mode = ['mcq', 'vqa']
    assert mode in valid_mode, f'LLM extract mode must be in {valid_mode}, but got {mode}!'

    df = df.copy()
    rows: List[Dict[str, Any]] = list(df.to_dict(orient='records'))

    def _one_sample(row: Dict[str, Any]):
        """
        Per-sample evaluation used by track_progress_rich.
        Returns (grade, extracted), where grade ∈ {'A', 'B', 'C'}.
        """
        row = pd.Series(row)
        grade, extracted = extract_ans_by_llm(
            model=model,
            row=row,
            mode=mode,
            max_retry=max_retry,
        )
        return grade, extracted

    # ===== case 1: no cache, plain parallel run =====
    if not cache_file:
        tasks = [dict(row=r) for r in rows]
        results = track_progress_rich(
            func=_one_sample,
            tasks=tasks,
            nproc=nproc,
        )
        grades = [g for g, _ in results]
        extracted_list = [e for _, e in results]
        return grades, extracted_list

    # ===== case 2: with cache, resume by key_col =====
    # cache format: {key: (grade, extracted)}
    cache: dict = {}
    if os.path.exists(cache_file):
        try:
            cache = load(cache_file)
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}

    grades: list = [None] * len(rows)
    extracted_list: list = [None] * len(rows)

    tasks: list[dict] = []
    keys: list = []
    task_pos: list[int] = []

    for i, row in enumerate(rows):
        key = row.get(key_col, None)
        if key is not None and key in cache:
            val = cache[key]
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                g, ex = val[0], val[1]
            else:
                g, ex = None, None
            grades[i] = g
            extracted_list[i] = ex
        else:
            tasks.append(dict(row=row))
            keys.append(key)
            task_pos.append(i)

    if tasks:
        results = track_progress_rich(
            func=_one_sample,
            tasks=tasks,
            nproc=nproc,
            save=cache_file,
            keys=keys,
        )
        for pos, (g, ex) in zip(task_pos, results):
            grades[pos] = g
            extracted_list[pos] = ex

    return grades, extracted_list
