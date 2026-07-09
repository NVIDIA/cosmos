"""
vlm2bench utils for eval

Input sample format: contains the following fields:
  - index        (original q_id)
  - question
  - answer       (correct answer, i.e., gt_answer)
  - category
  - prediction   (model output, i.e., model answer)

The categories of each sub-task include:
  gc-mat, gc-trk, oc-cpr, pc-cpr   --> tf pair task (the last character of the same index distinguishes positive or negative with _p or _n)
  oc-cnt, pc-cnt                  --> cnt type
  oc-grp, pc-grp                  --> grp (MCQ) type
"""  # noqa: E501

import re
from collections import defaultdict

##########################################
# 1. General Functions
##########################################


def common_doc_to_text(sample, **kwargs):
    """
    General: directly return the "question" field from the sample.
    """
    return sample.get("question", "")


def common_doc_to_target(sample, **kwargs):
    """
    General: return the "answer" field from the sample as the correct answer.
    """
    return sample.get("answer", "")


def common_process_results(results):
    """
    Since the input file fields are already index, question, answer, category, prediction,
    directly return the original results without field mapping conversion.
    """
    return results

##########################################
# 2. TF Pair Task Evaluation (suitable for gc-mat, gc-trk, oc-cpr, pc-cpr)
##########################################


def parse_tf_answer(model_answer):
    """
    Extract 'T' or 'F' from the tf type model_answer.
    Supports formats like 'T', 'F', 'True', 'False'; returns an error flag if multiple matches are found.
    """
    pattern = re.compile(r'\b(t|f|true|false)\b', re.IGNORECASE)
    matches = pattern.findall(model_answer)
    extracted = [match.upper()[0] for match in matches]
    if len(extracted) == 1:
        return extracted[0], None
    elif len(extracted) > 1:
        return None, 'multiple_answers_found'
    else:
        return None, 'no_answer_found'


def tf_pair_aggregate_accuracy(results):
    """
    Aggregate evaluation results for the tf pair task.
    Group by index, where the index format is like "pc-cpr_1_p" and "pc-cpr_1_n",
    taking the prefix (removing the last _p or _n) as the identifier for the same group.
    If all records in the group have predictions that match the answer ("T" or "F"), the group is considered correct,
    returning the ratio of correct groups to total groups.
    """
    groups = defaultdict(list)
    for item in results:
        idx = item.get("index", "")
        if "_" not in idx:
            continue
        base_id = "_".join(idx.split("_")[:-1])
        groups[base_id].append(item)
    total_groups = len(groups)
    correct_groups = 0
    for base_id, items in groups.items():
        # At least two records are required in the group
        if len(items) < 2:
            continue
        group_correct = True
        for item in items:
            gt = item.get("answer", "").strip().upper()
            pred = item.get("prediction", "").strip().upper()
            parsed, err = parse_tf_answer(pred)
            if parsed != gt:
                group_correct = False
                break
        if group_correct:
            correct_groups += 1
    return (correct_groups / total_groups) * 100 if total_groups > 0 else 0

##########################################
# 3. CNT Task Evaluation (suitable for oc-cnt, pc-cnt)
##########################################


NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
}
PENALTY_FACTOR = 10
L_MAX = 4


def words_to_num(s):
    s = s.lower().replace('-', ' ').replace('and', ' ')
    tokens = s.split()
    total = 0
    current = 0
    for token in tokens:
        if token in NUM_WORDS:
            scale = NUM_WORDS[token]
            if scale in (100, 1000):
                if current == 0:
                    current = 1
                current *= scale
                total += current
                current = 0
            else:
                current += scale
        else:
            return None
    total += current
    return total if total != 0 else None


def extract_numbers(text):
    text = text.lower()
    digit_numbers = re.findall(r'\d+', text)
    digit_numbers = [int(num) for num in digit_numbers]
    word_numbers = []
    pattern = re.compile(
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
        r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|'
        r'eighty|ninety|hundred|thousand)\b', re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        word_phrase = ' '.join(matches)
        num = words_to_num(word_phrase)
        if num is not None:
            word_numbers.append(num)
    return digit_numbers + word_numbers


def parse_model_answer(model_answer):
    numbers = extract_numbers(model_answer)
    if len(numbers) == 1:
        return numbers[0]
    else:
        return None


def cnt_aggregate_metric(results):
    """
    Aggregate evaluation results for the CNT task.
    For each sample, parse the numbers in the prediction and compare them with the answer (which should be an integer),
    calculate the score based on the error, and return the average score of all samples.
    """
    total_count = 0
    total_norm_score = 0.0
    for item in results:
        try:
            gt = int(item.get("answer", None))
        except Exception:
            gt = None
        if gt is None:
            continue
        total_count += 1
        model_ans_str = str(item.get("prediction", "")).strip()
        # Try to use the image_seq_len provided in the record; if not, default to 2
        image_seq_len = item.get("image_seq_len", 2)
        try:
            image_seq_len = int(image_seq_len)
        except Exception:
            image_seq_len = 2

        parsed = parse_model_answer(model_ans_str)
        if parsed is None:
            norm_score = 0.0
        else:
            raw_diff = abs(parsed - gt)
            if raw_diff == 0:
                norm_score = 100.0
            else:
                max_error = max(gt - 1, image_seq_len - gt)
                if max_error <= 0:
                    max_error = 1
                relative_error = raw_diff / max_error
                weight = L_MAX / image_seq_len
                penalty = weight * (relative_error ** (1.0 / PENALTY_FACTOR))
                norm_score = 100 * (1 - penalty) if penalty < 1 else 0.0
        total_norm_score += norm_score
    return total_norm_score / total_count if total_count > 0 else 0


##########################################
# 4. GRP Task Evaluation (suitable for oc-grp, pc-grp)
##########################################


def grp_clean_answer(answer):
    if ")" in answer:
        return answer.split(")")[0].strip()
    return answer.strip()


def grp_count_options(answer):
    return len(re.findall(r'\([A-Z]\)', answer))


def grp_aggregate_accuracy(results):
    """
    Aggregate evaluation results for the GRP task (MCQ).
    For each sample, if multiple options appear in the prediction, it is considered incorrect; otherwise, compare the cleaned answer letters.
    Return the accuracy.
    """  # noqa: E501
    total = 0
    correct = 0
    for item in results:
        total += 1
        model_ans = item.get("prediction", "")
        gt_ans = item.get("answer", "")
        if grp_count_options(model_ans) > 1:
            continue
        if grp_clean_answer(model_ans) == grp_clean_answer(gt_ans):
            correct += 1
    return (correct / total * 100) if total > 0 else 0
