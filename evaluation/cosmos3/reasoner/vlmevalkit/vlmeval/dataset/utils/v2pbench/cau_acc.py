import json
import re

import pandas as pd


# for mimo-vl
def remove_think_blocks(text: str) -> str:
    """
    Remove all occurrences of <think>...</think> or <think>...</think>
    (including the tags) from the input text.
    Handles multiline content and multiple blocks.
    """
    # 修复 W605: 使用原始字符串 r'' 来处理反斜杠转义
    pattern = r'<think>.*?(?:<\/think>|<\\/think>)'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]


def xlsx2json(xlsx_file, json_file):
    df = pd.read_excel(xlsx_file)
    df.to_json(json_file, orient='records')


def calu_acc_main(file_path, txt_file=None):
    # Load data
    with open(file_path, 'r', encoding='utf-8') as f_in:
        data = json.load(f_in)

    durations = [0, 240, 1800, 7200]
    dim_mapping = {
        1: "OA", 2: "HA", 3: "OD", 4: "FM", 5: "CR", 6: "PU", 7: "CI",
        9: "FT", 10: "RT", 12: "AS", 13: "SR", 14: "GC"
    }

    dim_nums = 16
    dim_list_sum = [0] * dim_nums
    dim_list_cor = [0] * dim_nums

    short_cor, short_sum = 0, 0
    medium_cor, medium_sum = 0, 0
    long_cor, long_sum = 0, 0

    f = open(txt_file, "w", encoding="utf-8") if txt_file else None

    def log(msg):
        """Print to both the console and the file"""
        print(msg)
        if f:
            f.write(msg + "\n")

    for line in data:
        dim = line["dimension"]
        dim_list_sum[dim - 1] += 1

        if line["duration"] < durations[1]:
            short_sum += 1
        elif line["duration"] < durations[2]:
            medium_sum += 1
        else:
            long_sum += 1

        if line["score"] == 1:
            dim_list_cor[dim - 1] += 1
            if line["duration"] < durations[1]:
                short_cor += 1
            elif line["duration"] < durations[2]:
                medium_cor += 1
            else:
                long_cor += 1

    for index, (dim_cor, dim_sum) in enumerate(zip(dim_list_cor, dim_list_sum)):
        if index + 1 not in [8, 11, 15, 16]:
            if dim_sum != 0:
                log(f"{dim_mapping[index + 1]}: {dim_cor / dim_sum:.3f}")
            else:
                log(f"Dimension is zero: {dim_mapping[index + 1]}")

    log("-" * 58)
    if short_sum != 0:
        log(f"Short\nCorrect: {short_cor}, Total: {short_sum}, Accuracy: {short_cor / short_sum:.3f}")
    if medium_sum != 0:
        log(f"Medium\nCorrect: {medium_cor}, Total: {medium_sum}, Accuracy: {medium_cor / medium_sum:.3f}")
    if long_sum != 0:
        log(f"Long\nCorrect: {long_cor}, Total: {long_sum}, Accuracy: {long_cor / long_sum:.3f}")

    log("-" * 58)
    cor_data = sum(dim_list_cor)
    all_data = sum(dim_list_sum)
    log(f"Total Correct: {cor_data}")
    log(f"Total Success: {all_data}")
    log(f"Accuracy:      {cor_data / all_data:.3f}")

    if f:
        f.close()
