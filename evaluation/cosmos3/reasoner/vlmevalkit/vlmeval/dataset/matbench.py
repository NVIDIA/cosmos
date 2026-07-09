import os
import re
import string

import pandas as pd
from tqdm import tqdm

from .image_base import ImageBaseDataset


def is_chinese(text):
    """判断文本是否含中文字符"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)


def normalize(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        chinese_punc = "！？｡＂＃＄％＆＇（）＊＋，－．／：；＜＝＞＠［＼］＾＿｀｛｜｝～“”‘’、。：《》【】"
        exclude = set(string.punctuation + chinese_punc)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    s = lower(s)
    s = remove_punc(s)

    if is_chinese(s):
        s = s.replace(" ", "")  # 中文一般去除所有空白
    else:
        s = remove_articles(s)
        s = white_space_fix(s)

    return s


def compute_f1(prediction, ground_truth):
    if prediction is None:
        return 0.0

    norm_pred = normalize(prediction)
    norm_gt = normalize(ground_truth)

    # 中文使用字符级，英文使用词级
    if is_chinese(norm_pred) or is_chinese(norm_gt):
        pred_tokens = list(norm_pred)
        gt_tokens = list(norm_gt)
    else:
        pred_tokens = norm_pred.split()
        gt_tokens = norm_gt.split()

    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    if prediction is None:
        return 0.0
    return int(normalize(prediction) == normalize(ground_truth))


def mat_evaluate(tsv_path, eval_file):
    # pass
    print("\nStarting evaluation Ocean-OCR Bench...")
    df = pd.read_excel(eval_file)

    print(df.iloc[300])
    if "id" in df.iloc[0].values:
        print("⚠️ 检测到首行是header，跳过该行")
        df = df.iloc[1:].reset_index(drop=True)

    results = df.to_dict(orient='records')
    print(results[0])

    eval_results = {
        "code_simple_f1": [],
        "code_simple_em": [],
        "code_hard_f1": [],
        "code_hard_em": [],
        "search_simple_f1": [],
        "search_simple_em": [],
        "search_hard_f1": [],
        "search_hard_em": [],
    }
    for item in tqdm(results):
        try:
            pred = item['prediction']
            gts = item['answer']
            split = item["split"]
            category = item["category"]
            # 若gt是str，统一转换为列表处理
            if isinstance(gts, str):
                gts = [gts]
            f1 = max([compute_f1(pred, gt) for gt in gts])
            em = max([exact_match_score(pred, gt) for gt in gts])
            if em == 1:
                f1 = 1
            keys_f1 = f"{category}_{split}_f1"
            keys_em = f"{category}_{split}_em"
            eval_results[keys_f1].append(f1)
            eval_results[keys_em].append(em)
        except Exception as e:
            print("Error:", {e})

    # 计算各项指标均值
    mean_results = {}
    for key, values in eval_results.items():
        if len(values) > 0:
            mean_results[key] = sum(values) / len(values)
        else:
            mean_results[key] = 0.0

    # 计算 code / search / overall 的平均值
    code_f1_values = eval_results["code_simple_f1"] + eval_results["code_hard_f1"]
    code_em_values = eval_results["code_simple_em"] + eval_results["code_hard_em"]
    search_f1_values = eval_results["search_simple_f1"] + eval_results["search_hard_f1"]
    search_em_values = eval_results["search_simple_em"] + eval_results["search_hard_em"]
    all_f1_values = code_f1_values + search_f1_values
    all_em_values = code_em_values + search_em_values

    def safe_mean(values):
        return sum(values) / len(values) if len(values) > 0 else 0.0

    mean_results["code_f1_avg"] = safe_mean(code_f1_values)
    mean_results["code_em_avg"] = safe_mean(code_em_values)
    mean_results["search_f1_avg"] = safe_mean(search_f1_values)
    mean_results["search_em_avg"] = safe_mean(search_em_values)
    mean_results["overall_f1_avg"] = safe_mean(all_f1_values)
    mean_results["overall_em_avg"] = safe_mean(all_em_values)

    # 打印结果
    for key, value in mean_results.items():
        print(f"{key}: {value:.4f}")

    # 保存为 CSV 文件
    csv_path = os.path.join(os.path.dirname(eval_file), "MATBench_eval_summary.csv")
    df_mean = pd.DataFrame(list(mean_results.items()), columns=['metric', 'mean_value'])
    df_mean.to_csv(csv_path, index=False)
    print(f"\n✅ saved to {csv_path}")


class MATBench(ImageBaseDataset):
    MODALITY = 'IMAGE'
    TYPE = 'QA'

    DATASET_URL = {'MATBench': 'https://opencompass.openxlab.space/utils/VLMEval/MATBench.tsv'}
    DATASET_MD5 = {'MATBench': '8c79a75ade70384c9918fad1c2a146cb'}  # 测试版本的tsv文件 4e1f4f80f753325f6a471d2ae0f9654e

    def __init__(self, dataset='MATBench', **kwargs):
        super().__init__(dataset, **kwargs)
        print(f'self.img_root:{self.img_root}')

    def build_prompt(self, line):
        input_question = line[2]
        image_path = self.dump_image(line)[0]
        msg = [
            dict(type='image', value=image_path),
            dict(
                type='text',
                value=(
                    input_question
                    + '\n'
                    + "Answer the question directly. "
                    "The answer should be very brief."
                ),
            ),
        ]
        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        tsv_path = self.data_path
        mat_evaluate(tsv_path, eval_file)
