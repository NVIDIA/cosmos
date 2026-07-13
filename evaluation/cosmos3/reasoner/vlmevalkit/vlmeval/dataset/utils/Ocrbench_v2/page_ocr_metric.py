import re

import jieba
import nltk
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score


def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


def cal_per_metrics(pred, gt):
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics


if __name__ == "__main__":

    # Examples for region text recognition and read all text tasks
    predict_text = "metrics['edit_dist'] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)

    predict_text = "metrics['edit_dist'] len(gt))"
    true_text = "metrics = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))"

    scores = cal_per_metrics(predict_text, true_text)
    print(scores)
