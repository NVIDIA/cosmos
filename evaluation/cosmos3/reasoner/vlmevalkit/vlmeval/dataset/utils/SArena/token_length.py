import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from .average_meter import AverageMeter
from .base_metric import BaseMetric


class TokenLengthCalculator(BaseMetric):
    def __init__(self, tokenizer_path: str):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.metric = self.count_token_length

        self.meter_gt_tokens = AverageMeter()
        self.meter_pred_tokens = AverageMeter()
        self.meter_diff = AverageMeter()

    def count_token_length(self, **kwargs):
        svg = kwargs.get('gt_svg')
        tokens = self.tokenizer.encode(svg)

        pred_svg = kwargs.get('pred_svg')
        pred_tokens = self.tokenizer.encode(pred_svg)

        diff = len(pred_tokens) - len(tokens)
        return len(tokens), len(pred_tokens), diff

    def calculate_score(self, batch, update=True):
        gt_svgs = batch['gt_svg']
        pred_svgs = batch['pred_svg']
        values = []
        valid_len = 0
        for gt_svg, pred_svg in tqdm(zip(gt_svgs, pred_svgs), total=len(gt_svgs), desc='Processing SVGs'):
            if gt_svg == '' or pred_svg == '':
                continue
            gt_tokens, pred_tokens, diff = self.count_token_length(gt_svg=gt_svg, pred_svg=pred_svg)
            self.meter_gt_tokens.update(gt_tokens, 1)
            self.meter_pred_tokens.update(pred_tokens, 1)
            self.meter_diff.update(diff, 1)

            values.append({
                'gt_tokens': gt_tokens,
                'pred_tokens': pred_tokens,
                'diff': diff
            })
            valid_len += 1
        avg_score = {
            'gt_tokens': self.meter_gt_tokens.avg,
            'pred_tokens': self.meter_pred_tokens.avg,
            'diff': self.meter_diff.avg
        }

        if not values:
            print("No valid SVGs found in the batch")
            return float('nan'), []

        if update:
            self.meter.update(avg_score['pred_tokens'], valid_len)

        return avg_score, values

    def reset(self):
        self.meter_gt_tokens.reset()
        self.meter_pred_tokens.reset()
        self.meter_diff.reset()

    def get_avg_gt_tokens(self):
        return self.meter_gt_tokens.avg

    def get_avg_pred_tokens(self):
        return self.meter_pred_tokens.avg

    def get_avg_diff(self):
        return self.meter_diff.avg
