from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.multimodal.clip_score import _clip_score_update
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .base_metric import BaseMetric


class CLIPScoreCalculator(BaseMetric):
    def __init__(self, task_type: Literal['T2I', 'I2I']):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
        self.clip_score.to(self.device)
        self.task_type = task_type

    def CLIP_Score(self, images, captions):
        if isinstance(captions, tuple):
            captions = list(captions)
        all_scores = _clip_score_update(images, captions, self.clip_score.model, self.clip_score.processor)
        return all_scores

    def collate_fn(self, batch):
        if self.task_type == 'T2I':
            pred_imgs, captions = zip(*batch)
            tensor_pred_imgs = [ToTensor()(img) for img in pred_imgs]
            return tensor_pred_imgs, captions
        else:
            pred_imgs, gt_imgs = zip(*batch)
            tensor_pred_imgs = [ToTensor()(img) for img in pred_imgs]
            tensor_gt_imgs = [ToTensor()(img) for img in gt_imgs]
            return tensor_pred_imgs, tensor_gt_imgs

    def calculate_score(self, batch, batch_size=64, update=True):
        if self.task_type == 'T2I':
            pred_images = batch['pred_im']
            captions = batch['caption']
            data_loader = DataLoader(list(zip(pred_images, captions)), collate_fn=self.collate_fn,
                                     batch_size=batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)
        else:
            pred_images = batch['pred_im']
            gt_images = batch['gt_im']
            data_loader = DataLoader(list(zip(pred_images, gt_images)), collate_fn=self.collate_fn,
                                     batch_size=batch_size, shuffle=False,
                                     num_workers=4, pin_memory=True)

        all_scores = []
        for batch_eval in tqdm(data_loader):
            if self.task_type == 'T2I':
                images, captions = batch_eval
                images = [img.to(self.device, non_blocking=True) * 255 for img in images]
                list_scores = self.CLIP_Score(images, captions)[0].detach().cpu().tolist()
                all_scores.extend(list_scores)
            else:
                pred_images, gt_images = batch_eval
                pred_images = [img.to(self.device, non_blocking=True) * 255 for img in pred_images]
                gt_images = [img.to(self.device, non_blocking=True) * 255 for img in gt_images]
                list_scores = self.CLIP_Score(pred_images, gt_images)[0].detach().cpu().tolist()
                all_scores.extend(list_scores)

        if not all_scores:
            print("No valid scores found for metric calculation.")
            return float("nan"), []

        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
        return avg_score, all_scores
