import os
import shutil

import cv2
import lpips
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

from vlmeval.smp.file import LMUDataRoot
from ..base_metric import BaseMetric
from .CLIP_video import sample_frames_from_video


def get_lpips_vgg_model(device):
    """Load LPIPS VGG model, downloading to aux_models if needed."""
    vgg_path = os.path.join(LMUDataRoot(), 'aux_models', 'vgg.pth')

    if os.path.exists(vgg_path):
        return lpips.LPIPS(net='vgg', model_path=vgg_path).to(device)

    # Download model (lpips uses torch hub cache)
    model = lpips.LPIPS(net='vgg').to(device)

    # Copy from torch hub cache to aux_models for future offline use
    aux_models_dir = os.path.dirname(vgg_path)
    os.makedirs(aux_models_dir, exist_ok=True)

    cache_path = os.path.expanduser('~/.cache/torch/hub/checkpoints/vgg_net_g.pth')
    if os.path.exists(cache_path):
        shutil.copy(cache_path, vgg_path)

    return model


class LPIPSVideoCalculator(BaseMetric):
    def __init__(self, num_frames=32, batch_size=32):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_lpips_vgg_model(self.device).eval()
        self.metric = self.LPIPS
        self.to_tensor = ToTensor()
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.num_frames = num_frames
        self.batch_size = batch_size

    def LPIPS(self, tensor_image1, tensor_image2):
        tensor_image1, tensor_image2 = tensor_image1.to(self.device), tensor_image2.to(
            self.device
        )
        return self.model(tensor_image1, tensor_image2)

    def to_tensor_transform(self, pil_img):
        return self.normalize(self.to_tensor(pil_img))

    def collate_fn(self, batch):
        gt_imgs, pred_imgs = zip(*batch)
        tensor_gt_imgs = torch.stack([self.to_tensor_transform(img) for img in gt_imgs])
        tensor_pred_imgs = torch.stack(
            [self.to_tensor_transform(img) for img in pred_imgs]
        )
        return tensor_gt_imgs, tensor_pred_imgs

    @torch.inference_mode()
    def calculate_score(self, batch, batch_size=None, update=True):
        if batch_size is None:
            batch_size = self.batch_size

        gt_videos = batch["gt_video"]
        pred_videos = batch["pred_video"]

        gt_imgs_pil = []
        pred_imgs_pil = []

        for gt_video, pred_video in zip(gt_videos, pred_videos):
            gt_frames = sample_frames_from_video(gt_video, self.num_frames)
            pred_frames = sample_frames_from_video(pred_video, self.num_frames)

            T = min(len(gt_frames), len(pred_frames))
            if T == 0:
                continue
            gt_frames = gt_frames[:T]
            pred_frames = pred_frames[:T]

            for g, p in zip(gt_frames, pred_frames):
                g_pil = Image.fromarray(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))
                p_pil = Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
                gt_imgs_pil.append(g_pil)
                pred_imgs_pil.append(p_pil)

        if len(gt_imgs_pil) == 0:
            print("No valid frames for metric calculation.")
            return float("nan"), []

        data_loader = DataLoader(
            list(zip(gt_imgs_pil, pred_imgs_pil)),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
        )

        values = []
        for tensor_gt_batch, tensor_pred_batch in tqdm(data_loader):
            lpips_values = self.LPIPS(tensor_gt_batch, tensor_pred_batch)
            lpips_values = lpips_values.view(-1).detach().cpu().tolist()
            values.extend(lpips_values)

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan"), []

        avg_score = float(sum(values) / len(values))
        if update:
            self.meter.update(avg_score, len(values))
        return avg_score, values
