import os
import shutil

import lpips
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm

from vlmeval.smp.file import LMUDataRoot
from .base_metric import BaseMetric


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


class LPIPSCalculator(BaseMetric):
    def __init__(self):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_lpips_vgg_model(self.device)
        self.metric = self.LPIPS
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def LPIPS(self, tensor_image1, tensor_image2):
        tensor_image1, tensor_image2 = tensor_image1.to(self.device), tensor_image2.to(self.device)
        return self.model(tensor_image1, tensor_image2)

    def to_tensor_transform(self, pil_img):
        return self.normalize(self.to_tensor(pil_img))

    def collate_fn(self, batch):
        gt_imgs, pred_imgs = zip(*batch)
        tensor_gt_imgs = torch.stack([self.to_tensor_transform(img) for img in gt_imgs])
        tensor_pred_imgs = torch.stack([self.to_tensor_transform(img) for img in pred_imgs])
        return tensor_gt_imgs, tensor_pred_imgs

    def calculate_score(self, batch, batch_size=8, update=True):
        gt_images = batch['gt_im']
        pred_images = batch['pred_im']

        data_loader = DataLoader(list(zip(gt_images, pred_images)), batch_size=batch_size,
                                 collate_fn=self.collate_fn, shuffle=False)

        values = []
        for tensor_gt_batch, tensor_pred_batch in tqdm(data_loader):
            lpips_values = self.LPIPS(tensor_gt_batch, tensor_pred_batch)
            values.extend([lpips_values.squeeze().cpu().detach().tolist()]
                          if lpips_values.numel() == 1 else lpips_values.squeeze().cpu().detach().tolist())

        if not values:
            print("No valid values found for metric calculation.")
            return float("nan"), []

        avg_score = sum(values) / len(values)
        if update:
            self.meter.update(avg_score, len(values))
        return avg_score, values
