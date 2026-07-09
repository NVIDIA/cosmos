import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from .base_metric import BaseMetric


class PSNRCalculator(BaseMetric):
    def __init__(self):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.metric = self.compute_psnr

    def compute_psnr(self, **kwargs):
        gt_im = kwargs.get('gt_im')
        pred_im = kwargs.get('pred_im')

        gt_im = np.array(gt_im)
        pred_im = np.array(pred_im)

        assert gt_im.shape == pred_im.shape, "GT and predicted images must have the same shape"

        psnr_score = psnr(gt_im, pred_im)

        if np.isinf(psnr_score):
            psnr_score = 100

        return psnr_score
