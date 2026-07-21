import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from ..base_metric import BaseMetric
from .CLIP_video import sample_frames_from_video


class PSNRVideoCalculator(BaseMetric):
    def __init__(self, num_frames=16):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.metric = self.compute_psnr
        self.num_frames = num_frames

    def compute_psnr(self, **kwargs):
        gt_video = kwargs.get("gt_video")
        pred_video = kwargs.get("pred_video")

        gt_frames = sample_frames_from_video(gt_video, self.num_frames)
        pred_frames = sample_frames_from_video(pred_video, self.num_frames)

        scores = []
        for gt_frame, pred_frame in zip(gt_frames, pred_frames):
            gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR)
            pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)

            gt_im = np.array(gt_frame)
            pred_im = np.array(pred_frame)

            assert (
                gt_im.shape == pred_im.shape
            ), "GT and predicted images must have the same shape"

            psnr_score = psnr(gt_im, pred_im)

            if np.isinf(psnr_score):
                psnr_score = 100

            scores.append(psnr_score)

        return np.mean(scores)
