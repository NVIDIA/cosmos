import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..base_metric import BaseMetric
from .CLIP_video import sample_frames_from_video


class SSIMVideoCalculator(BaseMetric):
    def __init__(self, num_frames=16):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.metric = self.compute_SSIM
        self.num_frames = num_frames

    def compute_SSIM(self, **kwargs):
        video1 = kwargs.get("gt_video")
        video2 = kwargs.get("pred_video")
        win_size = kwargs.get("win_size", 11)  # Increase win_size for more accuracy
        channel_axis = kwargs.get("channel_axis", -1)  # Default channel_axis to -1
        sigma = kwargs.get("sigma", 1.5)  # Add sigma parameter for Gaussian filter

        frames1 = sample_frames_from_video(video1, self.num_frames)
        frames2 = sample_frames_from_video(video2, self.num_frames)

        scores = []
        for frame1, frame2 in zip(frames1, frames2):
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

            # Convert images to numpy arrays if they aren't already
            img1_np = np.array(frame1)
            img2_np = np.array(frame2)

            # Check if images are grayscale or RGB
            if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
                # Compute SSIM for RGB images
                cur_score, _ = ssim(
                    img1_np,
                    img2_np,
                    win_size=win_size,
                    channel_axis=channel_axis,
                    sigma=sigma,
                    full=True,
                )
            else:
                # Convert to grayscale if not already
                if len(img1_np.shape) == 3:
                    img1_np = np.mean(img1_np, axis=2)
                    img2_np = np.mean(img2_np, axis=2)

                cur_score, _ = ssim(
                    img1_np, img2_np, win_size=win_size, sigma=sigma, full=True
                )
            scores.append(cur_score)

        return np.mean(scores)
