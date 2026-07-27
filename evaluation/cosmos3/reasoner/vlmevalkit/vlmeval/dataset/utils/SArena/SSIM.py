import numpy as np
from skimage.metrics import structural_similarity as ssim

from .base_metric import BaseMetric


class SSIMCalculator(BaseMetric):
    def __init__(self):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.metric = self.compute_SSIM

    def compute_SSIM(self, **kwargs):
        image1 = kwargs.get('gt_im')
        image2 = kwargs.get('pred_im')
        win_size = kwargs.get('win_size', 11)  # Increase win_size for more accuracy
        channel_axis = kwargs.get('channel_axis', -1)  # Default channel_axis to -1
        sigma = kwargs.get('sigma', 1.5)  # Add sigma parameter for Gaussian filter

        # Convert images to numpy arrays if they aren't already
        img1_np = np.array(image1)
        img2_np = np.array(image2)

        # Check if images are grayscale or RGB
        if len(img1_np.shape) == 3 and img1_np.shape[2] == 3:
            # Compute SSIM for RGB images
            score, _ = ssim(img1_np, img2_np, win_size=win_size, channel_axis=channel_axis, sigma=sigma, full=True)
        else:
            # Convert to grayscale if not already
            if len(img1_np.shape) == 3:
                img1_np = np.mean(img1_np, axis=2)
                img2_np = np.mean(img2_np, axis=2)

            score, _ = ssim(img1_np, img2_np, win_size=win_size, sigma=sigma, full=True)

        return score
