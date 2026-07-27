import cv2
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from ..base_metric import BaseMetric
from .CLIP_video import sample_frames_from_video


class DINOVideoCalculator(BaseMetric):
    def __init__(self, num_frames=16, model_size="base", batch_size=64, use_amp=True):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = self.get_DINOv2_model(model_size)
        self.model = self.model.to(self.device).eval()
        self.metric = self.calculate_DINOv2_similarity_score
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.use_amp = use_amp and self.device == "cuda"

    def get_DINOv2_model(self, model_size):
        model_map = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large",
        }
        name = model_map.get(model_size)
        if not name:
            raise ValueError(
                f"model_size should be either 'small', 'base' or 'large', got {model_size}"
            )
        model = AutoModel.from_pretrained(name)
        processor = AutoImageProcessor.from_pretrained(name)
        return model, processor

    @torch.inference_mode()
    def _frames_to_features(self, frames_bgr):
        if len(frames_bgr) == 0:
            return torch.empty(0, device=self.device)

        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

        feats = []
        for i in range(0, len(frames_rgb), self.batch_size):
            batch_imgs = frames_rgb[i: i + self.batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values=pixel_values)

            feat = outputs.last_hidden_state.mean(dim=1)
            feat = F.normalize(feat, dim=1)
            feats.append(feat)

        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def calculate_DINOv2_similarity_score(self, **kwargs):
        video1 = kwargs.get("gt_video")
        video2 = kwargs.get("pred_video")
        if video1 is None or video2 is None:
            raise ValueError("Please provide 'gt_video' and 'pred_video'.")

        frames1 = sample_frames_from_video(video1, self.num_frames)
        frames2 = sample_frames_from_video(video2, self.num_frames)
        if len(frames1) == 0 or len(frames2) == 0:
            return float("nan")

        feats1 = self._frames_to_features(frames1)
        feats2 = self._frames_to_features(frames2)
        if feats1.numel() == 0 or feats2.numel() == 0:
            return float("nan")

        T = min(feats1.size(0), feats2.size(0))
        feats1 = feats1[:T]
        feats2 = feats2[:T]

        sims = (feats1 * feats2).sum(dim=1)
        sims01 = (sims + 1.0) / 2.0
        sim_mean = float(sims01.mean().item())
        return sim_mean
