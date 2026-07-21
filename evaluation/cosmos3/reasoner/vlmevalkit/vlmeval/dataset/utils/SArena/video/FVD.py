import torch
from cdfvd import fvd

from ..base_metric import BaseMetric


class FVDCalculator(BaseMetric):
    """
    Calculate FVD Score
    """

    def __init__(
        self, model_name: str = "i3d", resolution: int = 128, sequence_length: int = 16
    ):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.resolution = resolution
        self.sequence_length = sequence_length

        if self.model_name == "i3d":
            self.evaluator = fvd.cdfvd(
                model="i3d", n_real="full", n_fake="full", device=self.device
            )
        elif self.model_name == "videomae":
            self.evaluator = fvd.cdfvd(
                model="videomae",
                n_real="full",
                n_fake="full",
                device=self.device,
                ckpt_path="PATH_TO_CKPT",
            )
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

    def calculate_score(self, batch, update: bool = True):
        pred_videos = batch.get("pred_video_fvd", [])
        overall_videos = batch.get("overall_video", [])
        if not pred_videos or not overall_videos:
            raise ValueError(
                "batch must contain the key 'pred_video_fvd' and 'overall_video'"
            )

        # get dir of video
        pred_video_dir = pred_videos[0]
        overall_video_dir = overall_videos[0]
        self.evaluator.compute_real_stats(
            self.evaluator.load_videos(
                overall_video_dir,
                data_type="video_folder",
                resolution=self.resolution,
                sequence_length=self.sequence_length,
            )
        )
        self.evaluator.compute_fake_stats(
            self.evaluator.load_videos(
                pred_video_dir,
                data_type="video_folder",
                resolution=self.resolution,
                sequence_length=self.sequence_length,
            )
        )

        fvd_score = self.evaluator.compute_fvd_from_stats()
        if update:
            self.meter.update(fvd_score, len(batch["gt_video"]))
        return fvd_score, []
