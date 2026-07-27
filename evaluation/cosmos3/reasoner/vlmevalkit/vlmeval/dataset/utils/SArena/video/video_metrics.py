import math
from dataclasses import dataclass
from typing import Callable, Dict

from ..base_metric import BaseMetric
from ..token_length import TokenLengthCalculator
from .CLIP_video import ViCLIPCalculator
from .DINO_video import DINOVideoCalculator
from .FVD import FVDCalculator
from .LPIPS_video import LPIPSVideoCalculator
from .PSNR_video import PSNRVideoCalculator
from .SSIM_video import SSIMVideoCalculator


@dataclass
class VideoMetricsConfig:
    use_FVD: bool = False
    use_ViCLIP_T2V: bool = False
    use_ViCLIP_V2V: bool = False
    use_DINO_Video: bool = False
    use_SSIM_Video: bool = False
    use_LPIPS_Video: bool = False
    use_PSNR_Video: bool = False
    use_token_length: bool = False


VICLIP_MODED_PATH = "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth"


class InternSVGVideoMetrics:
    def __init__(self, config: VideoMetricsConfig, tokenizer_path: str):
        self.config = config

        _registry: Dict[str, tuple[str, Callable[[], BaseMetric]]] = {
            "use_FVD": (
                "FVD",
                lambda: FVDCalculator(
                    model_name="i3d", resolution=128, sequence_length=16
                ),
            ),
            "use_ViCLIP_T2V": (
                "ViCLIP-T2V",
                lambda: ViCLIPCalculator(
                    model_name="viclip",
                    ckpt_path=VICLIP_MODED_PATH,
                    num_frames=8,
                    target_size=(224, 224),
                    task_type="T2V",
                ),
            ),
            "use_ViCLIP_V2V": (
                "ViCLIP-V2V",
                lambda: ViCLIPCalculator(
                    model_name="viclip",
                    ckpt_path=VICLIP_MODED_PATH,
                    num_frames=8,
                    target_size=(224, 224),
                    task_type="V2V",
                ),
            ),
            "use_DINO_Video": (
                "DINO-Video",
                lambda: DINOVideoCalculator(num_frames=8, batch_size=64),
            ),
            "use_SSIM_Video": ("SSIM-Video", lambda: SSIMVideoCalculator(num_frames=8)),
            "use_LPIPS_Video": (
                "LPIPS-Video",
                lambda: LPIPSVideoCalculator(num_frames=8, batch_size=32),
            ),
            "use_PSNR_Video": ("PSNR-Video", lambda: PSNRVideoCalculator(num_frames=8)),
            "use_token_length": (
                "Token-Length",
                lambda: TokenLengthCalculator(tokenizer_path=tokenizer_path),
            ),
        }

        self.active_metrics = {}
        for flag, (metric_name, builder) in _registry.items():
            if getattr(self.config, flag, False):
                self.active_metrics[metric_name] = builder()

    def reset(self):
        for metric in self.active_metrics.values():
            metric.reset()

    def calculate_metrics(self, batch):
        avg_results_dict = {}

        for metric_name, metric in self.active_metrics.items():
            print(f"Calculating {metric_name}...")
            avg_result, values = metric.calculate_score(batch)
            if not math.isnan(avg_result):
                avg_results_dict[metric_name] = avg_result

        return avg_results_dict

    def summarize_metrics(self):
        summary_scores = {}
        for name, calc in self.active_metrics.items():
            meter = getattr(calc, "meter", None)
            if meter is not None:
                summary_scores[name] = meter.avg
        return summary_scores

    def __len__(self) -> int:
        return len(self.active_metrics)

    def __repr__(self) -> str:
        metrics_list = ", ".join(self.active_metrics.keys())
        return f"<InternSVGVideoMetrics active=[{metrics_list}]>"
