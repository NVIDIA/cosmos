import math
from dataclasses import dataclass
from typing import Callable, Dict

from .base_metric import BaseMetric
from .CLIP_Score import CLIPScoreCalculator
from .DINO_Score import DINOScoreCalculator
from .FID import FIDCalculator
from .LPIPS import LPIPSCalculator
from .PSNR import PSNRCalculator
from .SSIM import SSIMCalculator
from .token_length import TokenLengthCalculator


@dataclass
class MetricsConfig:
    use_FID: bool = False
    use_FID_C: bool = False
    use_CLIP_Score_T2I: bool = False
    use_CLIP_Score_I2I: bool = False
    use_DINO_Score: bool = False
    use_LPIPS: bool = False
    use_SSIM: bool = False
    use_PSNR: bool = False
    use_token_length: bool = False


class InternSVGMetrics:
    def __init__(self, config: MetricsConfig, tokenizer_path: str):
        self.config = config

        # flag -> (metric_name, builder)
        _registry: Dict[str, tuple[str, Callable[[], BaseMetric]]] = {
            'use_FID': ('FID', lambda: FIDCalculator(model_name='InceptionV3')),
            'use_FID_C': ('FID-C', lambda: FIDCalculator(model_name='ViT-B/32')),
            'use_CLIP_Score_T2I': ('CLIP-Score-T2I', lambda: CLIPScoreCalculator(task_type='T2I')),
            'use_CLIP_Score_I2I': ('CLIP-Score-I2I', lambda: CLIPScoreCalculator(task_type='I2I')),
            'use_DINO_Score': ('DINO-Score', lambda: DINOScoreCalculator()),
            'use_LPIPS': ('LPIPS', lambda: LPIPSCalculator()),
            'use_SSIM': ('SSIM', lambda: SSIMCalculator()),
            'use_PSNR': ('PSNR', lambda: PSNRCalculator()),
            'use_token_length': ('Token-Length', lambda: TokenLengthCalculator(tokenizer_path=tokenizer_path)),
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
            if metric_name in ['FID', 'FID-C']:
                avg_result = metric.calculate_score(batch)
                if not math.isnan(avg_result):
                    avg_results_dict[metric_name] = avg_result
            else:
                avg_result, values = metric.calculate_score(batch)
                if not math.isnan(avg_result):
                    avg_results_dict[metric_name] = avg_result

        return avg_results_dict

    def summarize_metrics(self):
        summary_scores = {}
        for name, calc in self.active_metrics.items():
            meter = getattr(calc, 'meter', None)
            if meter is not None:
                summary_scores[name] = meter.avg
        return summary_scores

    def __len__(self) -> int:
        return len(self.active_metrics)

    def __repr__(self) -> str:
        metrics_list = ", ".join(self.active_metrics.keys())
        return f"<InternSVGMetrics active=[{metrics_list}]>"
