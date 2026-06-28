import torch

from .perceiver import DynamicCompressor


class IdentityMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_resampler_type": None}

def build_vision_resampler(model_args, delay_load=False, **kwargs):
    # import pdb;pdb.set_trace()
    resampler_type = getattr(model_args, 'mm_resampler_type', None)
    if resampler_type == 'dynamic_compressor':
        return DynamicCompressor(model_args, **kwargs)
    elif resampler_type is None:
        return IdentityMap()
    else:
        raise ValueError(f'Unknown resampler type: {resampler_type}')
