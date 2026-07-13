"""NVIDIA Nemotron VL family wrapper.

NanoNemotronVLProcessor 400s on any mm_processor_kwarg (do_sample_frames,
fps, ...), so prepare_inputs drops video_kwargs to omit the field.
"""

from .cosmos_reason import CosmosReason


class Nemotron(CosmosReason):
    """NVIDIA Nemotron VL endpoint wrapper (Nemotron-3-Nano-Omni and successors)."""

    def __init__(self, *args, **kwargs):
        # Required by CosmosReason.prepare_itlist video branch.
        kwargs.setdefault('image_patch_size', 16)
        super().__init__(*args, **kwargs)

    def prepare_inputs(self, inputs):
        # NanoNemotronVLProcessor 400s on any mm_processor_kwarg.
        input_msgs, _ = super().prepare_inputs(inputs)
        return input_msgs, None
