"""Xiaomi MiMo family wrapper (MiMo-Embodied-7B, MiMo-VL-7B-{SFT,RL}, ...).

All current MiMo VLMs ship a Qwen2.5-VL-7B backbone (HF vision_config.patch_size=14)
with hardcoded thinking-on; chat_template_kwargs.enable_thinking is a no-op. We
append ' /no_think' to the last user-text part (the trained-on signal per
MiMo-Embodied/lmms_eval/models/mivllm.py:217-222) so the server emits an empty
<think></think> envelope plus the answer (~50x cheaper than default thinking-on),
then strip the envelope client-side with the CosmosReason2 no-answer-tag pattern.
"""

import re

from .cosmos_reason import CosmosReason


class MiMo(CosmosReason):

    NO_THINK_SUFFIX = ' /no_think'
    _THINK_PATTERN = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>(?:\n|\n\n| |)([\s\S]*?)$"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('image_patch_size', 14)  # Qwen2.5-VL family
        super().__init__(*args, **kwargs)

    def generate_inner(self, inputs, **kwargs):
        inputs = self._append_no_think(inputs)
        return super().generate_inner(inputs, **kwargs)

    @classmethod
    def _append_no_think(cls, inputs):
        # BaseAPI.generate flattens to a listdict of {'type', 'value'} before
        # reaching generate_inner. Append the suffix to the last text part.
        for part in reversed(inputs):
            if isinstance(part, dict) and part.get('type') == 'text':
                part['value'] = part['value'].rstrip() + cls.NO_THINK_SUFFIX
                return inputs
        return inputs

    def parse_answer(self, answer: str) -> str:
        match = re.search(self._THINK_PATTERN, answer, re.DOTALL)
        return match.group(2).strip() if match else answer
