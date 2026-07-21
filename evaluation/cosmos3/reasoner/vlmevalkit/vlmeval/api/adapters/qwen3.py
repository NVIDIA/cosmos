from .base import ModelAdapter, register_adapter


@register_adapter('qwen3')
class Qwen3Adapter(ModelAdapter):

    def __init__(self, max_pixels=None):
        self.max_pixels = max_pixels

    def process_payload(self, payload, dataset=None):
        if self.max_pixels:
            payload = payload.copy()
            payload['mm_processor_kwargs'] = {'max_pixels': self.max_pixels}
        return payload
