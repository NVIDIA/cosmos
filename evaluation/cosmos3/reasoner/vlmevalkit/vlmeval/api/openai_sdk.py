import json

import requests

from ..smp import get_logger
from .base import BaseAPI

logger = get_logger(__name__)


class OpenAISDKWrapper(BaseAPI):
    """Base class for OpenAI-compatible API wrappers.

    Provides a standard ``generate_inner`` implementation with optional
    :class:`~vlmeval.api.adapters.ModelAdapter` support for model-specific
    prompt building and payload processing.

    Subclasses **must** implement :meth:`prepare_inputs` to convert the
    internal message list into the HTTP message format expected by their
    specific API endpoint.

    Attributes:
        adapter: An optional :class:`ModelAdapter` instance. Set to
            ``None`` to disable adapter hooks.
        key (str): API authentication key.
        api_base (str): Full URL of the chat completions endpoint.
        model (str): Model identifier sent in the request payload.
        timeout (int): HTTP request timeout in seconds.
    """

    is_api: bool = True

    def __init__(self,
                 retry=5,
                 wait=5,
                 verbose=True,
                 system_prompt=None,
                 model_adapter=None,
                 **kwargs):
        """
        Args:
            model_adapter: Either a :class:`ModelAdapter` instance, a
                registered adapter name (``str``), or ``None``.
        """
        self.adapter = None
        if model_adapter is not None:
            if isinstance(model_adapter, str):
                from .adapters import build_adapter
                self.adapter = build_adapter(model_adapter)
            else:
                self.adapter = model_adapter

        super().__init__(
            retry=retry,
            wait=wait,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Adapter integration
    # ------------------------------------------------------------------

    def set_dump_image(self, dump_image_func):
        if self.adapter is not None:
            self.adapter.dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset) -> bool:
        if self.adapter is not None:
            return self.adapter.use_custom_prompt(dataset, self.system_prompt)
        return False

    def build_prompt(self, line, dataset=None):
        if self.adapter is not None:
            return self.adapter.build_prompt(line, dataset)
        raise NotImplementedError

    # ------------------------------------------------------------------
    # HTTP formatting – subclasses must override
    # ------------------------------------------------------------------

    def prepare_inputs(self, inputs, system_prompt):
        """Convert the internal message list to HTTP message dicts.

        Args:
            inputs: List of ``{'type': ..., 'value': ...}`` dicts, or a
                list of role-keyed dicts for multi-turn conversations.
            system_prompt: System prompt string, or ``None``.

        Returns:
            List of OpenAI-style message dicts with ``role`` and
            ``content`` fields.
        """
        raise NotImplementedError(
            f'{type(self).__name__} must implement prepare_inputs()'
        )

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    def generate_inner(self, inputs, dataset=None, **kwargs) -> tuple:
        if self.adapter is not None:
            model_args = self.adapter.override_model_args(dataset)
            system_prompt = model_args.pop('system_prompt', self.system_prompt)
            inputs = self.adapter.process_inputs(inputs, dataset)
            kwargs.update(model_args)
        else:
            system_prompt = self.system_prompt

        input_msgs = self.prepare_inputs(inputs, system_prompt)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key}',
        }
        payload = dict(model=self.model, messages=input_msgs, n=1, **kwargs)

        if self.adapter is not None:
            payload = self.adapter.process_payload(payload, dataset=dataset)

        response = requests.post(
            self.api_base,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout * 1.1,
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            if self.adapter is not None:
                answer = self.adapter.postprocess(answer, dataset=dataset)
        except Exception:
            pass
        return ret_code, answer, response
