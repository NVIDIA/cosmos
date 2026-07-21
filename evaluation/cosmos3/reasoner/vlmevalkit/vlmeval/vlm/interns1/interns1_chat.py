from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))


def _best_image_value(v: Any) -> Optional[str]:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("path", "url", "value"):
            if isinstance(v.get(k), str):
                return v[k]
    return None


def _is_local_image_path(p: str) -> bool:
    ext = os.path.splitext(p)[-1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] and os.path.exists(p)


class InternS1Chat:
    INSTALL_REQ = False
    INTERLEAVE = True
    SUPPORT_CHAT = True
    IS_API = False

    def __init__(
        self,
        model_path: str = "internlm/Intern-S1-mini",
        device_map: str = "auto",
        torch_dtype: Union[str, torch.dtype] = "auto",
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        processor_dtype: Optional[torch.dtype] = torch.bfloat16,
        **kwargs,
    ):
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        if processor_dtype is None:
            processor_dtype = torch.bfloat16
        if not torch.cuda.is_available():
            processor_dtype = torch.float32
        self.processor_dtype = processor_dtype

        self._dump_image = False
        self._batch_size = 1
        self._output_dir: Optional[str] = None

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(
        self,
        msgs: Optional[Union[List[Dict[str, Any]], List[str]]] = None,
        message: Optional[Union[List[Dict[str, Any]], List[str]]] = None,
        dataset: Optional[str] = None,
        **kwargs,
    ) -> str:
        if message is not None:
            msgs = message
        if msgs is None:
            raise ValueError("generate() expects `message` or `msgs`.")
        return self.generate_inner(msgs, dataset=dataset)

    def chat(
        self,
        message: Optional[List[Dict[str, Any]]] = None,
        msgs: Optional[List[Dict[str, Any]]] = None,
        dataset: Optional[str] = None,
        **kwargs,
    ) -> str:
        if message is None:
            message = msgs
        if message is None:
            raise ValueError("chat() expects `message` or `msgs`.")
        return self.chat_inner(message, dataset=dataset)

    def set_dump_image(self, flag: bool) -> None:
        self._dump_image = bool(flag)

    def set_batch_size(self, bs: int) -> None:
        self._batch_size = int(bs) if bs and bs > 0 else 1

    def set_output_folder(self, out_dir: str) -> None:
        self._output_dir = out_dir

    def set_generation_config(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if temperature is not None:
            self.temperature = float(temperature)
        if top_p is not None:
            self.top_p = float(top_p)
        if top_k is not None:
            self.top_k = int(top_k)
        if min_p is not None:
            self.min_p = float(min_p)
        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)
        if do_sample is not None:
            self.do_sample = bool(do_sample)

    def generate_inner(
        self,
        msgs: Union[List[Dict[str, Any]], List[str]],
        dataset: Optional[str] = None
    ) -> str:
        hf_messages = self._to_hf_messages_from_mm_msgs(msgs)
        inputs = self._build_inputs(hf_messages)

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                pad_token_id=self.model.config.pad_token_id or self.processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        text = self.processor.decode(gen_ids[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def chat_inner(
        self,
        message: List[Dict[str, Any]],
        dataset: Optional[str] = None
    ) -> str:
        hf_messages = self._to_hf_messages_from_chat(message)
        inputs = self._build_inputs(hf_messages)

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                pad_token_id=self.model.config.pad_token_id or self.processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        text = self.processor.decode(gen_ids[0, input_len:], skip_special_tokens=True)
        return text.strip()

    def use_custom_prompt(self, dataset: Optional[str]) -> bool:
        return False

    def build_prompt(self, line: Dict[str, Any], dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _build_inputs(self, hf_messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        inputs = self.processor.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.processor_dtype)
        return inputs

    def _to_hf_messages_from_mm_msgs(
        self,
        msgs: Union[List[Dict[str, Any]], List[str]]
    ) -> List[Dict[str, Any]]:
        if len(msgs) > 0 and isinstance(msgs[0], str):
            msgs = self._strings_to_msg_dicts(msgs)

        content = []
        for m in msgs:
            t = m.get("type")
            v = m.get("value")
            v_str = _best_image_value(v) if t == "image" else v

            if t == "text":
                content.append({"type": "text", "text": str(v_str)})
            elif t == "image":
                if isinstance(v_str, str) and _is_url(v_str):
                    content.append({"type": "image", "url": v_str})
                elif isinstance(v_str, str) and _is_local_image_path(v_str):
                    img = Image.open(v_str).convert("RGB")
                    content.append({"type": "image", "image": img})
                else:
                    raise ValueError(f"Invalid image value: {v}")
            else:
                raise ValueError(f"Unsupported content type: {t}")

        return [{"role": "user", "content": content}]

    def _to_hf_messages_from_chat(self, chat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hf_messages: List[Dict[str, Any]] = []
        for turn in chat:
            role = turn.get("role")
            mm = turn.get("content")
            if len(mm) > 0 and isinstance(mm[0], str):
                mm = self._strings_to_msg_dicts(mm)

            content = []
            for m in mm:
                t = m.get("type")
                v = m.get("value")
                v_str = _best_image_value(v) if t == "image" else v

                if t == "text":
                    content.append({"type": "text", "text": str(v_str)})
                elif t == "image":
                    if isinstance(v_str, str) and _is_url(v_str):
                        content.append({"type": "image", "url": v_str})
                    elif isinstance(v_str, str) and _is_local_image_path(v_str):
                        img = Image.open(v_str).convert("RGB")
                        content.append({"type": "image", "image": img})
                    else:
                        raise ValueError(f"Invalid image value: {v}")
                else:
                    raise ValueError(f"Unsupported content type in chat: {t}")

            if role not in ("user", "assistant"):
                raise ValueError(f"Invalid role: {role}")
            hf_messages.append({"role": role, "content": content})
        return hf_messages

    def _strings_to_msg_dicts(self, items: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        text_buf: List[str] = []
        for s in items:
            if isinstance(s, str) and (_is_url(s) or _is_local_image_path(s)):
                if text_buf:
                    out.append({"type": "text", "value": " ".join(text_buf)})
                    text_buf = []
                out.append({"type": "image", "value": s})
            else:
                text_buf.append(str(s))
        if text_buf:
            out.append({"type": "text", "value": " ".join(text_buf)})
        return out

    def message_to_promptimg(
        self,
        message: Union[List[Dict[str, Any]], List[str]],
        dataset: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        if len(message) > 0 and isinstance(message[0], str):
            message = self._strings_to_msg_dicts(message)

        texts: List[str] = []
        first_img: Optional[str] = None
        for m in message:
            if m.get("type") == "image" and first_img is None:
                v_str = _best_image_value(m.get("value"))
                if isinstance(v_str, str) and (_is_url(v_str) or _is_local_image_path(v_str)):
                    first_img = v_str
            elif m.get("type") == "text":
                texts.append(str(m.get("value")))
        return (" ".join(texts).strip(), first_img)
