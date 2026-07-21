import logging

import torch
from PIL import Image

from .base import BaseModel


class DeepSeekOCR(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    VIDEO_LLM = False

    def __init__(
        self,
        model_path="deepseek-ai/DeepSeek-OCR",
        base_size=1024,
        image_size=640,
        crop_mode=True,
        output_path='/tmp/',
        prompt_template=None,
        torch_dtype=None,
        model_kwargs=None,
        infer_kwargs=None,
        use_vllm=False,
        vllm_kwargs=None,
        sampling_params=None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            logging.critical(
                "Please install transformers>=4.51 and the dependencies required by DeepSeek-OCR."
            )
            raise exc

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch_dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

        self.prompt_template = prompt_template or "<image>\n{text}"
        self.use_vllm = use_vllm

        model_kwargs = model_kwargs or {}
        attn_implementation = None
        if self.device.type == "cuda":
            attn_implementation = "flash_attention_2"
        if attn_implementation is not None:
            model_kwargs.setdefault("_attn_implementation", attn_implementation)
        model_kwargs.setdefault("trust_remote_code", True)
        model_kwargs.setdefault("use_safetensors", True)

        if not self.use_vllm:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_path, **model_kwargs)
            self.model = self.model.to(self.device).to(self.dtype).eval()
        else:
            from vllm import LLM, SamplingParams
            try:
                from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
                logits_processors = [NGramPerReqLogitsProcessor]
            except Exception:
                logits_processors = None

            vllm_defaults = dict(
                model=model_path,
                enable_prefix_caching=False,
            )
            if logits_processors:
                vllm_defaults["logits_processors"] = logits_processors

            vllm_kwargs = vllm_kwargs or {}
            vllm_defaults.update(vllm_kwargs)
            self.vllm_model = LLM(**vllm_defaults)

            sp_kwargs = dict(
                temperature=0.0,
                max_tokens=8192,
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},
                ),
                skip_special_tokens=False,
            )
            if sampling_params:
                sp_kwargs.update(sampling_params)
            self.vllm_sampling_params = SamplingParams(**sp_kwargs)

        self.output_path = output_path or ""
        self.infer_kwargs = dict(
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
        )
        if infer_kwargs:
            self.infer_kwargs.update(infer_kwargs)

    def generate_inner(self, message, dataset=None):
        prompt, image = self.message_to_promptimg(message, dataset)
        if image is None:
            raise ValueError("DeepSeek-OCR requires at least one image input.")

        full_prompt = self._render_prompt(prompt)

        if self.use_vllm:
            return self.generate_inner_vllm(full_prompt, image)

        with torch.inference_mode():
            result = self.model.infer(
                self.tokenizer,
                prompt=full_prompt,
                image_file=image,
                output_path=self.output_path,
                eval_mode=True,
                **self.infer_kwargs,
            )

        return self._normalize_result(result)

    def _render_prompt(self, prompt_text):
        text = prompt_text.strip() if prompt_text else "Free OCR."
        return self.prompt_template.format(text=text)

    def _normalize_result(self, result):
        if isinstance(result, (list, tuple)) and len(result) == 1:
            result = result[0]
        if isinstance(result, dict):
            if "text" in result:
                return result["text"]
            if "result" in result:
                return result["result"]
            return str(result)
        return str(result)

    def generate_inner_vllm(self, full_prompt, image_path):
        img = Image.open(image_path).convert("RGB")
        vllm_input = {
            "prompt": full_prompt,
            "multi_modal_data": {"image": img},
        }
        outputs = self.vllm_model.generate([vllm_input], self.vllm_sampling_params)
        if not outputs:
            return ""
        return outputs[0].outputs[0].text
