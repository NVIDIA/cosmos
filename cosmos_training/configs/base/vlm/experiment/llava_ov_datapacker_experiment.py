# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VLM training on lmms-lab/LLaVA-OneVision-Data via DataPackerDataLoader.

Inherits model, optimizer, scheduler, and parallelism from
``pre_exp012_000_phase2_vlm_smoke_4gpu_8b`` and replaces only the dataloader
with the OSS-facing DataPackerDataLoader + VLMDataPacker pattern.

The dataset is loaded in streaming mode from the HuggingFace Hub so no local
download is required.  Each record is converted from ShareGPT conversation
format to the OpenAI message format expected by Qwen3-VL's processor, then
tokenized in the DataLoader worker via ``processor.apply_chat_template``.

Usage (smoke test)::

    torchrun --nproc_per_node=4 --master_port=12344 -m scripts.train \\
        --config=configs/base/vlm/config.py -- \\
        experiment=pre_exp012_llava_ov_datapacker \\
        "model.config.policy.backbone.model_name=/path/to/Siglip2-Qwen3-1.7B-BF16-Alignment" \\
        trainer.max_iter=10 trainer.logging_iter=1 \\
        job.wandb_mode=disabled ckpt_type=dummy

See ``launch_vlm_llava_ov.sh`` for a ready-to-run shell script.
"""

from __future__ import annotations

from typing import Any

from hydra.core.config_store import ConfigStore

from cosmos.utils.lazy_config import LazyCall as L
from cosmos.utils.lazy_config import LazyDict, instantiate
from cosmos.data.vfm.data_packer import DataPacker
from cosmos.data.vfm.data_packer_dataloader import DataPackerDataLoader
from cosmos.data.vfm.processors import build_processor
from cosmos.utils.vlm.constant import IGNORE_INDEX, PROCESSOR_KEYS_TO_ADD

cs = ConfigStore.instance()


# ---------------------------------------------------------------------------
# LLaVA-OneVision-Data source factory
#
# Loads lmms-lab/LLaVA-OneVision-Data in streaming mode so no local download
# is needed.  streaming=True returns an IterableDataset which DataPackerDataLoader
# wraps directly.
# ---------------------------------------------------------------------------


def build_vlm_datapacker_dataloader(**kwargs) -> "DataPackerDataLoader":
    """Thin wrapper around DataPackerDataLoader that drops schema keys injected by
    OmegaConf when the parent experiment's VLMRecipeDataLoader schema merges with
    our DataPackerDataLoader config (e.g. ``storage_type``).
    """
    for _spurious in ("storage_type",):
        kwargs.pop(_spurious, None)
    return DataPackerDataLoader(**kwargs)


def get_llava_ov_streaming(
    subset: str = "si",
    split: str = "train",
) -> Any:
    """Load lmms-lab/LLaVA-OneVision-Data as a streaming HuggingFace IterableDataset.

    Args:
        subset: Dataset config/subset name.  ``"si"`` (single-image, ~1M samples)
            is the standard choice; pass any valid config name from the Hub.
        split: Dataset split (default ``"train"``).

    Returns:
        A streaming ``datasets.IterableDataset`` whose items have keys:
        ``id``, ``image`` (PIL.Image), ``conversations`` (ShareGPT format).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets to use lmms-lab/LLaVA-OneVision-Data") from exc

    ds = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data",
        name=subset,
        split=split,
        streaming=True,
    )
    # Pre-filter to remove records without an image or conversations so
    # sft_process_sample never receives unparseable samples (DataPacker's
    # packing engine does not tolerate None returns from sft_process_sample).
    return ds.filter(lambda x: x.get("image") is not None and len(x.get("conversations") or []) >= 2)


# ---------------------------------------------------------------------------
# VLMDataPacker
#
# Bridges lmms-lab/LLaVA-OneVision-Data (ShareGPT format) into the
# VLMModel training loop.
#
# Three-step pipeline per sample:
#   1. Convert ShareGPT (from/value) → OpenAI messages (role/content).
#   2. Apply processor.apply_chat_template → input_ids, pixel_values, etc.
#   3. Build labels by masking non-assistant tokens with IGNORE_INDEX.
# ---------------------------------------------------------------------------


class VLMDataPacker(DataPacker):
    """DataPacker adapter for lmms-lab/LLaVA-OneVision-Data + Qwen3-VL processor.

    Converts ShareGPT-format image+conversation samples into the
    ``input_ids / labels / pixel_values / image_grid_thw`` batch dict that
    ``VLMModel.training_step`` expects.

    Designed for ``max_batch_size=1`` — each packed batch is a single sample.
    The ``sft_collate_fn`` adds a leading batch dimension to 1-D tensors
    (``input_ids``, ``labels``, ``attention_mask``) while leaving
    ``pixel_values`` and ``image_grid_thw`` in their native flat shapes,
    matching what Qwen3-VL's forward pass expects.
    """

    def __init__(
        self,
        tokenizer_config: Any,
        max_seq_len: int = 16000,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self._max_seq_len = max_seq_len
        self._ignore_index = ignore_index
        # Instantiate if tokenizer_config is a Hydra LazyCall; use directly if already built.
        self._processor = (
            tokenizer_config if hasattr(tokenizer_config, "apply_chat_template") else instantiate(tokenizer_config)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_image(image: Any) -> Any:
        """Decode a HuggingFace streaming image to PIL.

        In streaming mode HuggingFace delivers images as
        ``{"bytes": bytes, "path": str}`` dicts rather than decoded PIL Images.
        """
        if isinstance(image, dict):
            import io

            from PIL import Image

            raw = image.get("bytes")
            if raw:
                return Image.open(io.BytesIO(raw)).convert("RGB")
            path = image.get("path")
            if path:
                return Image.open(path).convert("RGB")
            return None
        return image

    def _sharegpt_to_openai(self, item: dict) -> list[dict]:
        """Convert ShareGPT conversation to OpenAI message format.

        LLaVA-OneVision-Data records use ``from``/``value`` pairs where the
        human turn may contain a ``<image>`` placeholder.  We strip the
        placeholder and attach the PIL image as a separate content block.
        """
        conversations = item.get("conversations", [])
        image = self._decode_image(item.get("image"))  # PIL.Image or None
        messages: list[dict] = []
        image_inserted = False

        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            text = turn["value"].replace("<image>", "").strip()

            if role == "user" and not image_inserted and image is not None:
                content: Any = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ]
                image_inserted = True
            else:
                content = text

            messages.append({"role": role, "content": content})

        return messages

    # ------------------------------------------------------------------
    # DataPacker protocol
    # ------------------------------------------------------------------

    def sft_process_sample(self, item: dict) -> dict:
        """Convert one LLaVA-OV record to VLM training tensors."""
        messages = self._sharegpt_to_openai(item)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        input_ids = inputs["input_ids"]  # [N]

        token_mask = self._processor.add_assistant_tokens_mask(input_ids)  # [N] bool
        labels = input_ids.clone()  # [N]
        labels[~token_mask] = self._ignore_index

        result: dict = {
            "input_ids": input_ids,
            "labels": labels,
        }
        for key in PROCESSOR_KEYS_TO_ADD:
            if key in inputs and inputs[key] is not None:
                result[key] = inputs[key]

        return result

    def compute_num_tokens(self, sample: dict) -> int:
        """Token count = sequence length (input_ids)."""
        return int(sample["input_ids"].shape[0])  # [N] → scalar

    def sft_collate_fn(
        self,
        samples: list[dict],
        max_len: int,
        ignore_label_id: int = IGNORE_INDEX,
    ) -> dict:
        """Assemble one VLM training batch.

        Designed for ``max_batch_size=1``.  1-D sequence tensors get an
        unsqueezed batch dimension; ``pixel_values`` / ``image_grid_thw``
        stay in the flat format Qwen3-VL expects.
        """
        assert len(samples) == 1, f"VLMDataPacker expects max_batch_size=1, got {len(samples)}"
        s = samples[0]

        import torch

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        batch: dict = {
            "input_ids": s["input_ids"].unsqueeze(0),  # [1,N]
            "labels": s["labels"].unsqueeze(0),  # [1,N]
            "sample_worker_id": torch.tensor([worker_id]),  # [1]
            "sample_epoch": torch.tensor([0]),  # [1] streaming has no epoch concept
            "sample_index": torch.tensor([0]),  # [1] streaming has no global index
        }

        if "attention_mask" in s and s["attention_mask"] is not None:
            batch["attention_mask"] = s["attention_mask"].unsqueeze(0)  # [1,N]

        # Vision tensors: pixel_values [P,C] and image_grid_thw [1,3] stay flat.
        for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
            if key in s and s[key] is not None:
                batch[key] = s[key]

        return batch


# ---------------------------------------------------------------------------
# Experiment registration
# ---------------------------------------------------------------------------


pre_exp012_llava_ov_datapacker = LazyDict(
    dict(
        defaults=[
            "/experiment/pre_exp012_000_phase2_vlm_smoke_4gpu_8b",
            # Clear the parent's data_train/data_val defaults so their schemas
            # (including storage_type) don't bleed into our DataPackerDataLoader.
            {"override /data_train": None},
            {"override /data_val": None},
            "_self_",
        ],
        job=dict(
            name="pre_exp012_llava_ov_datapacker_${now:%Y-%m-%d}_${now:%H-%M-%S}",
            group="vlm_llava_ov_demo",
        ),
        trainer=dict(
            max_iter=10,
            logging_iter=1,
            run_validation=False,
        ),
        # Replace the S3 WebDataset-based dataloader with DataPackerDataLoader
        # pointing at lmms-lab/LLaVA-OneVision-Data streamed from HuggingFace Hub.
        dataloader_train=L(build_vlm_datapacker_dataloader)(
            data_source=L(get_llava_ov_streaming)(
                subset="ai2d(gpt4v)",
                split="train",
            ),
            data_packer=L(VLMDataPacker)(
                tokenizer_config=L(build_processor)(
                    tokenizer_type="${model.config.policy.backbone.model_name}",
                ),
                max_seq_len="${data_setting.max_tokens}",
                ignore_index=IGNORE_INDEX,
            ),
            max_tokens="${data_setting.max_tokens}",
            max_batch_size=1,
            pool_size=16,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=True,
        ),
        dataloader_val=None,
    ),
    flags={"allow_objects": True},
)

cs.store(
    group="experiment",
    package="_global_",
    name="pre_exp012_llava_ov_datapacker",
    node=pre_exp012_llava_ov_datapacker,
)
