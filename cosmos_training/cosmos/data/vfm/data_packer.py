# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, without express written consent of
# NVIDIA is strictly prohibited.
# -----------------------------------------------------------------------------

"""
Storage-agnostic sample-level transform protocol for OSS-compatible training.

Interface is name-compatible with cosmos-rl's ``BaseDataPacker`` SFT methods:

    sft_process_sample  ↔  sft_process_sample  (identical)
    sft_collate_fn      ↔  sft_collate_fn       (identical)
    compute_num_tokens  ←  NEW single-sample token cost for packing budget

After adding a one-line ``compute_num_tokens`` default to cosmos-rl's
``BaseDataPacker``, existing cosmos-rl packers (``HFVLMDataPacker``,
``Qwen3_VL_DataPacker``, etc.) become directly usable here with no other changes.

Usage
-----
Subclass ``DataPacker`` and implement three methods, then plug into
``DataPackerDataLoader``::

    class MyPacker(DataPacker):
        def sft_process_sample(self, item):
            return {"input_ids": tokenizer(item["text"]).input_ids}

        def compute_num_tokens(self, sample):
            return len(sample["input_ids"])

        def sft_collate_fn(self, samples, max_len, ignore_label_id=-100):
            # pad and stack
            ...

    loader = DataPackerDataLoader(
        data_source=my_dataset,
        data_packer=MyPacker(),
        max_tokens=16000,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataPacker(ABC):
    """Storage-agnostic protocol for transforming dataset items into training batches.

    OSS users subclass this to support any model or dataset format.
    Three abstract methods are required; the rest of the training infrastructure
    (packing, worker management, Hydra config) is inherited automatically.
    """

    @abstractmethod
    def sft_process_sample(self, item: Any) -> dict:
        """Convert one raw dataset item into a training-ready sample dict.

        Parameters
        ----------
        item:
            Whatever the user's ``data_source`` iterable yields —
            a HuggingFace record, a ``{"image": PIL.Image, "text": str}`` dict,
            or any other format.

        Returns
        -------
        dict
            Must contain at minimum the keys expected by ``sft_collate_fn``
            and must have a token-countable representation for
            ``compute_num_tokens``.
        """

    @abstractmethod
    def compute_num_tokens(self, sample: dict) -> int:
        """Return the token cost of one sample for the packing budget.

        For VLM/text models this is typically ``len(sample["input_ids"])``.
        For VFM models override with the VAE spatial/temporal formula.

        This method corresponds to the *per-sample* granularity needed by
        ``PackingIterableDataset._best_fit_batch``.  It differs from
        cosmos-rl's ``sft_compute_max_len`` (batch-level) intentionally.
        """

    @abstractmethod
    def sft_collate_fn(
        self,
        samples: list[dict],
        max_len: int,
        ignore_label_id: int = -100,
    ) -> dict:
        """Collate a list of packed samples into one training batch.

        Parameters
        ----------
        samples:
            List of dicts returned by ``sft_process_sample``.
        max_len:
            Maximum token length in this batch (for padding).
        ignore_label_id:
            Label value for masked/padding positions (default ``-100``).

        Returns
        -------
        dict
            Batch ready for ``model.forward()``.
        """
