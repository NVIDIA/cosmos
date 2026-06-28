# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub: dataset implementation not included in this public release."""

from __future__ import annotations


class LVSDataset:
    """Stub for LVSDataset."""

    @classmethod
    def supported_datasets(cls):
        return []

    def __init__(self, *args, **kwargs):
        raise RuntimeError("LVSDataset is not available in this public release.")


class LVSHallucinationDataset:
    """Stub for LVSHallucinationDataset."""

    @classmethod
    def supported_datasets(cls):
        return []

    def __init__(self, *args, **kwargs):
        raise RuntimeError("LVSHallucinationDataset is not available in this public release.")
