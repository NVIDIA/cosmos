# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub: dataset implementation not included in this public release."""

from __future__ import annotations


class SPARBench:
    """Stub for SPARBench."""

    @classmethod
    def supported_datasets(cls):
        return []

    def __init__(self, *args, **kwargs):
        raise RuntimeError("SPARBench is not available in this public release.")


class SPARBenchTiny:
    """Stub for SPARBenchTiny."""

    @classmethod
    def supported_datasets(cls):
        return []

    def __init__(self, *args, **kwargs):
        raise RuntimeError("SPARBenchTiny is not available in this public release.")
