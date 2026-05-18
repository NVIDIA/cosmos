# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Tokenizer utility helpers."""

# Keep this initializer dependency-light. Launcher scripts import
# ``projects.cosmos3.tokenizer.utils.paths`` before training dependencies such
# as Torch are installed, so utility symbols should be imported from their
# concrete submodules instead of re-exported here.
