#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DISTILL_MODE="i2v"
DISTILL_RECIPE_TOML="${SCRIPT_DIR}/toml/distillation_i2v.toml"
TEACHER_MODEL_REPO="nvidia/Cosmos3-Super-Image2Video"
STUDENT_MODEL_REPO="nvidia/Cosmos3-Super-Image2Video-4Step"
TEACHER_CHECKPOINT_NAME="Cosmos3-Super-Image2Video"
STUDENT_CHECKPOINT_NAME="Cosmos3-Super-Image2Video-4Step"

source "${SCRIPT_DIR}/common.sh"
run_distillation
