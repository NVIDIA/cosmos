#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DISTILL_MODE="t2i"
DISTILL_RECIPE_TOML="${SCRIPT_DIR}/toml/distillation_t2i.toml"
TEACHER_MODEL_REPO="nvidia/Cosmos3-Super-Text2Image"
STUDENT_MODEL_REPO="nvidia/Cosmos3-Super-Text2Image-4Step"
TEACHER_CHECKPOINT_NAME="Cosmos3-Super-Text2Image"
STUDENT_CHECKPOINT_NAME="Cosmos3-Super-Text2Image-4Step"

source "${SCRIPT_DIR}/common.sh"
run_distillation
