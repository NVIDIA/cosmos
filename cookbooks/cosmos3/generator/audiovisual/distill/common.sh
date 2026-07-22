#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Shared implementation for the T2I and I2V distillation launchers.

set -euo pipefail

DISTILL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_DATA2_REVISION="40d018ac1c1a2a4b9734f17fdb21f3d933c49a01"


print_command() {
    printf "+"
    printf " %q" "$@"
    printf "\n"
}


run_command() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command "$@"
    else
        "$@"
    fi
}


wait_for_file() {
    local path="$1"
    local description="$2"
    local waited=0
    local timeout="${ASSET_WAIT_TIMEOUT_SECONDS:-7200}"

    while [[ ! -f "${path}" ]]; do
        if (( waited >= timeout )); then
            echo "Timed out waiting for ${description}: ${path}" >&2
            return 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
}


validate_launch_environment() {
    : "${DISTILL_ROOT:?Set DISTILL_ROOT to a shared filesystem path visible from all eight nodes}"
    : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 host name or IP address}"
    : "${MASTER_PORT:?Set MASTER_PORT to a free rendezvous port}"
    : "${NODE_RANK:?Set NODE_RANK to the rank of this node from 0 through 7}"

    if [[ "${NNODES}" != "8" || "${NPROC_PER_NODE}" != "4" ]]; then
        echo "The first supported recipe requires NNODES=8 and NPROC_PER_NODE=4; got ${NNODES} and ${NPROC_PER_NODE}." >&2
        return 1
    fi
    if [[ ! "${NODE_RANK}" =~ ^[0-7]$ ]]; then
        echo "NODE_RANK must be an integer from 0 through 7; got ${NODE_RANK}." >&2
        return 1
    fi
}


prepare_assets() {
    local dataset_dir="$1"
    local vae_path="$2"
    local teacher_dcp="$3"
    local student_dcp="$4"
    local dataset_marker="${dataset_dir}/sft_dataset_bridge/train/video_dataset_file.jsonl"
    local teacher_marker="${teacher_dcp}/model/.metadata"
    local student_marker="${student_dcp}/model/.metadata"

    if [[ "${NODE_RANK}" == "0" ]]; then
        if [[ ! -f "${dataset_marker}" ]]; then
            run_command "${UVX_BIN}" hf@latest download \
                --repo-type dataset \
                "nvidia/BridgeData2-Subset-Synthetic-Captions" \
                --revision "${BRIDGE_DATA2_REVISION}" \
                --local-dir "${dataset_dir}"
        fi
        if [[ ! -f "${vae_path}" ]]; then
            run_command "${UVX_BIN}" hf@latest download \
                "Wan-AI/Wan2.2-TI2V-5B" \
                "Wan2.2_VAE.pth" \
                --local-dir "$(dirname "${vae_path}")"
        fi
        if [[ ! -f "${teacher_marker}" ]]; then
            run_command env \
                "WORLD_SIZE=1" \
                "RANK=0" \
                "LOCAL_RANK=0" \
                "LOCAL_WORLD_SIZE=1" \
                "${PYTHON_BIN}" -m cosmos_framework.scripts.convert_model_to_dcp \
                -o "${teacher_dcp}" \
                --checkpoint-path "${TEACHER_CHECKPOINT_NAME}"
        fi
        if [[ ! -f "${student_marker}" ]]; then
            run_command env \
                "WORLD_SIZE=1" \
                "RANK=0" \
                "LOCAL_RANK=0" \
                "LOCAL_WORLD_SIZE=1" \
                "${PYTHON_BIN}" -m cosmos_framework.scripts.convert_model_to_dcp \
                -o "${student_dcp}" \
                --checkpoint-path "${STUDENT_CHECKPOINT_NAME}"
        fi
    fi

    if [[ "${DRY_RUN}" != "1" ]]; then
        wait_for_file "${dataset_marker}" "BridgeData2 dataset"
        wait_for_file "${vae_path}" "Wan2.2 VAE"
        wait_for_file "${teacher_marker}" "teacher DCP checkpoint"
        wait_for_file "${student_marker}" "student DCP checkpoint"
    fi
}


build_training_config() {
    local config_path="$1"
    local dataset_path="$2"
    local teacher_dcp="$3"
    local student_dcp="$4"

    if [[ "${NODE_RANK}" == "0" ]]; then
        run_command "${PYTHON_BIN}" "${DISTILL_SCRIPT_DIR}/build_distillation_config.py" \
            --recipe-toml "${DISTILL_RECIPE_TOML}" \
            --dataset-path "${dataset_path}" \
            --teacher-checkpoint "${teacher_dcp}/model" \
            --student-checkpoint "${student_dcp}/model" \
            --output "${config_path}" \
            --validate-only
    fi
    if [[ "${DRY_RUN}" != "1" ]]; then
        wait_for_file "${config_path}" "generated ${DISTILL_MODE} config"
    fi
}


run_training_phase() {
    local config_path="$1"
    local invocation_root="$2"
    local max_iter="$3"
    local resume_option="$4"

    run_command env \
        "IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT}" \
        "WAN_VAE_PATH=${WAN_VAE_PATH}" \
        "COSMOS_TRAINING=1" \
        "PYTORCH_ALLOC_CONF=expandable_segments:True" \
        "LD_LIBRARY_PATH=" \
        "${TORCHRUN_BIN}" \
        "--nnodes=${NNODES}" \
        "--nproc-per-node=${NPROC_PER_NODE}" \
        "--node-rank=${NODE_RANK}" \
        "--master-addr=${MASTER_ADDR}" \
        "--master-port=${MASTER_PORT}" \
        -m cosmos_framework.scripts._train \
        -o "${invocation_root}" \
        --config-file "${config_path}" \
        "${resume_option}" \
        --config-overrides "trainer.max_iter=${max_iter}"
}


export_student_checkpoint() {
    local invocation_root="$1"
    local job_name="cosmos3_super_${DISTILL_MODE}_dmd2_smoke"
    local run_dir="${invocation_root}/${job_name}"
    local checkpoint_path="${run_dir}/job/checkpoints/iter_000000006"

    if [[ "${NODE_RANK}" == "0" && "${EXPORT_AFTER_TRAIN}" == "1" ]]; then
        run_command env \
            "WORLD_SIZE=1" \
            "RANK=0" \
            "LOCAL_RANK=0" \
            "LOCAL_WORLD_SIZE=1" \
            "${PYTHON_BIN}" -m cosmos_framework.scripts.export_model \
            --checkpoint-path "${checkpoint_path}" \
            --config-file "${run_dir}/config.yaml" \
            --student-only-checkpoint-metadata \
            -o "${run_dir}/student"
    fi
}


run_distillation() {
    NNODES="${NNODES:-8}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
    DRY_RUN="${DRY_RUN:-0}"
    EXPORT_AFTER_TRAIN="${EXPORT_AFTER_TRAIN:-1}"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
    UVX_BIN="${UVX_BIN:-uvx}"

    validate_launch_environment

    local dataset_dir="${DISTILL_ROOT}/data/BridgeData2-Subset-Synthetic-Captions"
    local dataset_path="${dataset_dir}/sft_dataset_bridge"
    local vae_path="${DISTILL_ROOT}/checkpoints/wan22_vae/Wan2.2_VAE.pth"
    # DMD2RFModel selects the regular ``net.*`` branch for converted checkpoints
    # whose model directory ends in ``.dcp/model``.
    local teacher_dcp="${DISTILL_ROOT}/checkpoints/${DISTILL_MODE}/teacher.dcp"
    local student_dcp="${DISTILL_ROOT}/checkpoints/${DISTILL_MODE}/student.dcp"
    local config_path="${DISTILL_ROOT}/configs/distillation_${DISTILL_MODE}.yaml"
    local invocation_root="${DISTILL_ROOT}/outputs/invocations/${DISTILL_MODE}"

    mkdir -p \
        "${DISTILL_ROOT}/configs" \
        "${DISTILL_ROOT}/outputs/invocations" \
        "$(dirname "${vae_path}")" \
        "$(dirname "${teacher_dcp}")"

    WAN_VAE_PATH="${vae_path}"
    IMAGINAIRE_OUTPUT_ROOT="${DISTILL_ROOT}/outputs/train"
    COSMOS_TRAINING=1
    LD_LIBRARY_PATH=""
    export WAN_VAE_PATH IMAGINAIRE_OUTPUT_ROOT COSMOS_TRAINING LD_LIBRARY_PATH

    echo "Mode: ${DISTILL_MODE}"
    echo "Teacher model: ${TEACHER_MODEL_REPO}"
    echo "Student model: ${STUDENT_MODEL_REPO}"
    echo "Topology: ${NNODES} nodes x ${NPROC_PER_NODE} GPUs"

    prepare_assets "${dataset_dir}" "${vae_path}" "${teacher_dcp}" "${student_dcp}"
    build_training_config "${config_path}" "${dataset_path}" "${teacher_dcp}" "${student_dcp}"
    run_training_phase "${config_path}" "${invocation_root}" 5 --no-resume
    run_training_phase "${config_path}" "${invocation_root}" 6 --resume
    export_student_checkpoint "${invocation_root}"
}
