#!/usr/bin/env bash
# Sequential 4-experiment OSS smoke runner.
# Each launch_*.sh already has PYTHONHASHSEED=42 and --deterministic baked in.
#
# Usage:
#   srun --overlap --jobid <JOB_ID> --container-name yangyangt_dev \
#        bash -s < /lustre/.../cosmos_training/run_4exps_oss.sh

set -uo pipefail

WORKDIR="/lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/cosmos_training"

for tag in \
    "launch_mixed_modality_sft_8b" \
    "launch_vlm_llava_ov" \
    "launch_action_libero" \
    "launch_t2w_sft_local_datapacker"
do
    echo ""
    echo ">>> $(date '+%H:%M:%S') OSS SMOKE: $tag"
    bash "$WORKDIR/$tag.sh"
    echo ">>> $(date '+%H:%M:%S') OSS Done $tag (exit ${PIPESTATUS[0]})"
done

echo ""
echo ">>> $(date '+%H:%M:%S') ALL 4 OSS SMOKES COMPLETE"
echo "Logs at: /lustre/fsw/portfolios/cosmos/users/yangyangt/Cosmos-prerelease/training_output/logs/"
