#!/usr/bin/env bash
# Alignment test: --dryrun the same job two ways and diff the resolved YAML.
#
#   A) direct CLI overrides (mirrors run_mixed_modality_sft_trace.sh)
#   B) --toml=toml/alignment_test.toml + the project-specific
#      jsonl/vae paths as CLI tail
#
# If interface_toml.py + the TOML are correctly aligned, both invocations
# produce byte-identical config.yaml files.
set -uo pipefail

WORKDIR="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/cosmos_training"
DATASET_JSONL="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/sft_dataset_bridge/train/video_dataset_file.jsonl"
DCP_LOAD_PATH="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/midtrain"
WAN_VAE_PATH="/nfs/sw/sw_aidot/users/pzeren/Cosmos-prerelease/workdir/cosmos_opensource/pretrained/tokenizers/video/wan2pt2/Wan2.2_VAE.pth"

OUT_A=/tmp/align_a
OUT_B=/tmp/align_b
rm -rf "$OUT_A" "$OUT_B"

cd "$WORKDIR"

# Pin job.name to a constant in BOTH runs so ${now:...} doesn't appear in the
# resolved config and create a false diff.
COMMON_OVERRIDES=(
    "dataloader_train.dataloader.datasets.video.dataset.jsonl_paths=[\"$DATASET_JSONL\"]"
    "model.config.tokenizer.vae_path=$WAN_VAE_PATH"
)

echo ">>> A) direct CLI overrides"
IMAGINAIRE_OUTPUT_ROOT="$OUT_A" PYTHONPATH=. \
    torchrun --nproc_per_node=1 --master_port=12381 -m scripts.train \
    --config=configs/base/config.py \
    --dryrun \
    -- \
    experiment=mixed_modality_sft_8b \
    job.name=alignment_test \
    job.wandb_mode=disabled \
    trainer.max_iter=2 \
    trainer.logging_iter=1 \
    trainer.run_validation=false \
    upload_reproducible_setup=false \
    "checkpoint.load_path=$DCP_LOAD_PATH" \
    model.config.parallelism.data_parallel_shard_degree=4 \
    "${COMMON_OVERRIDES[@]}" \
    > "$OUT_A.log" 2>&1
A_EXIT=$?
A_YAML=$(grep -oE '/tmp/align_a/[^ ]+/config\.yaml' "$OUT_A.log" | tail -1)
echo "    exit=$A_EXIT yaml=$A_YAML"

echo ">>> B) --toml=toml/alignment_test.toml"
IMAGINAIRE_OUTPUT_ROOT="$OUT_B" PYTHONPATH=. \
    torchrun --nproc_per_node=1 --master_port=12382 -m scripts.train \
    --config=configs/base/config.py \
    --toml=toml/alignment_test.toml \
    --dryrun \
    -- \
    "${COMMON_OVERRIDES[@]}" \
    > "$OUT_B.log" 2>&1
B_EXIT=$?
B_YAML=$(grep -oE '/tmp/align_b/[^ ]+/config\.yaml' "$OUT_B.log" | tail -1)
echo "    exit=$B_EXIT yaml=$B_YAML"

if [[ $A_EXIT -ne 0 || $B_EXIT -ne 0 ]]; then
    echo
    echo "ERROR: one of the dryruns failed. Last 40 lines of each log:"
    echo "--- A ---"; tail -40 "$OUT_A.log"
    echo "--- B ---"; tail -40 "$OUT_B.log"
    exit 1
fi

echo
echo ">>> diff (ignoring path_local prefix differences /tmp/align_a vs /tmp/align_b)"
diff <(sed 's|/tmp/align_a|/tmp/ALIGN|g' "$A_YAML") \
     <(sed 's|/tmp/align_b|/tmp/ALIGN|g' "$B_YAML")
DIFF_EXIT=$?
if [[ $DIFF_EXIT -eq 0 ]]; then
    echo ">>> ALIGNED: resolved config.yaml files are identical."
else
    echo ">>> NOT ALIGNED: see diff above."
fi
exit $DIFF_EXIT
