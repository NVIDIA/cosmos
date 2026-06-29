# Egocentric Hand Action Data Processing

This example converts an egocentric hand-pose annotation sample into the raw
57D Cosmos3 Action representation used by the hand-action policy data path.

The script expects one sample in this layout:

```text
example_root/
  videos/<sample_id>.mp4
  captions/<sample_id>.json
  cameras/<sample_id>.json
  human_annotation/<sample_id>.json
```

The checked-in example sample is `ESCALE_000374`.

## Setup

Install or clone `cosmos-framework` first so `cosmos_framework` is importable.
For a local checkout next to this repo:

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git ~/projects/cosmos-framework
```

Then run the converter from the `cosmos` repo root:

```bash
cd ~/projects/cosmos

PYTHONPATH=~/projects/cosmos-framework python \
  cookbooks/cosmos3/generator/action/finetune/data_processing_for_egocentric_hand_action.py \
  --example-root cookbooks/cosmos3/generator/action/assets/egocentric_hand_action_example \
  --sample-id ESCALE_000374 \
  --output-dir /tmp/egocentric_hand_action_example
```

If you installed `cosmos-framework` into the active Python environment, omit
the `PYTHONPATH=...` prefix.

Expected output for `ESCALE_000374`:

```text
raw action: (121, 57)
fingertip camera L2 max/mean: right about 4.3e-05 m, left about 3.2e-05 m
```

## 57D Action Layout

The raw action is saved as `<sample_id>_raw_action_57d.npy` with shape
`[num_pose_frames - 1, 57]`.

Each row is:

```text
[camera(9), right_wrist(9), right_fingertips(15), left_wrist(9), left_fingertips(15)]
```

Pose blocks are `[translation(3), rot6d(6)]`. The `rot6d` block is the first two
columns of the relative rotation matrix, following the convention implemented by
`cosmos_framework.data.vfm.action.pose_utils.pose_abs_to_rel`.

Fingertip blocks contain five 3D fingertip positions expressed in the
corresponding wrist frame at the future frame.

## Coordinate Conventions

The input camera pose in this example is `pose_world2cam`; the script inverts it
to camera-to-world before computing relative camera motion. Hand keypoints and
wrist poses are in the camera frame.

The script assumes the wrist-local frame already follows this convention:

```text
+X: thumb side toward pinky side
+Y: outward from the palm
+Z: wrist toward fingertips
```

If your source data uses a different wrist-local frame, edit the
`WRIST_FRAME_ALIGN` matrix in the script. Keep it as identity for data already
in this convention.

## Model-Space Action

By default the script writes only the raw 57D action. To also write the padded
model-space action, pass normalization stats from the matching training setup:

```bash
PYTHONPATH=~/projects/cosmos-framework python \
  cookbooks/cosmos3/generator/action/finetune/data_processing_for_egocentric_hand_action.py \
  --example-root cookbooks/cosmos3/generator/action/assets/egocentric_hand_action_example \
  --sample-id ESCALE_000374 \
  --output-dir /tmp/egocentric_hand_action_example \
  --normalizer-stats /path/to/action_stats.json \
  --normalizer-stats-key global_raw \
  --action-normalization quantile_rot \
  --max-action-dim 64
```

Use normalization stats from the same dataset/checkpoint configuration you plan
to train or run. Do not mix unrelated action statistics.

## Verification

The script runs a roundtrip check by default:

1. Encode source annotations into raw 57D action.
2. Decode the camera and wrist relative pose blocks back to absolute poses.
3. Transform the fingertip blocks back into camera coordinates.
4. Report pose and fingertip errors against the original source annotations.

The roundtrip check validates the geometry and indexing in the conversion. It
does not validate that unrelated source conventions, such as a different wrist
axis definition, are semantically correct; use `WRIST_FRAME_ALIGN` for that.
