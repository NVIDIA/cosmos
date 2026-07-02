# Egocentric Hand Action Data Processing

This example converts an egocentric hand-pose annotation sample into a raw
57-dimensional action array. Each action row describes the camera motion, both
wrist motions, and five fingertip positions for one transition between
consecutive video frames.

The script expects one sample in this layout:

```text
example_root/
  videos/<sample_id>.mp4
  captions/<sample_id>.json
  cameras/<sample_id>.json
  human_annotation/<sample_id>.json
```

The checked-in example sample is `ESCALE_000374`.

## Input Schema

The converter is intentionally small and expects the following JSON fields.
All pose arrays must have the same first dimension `N`, matching
`human_annotation/<sample_id>.json["num_frames"]`.

| File | Field | Shape / Type | Meaning |
| --- | --- | --- | --- |
| `human_annotation/<sample_id>.json` | `num_frames` | integer | Number of annotated pose frames. |
| `human_annotation/<sample_id>.json` | `left_hand.hand_keypoints` | `[N, 21, 3]` | Left-hand 3D keypoints in camera coordinates, meters. Joint `0` is the wrist. |
| `human_annotation/<sample_id>.json` | `right_hand.hand_keypoints` | `[N, 21, 3]` | Right-hand 3D keypoints in camera coordinates, meters. Joint `0` is the wrist. |
| `human_annotation/<sample_id>.json` | `left_ee_pose` | `[N, 7]` | Left wrist pose as `[qx, qy, qz, qw, x, y, z]` in camera coordinates. |
| `human_annotation/<sample_id>.json` | `right_ee_pose` | `[N, 7]` | Right wrist pose as `[qx, qy, qz, qw, x, y, z]` in camera coordinates. |
| `cameras/<sample_id>.json` | `camera.pose_world2cam` | `[N, 7]` | Camera world-to-camera pose as `[qx, qy, qz, qw, x, y, z]`; the script inverts it to camera-to-world. |
| `cameras/<sample_id>.json` | `camera.focal_length` | `[2]` | `[fx, fy]`; not used by the converter, but included for visualization checks. |
| `cameras/<sample_id>.json` | `camera.principal_point` | `[2]` | `[cx, cy]`; not used by the converter, but included for visualization checks. |
| `cameras/<sample_id>.json` | `camera.distortion` | `[4]` | Distortion coefficients; not used by the converter, but included for visualization checks. |
| `captions/<sample_id>.json` | `vlm_pipeline.long` or `vlm_pipeline.medium` | string | Optional caption copied into the output metadata. |
| `videos/<sample_id>.mp4` | video file | mp4 | Used only for frame-count reporting by this script. |

By default, wrist translation comes from keypoint `0` in
`hand_keypoints`. Pass `--wrist-position-source ee_pose` if your source should
use the translation stored in `left_ee_pose` and `right_ee_pose` instead.

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
  --output-dir /tmp/egocentric_hand_action_example
```

If you installed `cosmos-framework` into the active Python environment, omit
the `PYTHONPATH=...` prefix. The script defaults to the checked-in
`egocentric_hand_action_example` asset and its single sample, `ESCALE_000374`.

Key output lines for `ESCALE_000374`:

```text
sample_id: ESCALE_000374
pose frames: 122
video frames: 123
wrist position source: keypoint (left keypoint-vs-ee mean 0.0435 m, right 0.0435 m)
raw action: (121, 57) -> /tmp/egocentric_hand_action_example/ESCALE_000374_raw_action_57d.npy
round-trip check against source annotations:
  ...
  fingertip camera L2 max/mean: right 4.2e-05/2.0e-05, left 3.1e-05/1.5e-05
metadata: /tmp/egocentric_hand_action_example/ESCALE_000374_metadata.json
```

Small numerical differences across dependency versions are acceptable; the
roundtrip fingertip errors should remain below `1e-4` meters.

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

The script also writes `<sample_id>_metadata.json`. It records the output
paths, action shape, frame counts, wrist-position diagnostics, the roundtrip
verification metrics, and the copied caption text.

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
  --output-dir /tmp/egocentric_hand_action_example \
  --normalizer-stats /path/to/action_stats.json \
  --normalizer-stats-key global_raw \
  --action-normalization quantile_rot \
  --max-action-dim 64
```

Use normalization stats from the same dataset/checkpoint configuration you plan
to train or run. Do not mix unrelated action statistics.

The stats JSON is loaded with
`cosmos_framework.data.vfm.action.action_processing.load_action_stats`. It must
contain the keys required by the selected normalization method, either at the
top level or under `--normalizer-stats-key`. For `quantile` and `quantile_rot`,
provide `q01` and `q99` arrays of length `57`. For `meanstd`, provide `mean`
and `std`; for `minmax`, provide `min` and `max`.

## Downstream Use

This script is a data-conversion example for one sample. For training, run the
same conversion over your dataset, compute normalization statistics over the raw
57D actions, and connect those actions to the SFT dataset pipeline used by the
action policy recipe.

The surrounding cookbook entry point is
[`README.md`](./README.md). The canonical training implementation and config
reference live in
[`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework), especially the
Cosmos3 Action policy post-training and SFT docs linked from the parent README.

## Verification

The script runs a roundtrip check by default:

1. Encode source annotations into raw 57D action.
2. Decode the camera and wrist relative pose blocks back to absolute poses.
3. Transform the fingertip blocks back into camera coordinates.
4. Report pose and fingertip errors against the original source annotations.

The roundtrip check validates the geometry and indexing in the conversion. It
does not validate that unrelated source conventions, such as a different wrist
axis definition, are semantically correct; use `WRIST_FRAME_ALIGN` for that.
