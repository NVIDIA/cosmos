# Action Normalizers

This directory contains committed action-normalization statistics used by action
datasets during training and inference.

There are two formats:

- `BaseActionLeRobotDataset` stats JSONs, keyed by `global.mean/std/min/max/q01/q99`.
  Files generated with skipped rotation dimensions may also include `global_raw`,
  which preserves the raw pre-masking stats for `action_normalization="quantile_rot"`.
  These are produced by `projects/cosmos3/vfm/datasets/action/compute_action_stats.py`
  and are used by LeRobot-backed action datasets such as Bridge, DROID,
  Fractal, RoboMIND, HandPose, Embodiment_b, and AgiBot.
- UMI field normalizers, keyed by output field name with per-field
  `scale`/`offset` values. Normalization is `(x - offset) / scale`.

## LeRobot-backed datasets

Regenerate LeRobot-backed normalizers with `compute_action_stats.py`. By
default, output filenames match `BaseActionLeRobotDataset._normalizer_filename()`.
The script disables video loading for supported datasets; use fast init for
multi-shard datasets such as AgiBot:

```bash
PYTHONPATH=. python cosmos/data/vfm/action/compute_action_stats.py \
  --config configs/base/config.py \
  --split train \
  --enable-fast-init \
  --reservoir-size 5000000 \
  -- experiment=embodiment_c_gripper
```

For Embodiment C gripper, the committed file is:

```text
embodiment_c_gripper_backward_framewise_rot6d.json
```

This normalizer is shared by:

- `embodiment_c_gripper`
- `embodiment_c_gripper_ext`
- `agibotworld_beta`

The AgiBot FK-pose action layout is 29D:

```text
[head(9), right_wrist(9), right_gripper(1), left_wrist(9), left_gripper(1)]
```

Rotation dims are left unnormalized by writing identity stats for the rot6d
blocks when using `action_normalization="quantile"`. Use
`action_normalization="quantile_rot"` to load `global_raw` and normalize
rotation dimensions as well.

## UMI datasets

UMI normalizers are produced by `UMISingleTrajDataset.fit_normalizer()`, which
computes min/max statistics over the dataset. To regenerate:

```python
from cosmos.data.vfm.action.umi_dataset import get_umi_dataset

dataset = get_umi_dataset(
    dataset_name="<dataset_name>",
    dataset_type="single_task",
    is_val=False,
)
dataset.fit_normalizer()
```

This writes `<dataset_name>_normalizer.json` into the configured `normalizer_dir`. Copy the result into this directory.

## Regenerate When

- Changing action representation, pose convention, or rotation format.
- Changing gripper scaling or FK/action construction.
- Changing UMI `relative_pose_mode`, `use_relative_gripper_width`, or `eef_z_offset`.
- Adding a new task dataset or significantly expanding data with out-of-distribution actions.
