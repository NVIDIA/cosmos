# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge clean LeRobot experiment configs (post-training).

======================================================================================================================================================
Runnable experiments — Columns: action | rot | norm | chunk | model | iter | mode | resolution | notes
======================================================================================================================================================
8B backward_framewise policy from exp302_000 (SoT 36 base VFM checkpoint)
8B backward_framewise fdm/joint from exp506_000 midtraining_v1
  Policy loads from t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v7/iter_000043500
  FDM/joint loads from t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/iter_000040000

  bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy                   backward_framewise rot6d  quantile    c16  8B   4k   policy  256 (resize)  alr=5
  bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy                          backward_framewise rot6d  none        c16  8B   4k   policy  256 (resize)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm                      backward_framewise rot6d  quantile_rot c16  8B   4k   fdm     256 (resize)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm                             backward_framewise rot6d  none         c16  8B   4k   fdm     256 (resize)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint                    backward_framewise rot6d  quantile_rot c16  8B   4k   joint   256 (resize)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint                           backward_framewise rot6d  none         c16  8B   4k   joint   256 (resize)  alr=5
  bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy_highres           backward_framewise rot6d  quantile    c16  8B   4k   policy  480 (native)  alr=5
  bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy_highres                  backward_framewise rot6d  none        c16  8B   4k   policy  480 (native)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm_highres              backward_framewise rot6d  quantile_rot c16  8B   4k   fdm     480 (native)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm_highres                     backward_framewise rot6d  none         c16  8B   4k   fdm     480 (native)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint_highres            backward_framewise rot6d  quantile_rot c16  8B   4k   joint   480 (native)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint_highres                   backward_framewise rot6d  none         c16  8B   4k   joint   480 (native)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_rot_norm_idm_highres          backward_framewise rot6d  quantile_rot c16  8B   4k   idm     480 (native)  alr=5
  bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_idm_highres                     backward_framewise rot6d  none         c16  8B   4k   idm     480 (native)  alr=5

======================================================================================================================================================
"""

import copy

from hydra.core.config_store import ConfigStore

from configs.base.experiment.action.midtrain_config.action_datasets import DATASET_BRIDGE
from configs.base.experiment.action.pretrained_config.cosmos3_8b import make_8b_experiment

cs = ConfigStore.instance()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ACTION_KEYS_TO_SKIP = ["action2llm", "llm2action", "action_modality_embed", "action_pos_embed"]

_EXP302_000_CKPT = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v7/checkpoints/iter_000043500/"
)

_EXP506_000_MIDTRAIN_CKPT = (
    "cosmos3_vfm/t2w_mot_8b_qwen3_vl_runs/"
    "t2w_mot_exp506_000_qwen3_vl_8b_multires_recipe_midtraining_v1/checkpoints/iter_000040000/"
)

# Per-resolution batch budget. Highres pipeline uses ~6x more pixels (native
# Bridge 640x480 mapped to the 480 bucket "4,3" = 736x544), so we shrink batch.
_BATCH_BY_RES: dict[str, int] = {
    "256": 256,
    "480": 64,
}


# ---------------------------------------------------------------------------
# Dataset: derive from midtrain DATASET_BRIDGE (mode=joint, no norm, backward_framewise)
# Override mode and stamp the resolution bucket so wrap_dataset
# performs the corresponding resize + reflection-pad (see unified_dataset.py /
# VIDEO_RES_SIZE_INFO).
# ---------------------------------------------------------------------------
def _make_bridge_posttrain_dataset(
    *,
    pose_convention: str = "backward_framewise",
    action_normalization: str | None = "quantile",
    mode: str = "policy",
    resolution: str = "256",
) -> list:
    ds = copy.deepcopy(DATASET_BRIDGE)
    ds.dataset.mode = mode
    ds.dataset.pose_convention = pose_convention
    ds.dataset.action_normalization = action_normalization
    ds.resolution = resolution
    return [ds]


# ---------------------------------------------------------------------------
# Helper: common 8B posttrain overrides
# ---------------------------------------------------------------------------
def _apply_8b_posttrain_overrides(exp: dict, *, checkpoint_path: str, resolution: str = "256") -> dict:
    exp["checkpoint"]["save_iter"] = 1000
    exp["checkpoint"]["load_path"] = checkpoint_path
    exp["checkpoint"]["load_from_object_store"] = dict(bucket="nv-00-10206-checkpoint-experiments")

    exp["model"]["config"]["max_num_tokens_after_packing"] = -1
    exp["model"]["config"]["rectified_flow_training_config"]["shift"] = {"256": 3, "480": 5, "720": 10}
    exp["model"]["config"]["diffusion_expert_config"]["unified_3d_mrope_temporal_modality_margin"] = 15000
    exp["model"]["config"]["diffusion_expert_config"]["max_vae_latent_side_after_patchify"] = 52
    exp["model"]["config"]["tokenizer"]["encode_exact_durations"] = [17]

    exp["dataloader_train"]["max_sequence_length"] = None
    exp["dataloader_train"]["max_samples_per_batch"] = _BATCH_BY_RES[resolution]

    exp["optimizer"]["lr"] = 2e-4
    exp["scheduler"]["warm_up_steps"] = [100]
    exp["scheduler"]["f_min"] = [0.05]
    exp["trainer"]["run_validation"] = False
    exp["trainer"]["callbacks"]["compile_tokenizer"]["warmup_resolutions"] = [resolution]

    exp["job"]["project"] = "cosmos3_action"
    exp["job"]["group"] = "bridge_clean_lerobot"
    return exp


def _build_exp(
    exp_name: str,
    *,
    action_normalization: str | None,
    mode: str = "policy",
    resolution: str,
    checkpoint_path: str = _EXP302_000_CKPT,
) -> dict:
    return _apply_8b_posttrain_overrides(
        make_8b_experiment(
            exp_name,
            _make_bridge_posttrain_dataset(
                action_normalization=action_normalization,
                mode=mode,
                resolution=resolution,
            ),
            training_iterations=4_000,
            action_param_lr_multipliers=5.0,
            keys_to_skip_loading=_ACTION_KEYS_TO_SKIP,
        ),
        checkpoint_path=checkpoint_path,
        resolution=resolution,
    )


# ===========================================================================
# 8B backward_framewise from exp302_000 — 256 (resize)
# ===========================================================================

bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy = _build_exp(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy",
    action_normalization="quantile",
    resolution="256",
)
cs.store(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy",
    bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy = _build_exp(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy",
    action_normalization=None,
    resolution="256",
)
cs.store(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy",
    bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm",
    action_normalization="quantile_rot",
    mode="forward_dynamics",
    resolution="256",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm",
    action_normalization=None,
    mode="forward_dynamics",
    resolution="256",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint",
    action_normalization="quantile_rot",
    mode="joint",
    resolution="256",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint",
    action_normalization=None,
    mode="joint",
    resolution="256",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint,
    group="experiment",
    package="_global_",
)


# ===========================================================================
# 8B backward_framewise from exp302_000 — highres (480 bucket, native 640x480)
# ===========================================================================

bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy_highres",
    action_normalization="quantile",
    resolution="480",
)
cs.store(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy_highres",
    bridge_clean_8b_framewise_from_exp302_000_chunk16_quantile_norm_policy_highres,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy_highres",
    action_normalization=None,
    resolution="480",
)
cs.store(
    "bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy_highres",
    bridge_clean_8b_framewise_from_exp302_000_chunk16_nonorm_policy_highres,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm_highres",
    action_normalization="quantile_rot",
    mode="forward_dynamics",
    resolution="480",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm_highres",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_fdm_highres,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm_highres",
    action_normalization=None,
    mode="forward_dynamics",
    resolution="480",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm_highres",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_fdm_highres,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint_highres",
    action_normalization="quantile_rot",
    mode="joint",
    resolution="480",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint_highres",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_norm_joint_highres,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint_highres",
    action_normalization=None,
    mode="joint",
    resolution="480",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint_highres",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_joint_highres,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_rot_norm_idm_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_rot_norm_idm_highres",
    action_normalization="quantile_rot",
    mode="inverse_dynamics",
    resolution="480",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_rot_norm_idm_highres",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_quantile_rot_norm_idm_highres,
    group="experiment",
    package="_global_",
)

bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_idm_highres = _build_exp(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_idm_highres",
    action_normalization=None,
    mode="inverse_dynamics",
    resolution="480",
    checkpoint_path=_EXP506_000_MIDTRAIN_CKPT,
)
cs.store(
    "bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_idm_highres",
    bridge_clean_8b_framewise_from_exp506_000_chunk16_nonorm_idm_highres,
    group="experiment",
    package="_global_",
)
