# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib

from hydra.core.config_store import ConfigStore

from cosmos3._src.imaginaire.lazy_config import LazyCall as L
from cosmos3._src.vfm.datasets.vlm.joint_dataset_dynamic_batch_webloader import (
    JointDatasetDynamicBatchingWebLoader,
)
from cosmos3._src.vfm.processors import build_processor
from projects.cosmos3.vlm.datasets.augmentors.bytes_to_media import BytesToMedia
from projects.cosmos3.vlm.datasets.augmentors.filter_output_key import FilterOutputKey
from projects.cosmos3.vlm.datasets.augmentors.filter_seq_length import FilterSeqLength
from projects.cosmos3.vlm.datasets.augmentors.floating_number_format import FloatingNumberFormat
from projects.cosmos3.vlm.datasets.augmentors.format_describe_anything import FormatDescribeAnything
from projects.cosmos3.vlm.datasets.augmentors.prompt_format import PromptFormat
from projects.cosmos3.vlm.datasets.augmentors.shuffle_text_media_order import ShuffleTextMediaOrder
from projects.cosmos3.vlm.datasets.augmentors.timestamp import TimeStamp
from projects.cosmos3.vlm.datasets.augmentors.timestamp_with_subject_tracking import (
    TimeStampWithSubjectTracking,
)
from projects.cosmos3.vlm.datasets.augmentors.timestamp_without_augment_message import (
    TimeStampWithoutAugmentMessage,
)
from projects.cosmos3.vlm.datasets.augmentors.timestamp_without_end_time import TimeStampWithoutEndTime
from projects.cosmos3.vlm.datasets.augmentors.tokenize_data import TokenizeData
from projects.cosmos3.vlm.datasets.collate_fn import custom_collate
from projects.cosmos3.vlm.datasets.dataset_provider_sft import get_vlm_dataset
from projects.cosmos3.vlm.datasets.distributor_with_weight import (
    NoReplaceShardlistBasic,
    WeightedShardlistBasic,
)
from projects.cosmos3.vlm.datasets.joint_dataloader import IterativeJointDataLoader


def create_distributor_config(
    distributor_type: str,
    data_weight_dict: dict,
    url_to_category_fn,
    shuffle: bool = True,
    split_by_node: bool = False,
    split_by_worker: bool = False,
    resume_flag: bool = False,
    verbose: bool = True,
    is_infinite_loader: bool = True,
    seed: int = 1993,
    subsample_config: dict | None = None,
    split: str = "train",
):
    """
    Return a LazyCall to the distributor class based on distributor_type.

    Args:
        distributor_type: "with_replace" -> WeightedShardlistBasic, "no_replace" -> NoReplaceShardlistBasic
        data_weight_dict: category -> weight (or repetitions) mapping
        url_to_category_fn: maps URL path to category key
        split: "train" or "val" (used by NoReplaceShardlistBasic)
        Other args: passed to the distributor constructor.

    Returns:
        L(WeightedShardlistBasic)(...) or L(NoReplaceShardlistBasic)(...) — a LazyCall to the distributor class.
    """
    common = dict(
        data_weight_dict=data_weight_dict,
        url_to_category_fn=url_to_category_fn,
        shuffle=shuffle,
        split_by_node=split_by_node,
        split_by_worker=split_by_worker,
        resume_flag=resume_flag,
        verbose=verbose,
        is_infinite_loader=is_infinite_loader,
        subsample_config=subsample_config,
        split=split,
    )
    if distributor_type == "with_replace":
        return L(WeightedShardlistBasic)(**common)
    if distributor_type == "no_replace":
        return L(NoReplaceShardlistBasic)(seed=seed, **common)

    raise ValueError(f"distributor_type must be in ['with_replace', 'no_replace'], got {distributor_type!r}")


def create_data_augmentor_config():
    return {
        "bytes_to_media": L(BytesToMedia)(
            input_key="media",
            output_key="media",
            min_fps_thres=2,
            max_fps_thres=60,
            target_fps="${data_setting.qwen_target_fps}",  # type: ignore
            max_video_token_length="${data_setting.qwen_max_video_token_length}",  # type: ignore
            processor=processor,
            is_input_pickle_byptes=False,  # If True, it means the input "media" is pickled bytes that needs to be unpickled first; if False, it means the input "media" is raw bytes that can be directly decoded to image/video. Set to False for most cases, and only set to True for some special datasets where media is stored as pickled bytes.
        ),  # takes "videos" and output "videos"
        "prompt_format": L(PromptFormat)(  # takes text_keys and output "conversation"
            input_keys=["texts"],
            text_chat_order="${data_setting.text_chat_order}",
        ),
        "shuffle_text_media_order": L(ShuffleTextMediaOrder)(),
        # ============================
        # TL data augmentation
        # ============================
        "timestamp": L(TimeStamp)(
            input_key="media",
            # output_format="${data_setting.temporal_localization_output_format}",
            output_format="temporal_localization",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "av_reasoning_localization_20250627",
                "tl_activitynet_20250630",
                "tl_agibot_fisheye_20250630",
                "tl_2dvlm_20250627",
                "tl_2dvlm_20251121",
                "tl_youcook2_20250716",
                "tl_yt_cctv_warehouse_20250724",
            ],
            processor=processor,
        ),
        "TL_recaption": L(TimeStamp)(
            input_key="media",
            # output_format="${data_setting.temporal_localization_output_format}",
            output_format="caption",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "tl_2dvlm_recaption_20251121",
                "tl_2dvlm_recaption_20250627",
            ],
            processor=processor,
        ),
        # Special augmentors:
        # timestamp_without_end_time: nexar data does not contain end time
        # timestamp_with_subject_trackig: plm data has subject id + mask, and it's video data
        # format_describe_anything: dam data has subject id + mask + category label, and it's image data (does not need timestampt)
        # timestamp_without_augment_message: rft tl data require timestamp augmentation to video, but keep original text
        "timestamp_without_end_time": L(TimeStampWithoutEndTime)(
            input_key="media",
            # output_format="${data_setting.temporal_localization_output_format}",
            output_format="temporal_localization",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "tl_nexar_20250708",
                "mimicgen_temporal_localization",
            ],
            processor=processor,
        ),
        "timestamp_with_subject_trackig": L(TimeStampWithSubjectTracking)(
            input_key="media",
            output_format="temporal_location_subject",  # Only use temporal_localization tasks to keep the caption style of base model
            urls_needs_timestamp=[
                "tl_plm_sav_20250714",
            ],
            processor=processor,
        ),
        "floating_number_format": L(FloatingNumberFormat)(
            input_key="conversation",
            decimal_places=2,
            urls_needs_format=[
                "3d_grounding_av",
            ],
        ),
        "format_describe_anything": L(FormatDescribeAnything)(
            input_key="media",
            urls_needs_timestamp=[
                "describe-anything-dataset",
            ],
        ),
        "timestamp_without_augment_message": L(TimeStampWithoutAugmentMessage)(
            input_key="media",
            output_format="${data_setting.temporal_localization_output_format}",
            urls_needs_timestamp=[
                "rl_distill_tl_0729",
            ],
            processor=processor,
        ),
        # ============================
        # End of TL data augmentation
        # ============================
        "tokenize_data": L(TokenizeData)(
            processor=processor,
            max_video_token_length="${data_setting.qwen_max_video_token_length}",
            max_image_token_length="${data_setting.qwen_max_image_token_length}",
            add_system_prompt_if_missing=True,
            text_only=False,
        ),
        "filter_output_keys": L(FilterOutputKey)(
            text_only=False,
        ),
        "filter_seq_length": L(FilterSeqLength)(
            max_token_length="${data_setting.max_tokens}",
            processor=processor,
        ),
    }


processor = L(build_processor)(
    tokenizer_type="${model.config.policy.model_name_or_path}",
    credentials="${checkpoint.load_from_object_store.credentials}",
    bucket="${checkpoint.load_from_object_store.bucket}",
)


def get_vlm_dataset_from_module(
    data_module: str,
    split: str = "train",
    distributor_split: str = "train",
    object_store: str = "s3",
    augmentor_config: dict | None = None,
    distributor_type: str = "with_replace",
    distributor_seed: int = 1993,
    buffer_size: int = 2,
    detshuffle: bool = False,
):
    """Resolve data module at instantiation time instead of config registration time.

    This defers importlib.import_module to when the config is actually used (training time),
    avoiding the ~10+ minute startup penalty from eagerly importing all registered dataset
    modules during Hydra config store population.
    """
    data_weight_attr = data_module.split(".")[-1]
    module_path = ".".join(data_module.split(".")[:-1])
    data_weight_module = importlib.import_module(module_path)

    full_datainfo = data_weight_module.DATAINFO
    data_weight_dict = getattr(data_weight_module, data_weight_attr)
    url_to_category = data_weight_module.url_to_category
    subsample_config = getattr(data_weight_module, "subsample_config", None)

    distributor_config = create_distributor_config(
        distributor_type=distributor_type,
        data_weight_dict=data_weight_dict,
        url_to_category_fn=url_to_category,
        split=distributor_split,
        seed=distributor_seed,
        subsample_config=subsample_config,
    )

    return get_vlm_dataset(
        full_datainfo=full_datainfo,
        url_to_category_fn=url_to_category,
        buffer_size=buffer_size,
        object_store=object_store,
        data_weight_dict=data_weight_dict,
        split=split,
        augmentor_config=augmentor_config,
        distributor_config=distributor_config,
        detshuffle=detshuffle,
    )


def create_dataloader_config(
    data_module: str,
    split: str = "train",
    distributor_split: str = "train",
    object_store: str = "s3",
):
    """Create a lazy dataloader config that defers dataset module import to instantiation time.

    Args:
        data_module: Full dotted path to the data weight dict, e.g.
            "projects.cosmos3.vlm.datasets.data_sources_nanov2.data_weight.stage_1.data_weight_repeat".
            The module should export DATAINFO, url_to_category, and the named weight dict.
        split: Dataset split ("train" or "val").
        distributor_split: Distributor split for train/val sharding.
        object_store: Object store backend ("s3", "s3_vlmdb", "pdx", "neb_eu").

    Returns:
        L(get_vlm_dataset_from_module): a LazyCall that resolves the module at instantiation time.
    """
    return L(get_vlm_dataset_from_module)(
        data_module=data_module,
        split=split,
        distributor_split=distributor_split,
        object_store=object_store,
        augmentor_config=create_data_augmentor_config(),
        distributor_type="${data_setting.distributor_type}",
        distributor_seed="${data_setting.distributor_seed}",
        detshuffle="${data_setting.webdataset_detshuffle}",
    )


def register_data_weighted_url():
    cs = ConfigStore.instance()
    # This will register dataset:
    # reason1_v01_understanding_only_pdx
    # reason1_v01_understanding_only_s3
    # reason1_v01_understanding_only_neb_eu
    # eagle_v01_sft_no_text_only_pdx
    # eagle_v01_sft_no_text_only_s3
    # eagle_v01_sft_no_text_only_neb_eu
    # eagle_v02_grounding_2d_pdx
    # eagle_v02_grounding_2d_s3
    # eagle_v02_grounding_2d_neb_eu
    # eagle_v03_grounding_2d_v1_2_pdx
    # eagle_v03_grounding_2d_v1_2_s3
    # eagle_v03_grounding_2d_v1_2_neb_eu
    # eagle_v04_sft_no_text_only_no_grounding_2d_pdx
    # eagle_v04_sft_no_text_only_no_grounding_2d_s3
    # eagle_v04_sft_no_text_only_no_grounding_2d_neb_eu
    # joint_v01_cr1_understanding_eagle_sft_pdx
    # joint_v01_cr1_understanding_eagle_sft_s3
    # joint_v01_cr1_understanding_eagle_sft_neb_eu
    for dataset_id, data_module in {
        "01": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.understanding_only.data_weight_default",  # 01_reason1_understanding_only_default_s3
        "02": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full.data_weight_default",  # 02_eagle_sft_full_default_s3
        "03": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.grounding_2d_v1_1.data_weight_default",  # 03_eagle_grounding_2d_v1_1_default_s3
        "04": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.grounding_2d_v1_2.data_weight_default",  # 04_eagle_grounding_2d_v1_2_default_s3
        "05": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.cr1_eagle_sft.data_weight_full_5v5",  # 05_joint_cr1_eagle_sft_full_5v5_s3
        "06": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.cr1_eagle_sft.data_weight_2d_5v5",  # 06_joint_cr1_eagle_sft_2d_5v5_s3
        "07": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.pretrain.data_weight_default",  # 07_eagle_pretrain_default_s3
        "08": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full_mul_repeat.data_weight_default",  # 08_eagle_sft_full_mul_repeat_default_s3
        "09": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full_mul_repeat.data_weight_debug",  # 09_eagle_sft_full_mul_repeat_debug_s3
        "10": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.cr1_eagle_sft.data_weight_full_5v5_mul_repeat",  # 10_joint_cr1_eagle_sft_full_5v5_mul_repeat_s3
        "11": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full_mul_repeat.data_weight_single",  # 11_eagle_sft_full_mul_repeat_single_s3
        "12": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.understanding_only.data_weight_default",  # 12_reason1_understanding_only_default_s3
        "13": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.reason1p0_1p1.data_weight_mix_5v5",  # 13_reason1p0_1p1_mix_5v5_s3
        "14": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason1_2.data_weight_mix_5v5v5",  # 14_joint_reason1_2_mix_5v5v5_s3
        "15": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.reason1p0_1p1.data_weight_debug_tl",  # 15_reason1_reason1p0_1p1_debug_tl_s3
        "16": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason1_2.data_weight_mix_all_zero",  # 16_joint_reason1_2_mix_all_zero_s3
        "17": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full_mul_repeat.data_weight_only_vatex_subset",  # 17_eagle_sft_full_mul_repeat_only_vatex_subset_s3
        "18": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason1_2.data_weight_understand",  # 18_joint_reason1_2_data_weight_understand
        "19": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.reason1p0_1p1_721.data_weight_mix_5v5",  # 19_reason1_reason1p0_1p1_721_mix_5v5_s3
        "20": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.reason1p0_1p1_721.data_weight_debug_2dvlm",  # 20_reason1_reason1p0_1p1_721_debug_2dvlm_s3
        "21": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.reason1p0_1p1_721.data_weight_debug_av",  # 21_reason1_reason1p0_1p1_721_debug_av_s3
        "22": "projects.cosmos3.vlm.datasets.data_sources_reason1.data_weight.reason1p0_1p1_721.data_weight_no_rft",  # 22_reason1_reason1p0_1p1_721_no_rft_s3
        "23": "projects.cosmos3.vlm.datasets.data_sources_reason2.data_weight.reason2.data_weight_all_zero",  # 23_reason2_reason2_all_zero_s3
        # 24: Reason 2 data count 5% of total
        "24": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason1p0_1p1_2_721.data_weight_mix_475v475v005",  # 24_joint_reason1p0_1p1_2_721_mix_475v475v005_s3
        "25": "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.grounding_2d_v1_1.data_weight_no_robospatial",  # 25_eagle_grounding_2d_v1_1_no_robospatial_s3
        # via exploration
        "26": "projects.cosmos3.vlm.datasets.data_sources_via.data_weight.default.data_weight_spatial_suc_only",  # 26_via_default_spatial_suc_only_s3
        "27": "projects.cosmos3.vlm.datasets.data_sources_via.data_weight.default.data_weight_spatial_suc_only_round4",  # 27_via_default_spatial_suc_only_round4_s3
        "28": "projects.cosmos3.vlm.datasets.data_sources_via.data_weight.default.data_weight_90_suc_only_round2",  #
        "29": "projects.cosmos3.vlm.datasets.data_sources_via.data_weight.default.data_weight_all_zeros",  # 29_via_default_all_zeros_s3
        # reason2 release
        "54": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason2_release.data_weight_joint",  # 54_joint_reason2_release_joint_s3
        "55": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason2_release.data_weight_with_recaption",  # 55_joint_reason2_release_joint_with_recaption_s3
        "56": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason2_release.data_weight_with_recaption_wo_human",  # 56_joint_reason2_release_joint_with_recaption_wo_human_s3
        "57": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason2p0_2p1.data_weight_joint",  # 57_joint_reason2p0_2p1_joint_s3
        "101": "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason2_release.data_weight_debug_recaption",  # 101_joint_reason2_release_debug_recaption_s3
        # # taxonomy distillation
        # "100": "projects.cosmos3.vlm.datasets.data_sources_taxonomy_distill.data_weight.taxonomy_distill.data_weight_default",  # 100_taxonomy_distill_taxonomy_distill_default_s3
        # # interleave document scoring distillation
        # "102": "projects.cosmos3.vlm.datasets.data_sources_interleave_scoring.data_weight.interleave_scoring.data_weight_default",  # 102_interleave_scoring_interleave_scoring_default_s3
        # video taxonomy distillation
        "103": "projects.cosmos3.vlm.datasets.data_sources_video_taxonomy.data_weight.video_taxonomy.data_weight_default",  # 103_video_taxonomy_video_taxonomy_default_s3
        # nanov2 pre/post-training
        "200": "projects.cosmos3.vlm.datasets.data_sources_nanov2.data_weight.stage_1_0218_34m_uniform_pretrain.data_weight_repeat",  # 200_nanov2_stage_1_0218_34m_uniform_pretrain_repeat_s3_vlmdb
        "201": "projects.cosmos3.vlm.datasets.data_sources_nanov2.data_weight.stage_1_0218_34m_uniform_posttrain.data_weight_repeat",  # 201_nanov2_stage_1_0218_34m_uniform_posttrain_repeat_s3_vlmdb
        # Data ablation configs (below is a dummy example, do not uncomment)
        "202": "projects.cosmos3.vlm.datasets.data_sources_nanov2.data_weight.new_category_data_mixture.data_weight_repeat",  # 202_nanov2_new_category_data_mixture_repeat_s3_vlmdb
    }.items():
        data_source_name = data_module.split("data_sources_")[-1].split(".")[0]
        dataset_file_name = data_module.split(".")[-2]
        data_weight_name = data_module.split("data_weight_")[-1]
        for distributor_split in ["train", "val"]:
            for object_store in ["pdx", "s3", "s3_vlmdb", "neb_eu"]:
                dataset_name = f"{dataset_id}_{data_source_name}_{dataset_file_name}_{data_weight_name}_{object_store}"
                cs.store(
                    group=f"data_{distributor_split}",
                    package=f"dataloader_{distributor_split}",
                    name=dataset_name,
                    node=L(JointDatasetDynamicBatchingWebLoader)(
                        datasets_cfg={
                            "default": {
                                "dataset": create_dataloader_config(
                                    data_module=data_module,
                                    split="train",
                                    distributor_split=distributor_split,
                                    object_store=object_store,
                                ),
                                "ratio": 1,
                            }
                        },
                        # Arguments for the joint dataset
                        pool_size=16,
                        max_batch_size="${data_setting.max_batch_size}",
                        max_tokens="${data_setting.max_tokens}",
                        model_name_or_path="${model.config.policy.model_name_or_path}",  # "Qwen/Qwen3-VL-2B-Init"
                        long_threshold=6400,
                        length_key="input_ids",
                        batching_strategy="prefer_closest",
                        # Arguments for the webloader
                        batch_size=1,  # This is not the real batch size, it wont be used
                        num_workers="${data_setting.num_data_workers}" if distributor_split == "train" else 0,
                        sampler=None,
                        prefetch_factor="${data_setting.data_prefetch_factor}"
                        if distributor_split == "train"
                        else None,
                        persistent_workers=False,
                        pin_memory=True,
                        collate_fn=custom_collate,
                    ),
                )


def register_data_weighted_url_with_text():
    cs = ConfigStore.instance()

    # This will register dataset:

    for dataset_id, data_modules in {
        "m01": {
            "with_visual": (
                5,
                "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full_mul_repeat.data_weight_default",
            ),
            "text_only": (
                1,
                "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full_mul_repeat.data_weight_text_only",
            ),
        },  # m01_visual_5_mix_text_1__eagle_sft_full_mul_repeat_default_s3
        "m02": {
            "with_visual": (
                5,
                "projects.cosmos3.vlm.datasets.data_sources_joint.data_weight.reason2p0_2p1.data_weight_joint",
            ),
            "text_only": (
                1,
                "projects.cosmos3.vlm.datasets.data_sources_eagle.data_weight.sft_full_mul_repeat.data_weight_text_only",
            ),
        },  # m02_visual_5_mix_text_1__joint_reason2p0_2p1_joint_s3
    }.items():
        ratio_with_visual, data_module_with_visual = data_modules["with_visual"]
        ratio_text_only, data_module_text_only = data_modules["text_only"]
        data_source_name = data_module_with_visual.split("data_sources_")[-1].split(".")[0]
        dataset_file_name = data_module_with_visual.split(".")[-2]
        data_weight_name = data_module_with_visual.split("data_weight_")[-1]

        for distributor_split in ["train", "val"]:
            for object_store in ["pdx", "s3", "neb_eu"]:
                dataset_name = f"{dataset_id}_visual_{ratio_with_visual}_mix_text_{ratio_text_only}__{data_source_name}_{dataset_file_name}_{data_weight_name}_{object_store}"
                cs.store(
                    group=f"data_{distributor_split}",
                    package=f"dataloader_{distributor_split}",
                    name=dataset_name,
                    node=L(IterativeJointDataLoader)(
                        dataloaders={
                            "with_visual": {
                                "ratio": ratio_with_visual,
                                "dataloader": L(JointDatasetDynamicBatchingWebLoader)(
                                    datasets_cfg={
                                        "default": {
                                            "dataset": create_dataloader_config(
                                                data_module=data_module_with_visual,
                                                split="train",
                                                distributor_split=distributor_split,
                                                object_store=object_store,
                                            ),
                                            "ratio": 1,
                                        }
                                    },
                                    # Arguments for the joint dataset
                                    pool_size=16,
                                    max_batch_size="${data_setting.max_batch_size}",
                                    max_tokens="${data_setting.max_tokens}",
                                    model_name_or_path="${model.config.policy.model_name_or_path}",  # "Qwen/Qwen3-VL-2B-Init"
                                    long_threshold=6400,
                                    length_key="input_ids",
                                    batching_strategy="prefer_closest",
                                    # Arguments for the webloader
                                    batch_size=1,  # This is not the real batch size, it wont be used
                                    num_workers="${data_setting.num_data_workers}"
                                    if distributor_split == "train"
                                    else 0,
                                    sampler=None,
                                    prefetch_factor="${data_setting.data_prefetch_factor}"
                                    if distributor_split == "train"
                                    else None,
                                    persistent_workers=False,
                                    pin_memory=True,
                                    collate_fn=custom_collate,
                                ),
                            },
                            "text_only": {
                                "ratio": ratio_text_only,
                                "dataloader": L(JointDatasetDynamicBatchingWebLoader)(
                                    datasets_cfg={
                                        "default": {
                                            "dataset": create_dataloader_config(
                                                data_module=data_module_text_only,
                                                split="train",
                                                distributor_split=distributor_split,
                                                object_store=object_store,
                                            ),
                                            "ratio": 1,
                                        }
                                    },
                                    # Arguments for the joint dataset
                                    pool_size=16,
                                    max_batch_size="${data_setting.max_batch_size}",
                                    max_tokens="${data_setting.max_tokens}",
                                    model_name_or_path="${model.config.policy.model_name_or_path}",  # "Qwen/Qwen3-VL-2B-Init"
                                    long_threshold=6400,
                                    length_key="input_ids",
                                    batching_strategy="prefer_closest",
                                    # Arguments for the webloader
                                    batch_size=1,  # This is not the real batch size, it wont be used
                                    num_workers=2,
                                    sampler=None,
                                    prefetch_factor=1,
                                    persistent_workers=False,
                                    pin_memory=True,
                                    collate_fn=custom_collate,
                                ),
                            },
                        }
                    ),
                )


def register_data_recipe():
    from projects.cosmos3.vlm.datasets.recipe_dataloader import VLMRecipeDataLoader

    cs = ConfigStore.instance()
    # This will register recipe-based dataloaders using VLMRecipeDataLoader.
    # Recipe names and storage types are stored in the PostgreSQL recipe database.
    # Registered configs:
    #   cosmos_reason2_s3_vlmdb
    for recipe_name, storage_type in [
        ("cosmos_reason2_s3_vlmdb", "s3_vlmdb"),  # cosmos_reason2_s3_vlmdb_recipe
        (
            "nemotron_nanov2_stage_1_0218_34m_uniform_pretrain_s3_vlmdb",
            "s3_vlmdb",
        ),  # nemotron_nanov2_stage_1_0218_34m_uniform_s3_vlmdb__v1_recipe
        (
            "nemotron_nanov2_stage_1_0218_34m_uniform_posttrain_s3_vlmdb",
            "s3_vlmdb",
        ),  # nemotron_nanov2_stage_1_0218_34m_uniform_posttrain_s3_vlmdb__v1_recipe
    ]:
        config_name = f"{recipe_name.replace('/', '__')}_recipe"
        for distributor_split in ["train", "val"]:
            cs.store(
                group=f"data_{distributor_split}",
                package=f"dataloader_{distributor_split}",
                name=config_name,
                node=L(VLMRecipeDataLoader)(
                    recipe_name=recipe_name,
                    storage_type=storage_type,
                    model_name_or_path="${model.config.policy.model_name_or_path}",
                    max_tokens="${data_setting.max_tokens}",
                    max_batch_size="${data_setting.max_batch_size}",
                    pool_size=16,
                    long_threshold=6400,
                    length_key="input_ids",
                    batching_strategy="prefer_closest",
                    augmentor_config=create_data_augmentor_config(),
                    distributor_type="${data_setting.distributor_type}",
                    detshuffle="${data_setting.webdataset_detshuffle}",
                    split="train",  # use train split of the dataset
                    distributor_split=distributor_split,  # split training dataset into train and val splits
                    val_split_ratio="${data_setting.val_split_ratio}",
                    distributor_seed="${data_setting.distributor_seed}",
                    num_workers="${data_setting.num_data_workers}" if distributor_split == "train" else 0,
                    prefetch_factor="${data_setting.data_prefetch_factor}" if distributor_split == "train" else None,
                    persistent_workers=False,
                    pin_memory=True,
                    collate_fn=custom_collate,
                ),
            )
