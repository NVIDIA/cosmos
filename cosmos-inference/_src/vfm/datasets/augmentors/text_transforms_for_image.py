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

import json
import random
from typing import Optional

from cosmos3._src.imaginaire.datasets.augmentors.v3_text_transforms import pad_and_resize
from cosmos3._src.imaginaire.datasets.webdataset.augmentors.augmentor import Augmentor
from cosmos3._src.imaginaire.utils import log
from cosmos3._src.vfm.datasets.data_sources.data_registration import _CAPTION_EMBEDDING_KEY_MAPPING_IMAGES

# For the qwen captions, we have 3 variants: short, medium, long
# In addition, for synthetic data, we create prompt embeddings as well.
# There is quite a bit of entropy in the way prompt data is saved.
# Captions are saved as "prompts", while the corresponding embeddings are saved as "original_prompt"
# This part will be cleaned after synthetic data is cleaned to be in the same format as real data.
_AVAILABLE_QWEN_CAPTIONS = ["qwen2p5_7b_short", "qwen2p5_7b_medium", "qwen2p5_7b_long"]
_AVAILABLE_QWEN3_30B_A3B_CAPTIONS = [
    "qwen3_30b_a3b_short",
    "qwen3_30b_a3b_descriptive",
    "qwen3_30b_a3b_dense",
]
# used for new caption in Nov 2025
_AVAILABLE_CAPTIONS_V2 = ["caption_short", "caption_medium", "caption_long"]
# used for sft v1
_AVAILABLE_CAPTIONS_SFT_V1 = [
    "gemini_v1_dense",
    "gemini_v2_dense",
    "qwen3vl_30B_v1_dense",
    "qwen3vl_30B_v2_dense",
    "qwen3vl_235B_v1_dense",
    "qwen3vl_235B_v2_dense",
]
# used for genplan ablation
# captions are saved as "caption_long" as a JSON string, like {"dense": "xxx", "dense_bbox": "xxx"}
_AVAILABLE_CAPTIONS_GENPLAN = ["dense", "dense_bbox"]
_CAPTION_EMBEDDING_MAPPING = {
    "qwen2p5_7b_short": "qwen2p5_7b_short",
    "qwen2p5_7b_medium": "qwen2p5_7b_medium",
    "qwen2p5_7b_long": "qwen2p5_7b_long",
    "prompts": "original_prompt",
}


class TextTransformForImage(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs camera transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """

        caption_type = self.args["caption_type"]
        embedding_key_in_dict = _CAPTION_EMBEDDING_KEY_MAPPING_IMAGES[caption_type]
        embedding_type = self.args["embedding_type"]
        embedding_input_key_prefix = "" if embedding_type == "t5_xxl" else "umt5_"

        captions_key, embeddings_key = (
            f"captions_{caption_type}",
            f"{embedding_input_key_prefix}embeddings_captions_{embedding_key_in_dict}",
        )
        decoded_captions_ai = data_dict[captions_key]
        decoded_embeddings_ai = data_dict[embeddings_key]

        try:
            # Hotfix: Some captions are labeled as "captions" and some are labeled as "caption"
            # This issue needs to be fixed in the synthetic data. This is a hack and will be removed
            # once the data is cleaned.
            caption_key = "captions" if "captions" in decoded_captions_ai else "caption"
            embedding_key = "t5_xxl_fp8" if embedding_type == "t5_xxl" else "umt5_xxl"
            if caption_type == "qwen2p5_7b_v4":
                selected_caption_type = random.choice(_AVAILABLE_QWEN_CAPTIONS)
                data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                t5_embedding = decoded_embeddings_ai[selected_caption_type]["embeddings"][embedding_key]
                data_dict["selected_caption_type"] = selected_caption_type
            elif caption_type == "prompts":
                data_dict["ai_caption"] = decoded_captions_ai["caption"]["prompt"]
                t5_embedding = decoded_embeddings_ai[_CAPTION_EMBEDDING_MAPPING[caption_type]]["embeddings"][
                    embedding_key
                ]
                data_dict["selected_caption_type"] = caption_type
            else:
                assert caption_type == "ai_v3p1", f"Caption type {caption_type} not supported"
                if decoded_captions_ai["had_parse_issue"]:
                    data_dict["ai_caption"] = decoded_captions_ai["captions"]["kosmos_2"]
                    t5_embedding = decoded_embeddings_ai["kosmos2"]["embeddings"][embedding_key]
                else:
                    data_dict["ai_caption"] = decoded_captions_ai["captions"]["vfc"]
                    t5_embedding = decoded_embeddings_ai["vfc_fidelity"]["embeddings"][embedding_key]

            out_t5, out_t5_mask = pad_and_resize(
                t5_embedding,
                self.args["t5_tokens"]["num"],
                is_mask_all_ones=self.args["is_mask_all_ones"],
            )
            data_dict["t5_text_embeddings"] = out_t5
            data_dict["t5_text_mask"] = out_t5_mask
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}\n error {e}",
                rank0_only=False,
            )
            return None

        del data_dict[captions_key]
        del data_dict[embeddings_key]

        return data_dict


class TextTransformForImageWithoutEmbeddings(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.caption_prefix = args.get("caption_prefix", None) if args else None

    def _apply_caption_prefix(self, data_dict: dict) -> None:
        """Prepend caption_prefix to ai_caption if configured."""
        if not self.caption_prefix or not isinstance(data_dict.get("ai_caption"), str):
            return
        original = data_dict["ai_caption"]
        data_dict["ai_caption"] = self.caption_prefix + " " + original.lstrip()
        log.debug(
            f"[caption_prefix] before: {original[:120]!r}... | after: {data_dict['ai_caption'][:120]!r}...",
            rank0_only=False,
        )

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transform without any embedding loading.
        This is useful for online computation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """

        caption_type = self.args["caption_type"]
        captions_key = f"captions_{caption_type}"
        decoded_captions_ai = data_dict[captions_key]

        train_on_captions = self.args.get("train_on_captions", [])
        # if [], will infer based on the caption json
        # otherwise it will only use the captions in the list

        try:
            # Hotfix: Some captions are labeled as "captions" and some are labeled as "caption"
            # This issue needs to be fixed in the synthetic data. This is a hack and will be removed
            # once the data is cleaned.
            caption_key = "captions" if "captions" in decoded_captions_ai else "caption"
            if len(train_on_captions) == 0:
                # infer which caption types are there
                if caption_type in ("generated_gpt_oss_20b", "generated_gpt_oss_120b"):
                    selected_caption_type = "caption_long"
                    if caption_key in decoded_captions_ai:  # sharded with sila pipeline
                        data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                    else:
                        data_dict["ai_caption"] = decoded_captions_ai[selected_caption_type]
                    data_dict["selected_caption_type"] = selected_caption_type
                elif caption_type == "qwen3_30b_a3b":
                    selected_caption_type = random.choice(_AVAILABLE_QWEN3_30B_A3B_CAPTIONS)
                    data_dict["ai_caption"] = decoded_captions_ai[selected_caption_type]
                    data_dict["selected_caption_type"] = selected_caption_type
                elif any(
                    caption_type in _AVAILABLE_QWEN_CAPTIONS for caption_type in decoded_captions_ai[caption_key].keys()
                ):
                    # qwen2p5_7b_v4 captions
                    selected_caption_type = random.choice(_AVAILABLE_QWEN_CAPTIONS)
                    data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                    data_dict["selected_caption_type"] = selected_caption_type
                elif caption_type == "cosmos_captioner_v1p1":
                    selected_caption_type = "caption_cosmos_captioner_image"
                    if decoded_captions_ai[caption_key].get(selected_caption_type, "") == "":
                        # xingqianx: a temporary skip as some data is imperfect.
                        return None  # type: ignore
                    data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                    data_dict["selected_caption_type"] = selected_caption_type
                elif caption_type == "cosmos_captioner_v1p1_structured_json":
                    # this is made by mistake, should be removed in future.
                    # it is used for cosmos_lab_image_v1_human_sft. Once we fix it, this should be removed.
                    selected_caption_type = "caption_cosmos_captioner_image_structured_json"
                    data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                    data_dict["selected_caption_type"] = selected_caption_type
                elif any(
                    caption_type in _AVAILABLE_CAPTIONS_V2 for caption_type in decoded_captions_ai[caption_key].keys()
                ):
                    # v2 captions
                    selected_caption_type = random.choice(_AVAILABLE_CAPTIONS_V2)
                    data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                    data_dict["selected_caption_type"] = selected_caption_type
                elif caption_type == "prompts":
                    data_dict["ai_caption"] = decoded_captions_ai["caption"]["prompt"]
                    data_dict["selected_caption_type"] = caption_type
                else:
                    assert caption_type == "ai_v3p1", f"Caption type {caption_type} not supported"
                    if decoded_captions_ai["had_parse_issue"]:
                        data_dict["ai_caption"] = decoded_captions_ai["captions"]["kosmos_2"]
                    else:
                        data_dict["ai_caption"] = decoded_captions_ai["captions"]["vfc"]
            else:  # use the designated captions
                # Validate that all specified caption types exist (except genplan types which are nested)
                for cap_type in train_on_captions:
                    if cap_type not in _AVAILABLE_CAPTIONS_GENPLAN:
                        assert cap_type in decoded_captions_ai[caption_key].keys(), (
                            f"Caption type {cap_type} not found in data"
                        )

                selected_caption_type = random.choice(train_on_captions)

                if selected_caption_type in _AVAILABLE_CAPTIONS_GENPLAN:
                    # Genplan captions are nested inside caption_long as a JSON string
                    caption_long_data = json.loads(decoded_captions_ai[caption_key]["caption_long"])
                    data_dict["ai_caption"] = caption_long_data[selected_caption_type]
                else:
                    data_dict["ai_caption"] = decoded_captions_ai[caption_key][selected_caption_type]
                data_dict["selected_caption_type"] = selected_caption_type

        except Exception as e:
            log.warning(
                f"TextTransform dataloader error: {data_dict['__url__']}, {data_dict['__key__']}\n error {e}",
                rank0_only=False,
            )
            return None

        del data_dict[captions_key]

        self._apply_caption_prefix(data_dict)
        return data_dict


class TextTransformForImageJsonCaption(Augmentor):
    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.json_field_dropout_rate = args.get("json_field_dropout_rate", 0.0) if args else 0.0

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transform without any embedding loading.
        This is useful for online computation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with camera attributes added
        """

        caption_type = self.args["caption_type"]
        captions_key = f"captions_{caption_type}"

        if "cosmos_captioner_v1p1_structured_json" in data_dict[captions_key]:
            # this is made by mistake, should be removed in future.
            # it is used for cosmos_lab_image_v1_human_sft. Once we fix it, this should be removed.
            selected_caption_type = "caption_cosmos_captioner_image_structured_json"
        else:
            selected_caption_type = "caption_cosmos_captioner_image"
        caption_json = data_dict[captions_key]["caption"].get(selected_caption_type, "")
        if caption_json == "":
            # xingqianx: a temporary skip as some text data is imperfect.
            return None  # type: ignore
        caption_json = json.loads(caption_json)

        # In some erraneous cases, the caption_json is a list
        if isinstance(caption_json, list):
            caption_json = caption_json[0]

        assert isinstance(caption_json, dict), (
            f"Caption json is not a dict: {caption_json}, url: {data_dict['__url__']}, key: {data_dict['__key__']}"
        )

        # Randomly dropout json keys during training
        if self.json_field_dropout_rate > 0:
            for key in list(caption_json.keys()):
                if random.random() < self.json_field_dropout_rate:
                    caption_json.pop(key)

        data_dict["ai_caption"] = caption_json
        del data_dict[captions_key]

        return data_dict
