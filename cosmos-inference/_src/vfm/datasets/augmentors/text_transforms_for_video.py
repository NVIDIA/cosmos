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


class TextTransformForVideo(Augmentor):
    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)

        # our caption is saved in json with format: {"<key>": "xxx", "<caption_windows_key1>": [{"start_frame": x, "end_frame": x, "<caption_type>": xxx}, ...], "<caption_windows_key2>": [{"start_frame":...]}
        # our t5 embedding is saved in pickle with format: [{"<embedding_caption_type1>": array1, "<embedding_caption_type2>": array2}, ...]
        self.captions_key: str = args[
            "captions_key"
        ]  # s3 folder that saves the captions; this get mapped to the key in data_dict to fetch the caption field
        self.embeddings_key: Optional[str] = args[
            "embeddings_key"
        ]  # s3 folder that saves the embeddings; this get mapped to the key in data_dict to fetch the embedding field
        self.caption_windows_key: str = args[
            "caption_windows_key"
        ]  # key to get the caption windows from the caption field
        self.caption_type: str = args["caption_type"]  # key of caption type to fetch the caption from caption windows

        self._load_embeddings = self.embeddings_key is not None

        if not self._load_embeddings:
            # In this case, we don't load the embeddings
            log.info("No embeddings key provided, we will not load embeddings")
            self.embedding_caption_type = None
            self.t5_tokens_num = None
            self.is_mask_all_ones = None
            self.embedding_style_mapping = None
        else:
            self.embedding_caption_type: str = args[
                "embedding_caption_type"
            ]  # key to get the embedding of a particular caption type from the embedding field
            self.t5_tokens_num = args["t5_tokens"]["num"]  # number of tokens we cap after padding
            self.is_mask_all_ones = args["is_mask_all_ones"]  # if true, set mask for t5 to all ones

            self.embedding_style_mapping = {
                "long": self.embedding_caption_type,
                "short": f"{self.embedding_caption_type}_short",
                "medium": f"{self.embedding_caption_type}_medium",
                "user": f"{self.embedding_caption_type}_user",
            }

        self.caption_probs: dict[str, float] = args[
            "caption_probs"
        ]  # probabilities for user/short/medium/long captions
        self.caption_style_mapping = {
            "long": self.caption_type,
            "short": f"{self.caption_type}_short",
            "medium": f"{self.caption_type}_medium",
            "user": f"{self.caption_type}_user",
        }
        assert self.caption_probs.keys() == self.caption_style_mapping.keys(), (
            "The keys for caption_probs, caption_style_mapping, and embedding_style_mapping should match"
        )

        if self._load_embeddings:
            assert self.caption_style_mapping.keys() == self.embedding_style_mapping.keys(), (
                "The keys for caption_style_mapping and embedding_style_mapping should match"
            )

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """

        try:
            windows = data_dict[self.captions_key][self.caption_windows_key]
            n_windows = len(windows)
            chunk_index = data_dict["chunk_index"]

            if chunk_index == n_windows:
                # This will only happen when the number of captions does not match number of chunks due to re-transcoding the videos.
                log.warning(
                    f"TextTransform dataloader error: Found {data_dict['n_orig_video_frames']} in video but captioning is done with videos of {windows[-1]['end_frame']} frames. This mismatch is due to video re-transcoding.",
                    rank0_only=False,
                )
                chunk_index -= 1

            selected_caption_window = windows[chunk_index]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, chunk_index: {data_dict['chunk_index']}\n error {e}",
                rank0_only=False,
            )
            return None

        sampled_caption_style = None
        try:
            available_caption_styles = []
            for k in selected_caption_window.keys():
                caption_style = k.replace(self.caption_type, "").replace("_", "")
                if caption_style == "":  # it is long caption by default
                    available_caption_styles.append("long")
                elif caption_style in self.caption_style_mapping:
                    available_caption_styles.append(caption_style)
                else:
                    assert caption_style in ["startframe", "endframe"], f"Unsupported caption_type {caption_style}"

            probabilities_for_available_caption_styles = {
                k: v for k, v in self.caption_probs.items() if k in available_caption_styles
            }
            sampled_caption_style = random.choices(
                list(probabilities_for_available_caption_styles),
                weights=probabilities_for_available_caption_styles.values(),
            )[0]
            data_dict["ai_caption"] = selected_caption_window[self.caption_style_mapping[sampled_caption_style]]
        except Exception as e:
            log.warning(
                f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, selected_caption_window: {selected_caption_window}\n error {e}",
                rank0_only=False,
            )
            return None
        if data_dict["ai_caption"] == "":
            log.warning(
                f"TextTransform dataloader error -- empty caption! url: {data_dict['__url__']}, key: {data_dict['__key__']}, selected_caption_window: {selected_caption_window}",
                rank0_only=False,
            )
            return None

        assert data_dict["ai_caption"] is not None and sampled_caption_style is not None
        data_dict["sampled_caption_style"] = sampled_caption_style

        del data_dict[self.captions_key]  # delete the field as we have extracted ai_caption from it

        if self._load_embeddings:
            ai_caption_embedding_data = data_dict[self.embeddings_key]
            try:
                if self.embedding_caption_type == "vila_caption":
                    t5_embedding = ai_caption_embedding_data[chunk_index]
                else:
                    t5_embedding = ai_caption_embedding_data[chunk_index][
                        self.embedding_style_mapping[sampled_caption_style]
                    ]
            except Exception as e:
                log.warning(
                    f"TextTransform dataloader error -- url: {data_dict['__url__']}, key: {data_dict['__key__']}, chunk_index: {data_dict['chunk_index']}, n embeddings: {len(ai_caption_embedding_data)}, n captions: {n_windows} \n error {e}",
                    rank0_only=False,
                )
                return None
            out_t5, out_t5_mask = pad_and_resize(
                t5_embedding,
                self.t5_tokens_num,
                is_mask_all_ones=self.is_mask_all_ones,
            )
            data_dict["t5_text_embeddings"] = out_t5
            data_dict["t5_text_mask"] = out_t5_mask
            del data_dict[self.embeddings_key]  # delete the field as we have extracted t5 embedding from it

        return data_dict


class TextTransformForVideoWithFullFrames(Augmentor):
    """
    Pair use with VideoParsingWithFullFrames to get the full frames of the video.
    The caption is assumed to be for the entire video frames, rather than TextTransformForVideo
    which assumes captions are for a specific chunk of frames.

    Audio captions are handled separately by AudioCaptionAppender, which appends
    audio descriptions to the video caption after this augmentor runs.
    """

    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) == 3, "TextTransformForVideoWithFullFrames augmentor only supports three input keys"
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]
        self.sequence_plan_key = input_keys[2]
        self.args = args
        self.keep_metas = args.get("keep_metas", False) if args else False
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

    @staticmethod
    def _resolve_multi_chunk_caption(raw_caption: str) -> str:
        """Resolve a caption that may be in multi-chunk JSON format.

        Multi-chunk captions are JSON strings encoding a dict of chunks, e.g.:
            {"chunk_0_300": {"caption": "...", "start_frame": 0, "end_frame": 300}, ...}
        When detected, a chunk is randomly selected and its "caption" text returned.
        Plain string captions are returned unchanged.
        """
        if not isinstance(raw_caption, str):
            return raw_caption
        try:
            parsed = json.loads(raw_caption)
        except (json.JSONDecodeError, TypeError):
            return raw_caption
        if not isinstance(parsed, dict) or len(parsed) == 0:
            return raw_caption
        chunk = random.choice(list(parsed.values()))
        if isinstance(chunk, dict) and "caption" in chunk:
            return chunk["caption"]
        return raw_caption

    def __call__(self, data_dict: dict) -> dict:
        r"""Performs text transformation.

        Samples a video caption from metadata based on caption_config ratios.
        Supports both plain-string captions and multi-chunk JSON captions
        (randomly selects one chunk when multiple chunks are present).
        Audio captions are handled separately by AudioCaptionAppender.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """
        caption_config = self.args["caption_config"]
        meta_dict = data_dict[self.meta_key]

        for caption_type in caption_config:
            assert caption_type in meta_dict, (
                f"Caption type {caption_type} not found in meta_dict (keys = {meta_dict.keys()})"
            )

        # First check if we are doing image to world or video to world
        if self.sequence_plan_key in data_dict:
            sequence_plan = data_dict[self.sequence_plan_key]
            conditioning_frame_indexes_vision = sequence_plan.condition_frame_indexes_vision
            if len(conditioning_frame_indexes_vision) > 0:
                sampled_caption = self._resolve_multi_chunk_caption(meta_dict["caption_temporal"])
                data_dict["ai_caption"] = sampled_caption
                data_dict["sampled_caption_style"] = "caption_temporal"

                self._apply_caption_prefix(data_dict)
                if not self.keep_metas:
                    del data_dict[self.meta_key]
                return data_dict

        # Text-to-world: sample from short, medium, long captions
        caption_keys = list(caption_config.keys())
        caption_ratios = [caption_config[k]["ratio"] for k in caption_keys]
        sampled_caption_type = random.choices(caption_keys, weights=caption_ratios, k=1)[0]
        data_dict["ai_caption"] = self._resolve_multi_chunk_caption(meta_dict[sampled_caption_type])
        data_dict["sampled_caption_style"] = sampled_caption_type

        self._apply_caption_prefix(data_dict)

        # Clean up - delete the caption fields that were sampled from
        for caption_type in caption_config.keys():
            if caption_type in meta_dict:
                del meta_dict[caption_type]

        # Delete metas unless keep_metas=True (set when AudioCaptionAppender runs downstream)
        if not self.keep_metas:
            del data_dict[self.meta_key]

        return data_dict


class TextTransformForVideoTransferFullFrames(Augmentor):
    """Read structured captions for the full-frame transfer pipeline.

    Two-level lookup:

    1. A caption-source key is sampled (with weights) from ``caption_config``.
       This key identifies the WebDataset folder / metadata field whose value
       is a dict of annotations (e.g.
       ``"structured_captions_qwen3-vl-8b-lora-v1.5-merged"``). The sampled
       value is looked up first in ``data_dict`` (top-level) and then in
       ``meta_dict``.
    2. Inside that caption dict the field ``caption_structured`` is hardcoded
       as the JSON-encoded chunked annotation, of the form
       ``{"chunk_0_300": {"caption": "<json-encoded structured payload>",
       "start_frame": ..., "end_frame": ...}}``.

    The full-frame pipeline always decodes from the start of the video, so the
    first chunk is always selected and its inner JSON-encoded structured payload
    is parsed back into a dict before being serialized as ``ai_caption``.
    """

    CAPTION_FIELD = "caption_structured"

    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) >= 1, "TextTransformForVideoTransferFullFrames requires a metadata input key"
        self.meta_key = input_keys[0]
        self.args = args or {}
        self.caption_prefix = self.args.get("caption_prefix", None)
        self.keep_metas = self.args.get("keep_metas", False)
        self.caption_options = self._normalize_caption_config(self.args["caption_config"])

    @staticmethod
    def _normalize_caption_config(caption_config: dict | list) -> list[tuple[str, float]]:
        if isinstance(caption_config, dict):
            options: list[tuple[str, float]] = []
            for caption_key, config in caption_config.items():
                ratio = config.get("ratio", 1.0) if isinstance(config, dict) else float(config)
                options.append((caption_key, float(ratio)))
            return options

        options = []
        for item in caption_config:
            if isinstance(item, str):
                options.append((item, 1.0))
            elif isinstance(item, dict):
                caption_key = item.get("key") or item.get("caption_key") or item.get("caption_type") or item.get("name")
                if caption_key is None:
                    raise ValueError(f"Caption config entry is missing a caption key: {item}")
                options.append((caption_key, float(item.get("ratio", 1.0))))
            else:
                caption_key, ratio = item
                options.append((caption_key, float(ratio)))
        return options

    def _apply_caption_prefix(self, data_dict: dict) -> None:
        if not self.caption_prefix or not isinstance(data_dict.get("ai_caption"), str):
            return
        data_dict["ai_caption"] = self.caption_prefix + " " + data_dict["ai_caption"].lstrip()

    def _lookup_caption_dict(self, data_dict: dict, meta_dict: dict | None, caption_key: str) -> dict | None:
        candidate = data_dict.get(caption_key)
        if candidate is None and isinstance(meta_dict, dict):
            candidate = meta_dict.get(caption_key)
        if isinstance(candidate, dict):
            return candidate
        return None

    def __call__(self, data_dict: dict) -> dict | None:
        meta_dict = data_dict.get(self.meta_key)

        available_options: list[tuple[str, float]] = []
        for key, ratio in self.caption_options:
            if ratio <= 0:
                continue
            if self._lookup_caption_dict(data_dict, meta_dict, key) is not None:
                available_options.append((key, ratio))

        if not available_options:
            log.warning(
                f"TextTransformForVideoTransferFullFrames: none of the configured caption keys "
                f"{[key for key, _ in self.caption_options]} hold a caption dict in metadata/sample keys. "
                f"url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}",
                rank0_only=False,
            )
            return None

        sampled_caption_key = random.choices(
            [key for key, _ in available_options],
            weights=[ratio for _, ratio in available_options],
            k=1,
        )[0]
        caption_dict = self._lookup_caption_dict(data_dict, meta_dict, sampled_caption_key)
        if caption_dict is None or self.CAPTION_FIELD not in caption_dict:
            log.warning(
                f"TextTransformForVideoTransferFullFrames: caption dict for {sampled_caption_key} is missing the "
                f"hardcoded {self.CAPTION_FIELD} field. url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}",
                rank0_only=False,
            )
            return None

        try:
            chunks = json.loads(caption_dict[self.CAPTION_FIELD])
            first_chunk = next(iter(chunks.values()))
            structured = json.loads(first_chunk["caption"])
        except Exception as e:
            log.warning(
                f"TextTransformForVideoTransferFullFrames: failed to decode {sampled_caption_key}.{self.CAPTION_FIELD}. "
                f"url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}, error: {e}",
                rank0_only=False,
            )
            return None

        data_dict["ai_caption"] = json.dumps(structured)
        data_dict["sampled_caption_style"] = sampled_caption_key
        self._apply_caption_prefix(data_dict)

        if not self.keep_metas:
            data_dict.pop(self.meta_key, None)
        for caption_key, _ in self.caption_options:
            if caption_key in data_dict:
                del data_dict[caption_key]
        return data_dict


class TextTransformForVideoJsonCaption(Augmentor):
    """
    This augmentor is used to transform the caption from a json string to a string.
    The caption is assumed to be in the format of a json string.
    The caption is then transformed to a string by converting the json string to a dictionary and then converting the dictionary to a string.
    The caption is then returned as a string.

    When ``meta_dict["caption_audio"]`` is present and non-empty, its contents
    are injected into the caption dict under the ``"audio_description"`` key.
    This happens after the JSON field dropout so the audio description is
    preserved whenever upstream metadata provides it.
    """

    def __init__(self, input_keys: dict, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        assert len(input_keys) >= 2, (
            "TextTransformForVideoJsonCaption augmentor requires at least two input keys: [meta_key, video_key]"
        )
        self.meta_key = input_keys[0]
        self.video_key = input_keys[1]
        self.args = args
        self.keep_metas = args.get("keep_metas", False) if args else False

    def __call__(self, data_dict: dict) -> dict | None:
        r"""Performs text transformation.

        Parses the per-chunk caption JSON, randomly samples one chunk, and writes
        the chunk's frame range into ``data_dict`` so a downstream
        ``VideoParsingChunkedFrames`` can decode only that frame range. When a
        non-empty ``caption_audio`` field is present in the metadata, it is
        injected into the caption dict under the ``"audio_description"`` key.

        Args:
            data_dict (dict): Input data dict
        Returns:
            data_dict (dict): Output dict with captions and t5 embeddings added
        """
        caption_config = self.args["caption_config"]
        json_field_dropout_rate = caption_config["json_field_dropout_rate"]

        try:
            meta_dict = data_dict[self.meta_key]
            caption = json.loads(meta_dict["caption"])

            # Contents of caption
            # caption = {
            #    "chunk_0_300": {
            #        "caption": "...",
            #        "start_frame": 0,
            #        "end_frame": 300,
            #    },
            #    "chunk_300_435": {
            #        "caption": "...",
            #        "start_frame": 300,
            #        "end_frame": 435,
            #    },
            # }
            chunk_keys = list(caption.keys())
            if len(chunk_keys) == 0:
                log.warning(
                    f"TextTransformForVideoJsonCaption: empty caption dict. url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}",
                    rank0_only=False,
                )
                return None

            sampled_key = random.choice(chunk_keys)
            sampled_chunk = caption[sampled_key]

            data_dict["chunk_index"] = chunk_keys.index(sampled_key)
            data_dict["chunk_start_frame"] = int(sampled_chunk["start_frame"])
            data_dict["chunk_end_frame"] = int(sampled_chunk["end_frame"])

            caption_json = json.loads(sampled_chunk["caption"])
        except Exception as e:
            log.warning(
                f"TextTransformForVideoJsonCaption dataloader error -- url: {data_dict.get('__url__')}, key: {data_dict.get('__key__')}\n error {e}",
                rank0_only=False,
            )
            return None

        # Randomly dropout json keys during training
        if json_field_dropout_rate > 0:
            for key in list(caption_json.keys()):
                if random.random() < json_field_dropout_rate:
                    caption_json.pop(key)

        # Inject audio caption from metas as a new field when available. Added after the field
        # dropout above so it is preserved whenever upstream metadata provides it.
        audio_caption = meta_dict.get("caption_audio")
        if isinstance(audio_caption, str) and len(audio_caption) > 0:
            caption_json["audio_description"] = audio_caption

        data_dict["ai_caption"] = caption_json

        # Delete metas unless keep_metas=True (set when downstream augmentors still need them,
        # e.g. VideoParsingChunkedFrames needs framerate/width/height/nb_frames).
        if not self.keep_metas:
            del data_dict[self.meta_key]

        return data_dict
