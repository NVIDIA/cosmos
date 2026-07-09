# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

if sys.version_info >= (3, 10):
    print("Python version is above 3.10, patching the collections module.")
    import collections
    import collections.abc

    for type_name in collections.abc.__all__:
        setattr(collections, type_name, getattr(collections.abc, type_name))

from attrdict import AttrDict
from transformers import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))

class UrsaConfig(PretrainedConfig):
    model_type = "ursa"
    is_composition = False
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    text_config: PretrainedConfig

    def __init__(
        self,
        vision_config=None,
        aligner_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        **kwargs,
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        if vision_config is None:
            vision_config = VisionConfig()
            vision_config.cls = "HybridVisionTower"
            vision_config.params = {
                                        "concat_type": "tuple",
                                        "high_res_cfg": {
                                            "ckpt_path": "",
                                            "image_size": 1024,
                                            "model_name": "sam_b_downsample",
                                            "output_dim": 1024,
                                            "pixel_mean": [
                                            0.48145466,
                                            0.4578275,
                                            0.40821073
                                            ],
                                            "pixel_std": [
                                            0.26862954,
                                            0.26130258,
                                            0.27577711
                                            ],
                                            "select_feature": "same",
                                            "select_layer": -1
                                        },
                                        "low_res_cfg": {
                                            "ckpt_path": "",
                                            "image_size": 384,
                                            "model_name": "siglip_large_patch16_384",
                                            "output_dim": 1024,
                                            "pixel_mean": [
                                            0.5,
                                            0.5,
                                            0.5
                                            ],
                                            "pixel_std": [
                                            0.5,
                                            0.5,
                                            0.5
                                            ],
                                            "select_feature": "same",
                                            "select_layer": -1
                                        }
                                    }
        self.vision_config = vision_config
        self.aligner_config = aligner_config
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(**kwargs)
