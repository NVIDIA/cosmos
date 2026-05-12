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

import contextlib
from pathlib import Path

import attrs
import diffusers
import hydra
import omegaconf
import torch.distributed.checkpoint as dcp
import transformers
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from typing_extensions import TYPE_CHECKING, assert_never

from cosmos3.common.args import CheckpointType
from cosmos3.common.checkpoints import register_checkpoints
from cosmos3.common.config import structure_config, undo_config_dict_replacements, unstructure_config
from cosmos3._src.imaginaire.flags import SMOKE
from cosmos3._src.imaginaire.lazy_config.lazy_call import LazyCall
from cosmos3._src.imaginaire.utils import misc
from cosmos3._src.vfm.configs.base.defaults.model_config import ParallelismConfig

if TYPE_CHECKING:
    from cosmos3._src.vfm.models.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
    from cosmos3._src.vfm.models.omni_mot_model import OmniMoTModel
    from cosmos3._src.vfm.tokenizers.wan2pt2_vae_4x16x16 import WanVAE_


_ROOT_DIR = Path(__file__).parents[1].absolute()


class Cosmos3OmniConfig(transformers.PretrainedConfig):
    model_type = "cosmos3_omni"

    def __init__(self, model: dict | None = None, **kwargs):
        if model is not None:
            model = undo_config_dict_replacements(model)
        self.model = model or {}

        super().__init__(**kwargs)

        self.auto_map = {
            "AutoConfig": "cosmos3.model.Cosmos3OmniConfig",
            "AutoModel": "cosmos3.model.Cosmos3OmniModel",
        }

    @property
    def parallelism(self) -> dict:
        return self.model.get("config", {}).get("parallelism", {})

    @parallelism.setter
    def parallelism(self, value: dict | None):
        if value is None:
            return
        self.model.setdefault("config", {})["parallelism"] = unstructure_config(LazyCall(ParallelismConfig)(**value))


class Cosmos3OmniModel(transformers.PreTrainedModel):
    config_class = Cosmos3OmniConfig  # type: ignore

    def __init__(self, config: Cosmos3OmniConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.before_load_model()
        model_dict: "OmniMoTModel" = structure_config(config.model, omegaconf.DictConfig)


        if SMOKE:
            vlm_dict = model_dict.config.vlm_config.model_instance
            assert vlm_dict is not None
            with omegaconf.open_dict(vlm_dict.config):
                vlm_dict.config.num_hidden_layers = vlm_dict.config.num_window_layers = 2

        # The model loads some files by relative path 'cosmos3/...'
        with contextlib.chdir(_ROOT_DIR):
            self.model: "OmniMoTModel" = hydra.utils.instantiate(model_dict)
        self.after_load_model(self.model)

    @classmethod
    def from_pretrained_dcp(
        cls,
        checkpoint_path: Path,
        config: Cosmos3OmniConfig | None = None,
        parallelism_config: ParallelismConfig | None = None,
    ):
        if parallelism_config is None:
            parallelism_config = ParallelismConfig()
        if config is None:
            config = Cosmos3OmniConfig.from_pretrained(checkpoint_path, parallelism=attrs.asdict(parallelism_config))
        model = cls(config)
        checkpoint_type = CheckpointType.from_path(checkpoint_path)
        match checkpoint_type:
            case CheckpointType.DCP:
                state_dict = get_model_state_dict(model.model)
                storage_reader = FileSystemReader(str(checkpoint_path))
            case CheckpointType.HF:
                is_diffusers = next(checkpoint_path.glob("diffusion_pytorch_model.*"), None) is not None
                if is_diffusers:
                    state_dict = get_model_state_dict(model.model)
                else:
                    state_dict = get_model_state_dict(model)
                storage_reader = HuggingFaceStorageReader(str(checkpoint_path))
            case _:
                assert_never(checkpoint_type)
        dcp.load(state_dict=state_dict, storage_reader=storage_reader)
        return model

    @classmethod
    def before_load_model(cls):
        # Disable duck shapes, which triggers recompile.
        misc.set_torch_compile_options(use_duck_shape=False)

        register_checkpoints()

    @classmethod
    def after_load_model(cls, model: "OmniMoTModel"):
        pass


class Cosmos3OmniTransformer(diffusers.pipelines.pipeline_utils.ModelMixin, diffusers.ConfigMixin):
    @diffusers.configuration_utils.register_to_config
    def __init__(self, model: dict):
        super().__init__()

        self.net: "Cosmos3VFMNetwork | None" = None

    def forward(self, *args, **kwargs):
        pass


class Cosmos3OmniVisionTokenizer(diffusers.pipelines.pipeline_utils.ModelMixin, diffusers.ConfigMixin):
    @diffusers.configuration_utils.register_to_config
    def __init__(self, model: dict):
        super().__init__()

        self.model: "WanVAE_ | None" = None

    def forward(self, *args, **kwargs):
        pass


class Cosmos3OmniDiffusersPipeline(diffusers.pipelines.pipeline_utils.DiffusionPipeline):
    def __init__(
        self,
        transformer: Cosmos3OmniTransformer,
        text_tokenizer: transformers.PreTrainedTokenizerBase,
        vision_tokenizer: Cosmos3OmniVisionTokenizer,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, text_tokenizer=text_tokenizer, vision_tokenizer=vision_tokenizer)

    def __call__(self):
        pass
