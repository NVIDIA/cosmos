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

"""
This file is used to test the config of the cosmos3 vfm project.
It is used to verify the config is loadable.

To run the test, you can use the following command:
pytest -s configs/base/base_config_test.py
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from cosmos.utils.config import CheckpointConfig, JobConfig, ObjectStoreConfig
from cosmos.utils.config_helper import get_config_module, override


@pytest.fixture
def mock_checkpoint_config():
    """Create a mock checkpoint config for testing."""
    object_store = ObjectStoreConfig(
        enabled=False,
        bucket="test-bucket",
        credentials="test-credentials",
    )
    return CheckpointConfig(
        save_to_object_store=object_store,
        load_from_object_store=object_store,
        strict_resume=True,
        load_path=None,
        load_training_state=True,
    )


@pytest.fixture
def mock_job_config():
    """Create a mock job config for testing."""
    return JobConfig(
        project="test-project",
        group="test-group",
        name="test-job",
    )


@pytest.mark.L0
@pytest.mark.parametrize(
    "experiment_name",
    [
        "t2i_mot_exp001_009_qwen3_vl_2b_256res_frozen_llm",
    ],
)
def test_config_init_experiment_mot(experiment_name):
    """
    Parameterized test to verify config initialization for multiple experiments.
    PYTHONPATH=. torchrun --nproc_per_node=8 -m pytest -s configs/base/config_test_mot.py --L1
    """
    config_file = "configs/base/config.py"
    config_module = get_config_module(config_file)
    config = importlib.import_module(config_module).make_config()
    config = override(
        config,
        [
            "--",
            f"experiment={experiment_name}",
        ],
    )


def _build_validate_mocks(
    vlm_load_pretrained: bool,
    diffusion_load_weights: bool,
    mock_checkpoint_config,
    mock_job_config,
):
    """Build the (omni_config, root_config, vlm_config, diffusion_expert_config) mocks
    used by the OmniMoTModelConfig.validate tests below.

    The vlm_config and diffusion_expert_config mocks are wired onto BOTH the
    OmniMoTModelConfig instance (read side via self.*) AND the root Config
    instance (write side via root_config.model.config.*) so a single assertion
    on each mock observes both reads and writes.
    """
    from configs.base.config import Config
    from configs.base.defaults.model_config import OmniMoTModelConfig

    vlm_config = MagicMock()
    vlm_config.load_pretrained = vlm_load_pretrained

    diffusion_expert_config = MagicMock()
    diffusion_expert_config.load_weights_from_pretrained = diffusion_load_weights

    omni_config = MagicMock(spec=OmniMoTModelConfig)
    omni_config.vlm_config = vlm_config
    omni_config.diffusion_expert_config = diffusion_expert_config

    model_config = MagicMock()
    model_config.vlm_config = vlm_config
    model_config.diffusion_expert_config = diffusion_expert_config

    root_config = MagicMock(spec=Config)
    root_config.model = MagicMock()
    root_config.model.config = model_config
    root_config.checkpoint = mock_checkpoint_config
    root_config.job = mock_job_config

    return omni_config, root_config, vlm_config, diffusion_expert_config


@pytest.mark.L0
class TestOmniMoTValidateSkipPretrained:
    """Tests for OmniMoTModelConfig.validate(): disable pretrained loading when
    a training checkpoint exists or load_path is set.

    The Config.validate() entrypoint composes model.config from a DictConfig,
    materializes it via OmegaConf.to_object, and dispatches into the typed
    OmniMoTModelConfig.validate(self, root_config). These tests drive that
    inner dispatch directly to keep the unit small; the full polymorphic
    dispatch is covered separately in test_validate_dispatch.py.
    """

    _PATCH_TARGET = "cosmos.checkpoint.dcp.DistributedCheckpointer"

    def test_keeps_pretrained_when_no_checkpoint(self, mock_checkpoint_config, mock_job_config):
        """No checkpoint, no load_path: both flags stay True."""
        from configs.base.defaults.model_config import OmniMoTModelConfig

        omni_config, root_config, vlm_config, diffusion_expert_config = _build_validate_mocks(
            vlm_load_pretrained=True,
            diffusion_load_weights=True,
            mock_checkpoint_config=mock_checkpoint_config,
            mock_job_config=mock_job_config,
        )

        mock_checkpointer = MagicMock()
        mock_checkpointer._read_latest_checkpoint_file.return_value = None
        mock_checkpointer.load_path = None

        with patch(self._PATCH_TARGET, return_value=mock_checkpointer):
            OmniMoTModelConfig.validate(omni_config, root_config)

        assert vlm_config.load_pretrained is True
        assert diffusion_expert_config.load_weights_from_pretrained is True

    def test_skips_pretrained_when_checkpoint_exists(self, mock_checkpoint_config, mock_job_config):
        """Latest-checkpoint file exists: VLM pretrained loading disabled; diffusion-expert untouched."""
        from configs.base.defaults.model_config import OmniMoTModelConfig

        omni_config, root_config, vlm_config, diffusion_expert_config = _build_validate_mocks(
            vlm_load_pretrained=True,
            diffusion_load_weights=True,
            mock_checkpoint_config=mock_checkpoint_config,
            mock_job_config=mock_job_config,
        )

        mock_checkpointer = MagicMock()
        mock_checkpointer._read_latest_checkpoint_file.return_value = "/prev/checkpoint/010030"
        mock_checkpointer.load_path = None

        with patch(self._PATCH_TARGET, return_value=mock_checkpointer):
            OmniMoTModelConfig.validate(omni_config, root_config)

        assert vlm_config.load_pretrained is False
        assert diffusion_expert_config.load_weights_from_pretrained is True

    def test_skips_diffusion_when_load_path_exists(self, mock_checkpoint_config, mock_job_config):
        """load_path set, no checkpoint: VLM pretrained kept; diffusion-expert disabled."""
        from configs.base.defaults.model_config import OmniMoTModelConfig

        omni_config, root_config, vlm_config, diffusion_expert_config = _build_validate_mocks(
            vlm_load_pretrained=True,
            diffusion_load_weights=True,
            mock_checkpoint_config=mock_checkpoint_config,
            mock_job_config=mock_job_config,
        )

        mock_checkpointer = MagicMock()
        mock_checkpointer._read_latest_checkpoint_file.return_value = None
        mock_checkpointer.load_path = "/prev/checkpoint"

        with patch(self._PATCH_TARGET, return_value=mock_checkpointer):
            OmniMoTModelConfig.validate(omni_config, root_config)

        assert vlm_config.load_pretrained is True
        assert diffusion_expert_config.load_weights_from_pretrained is False

    def test_does_nothing_when_already_false(self, mock_checkpoint_config, mock_job_config):
        """Both flags already False: validate() early-returns without constructing a checkpointer."""
        from configs.base.defaults.model_config import OmniMoTModelConfig

        omni_config, root_config, vlm_config, diffusion_expert_config = _build_validate_mocks(
            vlm_load_pretrained=False,
            diffusion_load_weights=False,
            mock_checkpoint_config=mock_checkpoint_config,
            mock_job_config=mock_job_config,
        )

        with patch(self._PATCH_TARGET) as patched_checkpointer:
            OmniMoTModelConfig.validate(omni_config, root_config)
            patched_checkpointer.assert_not_called()

        assert vlm_config.load_pretrained is False
        assert diffusion_expert_config.load_weights_from_pretrained is False
