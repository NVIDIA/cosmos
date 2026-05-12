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

"""Hydra CLI."""

import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig

from cosmos3.common.config import CONFIG_DIR


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="base_config")
def main(cfg: omegaconf.DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    print(f"Config saved to: '{output_dir}/.hydra/config.yaml'")


if __name__ == "__main__":
    main()
