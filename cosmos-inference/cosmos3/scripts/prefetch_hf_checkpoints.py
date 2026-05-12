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

"""Pre-download all HF models used by the cosmos3 CI."""

import itertools

from cosmos3.common.checkpoints import register_checkpoints
from cosmos3._src.imaginaire.utils.checkpoint_db import CheckpointDirHf


def prefetch_all() -> None:
    register_checkpoints()

    from cosmos3.args import _CHECKPOINTS_EA, _CHECKPOINTS_EXPERIMENTAL

    for cfg in itertools.chain(_CHECKPOINTS_EXPERIMENTAL.values(), _CHECKPOINTS_EA.values()):
        cfg.hf.download()

    from cosmos3._src.imaginaire.utils.checkpoint_db import _CHECKPOINTS

    for cfg in _CHECKPOINTS.values():
        cfg.hf.download()

    for repo in [
        # 'cosmos3._src.imaginaire.auxiliary.guardrail.llamaGuard3.llamaGuard3',
        "meta-llama/Llama-Guard-3-8B",
        # 'cosmos3._src.imaginaire.auxiliary.guardrail.qwen3guard.qwen3guard',
        "Qwen/Qwen3Guard-Gen-0.6B",
        # 'cosmos3._src.imaginaire.auxiliary.guardrail.video_content_safety_filter.vision_encoder',
        "google/siglip-so400m-patch14-384",
    ]:
        CheckpointDirHf(repository=repo, revision="main").download()


def main():
    prefetch_all()


if __name__ == "__main__":
    main()
