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

import numpy as np

from cosmos3.common.args import GuardrailArgs
from cosmos3.common.inference import GuardrailRunners
from cosmos3._src.imaginaire.auxiliary.guardrail.common import presets


def test_guardrail_runners():
    guardrail_args = GuardrailArgs(guardrails=True, offload_guardrail_models=False)
    runners = GuardrailRunners.create(guardrail_args)
    assert runners.text is not None
    assert runners.video is not None

    assert presets.run_text_guardrail("test", runners.text)
    assert not presets.run_text_guardrail("Tesla Cybertruck", runners.text)

    frames_thwc = np.random.randint(0, 255, (1, 16, 16, 3), dtype=np.uint8)
    clean_frames_thwc = presets.run_video_guardrail(frames_thwc, runners.video)
    assert clean_frames_thwc is not None
    np.testing.assert_allclose(frames_thwc, clean_frames_thwc)
