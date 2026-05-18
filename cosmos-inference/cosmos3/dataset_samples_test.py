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

import json

import pytest

from cosmos3.dataset_samples import _normalize_caption


@pytest.mark.L0
def test_normalize_caption_string_unchanged() -> None:
    sample = {"ai_caption": "Pick up the cup."}
    assert _normalize_caption(sample) == "Pick up the cup."
    assert sample["ai_caption"] == "Pick up the cup."


@pytest.mark.L0
def test_normalize_caption_dict_serialized_and_written_back() -> None:
    caption = {
        "cinematography": {"framing": "third-person view"},
        "actions": [{"time": "0:00-0:02", "description": "Open the drawer."}],
        "duration": "2s",
        "fps": 8.0,
        "resolution": {"H": 192, "W": 320},
        "aspect_ratio": "16,9",
    }
    sample = {"ai_caption": caption}
    result = _normalize_caption(sample)
    assert json.loads(result) == caption
    assert sample["ai_caption"] == result


@pytest.mark.L0
def test_normalize_caption_missing_returns_empty() -> None:
    assert _normalize_caption({}) == ""


@pytest.mark.L0
def test_normalize_caption_raises_on_non_str_non_dict() -> None:
    with pytest.raises(TypeError, match="ai_caption must be str or dict"):
        _normalize_caption({"ai_caption": None})
    with pytest.raises(TypeError, match="ai_caption must be str or dict"):
        _normalize_caption({"ai_caption": 42})
