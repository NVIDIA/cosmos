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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cosmos.data.vfm.action.json_formatter import ActionPromptJsonFormatter
from cosmos.data.vfm.action.transforms import ActionTransformPipeline
from cosmos.data.vfm.augmentors.duration_fps_text_timestamps import DurationFPSTextTimeStamps
from cosmos.data.vfm.augmentors.resolution_text_info import ResolutionTextInfo


@pytest.mark.L0
def test_action_prompt_json_formatter_builds_requested_structure() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 12, 480, 640)  # [C,T,H,W]
    action = torch.zeros(11, 7)  # [T,D]
    image_size = torch.tensor([480, 640, 480, 640])  # [4]
    fps = torch.tensor(24)  # []
    idle_frames = torch.tensor(2)  # []
    data_dict = {
        "ai_caption": "Pick up the cup",
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
        "viewpoint": "concat_view",
        "additional_view_description": "The top row is the wrist camera and the bottom row is the scene camera.",
        "idle_frames": idle_frames,
    }

    result = formatter(data_dict)

    prompt = result["ai_caption"]
    assert list(prompt.keys()) == ["cinematography", "actions", "duration", "fps", "resolution", "aspect_ratio"]
    assert list(prompt["actions"][0].keys()) == ["time", "description", "idle_frame"]
    assert prompt == {
        "cinematography": {
            "framing": (
                "This video contains concatenated views from multiple camera perspectives. "
                "The top row is the wrist camera and the bottom row is the scene camera."
            )
        },
        "actions": [
            {
                "time": "0:00-0:00",
                "description": "Pick up the cup.",
                "idle_frame": "2 out of 11.",
            }
        ],
        "duration": "0s",
        "fps": 24.0,
        "resolution": {"H": 480, "W": 640},
        "aspect_ratio": "4,3",
    }
    assert "additional_view_description" not in result


@pytest.mark.L0
def test_action_prompt_json_formatter_drops_empty_fields() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 12, 480, 640)  # [C,T,H,W]
    action = torch.zeros(11, 7)  # [T,D]
    image_size = torch.tensor([480, 640, 480, 640])  # [4]
    fps = torch.tensor(24)  # []
    data_dict = {
        "ai_caption": "Pick up the cup.",
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
        "viewpoint": "third_person_view",
    }

    result = formatter(data_dict)

    assert result["ai_caption"]["actions"] == [
        {
            "time": "0:00-0:00",
            "description": "Pick up the cup.",
        }
    ]


@pytest.mark.L0
def test_action_prompt_json_formatter_drops_empty_viewpoint() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 12, 480, 640)  # [C,T,H,W]
    action = torch.zeros(11, 7)  # [T,D]
    image_size = torch.tensor([480, 640, 480, 640])  # [4]
    fps = torch.tensor(24)  # []
    data_dict = {
        "ai_caption": "Pick up the cup.",
        "video": video,
        "action": action,
        "conditioning_fps": fps,
        "image_size": image_size,
    }

    result = formatter(data_dict)

    assert "cinematography" not in result["ai_caption"]


@pytest.mark.L0
def test_action_transform_pipeline_json_prompt_toggle() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        format_prompt_as_json=True,
    )
    video = torch.zeros(3, 17, 192, 320)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "policy",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    prompt = result["ai_caption"]
    assert isinstance(prompt, dict)
    assert list(prompt.keys()) == ["cinematography", "actions", "duration", "fps", "resolution", "aspect_ratio"]
    assert list(prompt["actions"][0].keys()) == ["time", "description", "idle_frame"]
    assert prompt["cinematography"] == {
        "framing": "This video is captured from a third-person perspective looking towards the agent from the front."
    }
    assert prompt["actions"] == [
        {
            "time": "0:00-0:02",
            "description": "Open the drawer.",
            "idle_frame": "3 out of 16.",
        }
    ]
    assert prompt["duration"] == "2s"
    assert prompt["fps"] == 8.0
    assert prompt["resolution"] == {"H": 192, "W": 320}
    assert prompt["aspect_ratio"] == "16,9"
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_keeps_ai_caption_string_path() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_idle_frames=True,
        idle_frames_dropout=0.0,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "policy",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    assert result["ai_caption"] == (
        "Open the drawer. "
        "This video is captured from a third-person perspective looking towards the agent from the front. "
        "The video is 2.0 seconds long and is of 8 FPS. "
        "This video is of 256x256 resolution. "
        "IdleFrames: 3 out of 16."
    )
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_keeps_idle_frames_for_forward_dynamics() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_idle_frames=True,
        idle_frames_dropout=0.0,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "forward_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    assert "IdleFrames: 3 out of 16." in result["ai_caption"]
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_skips_idle_frames_for_inverse_dynamics_string_path() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        append_idle_frames=True,
        idle_frames_dropout=0.0,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "inverse_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    assert "IdleFrames" not in result["ai_caption"]
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_transform_pipeline_skips_idle_frames_for_inverse_dynamics_json_prompt() -> None:
    pipeline = ActionTransformPipeline(
        tokenizer_config=None,
        max_action_dim=4,
        format_prompt_as_json=True,
    )
    video = torch.zeros(3, 17, 256, 256)  # [C,T,H,W]
    action = torch.zeros(16, 2)  # [T,D]
    data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": action,
        "conditioning_fps": torch.tensor(8),  # []
        "mode": "inverse_dynamics",
        "domain_id": torch.tensor(0),  # []
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    result = pipeline(data_dict, resolution="256")

    prompt = result["ai_caption"]
    assert isinstance(prompt, dict)
    assert prompt["actions"] == [
        {
            "time": "0:00-0:02",
            "description": "Open the drawer.",
        }
    ]
    assert result["action"].shape == (16, 4)


@pytest.mark.L0
def test_action_prompt_json_formatter_matches_video_json_common_metadata() -> None:
    formatter = ActionPromptJsonFormatter()
    video = torch.zeros(3, 23, 192, 320)  # [C,T,H,W]
    image_size = torch.tensor([192, 320, 192, 320])  # [4]
    fps = torch.tensor(8.0)  # []
    action_data_dict = {
        "ai_caption": "Open the drawer.",
        "video": video,
        "action": torch.zeros(22, 2),  # [T,D]
        "conditioning_fps": fps,
        "image_size": image_size,
        "viewpoint": "third_person_view",
        "idle_frames": torch.tensor(3),  # []
    }

    action_prompt = formatter(action_data_dict)["ai_caption"]

    video_data_dict = {
        "ai_caption": {
            "cinematography": {
                "framing": "This video is captured from a third-person perspective looking towards the agent from the front."
            },
            "actions": [
                {
                    "time": "0:00-0:03",
                    "description": "Open the drawer.",
                }
            ],
        },
        "video": video,
        "conditioning_fps": fps,
        "image_size": image_size,
        "__url__": SimpleNamespace(meta=SimpleNamespace(opts={"aspect_ratio": "16,9"})),
    }
    duration_augmentor = DurationFPSTextTimeStamps(
        input_keys=["ai_caption", "video", "conditioning_fps"],
        args={"caption_key": "ai_caption", "video_key": "video", "fps_key": "conditioning_fps"},
    )
    resolution_augmentor = ResolutionTextInfo(
        input_keys=["ai_caption", "video", "image_size"],
        args={"caption_key": "ai_caption", "video_key": "video"},
    )
    duration_augmentor(video_data_dict)
    resolution_augmentor(video_data_dict)
    video_prompt = video_data_dict["ai_caption"]

    common_top_level_keys = ["cinematography", "duration", "fps", "resolution", "aspect_ratio"]
    assert {key: action_prompt[key] for key in common_top_level_keys} == {
        key: video_prompt[key] for key in common_top_level_keys
    }
    assert action_prompt["actions"][0]["time"] == video_prompt["actions"][0]["time"]
    assert action_prompt["actions"][0]["description"] == video_prompt["actions"][0]["description"]
    assert action_prompt["actions"][0]["idle_frame"] == "3 out of 22."
