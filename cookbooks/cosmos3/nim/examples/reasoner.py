# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run one canonical Cosmos3 Reasoner task through Chat Completions."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from common import media_to_data_url, require_runtime
from openai import OpenAI

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
ASSETS = Path(__file__).resolve().parents[2] / "reasoner" / "assets"
OUTPUTS = Path(__file__).parent / "outputs"

TEMPORAL_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "caption": {"type": "string"},
        },
        "required": ["start", "end", "caption"],
        "additionalProperties": False,
    },
}
GROUNDING_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
            "label": {"type": "string"},
        },
        "required": ["bbox_2d", "label"],
        "additionalProperties": False,
    },
}
TRAJECTORY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "point_2d": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 2,
                "maxItems": 2,
            },
            "label": {"type": "string"},
        },
        "required": ["point_2d", "label"],
        "additionalProperties": False,
    },
}

CASES: dict[str, dict[str, Any]] = {
    "image_caption": {
        "media_type": "image",
        "asset": "robot_153.jpg",
        "prompt": "Caption the image in detail.",
        "sampling": {"seed": 0},
    },
    "video_caption": {
        "media_type": "video",
        "asset": "video_caption.mp4",
        "prompt": "Describe the video in detail.",
    },
    "temporal_localization": {
        "media_type": "video",
        "asset": "temporal_localization_1.mp4",
        "prompt": (
            "List all action segments in the video. For each event, return its "
            "numeric start and end time in seconds and a concise caption."
        ),
        "schema_name": "temporal_events",
        "schema": TEMPORAL_SCHEMA,
    },
    "robotics_next_action": {
        "media_type": "video",
        "asset": "robotics_next_action.mp4",
        "prompt": "What can be the next immediate action?",
    },
    "robot_planning": {
        "media_type": "image",
        "asset": "robot_planning.png",
        "prompt": (
            "The task is to put the flower into the red bottle. Generate a plan "
            "of ordered subtasks to accomplish the task."
        ),
        "sampling": {"seed": 0},
    },
    "grounding_2d": {
        "media_type": "image",
        "asset": "grounding_2d.png",
        "prompt": (
            "Locate the bounding box of the load as a whole. Use coordinates "
            "normalized independently to 0-1000 on each image axis."
        ),
        "schema_name": "grounding_boxes",
        "schema": GROUNDING_SCHEMA,
        "sampling": {"seed": 0},
    },
    "trajectory_2d": {
        "media_type": "image",
        "asset": "action_cot_trajectory.png",
        "prompt": (
            "The task is to move the pink bowl to the right. Return the ordered "
            "2D trajectory the gripper should follow. Use coordinates normalized "
            "independently to 0-1000, with the origin at the top-left."
        ),
        "schema_name": "trajectory_points",
        "schema": TRAJECTORY_SCHEMA,
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 0.0,
        },
        "extra_body": {"top_k": 20, "repetition_penalty": 1.0},
    },
    "physical_plausibility": {
        "media_type": "video",
        "asset": "physical_plausibility.mp4",
        "prompt": (
            "Is this video physically plausible under normal laws of physics, "
            "including object permanence, shape constancy, and continuous object "
            "trajectories? Ignore simulation-rendering quality and do not judge "
            "the rising wall, which is part of the experiment. Answer (A) "
            "Possible or (B) Impossible."
        ),
    },
    "situation_understanding": {
        "media_type": "video",
        "asset": "situation_understanding.mp4",
        "prompt": (
            "What is the person doing with the skillet? What will the person "
            "likely do next in this situation?"
        ),
    },
}
CASE_ALIASES = {"image": "image_caption", "video": "video_caption"}


def response_format(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Build the OpenAI JSON-schema request shape for a structured case."""
    schema = spec.get("schema")
    if schema is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": spec["schema_name"], "schema": schema},
    }


def build_request(
    case: str,
    model: str,
    *,
    include_reasoning: bool = False,
    thinking_token_budget: int = 512,
) -> dict[str, Any]:
    """Build one NIM Chat Completions request from a canonical case."""
    case = CASE_ALIASES.get(case, case)
    spec = CASES[case]
    media_type = spec["media_type"]
    media = {
        "type": f"{media_type}_url",
        f"{media_type}_url": {"url": media_to_data_url(ASSETS / spec["asset"])},
    }
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    media,
                    {"type": "text", "text": spec["prompt"]},
                ],
            }
        ],
        "max_tokens": 4096,
        **spec.get("sampling", {}),
    }

    extra_body: dict[str, Any] = dict(spec.get("extra_body", {}))
    if media_type == "video":
        extra_body["media_io_kwargs"] = {"video": {"fps": 4.0}}
    if include_reasoning:
        extra_body.update(
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "include_reasoning": True,
                "thinking_token_budget": thinking_token_budget,
            }
        )
    if extra_body:
        request["extra_body"] = extra_body

    structured = response_format(spec)
    if structured is not None:
        request["response_format"] = structured
    return request


def validate_structured_output(case: str, value: object) -> None:
    """Validate the semantic invariants needed by the structured examples."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{case} output must be a non-empty JSON array")

    if case == "temporal_localization":
        for index, event in enumerate(value):
            if not isinstance(event, dict) or set(event) != {"start", "end", "caption"}:
                raise ValueError(f"Temporal event {index} has an unexpected shape")
            start, end, caption = event["start"], event["end"], event["caption"]
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end < start
                or not isinstance(caption, str)
                or not caption.strip()
            ):
                raise ValueError(f"Temporal event {index} contains invalid values")
        return

    key, width = ("bbox_2d", 4) if case == "grounding_2d" else ("point_2d", 2)
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {key, "label"}:
            raise ValueError(f"{case} item {index} has an unexpected shape")
        coordinates = item[key]
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != width
            or any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, int)
                or not 0 <= coordinate <= 1000
                for coordinate in coordinates
            )
            or not isinstance(item["label"], str)
            or not item["label"].strip()
        ):
            raise ValueError(f"{case} item {index} contains invalid values")
        if key == "bbox_2d":
            x1, y1, x2, y2 = coordinates
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Grounding box {index} has invalid corner order")


def response_dict(response: object) -> dict[str, Any]:
    """Serialize an OpenAI response without depending on its exact SDK version."""
    model_dump = getattr(response, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("OpenAI response does not provide model_dump()")
    value = model_dump(mode="json")
    if not isinstance(value, dict):
        raise TypeError("Serialized OpenAI response must be a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=tuple(CASES) + tuple(CASE_ALIASES),
        default="image_caption",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable thinking and request parsed reasoning from the NIM.",
    )
    parser.add_argument("--thinking-token-budget", type=int, default=512)
    args = parser.parse_args()
    if args.thinking_token_budget < 1:
        parser.error("--thinking-token-budget must be at least 1")
    case = CASE_ALIASES.get(args.case, args.case)
    spec = CASES[case]

    require_runtime(
        NIM_URL,
        expected_runtime="reasoner",
        expected_endpoint="/v1/chat/completions",
    )
    with OpenAI(base_url=f"{NIM_URL}/v1", api_key="not-used", timeout=1800) as client:
        model = client.models.list().data[0].id
        request = build_request(
            case,
            model,
            include_reasoning=args.reasoning,
            thinking_token_budget=args.thinking_token_budget,
        )
        response = client.chat.completions.create(**request)

    if not response.choices:
        raise ValueError("Reasoner response did not contain a completion choice")
    message = response.choices[0].message
    content = message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Reasoner response did not contain final text content")

    output_dir = OUTPUTS / f"reasoner_{case}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "case": case,
        "endpoint": "/v1/chat/completions",
        "model": model,
        "media_type": spec["media_type"],
        "asset": str((ASSETS / spec["asset"]).relative_to(Path(__file__).parents[2])),
        "prompt": spec["prompt"],
        "video_fps": 4.0 if spec["media_type"] == "video" else None,
        "sampling": {"max_tokens": 4096, **spec.get("sampling", {})},
        "request_extensions": spec.get("extra_body", {}),
        "reasoning_enabled": args.reasoning,
        "thinking_token_budget": args.thinking_token_budget if args.reasoning else None,
        "structured_output": spec.get("schema_name"),
    }
    (output_dir / "request.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "response.json").write_text(
        json.dumps(response_dict(response), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "output.txt").write_text(content.rstrip() + "\n", encoding="utf-8")

    if spec.get("schema") is not None:
        structured = json.loads(content)
        validate_structured_output(case, structured)
        (output_dir / "output.json").write_text(
            json.dumps(structured, indent=2) + "\n", encoding="utf-8"
        )

    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        (output_dir / "reasoning.txt").write_text(
            reasoning.rstrip() + "\n", encoding="utf-8"
        )

    print(content)
    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
