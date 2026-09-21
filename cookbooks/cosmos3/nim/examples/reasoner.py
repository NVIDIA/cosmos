# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Discover and run the Cosmos3 Reasoner task catalog through Chat Completions."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any

from common import media_to_data_url, require_runtime
from openai import OpenAI
import yaml

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")
ASSETS = Path(__file__).resolve().parents[2] / "reasoner" / "assets"
CATALOG_PATH = Path(__file__).with_name("reasoner_cases.yaml")
OUTPUTS = Path(__file__).parent / "outputs"

TEMPORAL_SECONDS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "number", "minimum": 0},
            "end": {"type": "number", "minimum": 0},
            "caption": {"type": "string", "minLength": 1},
        },
        "required": ["start", "end", "caption"],
        "additionalProperties": False,
    },
}
TEMPORAL_TIMECODES_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "start": {"type": "string", "pattern": r"^\d{2}:\d{2}\.\d{2,3}$"},
            "end": {"type": "string", "pattern": r"^\d{2}:\d{2}\.\d{2,3}$"},
            "caption": {"type": "string", "minLength": 1},
        },
        "required": ["start", "end", "caption"],
        "additionalProperties": False,
    },
}
TIMESTAMP_RANGE_SCHEMA = {
    "type": "object",
    "properties": {
        "start": {"type": "string", "pattern": r"^\d{2}:\d{2}\.\d{2,3}$"},
        "end": {"type": "string", "pattern": r"^\d{2}:\d{2}\.\d{2,3}$"},
    },
    "required": ["start", "end"],
    "additionalProperties": False,
}
GROUNDING_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
            "label": {"type": "string", "minLength": 1},
        },
        "required": ["bbox_2d", "label"],
        "additionalProperties": False,
    },
}
SUBJECT_CAPTIONS_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "subject_id": {"type": "string", "minLength": 1},
            "category": {"type": "string", "minLength": 1},
            "caption": {"type": "string", "minLength": 1},
        },
        "required": ["subject_id", "category", "caption"],
        "additionalProperties": False,
    },
}
TRAJECTORY_SCHEMA = {
    "type": "array",
    "minItems": 2,
    "items": {
        "type": "object",
        "properties": {
            "point_2d": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                "minItems": 2,
                "maxItems": 2,
            },
            "label": {"type": "string", "minLength": 1},
        },
        "required": ["point_2d", "label"],
        "additionalProperties": False,
    },
}
SCHEMAS = {
    "temporal_events_seconds": TEMPORAL_SECONDS_SCHEMA,
    "temporal_events_timecodes": TEMPORAL_TIMECODES_SCHEMA,
    "timestamp_range": TIMESTAMP_RANGE_SCHEMA,
    "grounding_boxes": GROUNDING_SCHEMA,
    "subject_captions": SUBJECT_CAPTIONS_SCHEMA,
    "trajectory_points": TRAJECTORY_SCHEMA,
}
CASE_ALIASES = {"image": "image_caption", "video": "video_caption"}


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Load and validate the endpoint-independent Reasoner case catalog."""
    catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("version") != 1:
        raise ValueError("Reasoner catalog must be a version 1 YAML mapping")
    prompt_source = catalog.get("prompt_source")
    defaults = catalog.get("defaults")
    baseline = catalog.get("validated_baseline")
    cases = catalog.get("cases")
    if not isinstance(prompt_source, dict) or not all(
        isinstance(prompt_source.get(field), str) and prompt_source[field].strip()
        for field in ("backend", "path", "contract")
    ):
        raise TypeError("Reasoner catalog prompt_source must identify its contract")
    if not isinstance(defaults, dict) or not isinstance(baseline, dict):
        raise TypeError("Reasoner catalog defaults and validated_baseline must be mappings")
    if not isinstance(defaults.get("sampling"), dict) or not isinstance(
        defaults.get("request_extensions"), dict
    ):
        raise TypeError("Reasoner catalog sampling defaults must be mappings")
    if not isinstance(cases, dict) or not cases:
        raise TypeError("Reasoner catalog cases must be a non-empty mapping")

    for case, spec in cases.items():
        if not isinstance(case, str) or not case or not isinstance(spec, dict):
            raise TypeError("Every Reasoner case must be a named mapping")
        for field in ("title", "category", "prompt"):
            if not isinstance(spec.get(field), str) or not spec[field].strip():
                raise ValueError(f"{case} must define a non-empty {field}")
        media = spec.get("media")
        if not isinstance(media, dict) or media.get("type") not in {"image", "video"}:
            raise ValueError(f"{case} must define image or video media")
        asset = media.get("asset")
        if not isinstance(asset, str) or not (ASSETS / asset).is_file():
            raise FileNotFoundError(f"{case} references a missing asset: {asset!r}")
        output = spec.get("output")
        if not isinstance(output, dict) or output.get("kind") not in {
            "text",
            "json_schema",
        }:
            raise ValueError(f"{case} must define a text or json_schema output")
        schema_name = output.get("schema")
        if output["kind"] == "json_schema" and schema_name not in SCHEMAS:
            raise ValueError(f"{case} references an unknown schema: {schema_name!r}")
        review = spec.get("review")
        if not isinstance(review, list) or not review or not all(
            isinstance(item, str) and item.strip() for item in review
        ):
            raise ValueError(f"{case} must define qualitative review criteria")
    return catalog


CATALOG = load_catalog()
PROMPT_SOURCE: dict[str, str] = CATALOG["prompt_source"]
CASES: dict[str, dict[str, Any]] = CATALOG["cases"]
DEFAULTS: dict[str, Any] = CATALOG["defaults"]
VALIDATED_BASELINE: dict[str, Any] = CATALOG["validated_baseline"]


def resolve_case(case: str) -> str:
    """Resolve a short compatibility alias to its catalog case ID."""
    return CASE_ALIASES.get(case, case)


def response_format(spec: dict[str, Any]) -> dict[str, Any] | None:
    """Build the OpenAI JSON-schema request shape for a structured case."""
    output = spec["output"]
    schema_name = output.get("schema")
    if schema_name is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": SCHEMAS[schema_name]},
    }


def case_thinking(spec: dict[str, Any], override: bool | None) -> bool:
    """Resolve a CLI native-thinking override against the catalog default."""
    if override is not None:
        return override
    thinking = spec.get("thinking", {})
    default = bool(DEFAULTS.get("native_thinking_enabled", False))
    return bool(thinking.get("enabled", default)) if isinstance(thinking, dict) else default


def build_request(
    case: str,
    model: str,
    *,
    enable_thinking: bool | None = None,
    thinking_token_budget: int | None = None,
    guided_output: bool = False,
) -> dict[str, Any]:
    """Build one NIM Chat Completions request from the declarative catalog."""
    case = resolve_case(case)
    spec = CASES[case]
    media = spec["media"]
    media_type = media["type"]
    media_content = {
        "type": f"{media_type}_url",
        f"{media_type}_url": {"url": media_to_data_url(ASSETS / media["asset"])},
    }
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    media_content,
                    {"type": "text", "text": spec["prompt"]},
                ],
            }
        ],
        "max_tokens": DEFAULTS["max_tokens"],
        **DEFAULTS["sampling"],
        **spec.get("sampling", {}),
    }

    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
        **DEFAULTS["request_extensions"],
    }
    extra_body.update(spec.get("request_extensions", {}))
    if media_type == "video":
        extra_body["media_io_kwargs"] = {
            "video": {"fps": float(DEFAULTS["video_fps"])}
        }
    thinking_enabled = case_thinking(spec, enable_thinking)
    if thinking_enabled:
        budget = thinking_token_budget or int(DEFAULTS["thinking_token_budget"])
        extra_body.update(
            {
                "chat_template_kwargs": {"enable_thinking": True},
                "thinking_token_budget": budget,
            }
        )
    if extra_body:
        request["extra_body"] = extra_body

    if guided_output:
        structured = response_format(spec)
        if structured is not None:
            request["response_format"] = structured
    return request


def parse_structured_content(content: str) -> object:
    """Extract the first JSON value from prompt-constrained model output."""
    text = content.rsplit("</think>", maxsplit=1)[-1]
    starts = sorted(
        index for index in (text.find("["), text.find("{")) if index >= 0
    )
    decoder = json.JSONDecoder()
    for start in starts:
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Reasoner response did not contain a complete JSON value")


def response_dict(response: object) -> dict[str, Any]:
    """Serialize an OpenAI response without depending on its exact SDK version."""
    model_dump = getattr(response, "model_dump", None)
    if not callable(model_dump):
        raise TypeError("OpenAI response does not provide model_dump()")
    value = model_dump(mode="json")
    if not isinstance(value, dict):
        raise TypeError("Serialized OpenAI response must be a JSON object")
    return value


def describe_case(case: str) -> dict[str, Any]:
    """Return one expanded, media-free case description for humans or tools."""
    case = resolve_case(case)
    spec = deepcopy(CASES[case])
    schema_name = spec["output"].get("schema")
    if schema_name is not None:
        spec["output"]["json_schema"] = deepcopy(SCHEMAS[schema_name])
    return {
        "case": case,
        "prompt_source": deepcopy(PROMPT_SOURCE),
        "validated_baseline": deepcopy(VALIDATED_BASELINE),
        "defaults": deepcopy(DEFAULTS),
        **spec,
    }


def list_cases(output_format: str) -> None:
    """Print the catalog inventory without requiring a NIM endpoint."""
    rows = [
        {
            "case": case,
            "category": spec["category"],
            "media": spec["media"]["type"],
            "title": spec["title"],
        }
        for case, spec in CASES.items()
    ]
    if output_format == "json":
        print(
            json.dumps(
                {
                    "prompt_source": PROMPT_SOURCE,
                    "validated_baseline": VALIDATED_BASELINE,
                    "cases": rows,
                },
                indent=2,
            )
        )
        return
    widths = {
        key: max(len(key), *(len(str(row[key])) for row in rows))
        for key in ("case", "category", "media")
    }
    print(
        f"{'CASE':<{widths['case']}}  "
        f"{'CATEGORY':<{widths['category']}}  "
        f"{'MEDIA':<{widths['media']}}  TITLE"
    )
    for row in rows:
        print(
            f"{row['case']:<{widths['case']}}  "
            f"{row['category']:<{widths['category']}}  "
            f"{row['media']:<{widths['media']}}  {row['title']}"
        )


def _request_summary(request: dict[str, Any]) -> dict[str, Any]:
    """Copy a request while replacing the embedded media payload."""
    summary = deepcopy(request)
    content = summary["messages"][0]["content"]
    media_item = content[0]
    media_key = next(key for key in media_item if key.endswith("_url"))
    media_item[media_key]["url"] = "<base64 data URL omitted>"
    return summary


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def annotate_spatial_output(
    spec: dict[str, Any], structured: object, output_path: Path
) -> None:
    """Draw normalized boxes or points over the source image."""
    visualization = spec["output"].get("visualization")
    if visualization not in {"boxes", "trajectory"}:
        return
    if not isinstance(structured, list):
        raise TypeError("Spatial output must be a list before annotation")

    from PIL import Image, ImageDraw

    image = Image.open(ASSETS / spec["media"]["asset"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    if visualization == "boxes":
        for item in structured:
            x1, y1, x2, y2 = item["bbox_2d"]
            box = [
                x1 / 1000 * width,
                y1 / 1000 * height,
                x2 / 1000 * width,
                y2 / 1000 * height,
            ]
            draw.rectangle(box, outline="red", width=max(3, width // 300))
            draw.text((box[0] + 4, max(0, box[1] - 14)), item["label"], fill="red")
    else:
        points = [
            (item["point_2d"][0] / 1000 * width, item["point_2d"][1] / 1000 * height)
            for item in structured
        ]
        draw.line(points, fill="lime", width=max(3, width // 200))
        radius = max(5, width // 100)
        for index, (x, y) in enumerate(points):
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill="red",
                outline="white",
                width=2,
            )
            draw.text((x + radius + 2, y - radius), str(index), fill="yellow")
    image.save(output_path)


def write_report(
    output_dir: Path,
    *,
    case: str,
    spec: dict[str, Any],
    model: str,
    selected_profile_id: object,
    content: str,
    validation: dict[str, Any],
) -> None:
    """Write a concise human-readable report alongside machine-readable artifacts."""
    criteria = "\n".join(f"- [ ] {item}" for item in spec["review"])
    annotation = (
        "\n- Annotated spatial output: `annotated.png`"
        if (output_dir / "annotated.png").exists()
        else ""
    )
    report = f"""# {spec['title']}

- Case: `{case}`
- Model: `{model}`
- Selected profile: `{selected_profile_id or 'not reported'}`
- Asset: `reasoner/assets/{spec['media']['asset']}`
- Format validation: `{validation['format_validation']['status']}`{annotation}

## Prompt

~~~~text
{spec['prompt'].rstrip()}
~~~~

## Final answer

~~~~text
{content.rstrip()}
~~~~

## Qualitative review

These task-level criteria require human or application-specific review; passing
format validation does not establish them.

{criteria}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def run_case(
    client: OpenAI,
    model: str,
    metadata: dict[str, Any],
    case: str,
    *,
    thinking_override: bool | None,
    thinking_token_budget: int | None,
    guided_output: bool,
) -> None:
    """Execute one catalog case and save machine- and human-readable artifacts."""
    spec = CASES[case]
    request = build_request(
        case,
        model,
        enable_thinking=thinking_override,
        thinking_token_budget=thinking_token_budget,
        guided_output=guided_output,
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
    for name in (
        "output.json",
        "reasoning.txt",
        "validation.json",
        "annotated.png",
        "report.md",
    ):
        (output_dir / name).unlink(missing_ok=True)

    selected_profile_id = metadata.get("selectedModelProfileId")
    thinking_enabled = case_thinking(spec, thinking_override)
    request_metadata = {
        "case": case,
        "title": spec["title"],
        "category": spec["category"],
        "endpoint": "/v1/chat/completions",
        "model": model,
        "selected_profile_id": selected_profile_id,
        "validated_baseline": VALIDATED_BASELINE,
        "request": _request_summary(request),
        "qualitative_review": spec["review"],
        "native_thinking_enabled": thinking_enabled,
        "guided_output_enabled": guided_output,
    }
    _write_json(output_dir / "request.json", request_metadata)
    _write_json(output_dir / "response.json", response_dict(response))
    (output_dir / "output.txt").write_text(
        content.rstrip() + "\n", encoding="utf-8"
    )

    validation: dict[str, Any] = {
        "api_response": {"status": "received"},
        "format_validation": {
            "status": "not_applicable",
            "schema": spec["output"].get("schema"),
        },
        "qualitative_review": {
            "status": "not_performed",
            "criteria": spec["review"],
        },
    }
    structured: object | None = None
    if spec["output"]["kind"] == "json_schema":
        try:
            structured = parse_structured_content(content)
        except (TypeError, ValueError) as exc:
            validation["format_validation"].update(
                {"status": "failed", "error": str(exc)}
            )
        else:
            validation["format_validation"]["status"] = "passed"
            _write_json(output_dir / "output.json", structured)
            try:
                annotate_spatial_output(
                    spec, structured, output_dir / "annotated.png"
                )
            except (TypeError, KeyError, IndexError, ValueError) as exc:
                validation["annotation"] = {"status": "skipped", "error": str(exc)}

    _write_json(output_dir / "validation.json", validation)
    write_report(
        output_dir,
        case=case,
        spec=spec,
        model=model,
        selected_profile_id=selected_profile_id,
        content=content,
        validation=validation,
    )

    print(content)
    print(f"Saved artifacts to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=tuple(CASES) + tuple(CASE_ALIASES) + ("all",),
        help="Run one case, or all cases sequentially. Defaults to image_caption.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List cases without contacting a NIM endpoint.",
    )
    parser.add_argument(
        "--describe",
        choices=tuple(CASES) + tuple(CASE_ALIASES),
        help="Describe one resolved case without contacting a NIM endpoint.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for --list-cases or --describe.",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        dest="thinking",
        action="store_true",
        help="Experimentally enable NIM-native thinking for every selected case.",
    )
    thinking.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="Keep NIM-native thinking disabled for every selected case.",
    )
    parser.set_defaults(thinking=None)
    parser.add_argument(
        "--thinking-token-budget",
        type=int,
        help="Override the thinking-token budget for an experimental reasoning request.",
    )
    parser.add_argument(
        "--guided-output",
        action="store_true",
        help="Use NIM JSON Schema guidance for structured cases.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    discovery_modes = int(args.list_cases) + int(args.describe is not None)
    if discovery_modes > 1 or (discovery_modes and args.case is not None):
        parser.error("choose one of --case, --list-cases, or --describe")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 1:
        parser.error("--thinking-token-budget must be at least 1")

    if args.list_cases:
        list_cases(args.format)
        return
    if args.describe is not None:
        description = describe_case(args.describe)
        if args.format == "json":
            print(json.dumps(description, indent=2))
        else:
            print(yaml.safe_dump(description, sort_keys=False).rstrip())
        return

    requested = args.case or "image_caption"
    cases = list(CASES) if requested == "all" else [resolve_case(requested)]
    metadata = require_runtime(
        NIM_URL,
        expected_runtime="reasoner",
        expected_endpoint="/v1/chat/completions",
    )
    with OpenAI(base_url=f"{NIM_URL}/v1", api_key="not-used", timeout=1800) as client:
        model = client.models.list().data[0].id
        if "super" not in model.lower():
            print(
                "Warning: this catalog's vLLM reference uses Super, but the "
                f"endpoint serves {model!r}.",
                file=sys.stderr,
            )
        failures: list[tuple[str, str]] = []
        for index, case in enumerate(cases):
            if len(cases) > 1:
                print(f"\n=== [{index + 1}/{len(cases)}] {case} ===")
            try:
                run_case(
                    client,
                    model,
                    metadata,
                    case,
                    thinking_override=args.thinking,
                    thinking_token_budget=args.thinking_token_budget,
                    guided_output=args.guided_output,
                )
            except Exception as exc:
                if requested != "all":
                    raise
                failures.append((case, str(exc)))
                print(f"Case failed: {case}: {exc}", file=sys.stderr)
    if failures:
        details = "; ".join(f"{case}: {error}" for case, error in failures)
        raise RuntimeError(f"{len(failures)} Reasoner catalog case(s) failed: {details}")


if __name__ == "__main__":
    main()
