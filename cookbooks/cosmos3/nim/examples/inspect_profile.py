# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Print the active Cosmos3 NIM profile without its artifact inventory."""

import os
from typing import Any

import requests
import yaml

NIM_URL = os.environ.get("NIM_URL", "http://localhost:8000").rstrip("/")


def get_json(path: str) -> dict[str, Any]:
    """Fetch one NIM management endpoint and require a JSON object."""
    response = requests.get(f"{NIM_URL}{path}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must return a JSON object")
    return payload


def selected_profile_summary(
    metadata: dict[str, Any],
    manifest_response: dict[str, Any],
) -> dict[str, Any]:
    """Match metadata's selected profile ID against the embedded YAML manifest."""
    selected_id = metadata.get("selectedModelProfileId")
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError("/v1/metadata did not report selectedModelProfileId")

    manifest_file = manifest_response.get("manifest_file")
    if not isinstance(manifest_file, str) or not manifest_file:
        raise ValueError("/v1/manifest did not return a non-empty manifest_file string")

    manifest = yaml.safe_load(manifest_file)
    if not isinstance(manifest, dict):
        raise TypeError("manifest_file must decode to a YAML mapping")
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list):
        raise TypeError("manifest_file must contain a profiles list")

    matches = [
        profile
        for profile in profiles
        if isinstance(profile, dict) and profile.get("id") == selected_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one manifest profile matching {selected_id!r}, found {len(matches)}"
        )

    profile = matches[0]
    tags = profile.get("tags")
    if not isinstance(tags, dict):
        raise TypeError("The selected manifest profile must contain a tags mapping")

    selected = {"id": selected_id, "tags": tags}
    workspace_hash = profile.get("workspace_hash")
    if workspace_hash is not None:
        selected["workspace_hash"] = workspace_hash
    return {
        "model": manifest.get("model"),
        "release": manifest.get("release"),
        "selected_profile": selected,
    }


def main() -> None:
    metadata = get_json("/v1/metadata")
    manifest_response = get_json("/v1/manifest")
    summary = selected_profile_summary(metadata, manifest_response)
    print(yaml.safe_dump(summary, sort_keys=False).rstrip())


if __name__ == "__main__":
    main()
