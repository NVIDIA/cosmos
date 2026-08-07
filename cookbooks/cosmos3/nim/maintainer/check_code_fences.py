# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Reject unresolved values in public runnable documentation blocks."""

from __future__ import annotations

import re
from pathlib import Path


COOKBOOK = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
UNRESOLVED = (
    (re.compile(r"<[^>\n]+:TBD>"), "unresolved <...:TBD> placeholder"),
    (re.compile(r"--case\s+CASE\b"), "literal CASE is not a supported runner choice"),
)
IMAGE_EXPORT = re.compile(r"^export NIM_IMAGE='([^']+)'$", re.MULTILINE)


def fenced_blocks(path: Path) -> list[tuple[int, str]]:
    """Return the starting line and content of each fenced block."""
    blocks: list[tuple[int, str]] = []
    marker: str | None = None
    start = 0
    content: list[str] = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if marker is None:
            if match:
                marker = match.group(1)
                start = line_number
                content = []
            continue

        if match and match.group(1)[0] == marker[0] and len(match.group(1)) >= len(marker):
            blocks.append((start, "\n".join(content)))
            marker = None
            continue
        content.append(line)

    if marker is not None:
        raise ValueError(f"{path}:{start}: unclosed fenced block")
    return blocks


def main() -> int:
    errors: list[str] = []
    for path in sorted(COOKBOOK.glob("*.md")):
        try:
            blocks = fenced_blocks(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        for line, content in blocks:
            for pattern, message in UNRESOLVED:
                if pattern.search(content):
                    errors.append(f"{path.relative_to(COOKBOOK)}:{line}: {message}")

    deployment = (COOKBOOK / "deployment.md").read_text(encoding="utf-8")
    image_match = IMAGE_EXPORT.search(deployment)
    if image_match is None:
        errors.append("deployment.md: missing single-quoted NIM_IMAGE export")
    else:
        image = image_match.group(1)
        if image.endswith(":latest"):
            errors.append("deployment.md: NIM_IMAGE must use a versioned tag, not latest")
        release_notes = (COOKBOOK / "release-notes.md").read_text(encoding="utf-8")
        if image not in release_notes:
            errors.append("release-notes.md: NIM_IMAGE does not match deployment.md")

    if errors:
        print("\n".join(errors))
        return 1
    print("Cosmos3 NIM documentation fence checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
