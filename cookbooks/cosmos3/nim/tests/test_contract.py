# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Contract tests: the request builders are the single source of truth.

These tests record each builder's request against a committed golden snapshot
(``snapshots/requests.json``). A change to a builder that alters the request a
script sends must update the snapshot deliberately — this catches drift between
the request builders and the served contract, and flags any accidental change.
"""

from __future__ import annotations

import json
from pathlib import Path

import request_builders as builders

SNAPSHOT = Path(__file__).resolve().parent / "snapshots" / "requests.json"


def _strip_media(payload: dict) -> dict:
    """Return ``payload`` with base64 data-url bodies replaced by ``"<media>"``."""
    out = dict(payload)
    for key, value in list(out.items()):
        if isinstance(value, str) and value.startswith("data:"):
            out[key] = "<media>"
    transfer = payload.get("transfer")
    if isinstance(transfer, dict):
        out["transfer"] = dict(transfer)
        for control, spec in transfer.items():
            if isinstance(spec, dict) and isinstance(spec.get("video"), str) and spec["video"].startswith("data:"):
                out["transfer"][control] = {**spec, "video": "<media>"}
    return out


def _golden() -> dict:
    if not SNAPSHOT.is_file():
        return {}
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_request_snapshots_match():
    golden = _golden()
    assert golden, "golden snapshot missing; run with --update-snapshots to create it"
    for rid in builders.GENERATOR_IDS:
        req = builders.request(rid)
        stripped = _strip_media(req)
        assert stripped == golden[rid], (
            f"{rid} request drifted from its golden snapshot.\n"
            f"  golden: {golden[rid]}\n"
            f"  actual: {stripped}"
        )


def test_all_builders_present_in_snapshot():
    golden = _golden()
    assert set(golden) == set(builders.GENERATOR_IDS), (
        "snapshot and builders out of sync",
        sorted(set(golden) ^ set(builders.GENERATOR_IDS)),
    )