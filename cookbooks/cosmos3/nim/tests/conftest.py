# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared fixtures for the Cosmos3 NIM example-client tests.

All tests are offline: no NIM server is contacted. HTTP is faked via
``monkeypatch`` of ``requests.post`` / ``requests.get`` (no extra deps).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the examples package importable from conftest so tests can ``import common``.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


@pytest.fixture
def fake_server(monkeypatch):
    """Patch ``requests.post``/``requests.get`` to return canned JSON.

    Returns a callable ``fake_server(method, url, payload)`` that tests use to
    register a response for a given method+URL body. Default responses for the
    metadata and infer endpoints are provided.
    """

    import requests as real_requests

    class _FakeResponse:
        def __init__(self, status_code, json_payload):
            self.status_code = status_code
            self._json_payload = json_payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise _HttpError(self.status_code)

        def json(self):
            return self._json_payload

    class _HttpError(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.status = status

    registry = {}

    def register(method, url, status=200, json_payload=None):
        registry[(method, url)] = (status, json_payload)

    def fake_call(method, url, *args, **kwargs):
        key = (method, url)
        if key not in registry:
            raise AssertionError(f"no registered response for {method} {url}")
        status, payload = registry[key]
        return _FakeResponse(status, payload)

    monkeypatch.setattr(real_requests, "post", lambda *a, **k: fake_call("POST", a[0], *a[1:], **k))
    monkeypatch.setattr(real_requests, "get", lambda *a, **k: fake_call("GET", a[0], *a[1:], **k))
    return register