# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Offline tests for the shared helpers in ``common.py`` (no NIM/GPU needed)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

import common


def test_get_default_nim_url_reads_env(monkeypatch):
    monkeypatch.setenv("NIM_URL", "http://example.com:8000/")
    assert common.get_default_nim_url() == "http://example.com:8000"


def test_get_default_nim_url_falls_back():
    # Ensure no NIM_URL is set for this test to isolate the default.
    import os

    os.environ.pop("NIM_URL", None)
    assert common.get_default_nim_url() == "http://localhost:8000"


def test_compact_json_file(tmp_path):
    p = tmp_path / "prompt.json"
    p.write_text('{"a": [1, 2, 3]}', encoding="utf-8")
    assert common.compact_json_file(p) == '{"a":[1,2,3]}'


def test_compact_json_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        common.compact_json_file(tmp_path / "nope.json")


def test_prompt_or_json_path(tmp_path):
    p = tmp_path / "p.json"
    p.write_text('{"k": "v"}', encoding="utf-8")
    assert common.prompt_or_json(str(p)) == '{"k":"v"}'


def test_prompt_or_json_plain_text():
    assert common.prompt_or_json("a plain prompt") == "a plain prompt"


def test_media_to_data_url_image(tmp_path):
    p = tmp_path / "img.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0")  # fake JPEG
    url = common.media_to_data_url(p)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > len("data:image/jpeg;base64,")


def test_media_to_data_url_unsupported(tmp_path):
    p = tmp_path / "x.xyz"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        common.media_to_data_url(p)


def test_media_to_data_url_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        common.media_to_data_url(tmp_path / "missing.mp4")


def test_decode_video_roundtrip():
    raw = b"\x00\x01\x02fake-mp4"
    encoded = base64.b64encode(raw).decode("ascii")
    assert common.decode_video(encoded) == raw
    # data-URL form
    assert common.decode_video(f"data:video/mp4;base64,{encoded}") == raw


def test_decode_video_bad_base64():
    with pytest.raises(ValueError):
        common.decode_video("!!!not-base64!!!")


def test_decode_video_empty():
    with pytest.raises(ValueError):
        common.decode_video(base64.b64encode(b"").decode("ascii"))


def test_decode_image_roundtrip():
    raw = b"\xff\xd8\xff\xe0"  # fake JPEG
    encoded = base64.b64encode(raw).decode("ascii")
    assert common.decode_image(encoded) == raw


def test_decode_image_bad_header():
    with pytest.raises(ValueError):
        common.decode_image("not-a-data-url")


def test_nim_infer(monkeypatch):
    import requests as real_requests

    class _R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"b64_video": "aGk="}

    def fake_post(url, *, json, timeout):
        assert url == "http://x/v1/infer"
        assert json == {"model_mode": "text2video"}
        assert timeout == 1800
        return _R()

    monkeypatch.setattr(real_requests, "post", fake_post)
    payload = common.nim_infer("http://x", request={"model_mode": "text2video"})
    assert payload == {"b64_video": "aGk="}


def test_nim_infer_non_object_response(monkeypatch):
    import requests as real_requests

    class _R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return "not-a-dict"

    monkeypatch.setattr(real_requests, "post", lambda *a, **k: _R())
    with pytest.raises(TypeError):
        common.nim_infer("http://x", request={})


def test_nim_infer_retries_then_succeeds(monkeypatch):
    """A transient 503 then a 200 -> the call succeeds with a retry."""
    import requests as real_requests
    import json as _json

    calls = []

    class _R:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def fake_post(*a, **k):
        calls.append(1)
        return _R(503 if len(calls) == 1 else 200)

    monkeypatch.setattr(real_requests, "post", fake_post)
    payload = common.nim_infer("http://x", request={}, retries=1, base_delay=0)
    assert payload == {"ok": True}
    assert len(calls) == 2


def test_nim_infer_no_retry_when_disabled(monkeypatch):
    import requests as real_requests

    class _R:
        status_code = 503

        def raise_for_status(self):
            raise _HttpError(self.status_code)

    class _HttpError(Exception):
        def __init__(self, code):
            super().__init__(f"HTTP {code}")
            self.status_code = code

    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return _R()

    monkeypatch.setattr(real_requests, "post", fake_post)
    with pytest.raises(_HttpError):
        common.nim_infer("http://x", request={}, retries=0, base_delay=0)
    assert len(calls) == 1


def test_default_output_dir():
    out = common.default_output_dir(Path("examples/t2v.py"))
    assert out.name == "outputs"
    assert out.parent.name == "examples"


def test_build_generator_infer_copies():
    base = {"a": 1, "b": 2}
    built = common.build_generator_infer(base, b=3, c=4)
    # base unchanged
    assert base == {"a": 1, "b": 2}
    assert built == {"a": 1, "b": 3, "c": 4}
    # mutations on the copy don't affect base
    del built["a"]
    assert base == {"a": 1, "b": 2}


def test_write_media_output(tmp_path):
    out_dir = tmp_path / "out"
    payload = {"b64_video": base64.b64encode(b"\x00\x01mp4").decode()}
    path = common.write_media_output(
        out_dir, name="clip.mp4", response_payload=payload, key="b64_video"
    )
    assert path.read_bytes() == b"\x00\x01mp4"


def test_write_media_output_missing_key(tmp_path):
    with pytest.raises(ValueError):
        common.write_media_output(
            tmp_path, name="x.mp4", response_payload={}, key="b64_video"
        )


def test_write_media_output_unique_no_clobber(tmp_path):
    payload = {"b64_video": base64.b64encode(b"\x00\x01mp4").decode()}
    first = common.write_media_output(
        tmp_path, name="clip.mp4", response_payload=payload, key="b64_video", unique=True
    )
    second = common.write_media_output(
        tmp_path, name="clip.mp4", response_payload=payload, key="b64_video", unique=True
    )
    assert first == tmp_path / "clip.mp4"
    assert second == tmp_path / "clip-1.mp4"
    # both exist and hold the same bytes
    assert first.read_bytes() == second.read_bytes()


def test_write_media_output_overwrite_default(tmp_path):
    # Without unique=True the existing file is overwritten.
    payload = {"b64_video": base64.b64encode(b"\x00\x01").decode()}
    first = common.write_media_output(
        tmp_path, name="x.mp4", response_payload=payload, key="b64_video"
    )
    second = common.write_media_output(
        tmp_path, name="x.mp4", response_payload=payload, key="b64_video"
    )
    assert first == second  # same path, overwritten in place


def test_require_runtime_ok(fake_server):
    fake_server(
        "GET",
        "http://nim/v1/metadata",
        json_payload={
            "model_type": "generator",
            "inference_endpoint": "/v1/infer",
        },
    )
    meta = common.require_runtime(
        "http://nim",
        expected_runtime="generator",
        expected_endpoint="/v1/infer",
    )
    assert meta["model_type"] == "generator"


def test_require_runtime_wrong_runtime(fake_server):
    fake_server(
        "GET",
        "http://nim/v1/metadata",
        json_payload={"model_type": "reasoner", "inference_endpoint": "/v1/chat/completions"},
    )
    with pytest.raises(RuntimeError):
        common.require_runtime(
            "http://nim",
            expected_runtime="generator",
            expected_endpoint="/v1/infer",
        )


def test_require_generator_profile(fake_server):
    fake_server(
        "GET",
        "http://nim/v1/metadata",
        json_payload={
            "model_type": "generator",
            "inference_endpoint": "/v1/infer",
            "selectedModelProfileId": "p1",
            "model_variant": "nano",
        },
    )
    meta = common.require_generator_profile("http://nim", allowed_variants=("nano", "super"))
    assert meta["model_variant"] == "nano"


def test_require_generator_profile_variant_rejected(fake_server):
    fake_server(
        "GET",
        "http://nim/v1/metadata",
        json_payload={
            "model_type": "generator",
            "inference_endpoint": "/v1/infer",
            "selectedModelProfileId": "p1",
            "model_variant": "super-i2v",
        },
    )
    with pytest.raises(RuntimeError):
        common.require_generator_profile("http://nim", allowed_variants=("nano",))