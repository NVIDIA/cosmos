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
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import fastapi
import pytest
import ray
import ray.serve
import requests
import yaml
from starlette.testclient import TestClient

from cosmos3.args import OmniSampleOverrides
from cosmos3.common.args import SampleOutput, SampleOutputs
from cosmos3.common.init import get_free_port
from cosmos3.ray.serve import OmniRouterDeployment, OmniRouterDeploymentArgs
from cosmos3.ray.submit import handle_response

_CONFIGS_DIR = Path(__file__).parent / "configs"


@pytest.mark.parametrize("config_path", sorted(_CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_config(monkeypatch: pytest.MonkeyPatch, config_path: Path):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = yaml.safe_load(config_path.read_text())
    args = OmniRouterDeploymentArgs.model_validate(config["applications"][0]["args"])
    assert "Cosmos3-Nano" in args.models


def _mock_response(status_code: int, json_body: dict | None = None) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    if json_body is not None:
        resp._content = json.dumps(json_body).encode()
    return resp


def test_handle_response_ok():
    handle_response(_mock_response(200))


def test_handle_response_error_with_detail():
    with pytest.raises(ValueError, match="Something went wrong"):
        handle_response(_mock_response(400, {"detail": "Something went wrong"}))


def test_handle_response_error_without_detail():
    with pytest.raises(requests.exceptions.HTTPError):
        handle_response(_mock_response(500))


def _make_generate_app(mock_handle: AsyncMock) -> fastapi.FastAPI:
    app = fastapi.FastAPI()

    @app.post("/generate")
    async def generate(sample_overrides: OmniSampleOverrides) -> SampleOutputs:
        return await mock_handle.generate.remote(sample_overrides)

    return app


def test_generate_e2e():
    mock_handle = AsyncMock()
    sample_kwargs = {"name": "test", "prompt": "a city"}
    mock_handle.generate.remote.return_value = SampleOutputs(
        args=sample_kwargs, status="success", outputs=[SampleOutput(content={}, files=[])]
    )
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post("/generate", json=sample_kwargs)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["args"] == sample_kwargs
    mock_handle.generate.remote.assert_called_once()


def test_generate_error_response():
    mock_handle = AsyncMock()
    sample_kwargs = {"name": "test", "prompt": "a city"}
    mock_handle.generate.remote.return_value = SampleOutputs(
        args=sample_kwargs, status="error", message="Out of memory", stack_trace="Traceback ..."
    )
    client = TestClient(_make_generate_app(mock_handle))
    response = client.post("/generate", json=sample_kwargs)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Out of memory"
    assert data["stack_trace"] is not None


@ray.serve.deployment
class _FakeModelDeployment:
    async def generate(self, sample_overrides: OmniSampleOverrides) -> SampleOutputs:
        return SampleOutputs(
            args=sample_overrides.model_dump(mode="json"),
            status="success",
            outputs=[SampleOutput(content={}, files=[])],
        )



@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(0)
def test_ray_serve_integration(tmp_path: Path):
    ray.init(num_cpus=2, num_gpus=0)  # type: ignore
    try:
        port = get_free_port()
        ray.serve.start(http_options={"host": "127.0.0.1", "port": port})

        app = cast(ray.serve.Deployment, OmniRouterDeployment).bind(
            {"test-model": cast(ray.serve.Deployment, _FakeModelDeployment).bind()},
            output_dir=tmp_path,
        )
        ray.serve.run(app, name="test")
        base_url = f"http://127.0.0.1:{port}"

        response = requests.get(f"{base_url}/models")
        assert response.status_code == 200
        assert "test-model" in response.json()

        sample_kwargs = {"name": "test", "prompt": "A city at night"}
        response = requests.post(
            f"{base_url}/generate",
            json=sample_kwargs,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        for k, v in sample_kwargs.items():
            assert data["args"][k] == v

        response = requests.post(
            f"{base_url}/generate",
            json={"name": "test", "prompt": "A city", "model": "nonexistent"},
        )
        assert response.status_code == 404
    finally:
        ray.serve.shutdown()
        ray.shutdown()  # type: ignore
