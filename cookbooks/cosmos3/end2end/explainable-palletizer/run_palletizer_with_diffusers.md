# Run Explainable Palletizer with Cosmos3 Diffusers

This markdown entry point validates the Cosmos3-Nano or Cosmos3-Super generation
path used by the explainable palletizer cookbook. It expects a small
Diffusers-compatible proxy with:

- `GET /health`
- `GET /v1/models`
- `POST /v1/infer`

The same request shape works for Nano and Super. The model is selected by the
server; verify it with `/v1/models` before running generation.

## 1. Choose an endpoint

Use the deployed Nano endpoint:

```bash
export COSMOS3_DIFFUSERS_BASE_URL="${COSMOS3_DIFFUSERS_BASE_URL:-http://127.0.0.1:8000}"
```

For a local Super deployment, point the variable at the Super service instead:

```bash
export COSMOS3_DIFFUSERS_BASE_URL="http://127.0.0.1:8000"
```

## 2. Smoke test health and model identity

```bash
curl -fsS "${COSMOS3_DIFFUSERS_BASE_URL}/health"
curl -fsS "${COSMOS3_DIFFUSERS_BASE_URL}/v1/models"
```

Expected Nano model response:

```json
{"object":"list","data":[{"id":"nvidia/Cosmos3-Nano","object":"model"}]}
```

For Super, the returned model ID should identify `nvidia/Cosmos3-Super`.

## 3. Generate a palletizing-scene artifact

```bash smoke-test
python3 - <<'PY'
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

base_url = os.environ.get("COSMOS3_DIFFUSERS_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
out_dir = Path(os.environ.get("COSMOS3_OUTPUT_DIR", "/tmp/cosmos3-palletizer-smoke"))
out_dir.mkdir(parents=True, exist_ok=True)

payload = {
    "prompt": (
        "A clean warehouse palletizing cell with a robot arm sorting mixed "
        "cardboard boxes, visible handling labels, a conveyor, and an "
        "operator review panel, technical demo style."
    ),
    "negative_prompt": "blurry, unsafe robot motion, broken gripper, low quality",
    "resolution": os.environ.get("COSMOS3_RESOLUTION", "256"),
    "num_output_frames": int(os.environ.get("COSMOS3_NUM_FRAMES", "1")),
    "fps": float(os.environ.get("COSMOS3_FPS", "1")),
    "steps": int(os.environ.get("COSMOS3_STEPS", "1")),
    "guidance_scale": float(os.environ.get("COSMOS3_GUIDANCE", "1.1")),
    "seed": int(os.environ.get("COSMOS3_SEED", "20260616")),
}

start = time.time()
req = urllib.request.Request(
    f"{base_url}/v1/infer",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=1800) as response:
        data = json.loads(response.read())
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    raise SystemExit(f"generation failed with HTTP {exc.code}: {body[:2000]}")

if data.get("error"):
    raise SystemExit(f"generation failed: {data['error']}")

media = base64.b64decode(data["b64_video"], validate=True)
suffix = ".jpg" if media.startswith(b"\xff\xd8\xff") else ".mp4"
artifact = out_dir / f"palletizer-cosmos3-{payload['seed']}{suffix}"
artifact.write_bytes(media)

print(json.dumps({
    "backend": data.get("backend"),
    "seed": payload["seed"],
    "bytes": len(media),
    "artifact": str(artifact),
    "elapsed_sec": round(time.time() - start, 2),
}, indent=2))

if len(media) < 1024:
    raise SystemExit("decoded media artifact is unexpectedly small")
PY
```

If the Diffusers service was just started, the first generation request may
spend several minutes loading model weights before it returns. Keep the request
timeout high enough for that warm load; subsequent smoke requests should be much
faster on the same process.

## 4. Run the Doosan full-stack smoke

This validates the reference robot loop without gated model access. Use alternate
ports if another service already occupies the defaults:

```bash
git clone https://github.com/doosan-robotics/explainable-palletizer.git
cd explainable-palletizer
cp docker/.env.example docker/.env
test -f uv.lock || uv lock

python3 - <<'PY'
from pathlib import Path
p = Path("docker/.env")
text = p.read_text()
for old, new in {
    "SIM_PORT=8100": "SIM_PORT=8310",
    "INFERENCE_PORT=8200": "INFERENCE_PORT=8320",
    "APP_PORT=8000": "APP_PORT=8330",
    "FRONTEND_PORT=3000": "FRONTEND_PORT=3340",
}.items():
    text = text.replace(old, new)
p.write_text(text)
PY

make docker-test
```

The current public Doosan Dockerfile expects `uv.lock`; the `uv lock` guard
generates it for fresh clones where upstream has not checked in the lockfile.
When the launcher reports all services healthy, open the configured frontend
port and verify that the simulated camera, reasoning panel, action panel, and
execution status are visible.

### Full-stack troubleshooting

If Docker reports `unknown or invalid runtime name: nvidia`, configure the NVIDIA
runtime before rerunning the stack:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

On a single-GPU host, Isaac Sim warmup can race with vLLM memory profiling when
all services start concurrently. Start the services sequentially if `make
docker-test` exits during inference-server profiling:

```bash
cd docker
set -a
. ./.env
set +a

docker compose -f docker-compose.yml -f docker-compose.test.yml up -d sim-server
until curl -fsS "http://127.0.0.1:${SIM_PORT}/sim/health"; do sleep 5; done
sleep 20

docker compose -f docker-compose.yml -f docker-compose.test.yml up -d inference-server
until curl -fsS "http://127.0.0.1:${INFERENCE_PORT}/health"; do sleep 5; done

docker compose -f docker-compose.yml -f docker-compose.test.yml up -d app-server frontend
until curl -fsS "http://127.0.0.1:${APP_PORT}/api/health"; do sleep 5; done
curl -fsS "http://127.0.0.1:${FRONTEND_PORT}/" >/dev/null
```

If `sim-server` exits with `AttributeError: module 'warp.types' has no attribute
'array'`, the image resolved a newer `warp-lang` than the current cuRobo/Isaac
Sim path expects. Pin `warp-lang==1.12.0` in the sim image, rebuild, and rerun
the smoke; do not count the robot-loop path as passed until all four services
are healthy.
