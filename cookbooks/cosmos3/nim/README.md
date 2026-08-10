<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM

Deploy Cosmos3 models for world generation and multimodal reasoning. One image
contains two runtimes, but each container runs only one:

- **Generator** serves image/video generation, Action, and Transfer through
  `POST /v1/infer`.
- **Reasoner** serves OpenAI-compatible image/video understanding through Chat
  Completions and, when enabled, the Responses API.

> **Documentation status:** API behavior follows the current implementation.
> The support matrix records semi-final source-profile hardware requirements;
> the target image manifest determines which rows it actually ships. Release
> identity, final approval, codecs, and checkpoint combinations that are not yet
> authoritative remain marked **TBD**.

## Choose what to run

Normal deployment does not require choosing a profile ID. Make a few
user-facing choices and let the NIM select the best compatible profile for the
visible GPUs.

1. Choose **Generator** or **Reasoner**.
2. Choose the model variant.
3. Optionally pin a precision. Automatic selection prefers FP8 when a
   compatible FP8 profile is available.
4. For Generator, choose **latency** or **throughput**.
5. Start the container. The NIM selects a compatible profile for the model,
   precision preference, performance objective, and host.
6. For each Generator request, set the explicit top-level `model_mode` for the
   task.

`NIM_MODEL_VARIANT` chooses the checkpoint at startup; request `model_mode`
chooses T2I, T2V, I2V, V2V, Transfer, or an Action operation.

### Generator variants

| Variant | Intended task contract |
| --- | --- |
| `nano` | General-purpose generation and compatible Generator tasks |
| `nano-droid` | DROID policy with an action-only response; BF16 only in the current implementation |
| `super` | General-purpose generation and compatible Generator tasks |
| `super-t2i` | Text-to-image only |
| `super-t2i-4step` | Four-step text-to-image only |
| `super-i2v` | Image-to-video only |
| `super-i2v-4step` | Four-step image-to-video only |

Select a Generator model with `NIM_MODEL_VARIANT`. For Generator, the variant
also determines Nano versus Super. Select `NIM_PERF_PROFILE=latency` to
prioritize individual request latency or `throughput` to prioritize aggregate
request rate. Latency is the software default, but make the choice explicit in
deployment automation.

Reasoner provides `nano` and `super`, also selected with
`NIM_MODEL_VARIANT`. Reasoner does not use `NIM_PERF_PROFILE`.

If `NIM_PRECISION` is omitted, selection prefers FP8 when the chosen model,
released profiles, and host support it, then falls back to another compatible
precision. Pin precision only when the workload requires it.

Exact profile IDs, low-level tags, and offload overrides are advanced controls.
See [Deploy the NIM](deployment.md#advanced-profile-controls).

## Choose a task

| Task | Runtime | Input | Output | Guide |
| --- | --- | --- | --- | --- |
| Text-to-image | Generator | Prompt | JPEG image | [Generation](generation.md#text-to-image) |
| Text-to-video | Generator | Prompt | MP4 video | [Generation](generation.md#text-to-video) |
| Image-to-video | Generator | Prompt + image | MP4 video | [Generation](generation.md#image-to-video) |
| Video-to-video | Generator | Prompt + video | MP4 video | [Generation](generation.md#video-to-video) |
| Forward dynamics | Generator | Image + action trajectory | Rollout video | [Action](action.md#forward-dynamics) |
| Policy | Generator | Image + task/state | Video + predicted action | [Action](action.md#policy) |
| Nano-DROID policy | Generator | Image + task/current state | Predicted action | [Action](action.md#nano-droid-policy) |
| Inverse dynamics | Generator | Video | Video + predicted action | [Action](action.md#inverse-dynamics) |
| Video Transfer | Generator | Prompt + spatial control | Controlled MP4 video | [Transfer](transfer.md) |
| Image/video reasoning | Reasoner | Media + text | Text or structured result | [Reasoning](reasoning.md) |
| Responses API | Reasoner | Responses input | Response object/text | [Reasoning](reasoning.md#use-the-responses-api) |

The public Generator request model does not expose image-to-image or sound
generation. Local examples publish either selected runtime at
`http://localhost:8000`; stop the active container before launching the other
runtime on the same host port.

## Get started

1. Verify the host against [Prerequisites](prerequisites.md).
2. Choose a model/hardware combination from the semi-final
   [Support matrix](support-matrix.md), then confirm that the target image
   contains that row.
3. Follow [Deployment](deployment.md) to authenticate, prepare the cache, and
   launch Generator or Reasoner.
4. Wait for readiness:

   ```bash
   export NIM_URL=${NIM_URL:-http://localhost:8000}
   until curl -fsS "$NIM_URL/v1/health/ready" >/dev/null; do sleep 10; done
   curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
   ```

   Before inference, confirm that metadata identifies the runtime and its
   primary endpoint: Generator uses `/v1/infer`; Reasoner uses
   `/v1/chat/completions`.

5. From the repository root, enter this cookbook directory. `uv` automatically
   creates the pinned client environment from `pyproject.toml` and `uv.lock`:

   ```bash
   cd cookbooks/cosmos3/nim
   ```

6. Run an example that matches the active runtime and model.

For a general-purpose Generator:

```bash
uv run python examples/t2v.py
```

For a Reasoner:

```bash
uv run python examples/reasoner.py --case image_caption
```

Specialist models accept only their documented task contract. Use
[`t2i_4step.py`](examples/t2i_4step.py),
[`i2v_4step.py`](examples/i2v_4step.py), or the
[Nano-DROID request](action.md#nano-droid-policy) only after launching the
matching variant.

Generator examples save decoded media and action JSON under
`examples/outputs/`. The Reasoner task runner prints its final answer and saves
request metadata, the raw response, final text, and validated JSON for
structured cases under the same ignored output directory.

## Documentation

### Deploy and configure

- [Prerequisites](prerequisites.md)
- [Deployment](deployment.md)
- [Configuration](configuration.md)
- [Support matrix](support-matrix.md)
- [Deploy with Helm](helm.md)
- [Bring your own checkpoint](bring-your-own-checkpoint.md)

### Use the APIs

- [API reference](api-reference.md)
- [Generation](generation.md)
- [Reasoning](reasoning.md)
- [Action](action.md)
- [Transfer](transfer.md)
- [Python examples](examples/)

### Operate and maintain

- [Operations and troubleshooting](operations.md)
- [Release notes](release-notes.md)
- [Acknowledgements](acknowledgements.md)

## Safety, license, and notices

Generator guardrails are enabled by default. Disabling them can remove
content-policy and face-privacy protections; see
[Guardrails](operations.md#guardrails).

This cookbook is licensed under the repository
[LICENSE](../../../LICENSE). The running NIM exposes bundled product license
information at `/v1/license`; use the released NGC model card for authoritative
product terms and intended-use information.

Third-party notices for the released image belong in
[acknowledgements.md](acknowledgements.md).
