# TensorRT-LLM Action and Transfer validation

This record reconciles the saved GPU runs with the notebook requests in
[cookbook commit `7f973ea81f4b37bab2d875fa34a90313c35a9990`](https://github.com/NVIDIA/cosmos/commit/7f973ea81f4b37bab2d875fa34a90313c35a9990).
The historical runs were audited on Fri 18 Sep 2026 (Pacific) using retained
logs and artifacts. A fresh run of the NLTK workaround is recorded separately
below; do not confuse its environment with the earlier runs.

## Compatible runtime

The historical Action and Nano Transfer notebook runs used TensorRT-LLM source revision
[`bca6761ab84fbcd58fc7f914eade7de48b32e35e`](https://github.com/NVIDIA/TensorRT-LLM/commit/bca6761ab84fbcd58fc7f914eade7de48b32e35e)
in a native Python 3.12 environment, with PyTorch `2.12.0+cu130` and
`cosmos_guardrail==0.3.0`. The runtime reported `1.3.0rc26`; that package
version alone does not identify the source revision. These were native runs,
so there is no container digest. The retained evidence does not establish a
fresh C++ rebuild or an immutable fingerprint of every installed dependency.

At this revision, the
[request schema](https://github.com/NVIDIA/TensorRT-LLM/blob/bca6761ab84fbcd58fc7f914eade7de48b32e35e/tensorrt_llm/serve/openai_protocol.py#L2067)
and [multipart parser](https://github.com/NVIDIA/TensorRT-LLM/blob/bca6761ab84fbcd58fc7f914eade7de48b32e35e/tensorrt_llm/serve/openai_video_routes.py#L311)
accept `image_reference` and `video_reference`.
The [conversion layer](https://github.com/NVIDIA/TensorRT-LLM/blob/bca6761ab84fbcd58fc7f914eade7de48b32e35e/tensorrt_llm/serve/visual_gen_utils.py#L482)
passes uploaded media to the worker, and the
[Action image decoder](https://github.com/NVIDIA/TensorRT-LLM/blob/bca6761ab84fbcd58fc7f914eade7de48b32e35e/tensorrt_llm/_torch/visual_gen/models/cosmos3/pipeline_cosmos3.py#L364)
handles encoded image bytes. The earlier `799d7d42` runs used `input_reference`;
they are historical evidence, not validation of the current request format.

The server used the checked-in `cosmos3-nano-1gpu.yaml`:

```bash
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args examples/visual_gen/configs/cosmos3-nano-1gpu.yaml \
  --host 127.0.0.1 --port 8000
```

The checkpoint resolved to snapshot
`7a312c868bcce8e40b3eb40861300a9d0ba3fde1`. Each run used one H100 80GB on
`ipp2-0160`. See the [shared setup](../README.md#tensorrt-llm-generator) for
the source pin and launch commands.

## Executed notebook requests

The runs below executed the checked-in code cells, including request creation
and response decoding. Action used the checked-in AV assets, prompt
`You are an autonomous vehicle planning system.`, and seed 0. Transfer used
the checked-in controls and captions, seed 2026, 50 steps, and flow shift 10.
Dimensions in this table are width × height.

| Notebook/case | Request to `POST /v1/videos/sync` | Decoded result | GPU evidence |
| --- | --- | --- | --- |
| [Forward dynamics](action/run_fd_with_trt_llm.ipynb) | Multipart `image_reference`, `format=safetensors` | `video[61,480,832,3]` uint8; `action[60,9]` float32; 10 fps | `20260909-135841-262ee2`, all 5 cells passed |
| [Inverse dynamics](action/run_id_with_trt_llm.ipynb) | Multipart `video_reference`, `format=safetensors` | `video[61,480,832,3]` uint8; `action[60,9]` float32; 10 fps | `20260909-135841-262ee2`, all 5 cells passed |
| [Transfer](transfer/run_video_transfer_with_trt_llm.ipynb): edge | JSON/base64 `extra_params.edge`, `format=mp4` | 1280×720, 121 frames, 30 fps | `20260909-140456-faff27` |
| Transfer: blur | JSON/base64 `extra_params.blur`, `size=1104x832`, `format=mp4` | **1104×832**, 121 frames, 30 fps | `20260909-140456-faff27` |
| Transfer: depth | JSON/base64 `extra_params.depth`, `format=mp4` | 1280×720, 121 frames, 30 fps | `20260909-140456-faff27` |
| Transfer: segmentation | JSON/base64 `extra_params.seg`, `format=mp4` | 1280×720, 121 frames, 30 fps | `20260909-140456-faff27` |
| Transfer: WSM | JSON/base64 `extra_params.wsm`, `format=mp4` | 1280×720, 100 frames, 10 fps | `20260909-140456-faff27` |

Both Action requests completed and decoded their tensors. The same run then
failed at the first Transfer request with HTTP 400 because `ffmpeg` was absent
from the server's `PATH`. After exposing the installed imageio-ffmpeg binary,
the full 12-cell Transfer notebook passed, with five HTTP 200 `video/mp4`
responses. Artifact validation run `20260909-150023-120a4f` decoded all seven
MP4s and checked the two safetensors payloads. The failed combined run is not
being counted as a successful Transfer run.

The blur server log records `height=832 width=1104`, followed by
`Cosmos3 transfer target=1104x832 (WxH)` and HTTP 200. Its MP4 is 688,026 bytes,
SHA-256 `f5dc9767915a37f9b3306fe2a55b15984589b025d01698cd843be5ff2fa8b21d`.
The old 1280×720 summary describes an earlier request, not this notebook.

For locating the exact retained evidence, gpu-run IDs refer to the author's
local `~/.config/gpu-run/runs/<id>/log` records. Videos, tensors, and Cosmos
Framework comparisons are under
`~/sim/cosmos3/transfer-action-output/matched-runs/<case>/`, with
`tensorrt-llm.mp4` and `cosmos-framework.mp4` in each case directory.
These local artifacts are not public downloads.

The notebook file SHA-256 values at the tested cookbook commit are:

| Notebook | SHA-256 |
| --- | --- |
| `action/run_fd_with_trt_llm.ipynb` | `13a243a6adfb1dba26c80c500aedf661ac7024edbf1823adfdde77f1eb17c885` |
| `action/run_id_with_trt_llm.ipynb` | `86232889e69e65dd73c1cc895896375ed586bbbc33ee15553511d2bfaaee2363` |
| `transfer/run_video_transfer_with_trt_llm.ipynb` | `61f960a34ae9e84e2eea7fa287df6d7cdae2206fa3b7c46a119f2e6d9651b63c` |

## NLTK workaround validation — Fri 18 Sep 2026

The [server-side NLTK setup](../README.md#server-side-nltk-data-setup) copies
the Guardrail snapshot's NLTK resources into a private directory as regular
files, then exports `NLTK_DATA` in the server shell. It does not disable NLTK
path security, change request guardrail settings, or patch either library.

The runtime was rebuilt natively from the same TensorRT-LLM source revision
`bca6761ab84fbcd58fc7f914eade7de48b32e35e`, using one H200 on `ipp2-0039`.
Build run `20260918-203438-1aaf41` passed and installed
`1.3.0rc26+bca6761ab8`. This is a native build, not a container validation.
The environment used Python 3.12.3, PyTorch `2.12.0+cu130`, Transformers
`5.5.4`, `cosmos_guardrail==0.3.0`, NLTK `3.10.3`, and driver `595.58.03`.
The installed imageio-ffmpeg 7.0.2 executable was exposed as `ffmpeg` in the
runtime's own `PATH`; no system package installation was performed.

Preflight run `20260918-205406-56f855` executed the README's NLTK setup
verbatim, then called the installed `Blocklist.is_safe()` on the FD prompt.
It passed with `nltk.pathsec.ENFORCE=True`, 383 blocklist entries and 1,430
exact-match entries loaded, and the tokenizer resolved inside the new NLTK
directory. This preflight is a CPU component check, not GPU inference.

Notebook run `20260918-205707-474ba5` executed the README setup and the FD
notebook's server command verbatim, then executed the checked-in code cells
of both Action notebooks and the full Transfer notebook. All requests retain
`use_guardrails=True`. It **passed**, exiting 0 after 3,149.8 seconds
(20:57–21:49 Pacific). All 5 FD, 5 ID, and 12 Transfer code cells executed
without cell errors; all seven requests returned HTTP 200. There was no
`pathsec` exception or HTTP 500. Notebook inference code was not changed for
this workaround; only setup instructions in Markdown cells were added.

Artifact-check run `20260918-215026-556c5a` decoded every frame of every MP4
and verified the following results. Both Action safetensors files contain
`video[61,480,832,3]` uint8 and finite `action[60,9]` float32 tensors.

| Case | Result | Decoded MP4 (width × height, frames, fps) |
| --- | --- | --- |
| Forward dynamics | PASS | 832×480, 61, 10 |
| Inverse dynamics | PASS | 832×480, 61, 10 |
| Transfer edge | PASS | 1280×720, 121, 30 |
| Transfer blur | PASS | 1104×832, 121, 30 |
| Transfer depth | PASS | 1280×720, 121, 30 |
| Transfer segmentation | PASS | 1280×720, 121, 30 |
| Transfer WSM | PASS | 1280×720, 100, 10 |

The executed notebook SHA-256 values identify the tested files independently
of later edits to this validation record:

| Notebook | SHA-256 |
| --- | --- |
| `action/run_fd_with_trt_llm.ipynb` | `2ee065cf6a96fdfdd64c00f2f89d972e41d2c198d1e03f6f0dbe8e304780cf52` |
| `action/run_id_with_trt_llm.ipynb` | `1f7d5c0e46ca19350cc7fb3a001ac774308a5f5762dcab55896065639a1f2638` |
| `transfer/run_video_transfer_with_trt_llm.ipynb` | `8946c3fdd1fb81bdde53bff67e76527cf693bf49155db832901a4fb4d3ec24d3` |

Full logs, dependency versions, executed notebooks, tensors, and artifact
hashes are retained under the author's scratch directory
`/home/scratch.ishovkun_gpu/gpu-run/artifacts/pr310-guardrail-20260918-2057/`.
Local videos are under
`~/sim/cosmos3/transfer-action-output/guardrail-workaround-20260918/<case>/`,
named `tensorrt-llm.mp4`, beside the previously saved `cosmos-framework.mp4`
reference. These are local evidence paths, not public downloads. The Framework
references were not rerun; visual comparison is not a pixel-parity claim.

The documented `nvidia/Cosmos3-Nano` model ID resolved to snapshot
`e59a53c25979a090fa8706c9acc0c254a6e89b92`. A read-only cache comparison
(`20260918-211533-9a4325`) confirmed that its model safetensors files share
the same cached data as the earlier `7a312c...` snapshot. The snapshot IDs
are nevertheless recorded separately.

This is not a warning-free runtime. Guardrail 0.3.0 still warns about its
intentionally disabled video-content classifier; its text checks and
RetinaFace postprocessing remain configured. Native MPI optional-plugin,
torchao compatibility, and upstream deprecation/shape-warmup warnings are
retained in the logs. The notebook execution harness also reported an
IPython history-file error on shared storage, and Python reported a leaked
semaphore at shutdown. None is silently filtered out or counted as validation
of the disabled video-content classifier.

## September 7 merge-hold issue

The runtime issue behind the
[Mon 7 Sep 2026 merge hold](https://github.com/NVIDIA/cosmos/pull/310#issuecomment-5571009086)
concerned the distilled model. The distilled T2I/I2V examples were removed from
this PR in [7ff4a86](https://github.com/NVIDIA/cosmos/commit/7ff4a8686f52e6fdd32ebf449171fec9b0695444),
so that issue no longer applies to this PR's scope. This addresses the review
concern by excluding the affected examples; it does not claim an upstream fix
or a passing distilled-model test.

## Separate runtime observations and validation limits

The following observations qualify the retained generation evidence. They are
separate from the distilled-model issue behind the September 7 merge hold.

| Reported issue | Evidence and current scope |
| --- | --- |
| Forward-dynamics visual corruption on the reviewer's GB200 runtime | Later H100 and GB200 runs produced coherent output. The native GB200 run `20260910-180258-e3d281` executed all FD cells, used default GPU RNG, and produced 61 frames at 832×480/10 fps. It used the ARM64 rc26 wheel plus the upstream action image-bytes decoder fix, not the reviewer's exact daily container. The original corruption was not reproduced, and its root cause is not established. |
| Transfer videos did not play in Firefox | The notebook explicitly requests MP4, checks the response MIME type and MP4 signature, and embeds the result. All five later Transfer responses were MP4. Actual Firefox playback was not tested in these runs. |
| VisualGen launch | The documented launch explicitly selects VisualGen with `--visual_gen_args`. The tested Nano server launched one VisualGen worker and served requests successfully. |
| Guardrails | The requests set `use_guardrails=True`. In `cosmos_guardrail==0.3.0`, `No safety models found, returning safe` comes from the intentionally empty video-content classifier list, not from failed loading of every safety component. Its constructor configures Blocklist and Qwen3Guard text checks plus RetinaFace face blurring. The separate [Thu 17 Sep 2026 follow-up](https://github.com/NVIDIA/cosmos/pull/310#issuecomment-5722101980) reports HTTP 500 from `pathsec` with Guardrail cache symlinks. The [server-side NLTK setup](../README.md#server-side-nltk-data-setup) materializes those resources without disabling path security. The fresh run above passed both Action notebooks and all five Transfer cases with that workaround and the unmodified guardrail 0.3.0 package. The disabled video-content classifier remains unvalidated. |

The Thu 17 Sep follow-up separately reports successful FD generation on H100
and H200 with rc26/rc27 images and a source overlay. It explicitly did not rerun
inverse dynamics, Transfer, or GB200; those results must not be extended to
those cases.

This validation matrix covers **Nano** Action and Transfer. It does not certify
the shared Super launch, every audiovisual example, optional raw-source control
preprocessing, or later TensorRT-LLM revisions. The evidence supports the
documented generation request contract, the dimensions above, and a working
default guardrail 0.3.0 configuration with the NLTK workaround. It is not a
safety-efficacy benchmark and does not certify the disabled video-content
classifier.
