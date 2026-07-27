# Cosmos3 Certified NIM documentation source map

> Local authoring artifact. Do not commit this file.
>
> Purpose: preserve source locations, authority, discrepancies, and the agreed
> information architecture while the public documentation is researched and
> written.

## Project objective

Produce standalone, human- and AI-readable documentation for the unified
Cosmos3 Certified NIM under `cookbooks/cosmos3/nim` on branch
`egor/cosmos3_nim_docs`.

The final guide set must cover deployment and NGC authentication, the Generator
and Reasoner runtime surfaces, generation, reasoning, action, transfer,
configuration, production operations, release identity, license notices, and
third-party acknowledgements. It should reuse established Cosmos cookbook
conventions without inheriting obsolete API or product limitations, and must
meet the previous official Generator documentation's topic-coverage floor.

This discovery phase does not replace the public `README.md` or draft the public
guides. It establishes the evidence map and page boundaries first.

## Reviewed repository snapshots

Reviewed on 2026-07-27.

| Repository | Local path | Branch | Reviewed commit | Role |
| --- | --- | --- | --- | --- |
| Cosmos cookbook | `/Users/ekrivov/projects/cosmos` | `egor/cosmos3_nim_docs` | `8061d42e261b666e2c7718a5fed63674f60c3db7` | Target and cookbook consistency references |
| Cosmos3 Certified NIM | `/Users/ekrivov/projects/cosmos-genai-nim` | `cosmos3` | `63578446d6c6eaeffc4b2a378f24bf1c9027494b` | Current product implementation and primary authority |
| NIM product documentation | `/Users/ekrivov/projects/documentation` | `main` | `9a81f6952ca0567b616ca7ce5c412950613e8dc7` | Previous Generator and Reasoner documentation |
| Cosmos framework | `/Users/ekrivov/projects/cosmos-framework` | `main` | `fbb5c9bf4b1298a09cabbe8d60389ef06ab60821` | Model terminology and non-NIM behavior only |

Before authoring or updating public docs, refresh the commit values and inspect
changes to the primary source files listed below.

## Authority and conflict resolution

Use this order whenever sources disagree:

1. Current runtime implementation, request models, profile/configuration code,
   and tests in `cosmos-genai-nim/cosmos3`.
2. OpenAPI returned by the reviewed Certified NIM image at runtime, when the
   image is available for validation.
3. Current scripts under `cosmos-genai-nim/cosmos3/examples`.
4. Current `cosmos-genai-nim/cosmos3/documentation.md`.
5. Previous public Generator and Reasoner product documentation.
6. Current Cosmos cookbook NIM and vLLM-Omni examples.
7. Cosmos Framework docs and examples for model concepts only.

Rules:

- Treat implementation and validation tests as API truth.
- Treat examples as the preferred source for minimal, known-good request bodies.
- Treat `documentation.md` as a broad operational inventory, not as an exact
  schema when it conflicts with code.
- Treat previous product docs as reusable explanations and information
  architecture, not as current names, defaults, limits, or support claims.
- Treat vLLM-Omni examples as workflow analogues. Never copy its endpoints or
  request field names into Certified NIM docs without translating and verifying
  them.
- Record unresolved release facts in the TBD ledger. Do not publish guesses.

## Resolved facts for source snapshot `63578446`

These facts are proven by the checked-in implementation and generated profile
export. They are safe inputs to the draft, but release-sensitive facts must
still be rechecked against the published image/manifest before publication.

### Runtime dispatch and defaults

- The exported manifest contains both `generator` and `reasoner` profiles.
- One selected profile starts one API/backend: Generator uses `pytriton` and
  Reasoner uses `vllm`.
- With no model-type selector, startup chooses `generator`.
- Unpinned selection softly prefers `model_size=nano` and `precision=fp8`.
- Generator additionally softly prefers `profile=latency` and then the offload
  order `none`, `model`, `layer`.
- Soft defaults are skipped rather than failing when they would empty the
  compatible candidate set.
- Generator and Reasoner share the `model_size`, `precision`, `n_gpus`, and
  `nim_*` selector axes. `profile=latency|throughput` is Generator-only and is
  rejected for Reasoner.
- `NIM_MODEL_TYPE`, `NIM_MODEL_SIZE`, `NIM_PRECISION`, `NIM_PERF_PROFILE`, and
  `NIM_OFFLOAD_MODE` are shorthands for selector tags. Explicit
  `NIM_MODEL_PROFILE` pins one exact manifest profile.

Primary evidence:

- `serving_stack/profile_selection/selection.py:14-27,48-78,128-219`
- `serving_stack/environment.py:224-351`
- `serving_stack/inference.py:49-113`

### Exported profile grid

`local_nimcraft/nimcraft_export/profiles.json` contains 39 rows:

- 32 Generator profiles.
- 7 Reasoner profiles.

Active Generator artifact/layout combinations:

| Size | Precision | Offload | Minimum GiB/device | GPU/profile behavior |
| --- | --- | --- | ---: | --- |
| Nano | BF16 | none | 42 | 1/2/4/8 GPU latency; 2/4/8 GPU throughput |
| Nano | FP8 | none | 39 | 1/2/4/8 GPU latency; 2/4/8 GPU throughput |
| Nano | FP8 | model | 31 | Single-GPU latency only |
| Nano | FP8 | layer | 26 | Single-GPU latency only |
| Super | BF16 | none | 150 | 1/2/4/8 GPU latency; 2/4/8 GPU throughput |
| Super | FP8 | none | 121 | 1/2/4/8 GPU latency; 2/4/8 GPU throughput |
| Super | FP8 | model | 79 | Single-GPU latency only |
| Super | FP8 | layer | 42 | Single-GPU latency only |

Generator NVFP4 artifact constants exist in `make_profiles.py`, but no Generator
NVFP4 row is active in the exported grid because `vram_profiles.yaml` contains no
active NVFP4 entry. Pure-TP fallback rows are also disabled by
`ENABLE_TP_PROFILES=False`.

Active Reasoner rows:

| Size | Precision | TP / GPUs | Minimum GiB/device | Served model name |
| --- | --- | ---: | ---: | --- |
| Nano | BF16 | 1 | 23.1 | `nvidia/cosmos3-nano-reasoner` |
| Nano | FP8 | 1 | 23.1 | `nvidia/cosmos3-nano-reasoner` |
| Nano | NVFP4 | 1 | 23.1 | `nvidia/cosmos3-nano-reasoner` |
| Super | BF16 | 2 | 41.25 | `nvidia/cosmos3-super-reasoner` |
| Super | BF16 | 1 | 82.5 | `nvidia/cosmos3-super-reasoner` |
| Super | FP8 | 1 | 66.55 | `nvidia/cosmos3-super-reasoner` |
| Super | NVFP4 | 1 | 72.05 | `nvidia/cosmos3-super-reasoner` |

Shared precision compute-capability gates are BF16 >= 8.7, FP8 >= 9.0, and
NVFP4 >= 10.0. Reasoner Super BF16 uses TP2 below the TP1 floor and TP1 at or
above 82.5 GiB/device. Reasoner profiles do not encode latency/throughput.

Primary evidence:

- `local_nimcraft/nimcraft_export/profiles.json`
- `local_nimcraft/make_profiles.py:9-54,122-241,327-400`
- `serving_stack/profile_selection/vram_profiles.yaml`
- `serving_stack/profile_selection/hardware.py:8-74`

### API capability boundary

- Generator exposes JSON `POST /v1/infer` and returns base64-encoded VP9/MP4,
  with optional action metadata.
- Current request-mode inference covers T2V, I2V, V2V, transfer, forward
  dynamics, policy, and inverse dynamics.
- Sound conditioning/output is not surfaced by this NIM source snapshot.
- Reasoner uses the OpenAI-compatible Chat Completions surface, inherited
  Responses routes, and streaming.
- Responses routes can be disabled, and persisted response state is not on by
  default.
- Reasoner served names are fixed by source to
  `nvidia/cosmos3-nano-reasoner` and `nvidia/cosmos3-super-reasoner`.
- Generator BYOC in this snapshot replaces the diffusion model only; Reasoner
  BYOC is not implemented by `NIM_FT_CHECKPOINT`.

Primary evidence:

- `serving_stack/data_models/generation.py:52-64,322-387`
- `serving_stack/data_models/actions.py:43-61`
- `serving_stack/data_models/transfer.py:124-186`
- `serving_stack/data_models/responses.py:20-34`
- `serving_stack/reasoner_inference.py:39-44,472-603`
- `serving_stack/environment.py:186-216`
- `examples/README.md:19-80`

### Authoring-ready Generator API contract

This inventory is intentionally concise. The public `api-reference.md` should
explain these fields in tables and link to task guides for combinations. It
should not reproduce Pydantic implementation details or error-message text.

All Generator capabilities use `POST /v1/infer`. Unknown fields are rejected.
The mode is inferred from the request shape:

| Mode | Required discriminator/input | Forbidden or mode-specific rule | Current example |
| --- | --- | --- | --- |
| T2V | Non-empty `prompt`; no media | Ordinary generation frame rules apply | `examples/t2v.py`; smaller smoke request in `examples/low_payload.py` |
| I2V | `image` | `image` and `video` are mutually exclusive | `examples/i2v.py` |
| V2V | `video` without `transfer` or `action_params` | V2V conditioning controls are valid only here | `examples/v2v.py` |
| Transfer | Non-empty `transfer` | Cannot combine with `image`, `action_params`, or V2V conditioning controls | `examples/transfer.py` |
| Forward dynamics | `image` plus `action_params.mode=forward_dynamics` | Requires an input action trajectory | `examples/action.py` |
| Policy | `image` plus `action_params.mode=policy` | Produces rather than accepts an action trajectory | `examples/action.py` |
| Inverse dynamics | `video` plus `action_params.mode=inverse_dynamics` | Produces rather than accepts an action trajectory | `examples/action.py` |

Shared top-level request fields:

| Field | Contract in source snapshot |
| --- | --- |
| `prompt` | Optional string, at most 20,000 characters. Required only when no image, video, or transfer input establishes a request. Normalized to an empty string for media/action requests when omitted. |
| `negative_prompt` | Optional string, at most 20,000 characters. Omission selects the vendored structured OSS negative prompt; an explicit empty string disables it. |
| `image` | Base64, image data URL, or public HTTP(S) URL; at most 20,000,000 encoded characters. Empty/whitespace input is treated as absent. |
| `video` | Base64, video data URL, or public HTTP(S) URL; at most 100,000,000 encoded characters and 75 MB after decoding. Empty/whitespace input is treated as absent. |
| `seed` | Optional integer >= 0. Generated by the service when omitted. Public examples always set `0` for reproducibility. |
| `guidance_scale` | Finite JSON number in `[1.0, 7.0]`; ordinary default `6.0`. |
| `steps` | JSON integer in `[1, 100]`; ordinary default `35`. |
| `flow_shift` | Finite JSON number; default `10.0`. No additional range constraint is present. |
| `resolution` | One of 18 keys across `256`, `480`, and `720` tiers, each with bare/`16_9`, `1_1`, `9_16`, `4_3`, and `3_4` spellings. Bare tiers mean 16:9. Default `720`. |
| `num_output_frames` | JSON integer on the `4k+1` cadence. Ordinary generation defaults to `189`, requires at least `25`, and caps output at `397`/`297`/`197` frames for the 256/480/720 tiers. |
| `fps` | Finite JSON number in `[1.0, 60.0]`; ordinary default `24.0`. Source recommends 10–30 for quality. |
| `condition_frame_indexes_vision` | V2V-only latent-frame indexes. Normalized to sorted, unique, non-negative integers; defaults to `[0, 1]`. The largest index must fit the requested output latent-frame range. |
| `condition_video_keep` | V2V-only `first` or `last`; defaults to `first`. |
| `transfer` | Nested transfer-control object described below. |
| `action_params` | Nested action-control object described below. |

Resolution shapes are canonical model shapes, not a mathematical resize of the
tier name. For example: `256_16_9` is 320x192, `480_16_9` is 832x480, and
`720_16_9` is 1280x720. The public reference should include the complete 18-row
key-to-shape table from `resolutions.py`, because the non-16:9 shapes are not
obvious from the key alone.

Primary evidence:

- `serving_stack/data_models/generation.py:38-205,208-410`
- `serving_stack/data_models/prompts.py:16-46`
- `serving_stack/resolutions.py:49-113`
- `serving_stack/generator_inference.py:164-314,528-537`

### Authoring-ready Action contract

`action_params` fields:

| Field | Contract in source snapshot |
| --- | --- |
| `mode` | Required: `forward_dynamics`, `policy`, or `inverse_dynamics`. |
| `domain_name` | Required: `av`, `bridge_orig_lerobot`, `droid_lerobot`, or `umi`. Maps to domain IDs 1, 7, 8, and 6 respectively. |
| `action_chunk_size` | Required positive multiple of 4. The service derives `num_output_frames = action_chunk_size + 1`. Typical values are 60 for AV and 16 for the robot domains. |
| `action` | `[T, D]` numeric trajectory. Required only for forward dynamics; forbidden for policy and inverse dynamics. `T` must equal `action_chunk_size`; `D` must equal `raw_action_dim`. |
| `raw_action_dim` | Optional but, if set, must match the domain default: 9 for AV, 10 for the other three domains. |
| `action_space` | `joint_pos` or `midtrain`; default `joint_pos`. |
| `image_size` | `256`, `480`, `704`, or `720`; default `480`. Integer spellings are normalized to strings. This is distinct from top-level `resolution`. |
| `action_fps` | Optional number in `[1.0, 60.0]`. |
| `history_length` | Optional integer >= 1; policy-only. |
| `use_state` | Optional boolean; policy-only. |
| `observation` | Optional free-form object passed to the pipeline without schema validation; policy-only. |

Action requests prohibit top-level `resolution`, `transfer`, and V2V
conditioning controls. Unless explicitly overridden, action mode changes the
top-level defaults to `steps=30`, `guidance_scale=1.0`, and `fps=10.0`.
`num_output_frames`, if supplied, must equal `action_chunk_size + 1`.

The response always contains video. Policy and inverse dynamics additionally
return `action` with `data`, two-element `shape`, `dtype="float32"`,
`raw_action_dim`, `action_mode`, and `domain_id`; forward dynamics returns
`action: null`.

Primary evidence:

- `serving_stack/data_models/actions.py:24-167,169-307`
- `serving_stack/data_models/generation.py:232-239,337-345`
- `examples/action.py:35-140`

### Authoring-ready Transfer contract

`transfer` can enable one or more of `edge`, `blur`, `depth`, `seg`, and `wsm`:

| Control | Accepted input | Special rule |
| --- | --- | --- |
| `edge` | `true` or object with optional `video` / `preset_edge_threshold` | Without a nested video, derives control from top-level `video`. Presets: `very_low`, `low`, `medium`, `high`, `very_high`. A preset cannot accompany a nested video. |
| `blur` | `true` or object with optional `video` / `preset_blur_strength` | Without a nested video, derives control from top-level `video`. Presets: `none`, `very_low`, `low`, `medium`, `high`, `very_high`. A preset cannot accompany a nested video. |
| `depth` | Object with required non-empty `video` | Precomputed control only. |
| `seg` | Object with required non-empty `video` | Precomputed control only. |
| `wsm` | Object with required non-empty `video` | Precomputed control only. |

At least one control must be enabled. A nested control video uses the same
base64/data-URL/public-URL contract and 100,000,000-character ceiling as the
top-level video. Transfer-level tuning fields are:

| Field | Constraint/default |
| --- | --- |
| `control_guidance` | Optional number in `[0.0, 10.0]`; effective default 1.5 generally, 2.0 for segmentation-only, and 3.0 for WSM-only. |
| `num_video_frames_per_chunk` | Integer >= 1; effective default 93 generally and 101 for WSM-only. |
| `num_conditional_frames` | Integer >= 0, default 1, and strictly smaller than the effective chunk size. |
| `num_first_chunk_conditional_frames` | Integer >= 0, default 0; cannot exceed chunk size or output frames. Values above 0 require top-level `video`. |

When the caller omits shared sampling fields, transfer mode uses `steps=50` and
`flow_shift=10.0`. Edge/blur/depth and mixed-control requests default to
`num_output_frames=121`, `fps=30.0`, and `guidance_scale=3.0`.
Segmentation-only uses the same values but a 2.0 control guidance default.
WSM-only defaults to 101 frames, 10 fps, guidance 1.0, control guidance 3.0,
and a 101-frame chunk. Multi-control requests deliberately use the general
edge-family defaults rather than combining per-control defaults.

Primary evidence:

- `serving_stack/data_models/transfer.py:28-169,172-253`
- `examples/transfer.py:23-181`

### Authoring-ready response and Reasoner contract

Generator success response:

| Field | Contract |
| --- | --- |
| `b64_video` | Required raw base64 string containing a VP9-encoded MP4. |
| `action` | Predicted action object for policy/inverse dynamics; otherwise `null`. |

Generator schema/media/guardrail validation generally returns HTTP 422;
unexpected internal failures return HTTP 500. Public docs should describe the
stable status semantics and example error envelope, not promise exact message
text.

Reasoner profiles expose OpenAI-compatible completion APIs rather than
`/v1/infer`. The authoring baseline is:

| Surface | Proven behavior/example |
| --- | --- |
| `POST /v1/chat/completions` | Image and video messages, non-streaming; `examples/reasoner.py`. |
| `POST /v1/chat/completions` with `stream=true` | Streaming deltas; `examples/reasoner_stream.py`. |
| `POST /v1/completions` | Registered and passed through the same normalization/validation middleware; no first-party example yet. |
| `POST /v1/responses` | Image input using `input_image` before `input_text`; `examples/reasoner_responses.py`. |
| `GET /v1/responses/{response_id}` | Registered through NIMlib when Responses routes are enabled; requires stored state for meaningful retrieval. |
| `POST /v1/responses/{response_id}/cancel` | Registered through NIMlib when Responses routes are enabled; background/stored response behavior depends on storage configuration. |

Completion requests must include a non-empty `model`. The service injects
sampling defaults `temperature=0.7`, `top_k=20`, and `top_p=0.8` when omitted.
It validates temperature in `[0,2]`, top-p in `(0,1]`, and top-k as `-1` or an
integer >= 1. It normalizes OpenAI `response_format` and legacy guided-decoding
fields into vLLM structured outputs. The full inherited request/response schema
must come from the released image's live OpenAPI instead of being hand-copied.

Current first-party examples establish these multimodal shapes:

- Chat image: `image_url` content before `text`, with a data URL.
- Chat video: `video_url` content before `text`, with per-request
  `media_io_kwargs.video.fps=4.0`.
- Responses image: `input_image` before `input_text`, `detail=auto`, and
  `store=false`. Keep video examples on Chat Completions until verified on the
  Responses surface.
- Default per-prompt limits are five images and one video, configurable with
  `NIM_MAX_IMAGES_PER_PROMPT` and `NIM_MAX_VIDEOS_PER_PROMPT`.
- `NIM_DISABLE_RESPONSES_ROUTE=true` removes all three Responses routes.
  Persistent state is disabled by default; NIMlib/vLLM uses
  `VLLM_ENABLE_RESPONSES_API_STORE=1` when retrieval, cancellation,
  background responses, or `previous_response_id` require it.

Primary evidence:

- `serving_stack/data_models/responses.py:20-34`
- `serving_stack/generator_inference.py:106-160,298-314`
- `serving_stack/workflow.py:122-140`
- `serving_stack/reasoner_inference.py:39-50,332-388,391-587`
- `serving_stack/environment.py:419-508`
- `serving_stack/tests/test_reasoner_inference.py:760-792`
- `examples/reasoner.py:15-57`
- `examples/reasoner_stream.py:21-82`
- `examples/reasoner_responses.py:15-35`
- `examples/README.md:56-68`

### Example coverage and cookbook adaptation rules

| First-party source example | Coverage to preserve in cookbook form |
| --- | --- |
| `examples/t2v.py` | Canonical 720p/189-frame T2V request with an explicit seed. |
| `examples/low_payload.py` | Faster 480p/49-frame smoke request. Keep it distinct from the full-quality example. |
| `examples/i2v.py` | Local image converted to a data URL, then I2V request. |
| `examples/v2v.py` | Pinned public video URL plus explicit latent conditioning controls. |
| `examples/action.py` | Three AV forward trajectories, AV policy, and Bridge inverse dynamics. |
| `examples/transfer.py` | Five precomputed control types plus server-derived edge and blur. |
| `examples/reasoner.py` | Chat image caption and Chat video caption via the OpenAI client. |
| `examples/reasoner_stream.py` | Streaming Chat image caption. |
| `examples/reasoner_responses.py` | Responses API image caption with stateless storage setting. |
| `examples/common.py` | Readiness check, media-to-data-URL helper, long timeout, request/response validation, and artifact layout. |

Cookbook scripts should be standalone copies adapted to the cookbook directory,
not imports from the private NIM source tree. Preserve explicit request
dictionaries, a deterministic seed for Generator examples, readiness checks,
large-request timeouts, compact logging for base64 inputs, exact request
artifacts, decoded `vision.mp4`, predicted `action.json`, and full Reasoner
response artifacts. Centralize those mechanics in `examples/common.py` so each
task script stays readable.

### Ports and identity caveat

- Container-internal ports are 8000 HTTP, 8001 gRPC, and 8002 metrics.
- The local Makefile maps these to 18000/18001/18002 by default; this is a local
  collision-avoidance convention, not the public container contract.
- Local build identity is inconsistent and not sufficient to establish the
  public release image: the old NIMCraft YAML says `cosmos3-generator`, the
  local Makefile uses `cosmos3-gen`, the build-context `VERSION` is `0.1.0`, and
  the local Makefile defaults to `1.0.0`.
- `local_nimcraft/nim-config.yaml` predates the July Reasoner/profile work and
  still describes a Generator/text-to-video NIM. NIMCraft service state is the
  external authority for public image identity.

Therefore the public image repository/tag and final product name remain TBD.

## Primary Certified NIM sources

All paths in this section are relative to
`/Users/ekrivov/projects/cosmos-genai-nim/cosmos3`.

### Product guide and machine-readable API

| Source | Use it for | Caveats |
| --- | --- | --- |
| `documentation.md` | Broad inventory: overview, endpoints, request fields, deployment, configuration, profiles, support, BYOC, Helm, observability, guardrails, troubleshooting | Still titled `Cosmos3-Generator API Guide`; portions describe only T2V/I2V and contain launch placeholders. Reconcile every value with code. |
| `api_spec.yaml` | Management endpoints and a static OpenAPI snapshot | Older than several current schema changes and does not represent the dynamically mounted Reasoner API. Prefer current models and live `/openapi.json`. |
| `README.md` | NIM developer/build context | Explicitly not the public user guide; use only to understand architecture and build/runtime terminology. |
| `AGENTS.md` | Repository ownership and runtime separation | Key invariant: one selected profile launches either Reasoner or Generator; the two API surfaces are not simultaneously active. |

Important `documentation.md` regions:

- Lines 1-43: overview and hardware summary.
- Lines 45-59: management and Generator endpoint inventory.
- Lines 61-325: quickstarts, request fields, validation, response, and errors.
- Lines 327-479: Docker deployment.
- Lines 481-665: environment variables, profile selection, and support matrix.
- Lines 667-735: Generator BYOC.
- Lines 737-813: Helm/Kubernetes and observability.
- Lines 815-868: guardrails and troubleshooting.
- Lines 870-880: unresolved placeholders and further sources.

### Runtime dispatch, configuration, and profiles

| Source | Use it for |
| --- | --- |
| `serving_stack/inference.py` | Unified dispatch from the selected manifest profile to `generator_inference` or `reasoner_inference` |
| `serving_stack/generator_inference.py` | Generator HTTP interface and workflow wiring |
| `serving_stack/reasoner_inference.py` | Reasoner NIMlib/vLLM interface, request normalization, streaming, and route behavior |
| `serving_stack/environment.py` | Environment-variable parsing, profile selector shorthands, Reasoner tuning, BYOC, guardrails, logging, and media URL policy |
| `serving_stack/prompt_upsampling.py` | Optional Generator T2V/I2V prompt rewriting through an operator-supplied OpenAI-compatible endpoint |
| `serving_stack/profile_selection/` | Hardware discovery, selection criteria, supported tags, and selection cascade |
| `local_nimcraft/make_profiles.py` | Profiles emitted into the NIM manifest, model identities, GPU layouts, and VRAM constraints |
| `local_nimcraft/nimcraft_export/profiles.json` | Generated profile inventory for the reviewed source snapshot |
| `benchmarking/baseline_performance_benchmarks.csv` | Evidence for tested performance configurations; not a substitute for the supported-profile manifest |

### Generator request and response contract

| Source | Use it for |
| --- | --- |
| `serving_stack/data_models/generation.py` | Top-level `/v1/infer` request, defaults, limits, supported resolution keys, mode inference, mutual exclusions, and validation |
| `serving_stack/data_models/actions.py` | Forward dynamics, policy, and inverse dynamics fields, domains, shapes, and validation |
| `serving_stack/data_models/transfer.py` | Transfer control types, derived/precomputed inputs, multi-control rules, and chunking parameters |
| `serving_stack/data_models/responses.py` | Generator response, base64 video, and optional action metadata |
| `serving_stack/resolutions.py` | Canonical resolution-key to pixel-shape mapping |
| `serving_stack/media_utils.py` | Accepted media representations, URL fetching, base64/data-URL handling, and size constraints |

Current top-level mode inference in `generation.py`:

| Inputs | Mode | Output surface |
| --- | --- | --- |
| Non-empty prompt, no media/control | Text-to-video | Base64 MP4 |
| `image` | Image-to-video | Base64 MP4 |
| `video`, without transfer/action | Video-to-video | Base64 MP4 |
| `transfer` | Transfer generation | Base64 MP4 |
| `action_params.mode=forward_dynamics` | Action-conditioned rollout | Base64 MP4 |
| `action_params.mode=policy` | Policy rollout | Base64 MP4 plus action metadata |
| `action_params.mode=inverse_dynamics` | Inverse dynamics | Base64 MP4 plus action metadata |

### Reasoner contract

The Reasoner surface is inherited from NIMlib/vLLM rather than described by the
Generator Pydantic models. Verify it using these sources together:

| Source | Use it for |
| --- | --- |
| `serving_stack/reasoner_inference.py` | Enabled routes, engine arguments, model-name resolution, normalization, errors, and streaming behavior |
| `serving_stack/environment.py` (`ReasonerEngineOptions`) | Prompt media limits and Reasoner-specific runtime variables |
| `examples/reasoner.py` | Image and video Chat Completions requests |
| `examples/reasoner_stream.py` | Streaming Chat Completions request and delta handling |
| `examples/reasoner_responses.py` | Responses API image request |
| `serving_stack/tests/test_reasoner_inference.py` | Route inheritance, disable flags, normalization, errors, and streaming coverage |
| `serving_stack/tests/test_examples.py` | Request ordering, local assets, model names, and example execution behavior |

Known current Reasoner routes from the examples and tests:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `POST /v1/responses/{response_id}/cancel`
- `GET /v1/models`
- NIM management and health endpoints exposed by the selected interface

The Responses storage/retrieve/background features depend on runtime flags. Do
not imply persistent response storage is on by default.

### Canonical runnable examples

| Source | Capability |
| --- | --- |
| `examples/t2v.py` | Text-to-video |
| `examples/low_payload.py` | Minimal/default-heavy Generator request |
| `examples/i2v.py` | Image-to-video using local media |
| `examples/v2v.py` | Video-to-video using URL media |
| `examples/reasoner.py` | Image and video Chat Completions |
| `examples/reasoner_stream.py` | Reasoner streaming |
| `examples/reasoner_responses.py` | Reasoner Responses API |
| `examples/action.py` | AV forward dynamics, AV policy, and Bridge inverse dynamics |
| `examples/transfer.py` | Precomputed and server-derived transfer controls |
| `examples/common.py` | Readiness, request transport, OpenAI client adaptation, output decoding, and artifact layout |
| `examples/README.md` | Intended execution flow, server URL, output artifacts, limits, and route notes |

Prefer the dictionaries in these files over examples embedded in
`documentation.md` when they differ. The scripts are covered offline by
`serving_stack/tests/test_examples.py`.

### Current `documentation.md` is a first-party coverage floor

Every numbered source-guide section must have a deliberate destination or
correction. The implementation remains authoritative when the prose conflicts
with code or the current profile export.

| Source-guide section | Required treatment | Planned destination/status |
| --- | --- | --- |
| 1. Overview and prerequisites | Rewrite around the unified selected Generator/Reasoner runtime; preserve prerequisites categories but not unverified counts, versions, or stale Generator-only profile summaries | `README.md`, `deployment.md`; release identity/support values TBD |
| 2. Endpoints at a glance | Separate endpoints by active runtime and distinguish shared management routes from Generator/Reasoner inference | `api-reference.md`, `operations.md`; live OpenAPI capture pending |
| 3. T2V/I2V quickstart | Preserve minimal requests and MP4 decoding; use the portable Python encoder/decoder and add V2V/current modes elsewhere | `README.md`, `generation.md`, `examples/` |
| 4. Request reference | Rebuild from current Pydantic models, including `video`, action, and transfer; correct the stale 2,000-character heading | `api-reference.md` |
| 5. Frame-count math | Preserve 4k+1 explanation, per-tier caps, and action/transfer-specific frame rules | `api-reference.md`, summarized in task guides |
| 6. Image input requirements | Preserve raw base64/data URL/public URL concepts and media size/error behavior; prefer MIME-aware helpers | `api-reference.md`, `generation.md` |
| 7. Supported resolutions | Generate exact key-to-pixel mappings from `resolutions.py` and distinguish ordinary generation from action templates | `api-reference.md` |
| 8. Response format | Preserve `b64_video`, optional action metadata, decoding, artifact output, and playback caveat | `api-reference.md`, `examples/common.py`, `operations.md` |
| 9. Validation cheat sheet | Replace the T2V/I2V-only table with mode-complete current constraints and link task-specific rules | `api-reference.md`, task guides |
| 10. Error envelope | Preserve stable status/type semantics without copying exact mutable messages; add Reasoner 400/422 behavior | `api-reference.md`, `operations.md` |
| 11. Deployment | Preserve NGC login, cache, ports, Docker flags, single/multi-GPU concepts, throughput/latency, cleanup, and cold-start notes | `deployment.md`; image/profile values release-gated |
| 12. Environment variables | Re-audit every variable against `environment.py`; split launch/profile settings from diagnostics and backend tuning | `deployment.md`, `operations.md` |
| 12.1 Prompt upsampling | Document the current optional, Generator-only T2V/I2V flow, secret handling, supported template styles, failure fallback, and non-applicable modes | `deployment.md`, `generation.md`, `operations.md` |
| 13. Profile selection | Preserve selectors, conflicts, pinning, soft defaults, layouts, and selection cascade from current code | `deployment.md`; released manifest recheck required |
| 14. Support matrix | Replace the stale prose grid with the released manifest-derived tested/compatible matrix | `deployment.md`; release matrix TBD |
| 15. BYOC | Preserve layout, mount, selector cross-check, verification, and operational notes; state Generator/diffusion-only boundary | `deployment.md`; published BYOC statement TBD |
| 16. Helm/Kubernetes | Preserve topic and operational categories but not placeholder chart identity or unverified values | `deployment.md`; chart/values TBD |
| 17. Observability | Preserve inspection, log, distributed-diagnostic, and Prometheus/Grafana workflows; capture current endpoints/metrics from release | `operations.md`; live scrape/log validation pending |
| 18. Guardrails | Preserve text/video/SigLIP controls, ordering, 422 behavior, BYOC separation, and risk of disabling; scope to Generator runtime | `operations.md` |
| 19. Troubleshooting | Preserve symptom/cause/fix organization, revalidate every command/variable, and add current Reasoner/action/transfer failures | `operations.md` |
| 20. Outstanding TBDs | Merge into the maintained release-dependent ledger; never silently resolve from old image facts | This `SOURCES.md`; public values visibly TBD |
| 21. See also | Link only public authorities useful to cookbook users; exclude private build/NIMCraft guidance | `README.md`, relevant focused pages |

Coverage is complete only when every row has evidence in its mapped public page
or remains visibly deferred. The source guide's numbered structure should not be
copied; the chosen hub-and-spoke page ownership remains canonical.

### Prompt upsampling contract discovered from current code

- `NIM_ENABLE_PROMPT_UPSAMPLING` is off by default and consumed only by the
  Generator backend.
- The current request path applies it only when `action_params`, `transfer`, and
  top-level `video` are absent: ordinary T2V and I2V. V2V, action, and transfer
  requests are not upsampled.
- When enabled, current startup validation requires an endpoint URL, model, API
  key value, a valid `external_api|reasoner` template style, and the bundled
  template files. A Reasoner-profile container does not require this config.
- The endpoint is normalized to an OpenAI-compatible
  `/v1/chat/completions` URL and receives Bearer authorization. Public examples
  must not imply compatibility with a provider's native, non-OpenAI API.
- I2V sends the conditioning image to a vision-capable upsampler as a data URL.
- The NIM strips the Reasoner-only `scene_imagination` scratch field and pins
  resolution, aspect ratio, duration, and FPS from the original generation
  request so the upsampler cannot change output shape.
- Request-time endpoint, timeout, or response-parsing failures log a warning and
  fall back to the original prompt instead of failing generation.
- `NIM_PROMPT_UPSAMPLING_API_KEY` is a separate external-service secret from
  `NGC_API_KEY`. Neither value may appear in logs, examples, saved payloads, or
  source control.

Primary evidence:

- `serving_stack/environment.py:866-949`
- `serving_stack/prompt_upsampling.py:11-229`
- `serving_stack/generator_inference.py:238-260`
- `serving_stack/tests/test_prompt_upsampling.py:38-307`
- `documentation.md:513-549`

### First-party example adaptation matrix

| First-party source | Public treatment | Asset/port/runtime correction |
| --- | --- | --- |
| `examples/README.md` | Preserve directly editable dictionaries, output-artifact contract, route notes, media limits, and URL policy | Rewrite for cookbook execution; do not expose `make reasoner-brr` or NIM developer setup |
| `examples/common.py` | Adapt readiness, portable data URLs, JSON transport, OpenAI extension handling, base64 MP4/action/text artifact writing, and summarized logs | Default public `NIM_URL` to `http://localhost:8000`; use target-repo license and dependencies |
| `examples/low_payload.py` | Merge its short/default-heavy request into the landing or generation quickstart | Do not create a second near-duplicate T2V script unless review finds a distinct user need |
| `examples/t2v.py` | Adapt as the canonical complete T2V script | Keep deterministic seed and current NIM field names |
| `examples/i2v.py` | Adapt as the canonical I2V script | Use an existing approved cookbook image rather than copying the NIM-only `sports_car.png` blindly |
| `examples/v2v.py` | Adapt as the canonical V2V script | Preserve conditioning controls; use local/pinned approved media and validate URL policy |
| `examples/action.py` | Adapt all three modes; retain one AV forward case rather than three directions in the default run | Correct source prose: policy is AV and inverse dynamics is Bridge; reuse existing cookbook AV assets |
| `examples/transfer.py` | Adapt one request per distinct precomputed/derived shape and expose direct case selection instead of always running seven expensive cases | Reuse existing cookbook prompts/control videos; do not copy the moving `main` asset URL as if pinned |
| `examples/reasoner.py` | Adapt image/video Chat Completions | Reuse existing cookbook media; discover the served model dynamically |
| `examples/reasoner_stream.py` | Adapt streaming delta handling | Keep separate from non-streaming artifact helper for clarity |
| `examples/reasoner_responses.py` | Adapt the non-streaming image Responses request and state caveats | Keep video on Chat Completions until current Responses video support is proven |
| `examples/__init__.py` | No public artifact required | Public scripts can import sibling `common.py` using the documented execution directory |

### Reuse existing cookbook assets instead of copying private-source files

Static SHA-256 comparison confirmed these first-party NIM assets are already
present byte-for-byte in the target cookbook repository:

| NIM example asset | Existing cookbook asset |
| --- | --- |
| `av_action.jpg` | `generator/action/assets/images/av_0.jpg` |
| `av_forward_trajectory.json` | `generator/action/assets/actions/av_traj_forward.json` |
| `av_left_trajectory.json` | `generator/action/assets/actions/av_traj_left.json` |
| `av_right_trajectory.json` | `generator/action/assets/actions/av_traj_right.json` |
| `robot_153.jpg` | `reasoner/assets/robot_153.jpg` |
| `video_caption.mp4` | `reasoner/assets/video_caption.mp4` |
| Five transfer `prompt.json` files | Matching `generator/transfer/assets/<hint>/prompt.json` files |

The cookbook also already contains the corresponding transfer control videos
and suitable audiovisual conditioning images. Prefer reading these existing
paths and converting them to data URLs at runtime. Do not copy files from the
proprietary NIM source tree into the public cookbook merely because their bytes
match; retain the target repository's existing provenance and license history.

## Previous product documentation

### Previous Cosmos3 Generator and older Cosmos NIMs

Repository root: `/Users/ekrivov/projects/documentation`.

The previous Generator documentation is integrated with Predict and Transfer
under `docs/cosmos/`. Use the page split and operational explanations as a
structural reference:

| Source | Reusable material |
| --- | --- |
| `docs/cosmos/index.rst` | Product landing and navigation pattern |
| `docs/cosmos/introduction.rst` | NIM concepts, Cosmos architecture, runtime positioning |
| `docs/cosmos/prerequisites.rst` | Host, driver, Docker, toolkit, storage, and credential organization |
| `docs/cosmos/quickstart-guide.rst` | NGC login, cache, launch, readiness, and first request flow |
| `docs/cosmos/api-reference.rst` | Endpoint/reference organization and example placement |
| `docs/cosmos/sampling-params.rst` | Lookup-oriented parameter documentation |
| `docs/cosmos/configuration.rst` | Environment variables, GPU selection, shared memory, and volumes |
| `docs/cosmos/support-matrix.rst` | Hardware/profile presentation |
| `docs/cosmos/bring-your-own-checkpoint.rst` | Generator BYOC organization |
| `docs/cosmos/helm.rst` | Kubernetes deployment structure |
| `docs/cosmos/observability.rst` | Metrics and logging structure |
| `docs/cosmos/troubleshooting.rst` | Symptom/cause/fix organization |
| `docs/cosmos/release-notes.rst` | Release identity, version, and capability-change placement |
| `docs/cosmos/EULA.rst` | License notice and in-container license location |
| `docs/cosmos/acknowledgements-cosmos3.rst` | Notice-page placement; includes the generated Markdown acknowledgement inventory |
| `docs/cosmos/acknowledgements-cosmos3-markdown.md` | Historical third-party component/license inventory; do not reuse for a different image build |
| `docs/cosmos/_static/yaml/cosmos3-generator.openapi.yaml` | Historical Generator API only |

The branch `egor/cosmos3-gen` contains the previous Generator documentation
development history. Prefer `main` for the latest merged wording unless a
specific historical rationale is needed.

### Previous Generator documentation is the coverage floor

The new cookbook documentation must cover no less than the durable user-facing
topics in the previous official `Cosmos3-Generator` documentation. This is a
topic-coverage requirement, not permission to copy obsolete product facts.
Every previous topic must be handled in one of four ways:

1. Document the current Certified NIM behavior in the mapped page.
2. Link to an authoritative external source when reproducing generic or legal
   material would create drift.
3. Keep the topic visibly TBD until release evidence exists.
4. Mark it obsolete with the current replacement; never silently omit it.

Parity matrix:

| Previous official topic | Required current treatment | Planned destination/status |
| --- | --- | --- |
| Product landing and navigation | Explain the Certified NIM, selected-profile runtime, supported task families, and routes to every guide | `README.md` |
| NIM container and profile architecture | Explain image pull, first-boot artifact materialization/cache, hardware-aware profile selection, and one active Generator or Reasoner backend | `README.md`, `deployment.md` |
| Cosmos3 model/pipeline architecture | Give enough Reasoner/Generator context to understand capabilities without making unverified parameter-count claims | `README.md` |
| Safety and guardrails | Explain text guardrails, output-video checks/face handling, failure behavior, and the risk of disabling controls | `operations.md` |
| API Catalog, NGC Catalog, and container security reports | Link to the final release catalog/model-card pages once known | `README.md`; release URL TBD |
| NVIDIA Developer Program / entitlement context | Provide a concise external link if still applicable to the released NIM | `README.md`; wording/link recheck required |
| Hardware prerequisites | CPU architecture, RAM, disk, shared memory, supported GPU architecture/count, homogeneity, and VRAM floors | `deployment.md`; release matrix TBD |
| Software prerequisites | Linux, driver, Docker, NVIDIA Container Toolkit, and setup verification with `nvidia-smi` | `deployment.md`; exact versions TBD |
| NGC credential creation | Explain how to create an NGC personal API key with NGC Catalog access | `deployment.md` |
| Credential export and safe handling | Use `NGC_API_KEY`; never place a real key in examples, logs, source control, notebooks, or output artifacts | `README.md`, `deployment.md` |
| Docker authentication | Preserve `echo "$NGC_API_KEY" \| docker login nvcr.io --username '$oauthtoken' --password-stdin` and explain the special username | `README.md`, `deployment.md` |
| Discovering available image tags | Show NGC Catalog/CLI discovery or point to the exact release page rather than hard-coding `latest` | `deployment.md`; public repository/tag TBD |
| Docker launch | Cover image, cache, GPU exposure, shared memory, ulimits, credential injection, port mapping, and selectors for both runtime modes | `README.md` minimal path; `deployment.md` full path |
| Docker flag explanations | Explain every non-obvious flag used by the launch command | `deployment.md` |
| Cache and volume permissions | Explain `/opt/nim/.cache`, persistent reuse, container UID/write access, and cold-cache behavior | `deployment.md` |
| Startup verification | Liveness versus readiness, cold download/compile/warmup delay, and 200-response checks | `README.md`, `deployment.md`, `operations.md` |
| First T2V and I2V requests | Preserve short curl/Python examples and base64 MP4 decoding | `README.md`, `generation.md` |
| Local media encoding | Cover data URLs/base64 and public URL inputs without shell-specific base64 assumptions | `generation.md`, `reasoning.md`, shared `examples/common.py` |
| Stopping and cleaning up | Cover `docker stop`, and only mention kill/removal as recovery/cleanup | `deployment.md` |
| GPU selection | Cover `--gpus`, `CUDA_VISIBLE_DEVICES`, homogeneous devices, and selector-visible GPU count | `deployment.md` |
| IPC/shared memory | Explain `--shm-size`, when `--ipc=host` is relevant, and `/dev/shm` media staging | `deployment.md`; release values TBD |
| Environment-variable reference | Include shared NIM variables plus Generator- and Reasoner-specific variables with defaults, scope, and conflicts | `deployment.md`, `operations.md` |
| Model/profile selectors | Cover `NIM_MODEL_TYPE`, size, precision, Generator performance profile/offload, exact profile pinning, tag selectors, and parallelism aliases | `deployment.md` |
| Volumes | Cover persistent cache and read-only BYOC mounts separately | `deployment.md` |
| Sampling reference | Preserve lookup tables for defaults, ranges, constraints, media types, frame cadence, resolutions, and errors; expand for V2V/Action/Transfer | `api-reference.md` and task guides |
| Endpoint inventory | Separate shared management endpoints, Generator `/v1/infer`, and Reasoner OpenAI-compatible routes | `api-reference.md`, `operations.md` |
| Request and response examples | Include complete request bodies, expected status/shape, decoding, and saved artifacts | Every task guide plus `examples/` |
| Error handling | Cover stable 4xx/5xx semantics and mode-specific validation without promising exact error strings | `api-reference.md`, task guides, `operations.md` |
| OpenAPI | Explain live `/openapi.json`; capture/validate it under both Generator and Reasoner profiles | `api-reference.md`; runtime capture pending |
| Resolution and frame-cap tables | Preserve exact key-to-WxH shapes and per-tier frame caps | `api-reference.md`, summarized in `generation.md` |
| Model sizes, precisions, and VRAM | Provide the release manifest-derived matrix, tested versus compatible distinction, and compute-capability gates | `deployment.md`; release matrix TBD |
| Parallelism/profile selection | Explain latency versus throughput, replicas/sharding, offload, selection cascade, and explicit pinning | `deployment.md` |
| Input/output codecs | State VP9-in-MP4 output and validate all claimed image/video inputs against the release | `api-reference.md`, `generation.md`, `operations.md` |
| BYOC | Preserve mount pattern, expected checkpoint layout, profile cross-check, readiness, cache/ulimits, path rules, and metadata verification | `deployment.md`; Generator-only unless release proves more |
| Helm prerequisites and chart selection | Cover GPU Operator/cluster needs and select the released chart version | `deployment.md`; chart name/version TBD |
| Kubernetes NGC secrets | Cover both the `nvcr.io` image-pull secret and the generic secret whose key is `NGC_API_KEY` | `deployment.md` |
| Helm values and GPU resources | Provide NIM-specific image, secret, cache, ports, environment, GPU count, probes, and service values | `deployment.md`; validate against released chart |
| Kubernetes storage | Cover PVC/RWX implications and persistent model-cache tradeoffs; link to chart documentation for generic values | `deployment.md` |
| Kubernetes monitoring | Cover ServiceMonitor/metrics and OpenTelemetry only if the released chart/runtime supports the documented settings | `deployment.md`, `operations.md`; release validation pending |
| Helm launch, readiness, port-forward, and inference | Preserve a minimal end-to-end deployment verification flow | `deployment.md` |
| Helm troubleshooting | Cover pending pods, GPU scheduling, storage mounts, and startup-probe failures | `operations.md` |
| Metrics endpoint and metric families | Document endpoint(s) and release-observed metrics; do not copy stale metric names without scraping the released image | `operations.md` |
| Prometheus and Grafana | Include a minimal scrape example and dashboard workflow; avoid pinning old tool versions unnecessarily | `operations.md` |
| Inspection endpoints | Cover health, metrics, metadata, models, manifest, version, and license per active runtime | `api-reference.md`, `operations.md` |
| Logging and distributed diagnostics | Cover service/backend log levels, JSON logs, NCCL/debug knobs, and performance costs | `operations.md` |
| Troubleshooting | Preserve prerequisite, Docker, profile, NGC download, readiness, BYOC, OOM, timeout, metrics, air-gap/cache, and playback cases; add current mode/API failures | `operations.md` |
| Release notes | Identify the documented release and link to authoritative release notes; summarize only changes relevant to these cookbooks | `README.md`; optional `release-notes.md` remains TBD |
| EULA/license notice | Link to the repository license, NGC model card, and the running NIM's `/v1/license`; do not copy a mutable legal agreement into the cookbook | `README.md`, `operations.md` |
| Third-party acknowledgements | Provide a dedicated acknowledgement destination sourced from the released image/build's approved notice inventory | Planned `acknowledgements.md`; content/source TBD |

Coverage is not complete merely because each planned page exists. Before
publication, re-run this matrix against the actual page contents and require one
evidence-backed destination for every row.

### NGC credential terminology and required flow

- Official Generator docs, current NIM source docs, the build Makefile, and
  nearby Cosmos3 cookbooks all use the runtime variable `NGC_API_KEY`.
- No reviewed source defines `NGC_TOKEN` as a container setting. The prose may
  call the credential an "NGC API key" or "NGC token," but commands must use
  `NGC_API_KEY` unless a released runtime explicitly adds another alias.
- Normal cold-start deployment needs the same credential in two places:
  Docker login uses it as the password for the literal `$oauthtoken` user, and
  the container receives it as `NGC_API_KEY` to materialize model artifacts.
- A complete pre-populated cache can remove the need for runtime download
  credentials, but that is an advanced/offline path and must not weaken the
  normal quickstart prerequisite.
- Kubernetes normally needs two differently formatted secrets: a Docker
  registry pull secret for `nvcr.io`, and a generic secret containing a key
  literally named `NGC_API_KEY` for model downloads.
- Examples must show placeholders only and must not echo, save, commit, or
  serialize real credential values. Prefer `--password-stdin` for registry
  login and secret references rather than literal values in Helm YAML.

Primary evidence:

- `docs/cosmos/quickstart-guide.rst:20-70,199-273,277-318`
- `docs/cosmos/prerequisites.rst:138-223`
- `docs/cosmos/configuration.rst:179-254`
- `docs/cosmos/helm.rst:10-97`
- `docs/cosmos/troubleshooting.rst:81-91`
- `cosmos3/documentation.md:327-403`
- `cosmos3/README.md:30-45`
- `cosmos3/Makefile:330-346,398-410`
- `cookbooks/cosmos3/README.md:31-38,496-555`
- `cookbooks/cosmos3/generator/audiovisual/README.md:261-297`

### Previous Cosmos3 Reasoner

The Reasoner lived alongside other VLM NIM documentation. Its dedicated API
page is historical rather than present in the current `main` tree:

- Commit: `783ddb33f455af8f75105bd9577bd73bb91d0f37`
- Path: `docs/vlm/examples/cosmos-reason3/api.rst`

Reusable topics:

- Nano/Super launch pattern.
- OpenAI Chat Completions with public URL and base64 image input.
- Streaming output.
- Video URL, base64 video, and pre-decoded video frames.
- `media_io_kwargs` and `mm_processor_kwargs`.
- Reasoning prompt format and task-oriented examples.
- Text-only queries.

Verify every field and model name against the current Reasoner implementation.
Do not carry over separate-container assumptions into the unified NIM guide.

### Previous Reasoner documentation is also a coverage floor

The historical Reasoner page and its surrounding VLM NIM guide define a second
minimum-coverage gate. As with the Generator audit, this is a topic inventory,
not an authority for current image names, hardware, defaults, formats, or API
behavior. Each topic must be documented from current evidence, linked to a
current authority, explicitly marked TBD, or explained as superseded.

| Previous Reasoner/VLM topic | Current evidence and required treatment | Planned destination/status |
| --- | --- | --- |
| Separate Nano/Super Reasoner image | The current dispatcher and profile export select Reasoner inside the Certified NIM; describe one selected backend per container and do not reuse the old image URL | `README.md`, `deployment.md`; final image URL/tag TBD |
| Model selection and served names | Current code maps Nano and Super to `nvidia/cosmos3-nano-reasoner` and `nvidia/cosmos3-super-reasoner`; prefer `/v1/models` discovery in runnable examples | `deployment.md`, `reasoning.md`, `api-reference.md` |
| NGC authentication, cache, shared memory, UID, and Docker launch | Merge with the unified launch flow and use `NGC_API_KEY`; explain cold-start materialization and writable cache permissions | `README.md`, `deployment.md` |
| Liveness, readiness, and startup delay | Preserve the distinction and explain that model download/load can outlast liveness | `README.md`, `deployment.md`, `operations.md` |
| Chat Completions with curl | Provide one complete non-streaming multimodal request against `/v1/chat/completions` | `reasoning.md` |
| OpenAI Python client | Use `base_url=<NIM_URL>/v1`, a non-secret placeholder client key, model discovery, and current `extra_body` handling | `reasoning.md`, `examples/common.py`, `examples/reasoner.py` |
| Streaming Chat Completions | Current first-party example assembles `delta.content`; retain both incremental output and final assembly | `reasoning.md`, `examples/reasoner_stream.py` |
| Image by public URL | Retain only after release smoke testing of remote URL access and failure behavior | `reasoning.md`; live validation pending |
| Image by data URL/base64 | Current first-party example proves request construction with `image_url`; document MIME-aware data URLs and media-before-text ordering | `reasoning.md`, `examples/reasoner.py` |
| Accepted image formats | Historical JPG/JPEG/PNG claims are not a current release contract | `api-reference.md`; exact release formats TBD |
| Video by public URL | Retain only after release smoke testing of download, timeout, and decode behavior | `reasoning.md`; live validation pending |
| Video by data URL/base64 | Current first-party example proves request construction with `video_url`; document payload-size implications | `reasoning.md`, `examples/reasoner.py` |
| Accepted video containers/codecs | Do not copy the historical MP4/MKV/FLV/3GP and H264/H265/VP9/FLV matrix; validate the released decoder path | `api-reference.md`, `operations.md`; exact matrix TBD |
| Pre-decoded `video_frames` input | Present only in the historical page; no current first-party example or test establishes it | Omit from runnable guidance until live verification; support TBD |
| Media before prompt text | Current cookbook prompt guide and first-party requests both place image/video content before text | `reasoning.md` |
| Request-level video sampling | Current example uses `media_io_kwargs.video.fps`; current startup default is 4 FPS | `reasoning.md`, `api-reference.md` |
| `fps` versus `num_frames` constraints | Historical docs say the fields are mutually exclusive; confirm accepted shape, bounds, and precedence against the released API | `api-reference.md`; live validation pending |
| `mm_processor_kwargs` pixel budgets | Historical-only in reviewed sources; current NIM environment/options and example helper do not expose it | Do not publish as supported until live validation; support/defaults TBD |
| Operator-level `NIM_MEDIA_IO_KWARGS` | Current Reasoner engine accepts the JSON object and the startup layer provides a complete 4-FPS `pynvvc` video default; make clear that an operator override replaces the complete object | `deployment.md`, `operations.md` |
| Per-prompt media limits | Current defaults are five images and one video, configurable with `NIM_MAX_IMAGES_PER_PROMPT` and `NIM_MAX_VIDEOS_PER_PROMPT` | `reasoning.md`, `api-reference.md`, `deployment.md` |
| Sampling defaults and validation | Current normalization supplies temperature 0.7, top-k 20, and top-p 0.8; current validation defines their accepted ranges | `api-reference.md`, summarized in `reasoning.md` |
| `extra_body` request extensions | Current helper routes `media_io_kwargs`, `top_k`, structured-output fields, and `nvext` through the OpenAI client `extra_body` | `reasoning.md`, `examples/common.py` |
| Structured/JSON outputs | Current middleware supports `response_format`, `structured_outputs`, and legacy guided-output normalization for completion routes | `reasoning.md`, `api-reference.md`; add a current example |
| Responses API | Current NIM exposes create/retrieve/cancel routes through NIMlib; use the standard `input_image`/`input_text` request shape | `reasoning.md`, `examples/reasoner_responses.py` |
| Responses storage/background features | Storage is disabled by default and release behavior is still an explicit validation item | `reasoning.md`, `operations.md`; published-image validation pending |
| Reasoning prompt format and `<think>` output | The historical page and current cookbook prompt guide contain explicit reasoning traces; do not promise hidden chain-of-thought or reproduce it as a general API guarantee. Document task prompting and final-answer schemas with approved wording | `reasoning.md`; product/policy wording review required |
| Image/video captioning and VQA | Retain as foundational task examples using current request shapes and vendored assets | `reasoning.md` |
| Temporal localization, event timelines, timestamp, and interval queries | Preserve the current cookbook task taxonomy and use structured final-answer schemas where useful | `reasoning.md` |
| Synthetic-data critic / physical plausibility | Preserve as a task example without implying a separate endpoint or deterministic judgment | `reasoning.md` |
| Embodied, common-sense, planning, and situation-understanding prompts | Preserve representative examples and link to the detailed prompt guide rather than duplicating its full gallery | `reasoning.md` |
| 2D grounding and action trajectories | Preserve the normalized 0-1000 coordinate convention, output schema, and pixel-conversion explanation after model-release verification | `reasoning.md` |
| Text-only queries | Historical support is plausible but is not covered by the current first-party examples/tests reviewed here | `reasoning.md`; release smoke test pending |
| Media and sampling errors | Current interface maps media-related failures to 422 and sampling/validation failures to 400; document stable semantics, not exact messages | `api-reference.md`, `operations.md` |
| Profiles, GPU/precision support, and KV-cache behavior | Replace the historical Reasoner 1.7.0 hardware tables with the exact released manifest and distinguish tested from merely compatible configurations | `deployment.md`; release matrix TBD |
| Environment-variable reference | Carry forward only variables present in the current runtime; include request logging, caching, sequence/token limits, media limits, video preprocessing, compilation, and attention controls | `deployment.md`, `operations.md` |
| Metrics, logging, Helm, and troubleshooting | Reuse the surrounding VLM guide's organization, but validate endpoints, chart values, metric names, and failure modes against this release | `deployment.md`, `operations.md`; release validation pending |

Primary current evidence:

- `cosmos3/serving_stack/reasoner_inference.py:39-50,270-329,336-388,391-454,472-545,574-603`
- `cosmos3/serving_stack/environment.py:419-508`
- `cosmos3/serving_stack/profile_selection/startup.py:25-46`
- `cosmos3/serving_stack/patches/vllm/README.md:29-42`
- `cosmos3/serving_stack/tests/test_reasoner_inference.py:620-934`
- `cosmos3/examples/README.md:19-68`
- `cosmos3/examples/common.py:22-33,355-455`
- `cosmos3/examples/reasoner.py:15-56`
- `cosmos3/examples/reasoner_responses.py:15-39`
- `cosmos3/examples/reasoner_stream.py:21-86`
- `cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md:6-99,101-659`

Before publication, audit every row against the completed guides just as for
the Generator matrix. Historical Reasoner release 1.7.0 facts are provenance,
not fallback values for unresolved release TBDs.

The surrounding current VLM pages remain useful for generic NIM organization:
`docs/vlm/api-reference.rst`, `configuration.rst`,
`environment-variables.rst`, `profiles.rst`, `observability.rst`,
`deploy-helm.rst`, and `support-matrix.rst`.

## Current Cosmos cookbook references

Repository root: `/Users/ekrivov/projects/cosmos`.

### Style and shared context

| Source | Use it for |
| --- | --- |
| `README.md` | Public project tone, model-family overview, integration matrix, and cookbook index |
| `cookbooks/cosmos3/README.md` | Shared environment/setup conventions and cross-cookbook linking style |
| `cookbooks/cosmos3/generator/audiovisual/README.md` | Generator workflow headings, quickstart/table style, and current legacy Generator NIM example |
| `cookbooks/cosmos3/reasoner/README.md` | Reasoner task organization and current legacy Reasoner NIM example |
| `cookbooks/cosmos3/generator/transfer/README.md` | Transfer terminology, controls, and troubleshooting style |
| `cookbooks/cosmos3/generator/action/README.md` | Action capability organization and domain terminology |
| `cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md` | Prompt and media ordering guidance |

### Existing NIM examples

| Source | Reuse carefully |
| --- | --- |
| `cookbooks/cosmos3/generator/audiovisual/run_with_nim.ipynb` | Container launch, readiness, base64 video decoding, and visual walkthrough |
| `cookbooks/cosmos3/reasoner/run_with_nim.ipynb` | OpenAI client setup, image/video encoding, and Reasoner tasks |

These notebooks target the previous separate Generator and Reasoner NIMs. They
are not authorities for the unified image, profile selector, or current feature
surface.

### vLLM and vLLM-Omni analogues

| Source | Conceptual use |
| --- | --- |
| `cookbooks/cosmos3/generator/audiovisual/run_with_vllm_omni.ipynb` | T2V, I2V, V2V workflow and media/output handling |
| `cookbooks/cosmos3/generator/transfer/run_video_transfer_with_vllm_omni.ipynb` | Transfer control cases and assets |
| `cookbooks/cosmos3/generator/action/run_fd_with_vllm_omni.ipynb` | Forward dynamics domains and autoregressive action workflows |
| `cookbooks/cosmos3/generator/action/run_id_with_vllm_omni.ipynb` | Inverse dynamics workflow |
| `cookbooks/cosmos3/reasoner/run_with_vllm.ipynb` | OpenAI-compatible Reasoner tasks and sampling controls |

Translation warning: vLLM-Omni generation commonly uses
`/v1/videos` or `/v1/videos/sync` with multipart fields and `extra_params`; the
Certified NIM Generator uses JSON `POST /v1/infer` and returns `b64_video`.

### vLLM/vLLM-Omni to Certified NIM translation contract

The vLLM and vLLM-Omni notebooks are workflow and asset sources, not request
templates for the Certified NIM. Translate every example at the semantic level.

Generator field mapping:

| vLLM-Omni field or behavior | Certified NIM equivalent | Translation rule |
| --- | --- | --- |
| `POST /v1/videos` or `/v1/videos/sync` | `POST /v1/infer` | Replace the endpoint; do not preserve async video-job semantics |
| Multipart `data` plus `files` | One JSON request body | Encode local media as a data URL/base64 string or use a validated public URL |
| `prompt` | `prompt` | Same concept; preserve structured prompt content only if it remains within the NIM prompt contract |
| `negative_prompt` | `negative_prompt` | Same concept and spelling; confirm default behavior in the NIM reference |
| `size="<width>x<height>"` | `resolution` enum such as `720_16_9` | Translate to an accepted NIM key; never pass an arbitrary vLLM size string |
| `num_frames` | `num_output_frames` | Rename and revalidate 4k+1 cadence and per-resolution caps |
| `num_inference_steps` | `steps` | Rename and revalidate the NIM range/default |
| `guidance_scale` | `guidance_scale` | Same spelling, but use NIM mode-specific defaults/limits |
| `flow_shift` | `flow_shift` | Same spelling, but use NIM mode-specific defaults/limits |
| `fps` | `fps` | Same spelling; revalidate mode constraints |
| `seed` | `seed` | Same spelling; retain deterministic examples |
| Multipart `input_reference` image | Top-level `image` | Use a data URL/base64 or validated public URL, not a host filesystem upload |
| Multipart `input_reference` video | Top-level `video` | Use a data URL/base64 or validated public URL |
| `extra_params.action_mode` | `action_params.mode` | Move into the typed nested action block and use `forward_dynamics`, `policy`, or `inverse_dynamics` |
| Other action `extra_params` | Typed `action_params` fields | Translate only fields accepted by `ActionParams`; unknown fields are rejected |
| Transfer hint `control_path` | `transfer.<hint>.video` | Replace server-local paths with a data URL/base64 or validated public URL |
| Transfer controls in `extra_params` | Typed top-level `transfer` block | Keep only fields exposed by `TransferParams`; move resolution to the top-level request |
| vLLM `guardrails` request flag | Operator-level NIM guardrail configuration | Do not send it in `/v1/infer`; current request models reject unknown fields |
| Direct MP4 response from `/videos/sync` | JSON `b64_video` | Decode base64 and save the MP4 explicitly |
| Async video id, status polling, content download | One synchronous `/v1/infer` response | Remove create/poll/content loops from NIM examples |

Fields that look similar still require NIM validation. In particular, a vLLM
notebook's successful `num_frames`, resolution, or control combination does not
prove that the same value is accepted by the NIM request model or released
profile.

Workflow-by-workflow reuse matrix:

| Cookbook source | Reuse | Translate or exclude |
| --- | --- | --- |
| Audiovisual vLLM-Omni notebook | Structured prompts, negative prompts, T2V/I2V assets, sampling intent, output preview patterns | Translate multipart video requests to JSON `/v1/infer`; use NIM resolution keys and field names. Exclude text-to-image and sound because the current NIM surface does not expose them. Use current first-party NIM `v2v.py` for V2V rather than inferring it from this notebook. |
| Forward-dynamics vLLM-Omni notebook | AV, DROID, and UMI action data concepts; chunk validation; extracting the last generated frame for client-side autoregressive continuation; video stitching/preview ideas | Move action fields under `action_params`; omit `view_point` and request-level `guardrails`; do not copy polling routes. `hand_pose` is not in the current NIM domain enum. Make clear that multi-chunk continuation is client orchestration, not one server request. |
| Inverse-dynamics vLLM-Omni notebook | Video-in/action-out workflow, AV action visualization, artifact layout | Use top-level `video` plus `action_params.mode="inverse_dynamics"`; remove multipart upload, async polling, and vLLM response reshaping. Respect the NIM domain-to-`raw_action_dim` contract. |
| Transfer vLLM-Omni notebook | Edge, blur, depth, segmentation, and WSM assets/prompts; per-control workflow intent; compact preview code | Replace `control_path` with nested `video`; omit vLLM-only `max_frames`, `share_vision_temporal_positions`, `show_*`, nested `resolution`, and `guardrails`. Use the current NIM examples for derived edge/blur and the exact supported multi-control rules. |
| Reasoner vLLM notebook | OpenAI client pattern, `/v1/models` discovery, media-before-text order, task prompts, 0-1000 visualization logic, output parsing ideas | Do not copy the standalone vLLM launch, HF model names, Edge model assumption, `file://` media paths, or `mm_processor_kwargs`. Use NIM data/public URLs, current served names/discovery, and request-level `media_io_kwargs` only where currently supported. Apply the approved reasoning-trace policy. |

Action-specific differences requiring explicit review:

- Current NIM action domains are `av`, `umi`, `bridge_orig_lerobot`, and
  `droid_lerobot`. The vLLM-Omni notebook's `hand_pose` case is not portable.
- `view_point` is present in vLLM-Omni notebook records but absent from the
  current NIM `ActionParams`; do not include it.
- `guardrails` is a vLLM-Omni request extension. Certified NIM guardrail
  controls are operator configuration and must not be placed in `/v1/infer`.
- NIM forward dynamics requires an action array whose row count equals
  `action_chunk_size` and whose width matches the selected domain's
  `raw_action_dim`. `num_output_frames` must equal `action_chunk_size + 1`.
- NIM policy and inverse dynamics return the predicted action in the synchronous
  response; forward dynamics does not return a predicted action.
- The autoregressive DROID/UMI notebook loops are reusable as advanced client
  orchestration only after each individual NIM request shape is validated.

Transfer-specific differences requiring explicit review:

- Precomputed controls become `transfer.<hint>.video`; they are not server-local
  `control_path` values.
- Derived edge and blur use a top-level input `video` plus an edge/blur boolean
  or preset. Depth, segmentation, and WSM require precomputed control video.
- Current NIM `TransferParams` exposes `control_guidance`,
  `num_video_frames_per_chunk`, `num_conditional_frames`, and
  `num_first_chunk_conditional_frames`. Other vLLM-Omni `extra_params` are not
  automatically part of the NIM contract.
- Multi-control behavior must come from current NIM validation and a release
  smoke test, not the vLLM-Omni notebook's broader statement.

Reasoner-specific differences requiring explicit review:

- The OpenAI Chat Completions shape is largely reusable, including dynamic
  `/v1/models` discovery and media-before-text ordering.
- The notebook's `file://` URLs rely on a separately launched vLLM container
  with `--allowed-local-media-path`; use data URLs or validated public URLs in
  Certified NIM examples.
- The notebook sends `mm_processor_kwargs`. Current first-party NIM code and
  helpers do not establish that extension; use the proven
  `media_io_kwargs.video.fps` shape and keep `mm_processor_kwargs` TBD.
- The notebook's direct HF model names and Edge/Super launch options are not NIM
  profile or served-model names.
- Prompt/task galleries and coordinate visualization code are reusable, but
  explicit `<think>` parsing is not a general API guarantee and requires the
  approved reasoning-output policy described above.

Primary evidence:

- `cookbooks/cosmos3/generator/audiovisual/run_with_vllm_omni.ipynb:30,444-576`
- `cookbooks/cosmos3/generator/action/run_fd_with_vllm_omni.ipynb:403-466,684-792,982-1188,1357-1414`
- `cookbooks/cosmos3/generator/action/run_id_with_vllm_omni.ipynb:272-326`
- `cookbooks/cosmos3/generator/transfer/run_video_transfer_with_vllm_omni.ipynb:139-320`
- `cookbooks/cosmos3/reasoner/run_with_vllm.ipynb:108-230,287-443,471-1104`
- `cosmos3/serving_stack/data_models/generation.py:52-401`
- `cosmos3/serving_stack/data_models/actions.py:24-232`
- `cosmos3/serving_stack/data_models/transfer.py:28-252`
- `cosmos3/serving_stack/data_models/responses.py:20-34`
- `cosmos3/examples/{t2v,i2v,v2v,action,transfer,reasoner}.py`
- `cosmos3/examples/common.py:62-97,164-224,302-325`

## Cookbook consistency and repository integration

The new guide should look native to the Cosmos cookbook repository while
intentionally correcting patterns that cause duplication or make content hard
to retrieve. Nearby pages are style references, not authorities for current NIM
facts.

### Adopt, adapt, and avoid

| Observed cookbook convention | Decision for `cookbooks/cosmos3/nim` | Rationale |
| --- | --- | --- |
| Title followed by a concise scope paragraph | Adopt | Establishes product/task context immediately |
| Shared setup linked from task pages | Adapt | Centralize Certified NIM launch in `deployment.md`; task guides link there rather than to the legacy shared NIM section |
| `Quickstart` followed by a fuller walkthrough/reference | Adopt | Supports both fast use and deeper reading |
| Descriptive `##`/`###` headings | Adopt | Stable anchors help humans, repository search, and AI retrieval |
| Relative links to sibling guides, scripts, and assets | Adopt | Keeps branch/fork rendering portable |
| Tables for exact fields, supported cases, and backend differences | Adopt | Improves lookup and prevents constraints from being buried in prose |
| Fenced `bash`, `python`, `json`, and `text` blocks | Adopt | Match the dominant nearby style and enable syntax-aware rendering |
| Checked-in assets under an `assets/` directory | Adapt | Reuse existing suitable assets or vendor only small approved assets under `examples/assets/`; prefer commit-pinned public media for large files |
| Outputs written under an ignored `outputs/` directory | Adopt | Keep generated media and exact transmitted payloads out of source control |
| Notebook as the primary tutorial | Avoid for this guide set | Plain Python remains directly inspectable, diffable, and reusable by humans and AI; notebooks stay supporting sources |
| Manual table of contents on selected long pages | Adapt | Use the landing-page guide table and strong headings; add a manual page TOC only when the rendered page is genuinely difficult to scan |
| Large `<details>` blocks in the root README | Avoid inside focused guides | Hidden/collapsed content is less discoverable and unnecessary when pages are already split by concern |
| Repeating complete backend setup in multiple READMEs | Avoid | One canonical deployment page prevents image, selector, and credential drift |
| Unqualified model size/count and hardware claims | Avoid | Use released manifest/model-card evidence or an explicit TBD |
| Backend-specific request examples copied between integrations | Avoid | Use the translation contract above; preserve workflow intent, not transport syntax |

### File and example presentation rules

- New Markdown files should use the current repository notice form:
  `SPDX-FileCopyrightText` plus `SPDX-License-Identifier: OpenMDW-1.1` in an HTML
  comment before the title. New Python files should use the corresponding `#`
  comment form.
- Use `Cosmos3` consistently in product prose unless quoting an approved title
  that uses `Cosmos 3`. Use exact code identifiers such as
  `nvidia/cosmos3-nano-reasoner` unchanged.
- State the expected working directory before any path-sensitive command. Prefer
  commands runnable from `cookbooks/cosmos3/nim` or explicitly from the
  repository root; do not make the reader infer it.
- Use `bash` for shell fences, `python` for scripts, `json` for bodies/responses,
  and `text` for non-executable output. Do not use an unlabeled fence for a
  runnable example.
- Use one command per conceptual action where possible. Explain non-obvious
  Docker flags immediately after the launch block in `deployment.md`.
- Use semantic placeholders and exported variables for release-owned values and
  credentials. Examples must use `NGC_API_KEY`; never show realistic credential
  strings.
- Keep examples platform-aware. Do not depend on GNU-only `base64 -w 0` for the
  primary local-media path; the shared Python encoder is the portable source.
- Prefer vendored small assets. For large remote media, pin the URL to an
  immutable commit or artifact version and record provenance/licensing before
  publication. Never silently rely on a moving `main` branch asset.
- Save each run's exact request, response or compact metadata, decoded media,
  and predicted action where applicable. Keep full base64 values out of console
  summaries.
- `examples/outputs/` is covered by the repository-wide `outputs/` ignore rule;
  confirm with `git check-ignore` after the directory exists.
- Do not add nbviewer links for plain Python scripts. Link the script directly
  from its owning guide and from the landing-page guide table.

### Repository integration surfaces

The public guide set lives under `cookbooks/cosmos3/nim`, but several existing
pages currently duplicate or contradict the future unified Certified NIM
contract. Once drafting is authorized, review these integration edits as part
of the same documentation change. If the user wants a strict path-only change,
leave them untouched but report the resulting stale inbound documentation.

| Existing surface | Current problem | Planned integration treatment |
| --- | --- | --- |
| Root `README.md` quickstart navigation and Generator/Reasoner NIM sections | Two separate image/launch guides, old Generator T2V/I2V-only boundary, legacy Reasoner 1.7.0 fields and links | Replace detailed duplicated setup with a concise Certified NIM summary and link to `cookbooks/cosmos3/nim/README.md`; retain legacy material only if clearly labeled historical |
| Root `README.md` integration chooser and examples table | Describes NIM as two separate, narrower integrations and has notebook-only entries | Add one Certified NIM guide entry covering selected Generator/Reasoner profiles and supported current tasks; update capability text from released evidence |
| `cookbooks/cosmos3/README.md` backend table and `## NIM` section | Treats Reasoner and Generator as separate containers and hard-codes legacy images/selectors | Point to the new deployment/overview pages and keep only a short shared-navigation summary |
| Audiovisual `README.md` `Run with NIM` section | Duplicates launch/schema content and ends with the previous T2V/I2V-only limitation | Link to `nim/generation.md`; label the existing notebook's separate-image scope if retained; remove or qualify stale limitations |
| Reasoner `README.md` `Run with NIM` section | Links old 1.7.0 docs and claims `mm_processor_kwargs` support | Link to `nim/reasoning.md`; use current media controls and explicit legacy/TBD status |
| Action `README.md` | Covers Framework/vLLM-Omni but has no Certified NIM route | Add a concise link to `nim/action.md` when the published image/profile support is confirmed; do not copy the full NIM request reference |
| Transfer `README.md` | Covers Framework/vLLM-Omni but has no Certified NIM route | Add a concise link to `nim/transfer.md` when published-image support is confirmed; keep backend-specific fields separate |

Primary style/integration evidence:

- `README.md:20-42,566-648,970-1073,1102-1135`
- `cookbooks/cosmos3/README.md:1-41,486-614`
- `cookbooks/cosmos3/generator/audiovisual/README.md:1-25,248-371`
- `cookbooks/cosmos3/reasoner/README.md:1-33,141-203`
- `cookbooks/cosmos3/generator/action/README.md:1-39,92-144`
- `cookbooks/cosmos3/generator/transfer/README.md:1-47,136-270`
- `cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md:1-6`
- `cookbooks/cosmos3/generator/transfer/preview_helpers.py:1-15`
- `.gitignore:181-192`

## Capability-to-page map

| Capability or topic | Primary source | Supporting source | Public destination |
| --- | --- | --- | --- |
| Unified image and profile-selected runtime mode | `serving_stack/inference.py`, profile code | `AGENTS.md`, `documentation.md` | `README.md`, `deployment.md` |
| NGC authentication, cache, ports, first launch | `documentation.md`, runtime configuration | Previous quickstart, existing cookbook launch cells | `README.md`, `deployment.md` |
| Catalog/model-card identity and release notes | Published NGC release | Previous introduction/release notes | `README.md`; release URL/version TBD |
| Hardware and profile selection | `make_profiles.py`, profile selection, exported profiles | `documentation.md`, support matrices | `deployment.md` |
| Shared management endpoints | live OpenAPI, `api_spec.yaml`, NIM interface | `documentation.md` | `api-reference.md`, `operations.md` |
| Generator request/response and validation | `data_models/*.py`, tests | current examples | `api-reference.md` |
| T2V, I2V, V2V | `generation.py`, `t2v.py`, `i2v.py`, `v2v.py` | vLLM-Omni audiovisual cookbook | `generation.md` |
| Generator prompt upsampling | `prompt_upsampling.py`, `generator_inference.py`, `environment.py`, tests | `documentation.md` | `deployment.md`, `generation.md`, `operations.md` |
| Chat Completions | `reasoner_inference.py`, `reasoner.py`, tests | previous Reasoner API page | `reasoning.md` |
| Responses API | `reasoner_responses.py`, Reasoner tests | NIMlib behavior noted in examples README | `reasoning.md` |
| Reasoner streaming | `reasoner_stream.py`, Reasoner tests | previous Reasoner API page | `reasoning.md` |
| Reasoner media limits/preprocessing | `environment.py`, current examples/tests | previous Reasoner API page, prompt guide | `reasoning.md`, `api-reference.md` |
| Forward dynamics | `actions.py`, `action.py`, tests | action/vLLM-Omni cookbook | `action.md` |
| Policy | `actions.py`, `action.py`, tests | action cookbooks | `action.md` |
| Inverse dynamics | `actions.py`, `action.py`, tests | action/vLLM-Omni cookbook | `action.md` |
| Transfer controls | `transfer.py`, example and tests | transfer/vLLM-Omni cookbook | `transfer.md` |
| Environment variables | `environment.py` and NIM framework contract | `documentation.md`, previous config pages | `deployment.md`, `operations.md` |
| BYOC | BYOC validation in `environment.py` | `documentation.md`, previous BYOC page | `deployment.md` |
| Helm/Kubernetes | released chart contract when confirmed | previous Helm page, `documentation.md` | `deployment.md` |
| Health, metrics, logging, guardrails | runtime code and NIM interface | `documentation.md`, previous observability page | `operations.md` |
| Troubleshooting | actual validation/errors/tests | source guide and previous troubleshooting pages | `operations.md` |
| License/EULA notice | Published release/model card and `/v1/license` | Previous EULA page, repository `LICENSE` | `README.md`, `operations.md` |
| Third-party acknowledgements | Approved notice inventory for the released image | Historical generated Cosmos3 acknowledgements | `acknowledgements.md`; source/content TBD |

## Documentation architecture decision

Use a hub-and-spoke Markdown guide set: a concise `README.md` is the entry point,
focused guides own distinct classes of facts, and runnable examples live under
`examples/`. This is the agreed structure for the authoring phase.

Alternatives considered:

| Alternative | Decision | Reason |
| --- | --- | --- |
| One large `README.md` | Reject | It would mix launch, reference, workflows, and operations into a document that is difficult to navigate, review, render, or fit into an AI context window. |
| One file per endpoint or example | Reject | It would fragment shared setup and concepts, create excessive cross-linking, and make defaults easy to duplicate inconsistently. |
| Generated API reference only | Reject | OpenAPI can establish request shapes but cannot replace deployment, profile selection, media handling, task guidance, operational caveats, or release TBDs. |
| Markdown guides plus a separate machine-readable documentation manifest | Defer | A second hand-maintained index would duplicate `README.md`. Add one only if a real documentation consumer requires it; do not invent `llms.txt` or YAML metadata pre-emptively. |

Consequences of the chosen structure:

- Every fact class has one canonical owner. Other pages summarize it only when
  needed to complete a workflow and link back to the owner.
- `README.md` acts as both human landing page and AI-readable document index.
- Each focused page is independently understandable: it states its runtime
  mode, endpoint, prerequisites, inputs, output, and links to canonical fields.
- The complete API tables live in `api-reference.md`; task pages contain only
  the subset needed to explain a workflow.
- Deployment owns launch-time configuration. Operations owns observation,
  diagnostics, failure recovery, and production caveats.
- Runnable scripts are the canonical complete examples. Embedded snippets stay
  short but must remain directly runnable and consistent with those scripts.
- Release-dependent facts remain visibly marked `TBD (release-dependent)` until
  authoritative release evidence is available. Historical values never fill a
  TBD implicitly.

## Agreed public information architecture

### `README.md`

- What the unified Cosmos3 Certified NIM is.
- One-profile-at-a-time Generator versus Reasoner explanation.
- Capability, input/output, and endpoint matrix.
- Prerequisites, `NGC_API_KEY` authentication, and links to full deployment
  details.
- Shortest supported launch and readiness flow.
- One minimal Generator request and one minimal Reasoner request.
- Choose-your-task navigation to every focused guide.
- Release/model-card, license, and acknowledgements links once approved.

Keep this page concise. It is a landing page, not the full reference.

### `deployment.md`

- Prerequisites and version requirements.
- NGC API key and Docker login.
- Cache and volume permissions.
- Container image, ports, shared memory, and ulimits.
- Choosing `generator` or `reasoner`, model size, precision, performance profile,
  GPU exposure, and explicit profile selection.
- Nano/Super and supported-hardware tables.
- Readiness and cold-start expectations.
- Optional Generator prompt-upsampling configuration, including the separate
  external-service secret and supported endpoint contract.
- BYOC, clearly scoped to supported runtime modes.
- Helm/Kubernetes deployment and persistent cache.

### `api-reference.md`

- Endpoint matrix separated by runtime mode.
- Shared management endpoints.
- Generator `/v1/infer` top-level request fields.
- Nested action and transfer schema links, with summary tables only.
- Generator response and optional action output.
- Accepted image/video representations and URL-input policy.
- Validation behavior and error envelope.
- Model discovery and live `/openapi.json` usage.

This is the canonical location for field defaults, ranges, and constraints.
Workflow pages should link here instead of repeating full field tables.

### `generation.md`

- Common setup and output decoder.
- Text-to-video.
- Image-to-video with local base64/data URL and public URL.
- Video-to-video and conditioning-frame controls.
- Optional prompt upsampling for T2V/I2V, including its mode boundary and
  original-prompt fallback behavior.
- Reproducibility, frame cadence, resolution, FPS, quality, and input/output
  media guidance.
- Capability-specific failure cases.

### `reasoning.md`

- Reasoner profile and served-model discovery.
- Chat Completions for text, image, and video.
- Responses API and its storage limitations.
- Streaming Chat Completions.
- Media ordering and prompt guidance.
- Request-level `media_io_kwargs`, operator defaults, prompt media limits, and
  the live-validation status of legacy `mm_processor_kwargs`.
- Sampling controls, structured outputs, and stable 400/422 error semantics.
- Captioning/VQA, temporal localization, embodied and common-sense reasoning,
  physical plausibility, situation understanding, 2D grounding, and action
  trajectory examples.
- Task prompting and final-answer formats without promising or exposing hidden
  chain-of-thought as an API guarantee.

### `action.md`

- Shared `action_params` contract.
- Supported domains and action dimensions.
- Forward dynamics input and video result.
- Policy input and video/action result.
- Inverse dynamics input and video/action result.
- Shape, chunk-size, frame-count, action-space, and mode-specific validation.
- One complete runnable case for each mode.

### `transfer.md`

- Transfer concepts and supported control types.
- Precomputed edge, blur, depth, segmentation, and WSM controls.
- Server-derived edge and blur controls.
- Single-control and supported multi-control behavior.
- Control video/media constraints and chunking fields.
- One compact example per distinct request shape, not per asset.

### `operations.md`

- Environment-variable reference not already explained in deployment.
- Health, readiness, model, version, metadata, manifest, license, and metrics.
- Logging and multi-GPU diagnostics.
- Guardrails and the consequences of disabling them.
- Prompt-upsampling diagnostics, timeout/response failures, and fallback
  interpretation without exposing its API key.
- Profile inspection and configuration confirmation.
- Prometheus/Grafana integration.
- Symptom/cause/fix troubleshooting tables.
- Known limitations and release-specific caveats.

### `acknowledgements.md`

- Dedicated destination for third-party components and applicable notices for
  the exact released Certified NIM image.
- Content must come from the approved release/build acknowledgement inventory,
  not from package inference, source-tree dependency files, or the previous
  38,574-line Generator acknowledgement artifact.
- Until that release artifact is supplied, keep the page in the plan and the
  content explicitly TBD; do not create a public placeholder yet.
- Keep the product EULA/license separate: link to repository/NGC terms and the
  running NIM's `/v1/license` endpoint from `README.md` and `operations.md`.

### `examples/`

Use plain Python, not notebooks. Keep request dictionaries directly editable and
avoid a large CLI abstraction.

Planned scripts:

- `common.py`
- `t2v.py`
- `i2v.py`
- `v2v.py`
- `reasoner.py`
- `reasoner_stream.py`
- `reasoner_responses.py`
- `action.py`
- `transfer.py`

Each public guide also embeds one short curl or Python example so it remains
useful without opening a second file.

## Page-level authoring contracts

These contracts define what each public artifact owns, which sources establish
its content, and what it must not absorb from neighboring pages. They are the
handoff from research to drafting.

| Artifact | Canonical ownership | Primary current evidence | Release/TBD inputs | Non-goals |
| --- | --- | --- | --- | --- |
| `README.md` | Product scope, selected-runtime mental model, capability/endpoint index, minimum launch, first Generator/Reasoner requests, navigation | `serving_stack/inference.py`, profile export/selection, current examples, `documentation.md` | Final image/tag, product/model-card/release URLs, approved public naming | Exhaustive fields, full environment reference, profile internals, long troubleshooting |
| `deployment.md` | Prerequisites, NGC key and login, cache/permissions, Docker flags, ports, selectors, profiles/hardware, readiness, prompt-upsampling launch configuration, BYOC, Helm | `documentation.md`, `environment.py`, `prompt_upsampling.py`, profile-selection code, `make_profiles.py`, `profiles.json`, BYOC and prompt-upsampling tests | Driver/toolkit floor, released profile matrix, image/tag, chart/version/values, published BYOC boundary, released external-endpoint compatibility | Request field tables, task prompt galleries, metrics interpretation |
| `api-reference.md` | Endpoint inventory, canonical request/response fields, defaults/ranges, media representations, validation/error semantics, model/OpenAPI discovery | `data_models/*.py`, `reasoner_inference.py`, data-model/Reasoner tests, current examples | Generator and Reasoner live OpenAPI, released media/codec support, public-URL behavior | End-to-end deployment, repeated task walkthroughs, generic OpenAI documentation |
| `generation.md` | T2V/I2V/V2V workflows, conditioning media, optional T2V/I2V prompt upsampling, output decoding, reproducibility, generation-specific caveats | `data_models/generation.py`, `generator_inference.py`, `prompt_upsampling.py`, `examples/{t2v,i2v,v2v,common}.py`, tests | Published capability set, live media/URL validation, prompt-upsampling integration smoke test, validated output playback | Full schema tables, action/transfer modes, vLLM-Omni endpoints |
| `reasoning.md` | Chat Completions, Responses, streaming, media ordering, task prompts, structured outputs, Reasoner-specific sampling/media guidance | `reasoner_inference.py`, Reasoner environment/startup, Reasoner examples/tests, current prompt guide | Text-only/public-URL/media-format checks, Responses state features, approved reasoning-trace wording | Generic vLLM tuning reference, unsupported legacy `video_frames`/`mm_processor_kwargs`, hidden chain-of-thought promises |
| `action.md` | Shared action contract and forward-dynamics, policy, inverse-dynamics workflows and validation | `data_models/actions.py`, `examples/action.py`, data-model/example tests, current action cookbooks | Published-image/profile smoke tests and released domain/model boundary | General generation tutorial, one page per action asset, framework/vLLM-Omni request syntax |
| `transfer.md` | Transfer control taxonomy, precomputed/server-derived controls, combinations, media/chunking fields, distinct request shapes | `data_models/transfer.py`, `examples/transfer.py`, tests, current transfer cookbook | Published-image/profile smoke tests, exact supported combination matrix | Duplicate example for every asset, vLLM-Omni multipart syntax, generic video generation reference |
| `operations.md` | Health/readiness, inspection endpoints, metrics/logs, guardrails, prompt-upsampling failure/fallback diagnostics, profile confirmation, production diagnostics, troubleshooting | Runtime interface/environment/prompt-upsampling code, `documentation.md`, tests, previous operations docs | Live metrics scrape, log/error samples, prompt-upsampling failure sample, chart probes, release limitations | Basic launch tutorial, duplicate schema tables, unverified metric names, secret values |
| `acknowledgements.md` | Third-party components and notices for the exact released image | Approved release/build acknowledgement inventory only | Entire content remains TBD until that artifact is supplied and approved | Product EULA copy, inferred dependency inventory, historical Generator notice reuse |
| `examples/common.py` | Shared URL, health check, media encoding, request dispatch, response decoding, safe artifact writing | Current NIM `examples/common.py`, current cookbook helpers | Final default host-port convention and live response shapes | Large CLI framework, implicit downloads, embedded secrets |
| Task example scripts | One editable, complete, representative request per distinct request shape | Current first-party NIM examples plus translated cookbook/vLLM-Omni workflows | Live release smoke results and stable asset locations | Exhaustive parameter combinations, duplicated common helpers, notebook-only execution |

### Canonical fact ownership

Use this table during drafting and review to resolve tempting duplication:

| Fact class | Canonical owner | Allowed repetition elsewhere |
| --- | --- | --- |
| Product identity, release compatibility, guide navigation | `README.md` | One-sentence context and links |
| Image name/tag, prerequisites, Docker/Helm launch, selectors, hardware profiles, launch-time environment | `deployment.md` | Minimal quickstart subset in `README.md`; no copied full tables |
| Endpoint list, request/response fields, defaults, ranges, media types, HTTP semantics | `api-reference.md` | Task-specific subsets with links to the canonical rows |
| Workflow order, task intent, representative payload, artifact handling | Corresponding task guide | One minimal first request in `README.md` |
| Reusable complete request/decoder implementation | `examples/` | Short synchronized snippets in guides |
| Health interpretation, metrics, logs, guardrails, diagnostics, known limitations | `operations.md` | Readiness command in quickstarts and links for detail |
| Legal/product license location | `README.md` and `operations.md` links | Do not reproduce mutable legal text |
| Third-party notice inventory | `acknowledgements.md` | Link only |

### Human- and AI-readable Markdown conventions

- Begin each page with a one-paragraph scope statement and a compact
  "Use this page when..." navigation sentence where helpful.
- Use stable descriptive headings such as `Prerequisites`, `Launch the NIM`,
  `Request`, `Response`, `Parameters`, `Errors`, and `Troubleshooting`. Avoid
  heading text that depends on numbering or vague context such as "More."
- State the active runtime mode and endpoint immediately before every request.
- Introduce acronyms and selector names before using them. Use the exact API or
  environment-variable spelling in backticks.
- Prefer tables for exact mappings, defaults, ranges, conflicts, profile rows,
  and status matrices. Keep explanations in prose when order or rationale is
  more important than lookup.
- Keep JSON bodies complete and valid. Use fenced blocks with language labels,
  semantic placeholders such as `<NIM_IMAGE:TBD>`, and no realistic-looking
  secret values.
- Label evidence status explicitly when it affects correctness:
  `Current source snapshot`, `Validated against release`, `Historical`, or
  `TBD (release-dependent)`. Public docs should not expose internal commit IDs
  or local paths; the source map retains that provenance.
- Avoid context-dependent phrases such as "the command above" when a stable
  section link or named command is clearer to both humans and retrieval tools.
- Link to canonical sections instead of copying whole tables. Relative links
  must include anchors and be checked after headings stabilize.
- Every runnable workflow states prerequisites, request, success status/output
  shape, saved artifact, and the most likely capability-specific failure.
- Prefer `/v1/models` discovery over a hard-coded Reasoner model name unless the
  served name is intentionally being explained as part of the current contract.
- Do not add YAML front matter, a second machine-readable index, or generated
  navigation until repository tooling or a concrete consumer requires it.

## Draft readiness and authoring sequence

Release-dependent TBDs do not block drafting. They block only the specific
claim or command that needs the missing fact. Use these readiness labels:

- **Source-ready**: current implementation, tests, and examples are sufficient
  to draft the page; live-image validation is still required before making
  release-tested claims.
- **Ready with TBDs**: the page can be drafted, but named release-owned facts
  must remain visibly TBD.
- **Artifact-dependent**: substantive content cannot be derived safely until an
  authoritative external artifact is supplied.

| Artifact | Readiness | Draftable from current evidence | Must remain TBD or validation-gated |
| --- | --- | --- | --- |
| `README.md` | Ready with TBDs | Unified selected-runtime model, capability/endpoint index, NGC credential flow, guide navigation, minimal request shapes | Final image/tag, approved product wording, catalog/model-card/release URLs, final support summary |
| `deployment.md` | Ready with TBDs | Docker/cache/credential mechanics, selector behavior, current profile model, readiness concepts, prompt-upsampling configuration/secret separation, current Generator BYOC boundary | Final image/tag, driver/toolkit floors, released profile matrix, Helm chart/values, published prompt-upsampling endpoint compatibility, published BYOC statement |
| `api-reference.md` | Source-ready with live gates | Generator models, current Reasoner normalization/routes, defaults, ranges, validation, response shapes | Live OpenAPI for both modes, media URL behavior, exact released formats/codecs, NIMlib-generated route details |
| `generation.md` | Source-ready with live gates | T2V/I2V/V2V request construction, current fields/constraints, prompt-upsampling mode boundary and fallback, base64 response decoding, deterministic-seed guidance | Published-image capability and prompt-upsampling smoke results, remote-input behavior, playback/codec observations |
| `reasoning.md` | Ready with TBDs | Chat Completions, Responses create flow, streaming, data URLs, media ordering, sampling, structured outputs, task taxonomy | Text-only and public-URL smoke tests, exact media formats, Responses persistence features, approved reasoning-trace wording |
| `action.md` | Source-ready with live gates | Forward dynamics, policy, inverse dynamics, action shapes/domains, current representative payloads | Published-profile availability and one smoke result per documented mode/domain boundary |
| `transfer.md` | Source-ready with live gates | Precomputed and derived controls, current fields/validators, representative request shapes | Published-profile availability, exact supported combination matrix, smoke results |
| `operations.md` | Ready with TBDs | Health/readiness concepts, current environment controls, stable error classes, prompt-upsampling failure/fallback semantics, historical troubleshooting categories | Live endpoint inventory, metrics scrape, log and prompt-upsampling failure samples, chart probes, release-specific known limitations |
| `acknowledgements.md` | Artifact-dependent | Page purpose and link placement only | Entire third-party inventory and notice text until the approved released-image artifact exists |
| `examples/` | Source-ready with live gates | Adaptable current NIM request dictionaries/helpers and cookbook assets/workflows | Final host-port convention, image-dependent model/profile selection, live response verification |

### Recommended authoring order

When the user authorizes drafting, use this dependency order:

1. Refresh the reviewed snapshots and reconcile any source drift.
2. Create only the agreed file scaffold and relative navigation targets.
3. Draft `api-reference.md` from current request models and Reasoner routing;
   this establishes the canonical field names, defaults, and error semantics.
4. Adapt `examples/common.py` and the task scripts from the current first-party
   examples, preserving complete editable request dictionaries.
5. Draft `generation.md`, `reasoning.md`, `action.md`, and `transfer.md` against
   the canonical API tables and runnable scripts. Keep prompt upsampling in the
   generation/deployment/operations pages rather than treating it as a new
   generation endpoint.
6. Draft `deployment.md`, keeping final image, released hardware, driver, chart,
   and release URLs explicitly TBD where evidence is unavailable.
7. Draft `operations.md` from current controls and historical coverage, marking
   live metrics, logs, probes, and release limitations as validation-gated.
8. Assemble `README.md` last so its guide index, capability matrix, and minimal
   examples summarize artifacts that actually exist.
9. Add acknowledgement content only when the approved artifact is supplied;
   otherwise retain the planned destination and explicit TBD.
10. Run the parity, link, syntax, secret, placeholder, and live-validation gates
    in the checklist below.

The scaffold in step 2 must not be created during the current discovery-only
phase. The order is recorded now so a later authoring turn can proceed without
re-deciding page dependencies.

### Source refresh procedure

Before the first public edit and again before publication:

1. Record `git rev-parse HEAD`, branch, and tracked worktree state for all four
   reviewed repositories.
2. Diff the current NIM commit against the snapshot recorded above, scoped to:
   `cosmos3/documentation.md`, `cosmos3/examples/`,
   `cosmos3/serving_stack/data_models/`, `reasoner_inference.py`,
   `environment.py`, profile selection/generation files, tests, and the exported
   profile manifest.
3. Diff the cookbook target and referenced Generator/Reasoner/vLLM-Omni pages
   for structural or workflow changes.
4. Diff the previous product documentation only to detect corrected reusable
   explanations or newly published release facts; it remains lower authority
   than the current runtime.
5. Update the snapshot table, resolved-fact inventories, parity matrices, and
   TBD ledger before carrying any changed fact into public prose.
6. If a release image becomes available, capture evidence separately under one
   Generator and one Reasoner profile: readiness, `/v1/models`, live OpenAPI,
   representative requests, errors, inspection endpoints, metrics, and logs.
7. Record which claims are source-derived versus release-validated. Do not
   silently upgrade a source-derived claim to "tested."

### Phase handoff criteria

Research is ready to hand off to drafting when:

- source authority, snapshots, parity floors, page ownership, and the TBD ledger
  are recorded;
- every planned page has primary source coverage or an explicit
  artifact-dependent/TBD state;
- unresolved release facts are isolated and do not force guessed commands; and
- the user explicitly authorizes public documentation edits.

Drafting is ready to hand off to validation when every planned non-placeholder
artifact exists, canonical facts are not duplicated inconsistently, examples
parse or compile offline, and every intentional TBD maps to the ledger.

Publication readiness is a separate decision. It requires the checklist below
plus an explicit review of which release-dependent TBDs are acceptable to leave
visible. A page may be structurally complete while some release-owned values
remain TBD by agreement.

## Known discrepancies and risks

| Issue | Evidence | Required resolution |
| --- | --- | --- |
| Source guide title and early overview are Generator-only | `documentation.md` title and T2V/I2V table | Use unified runtime dispatch and current request models; rewrite overview rather than copying it |
| Source guide says prompt max 2,000 in one field heading while current code allows 20,000 | `documentation.md` request reference vs `generation.py` | Publish the current validated value after testing live OpenAPI |
| Old cookbook says Generator NIM supports only T2V/I2V | audiovisual README NIM limitations | Do not carry this limitation into unified NIM docs; verify current image capabilities |
| Old cookbook uses separate `cosmos3-generator` and `cosmos3-reasoner` images | existing NIM notebooks and setup | Confirm final unified Certified NIM image name/tag and profile selectors |
| Static `api_spec.yaml` predates recent action, transfer, and Reasoner work | file history and missing dynamic routes | Generate/capture live OpenAPI for both Generator and Reasoner profiles |
| Product/model parameter counts differ across sources | NIM source guide and public Cosmos README | Use approved public naming/size language; avoid counts until reconciled |
| Source examples default to port 18000 while public cookbook examples commonly use 8000 | `examples/common.py`, cookbook NIM pages | Standardize docs on `NIM_URL`, defaulting to `http://localhost:8000`; explain host-port remapping once |
| Generated profile export includes Generator BF16, FP8, and FP8 offload rows while `documentation.md` says the active Generator grid is BF16-only | `profiles.json`, `vram_profiles.yaml`, documentation overview/support matrix | Use the release manifest for the published table; treat the current prose table as stale |
| Developer README says text generation lives in a separate Reasoner NIM while the current manifest and dispatcher include Reasoner profiles in the same image | `README.md:3` vs `profiles.json`, `inference.py`, `documentation.md:25` | Describe one-profile-at-a-time unified behavior only after confirming the published image; retain this as a release identity conflict |
| Local NIM identities and versions disagree | `nim-config.yaml`, Makefile, build-context `VERSION` | Never derive the public image/tag from local build defaults; obtain it from the published NIM/NIMCraft release |
| Hardware/profile tables are generated and release-sensitive | profile generator/export vs older documentation | Derive tables from the release manifest and record the reviewed version |
| Helm URL and some launch values are placeholders in `documentation.md` | outstanding TBDs | Confirm released chart and image details before publication |
| Reasoner routes are dynamic and not fully represented by Generator OpenAPI | Reasoner NIMlib interface | Validate live OpenAPI separately under a Reasoner profile |
| Historical Reasoner docs claim `video_frames` and `mm_processor_kwargs`, but current first-party code/examples do not establish them | old Reasoner API page vs current Reasoner options/helper | Keep both out of supported guidance until a published-image smoke test proves the exact request shapes |
| Historical Reasoner media support matrix belongs to release 1.7.0 | old VLM support matrix | Rebuild image/video format and codec claims from the released image; leave them TBD meanwhile |
| Historical and current prompt guides show explicit `<think>` traces | Reasoner API page and current prompt guide | Preserve task and output-format guidance, but obtain approved wording before documenting reasoning traces or chain-of-thought behavior |
| Current vLLM/vLLM-Omni cookbooks expose broader transports, fields, domains, and modalities than the Certified NIM request models | audiovisual/action/transfer/Reasoner notebooks vs current NIM models/examples | Reuse workflow intent and assets only through the translation contract above; never treat backend parity as API parity |
| Source-guide prompt-upsampling example names a provider-specific native endpoint, while the implementation normalizes and calls an OpenAI-compatible Chat Completions endpoint | `documentation.md` vs `prompt_upsampling.py` | Document the generic OpenAI-compatible contract; do not claim native provider compatibility without a published-image integration test |
| Existing inbound cookbook/root documentation duplicates separate legacy NIM launches and limitations | root README, shared Cosmos3 setup, audiovisual and Reasoner READMEs | Make the new guide canonical and, if scope permits, replace duplicated details with current summaries and links; otherwise report the remaining contradictions explicitly |
| BYOC material is Generator/diffusion-specific | environment code and source guide | Do not imply Reasoner BYOC unless current implementation and release support it |
| Broader Cosmos3 supports audio/image generation, but current Certified NIM examples center on video, text, and action | model docs vs current NIM request/response | Document only capabilities proven by the Certified NIM runtime and release profile |

## Release-dependent TBD ledger

Do not close these from memory or legacy docs:

- Final NGC image repository and tag for the unified Certified NIM.
- Public product name and whether the release uses one unified image externally.
- Confirm that the published manifest matches the locally proven selector axes
  and the 39-row profile export above.
- Tested GPU/profile/support matrix and minimum driver/toolkit versions.
- Final Helm chart name, version, and values.
- Final NGC Catalog/model-card URL, authoritative release-notes URL, and release
  version used by these cookbooks.
- Confirm the published-image BYOC boundary and checkpoint layout. Current code
  proves Generator/diffusion-only BYOC.
- Generator support for V2V, action, and transfer in the exact published image.
- Reasoner Responses storage/background/retrieve feature support in the exact
  published image.
- Reasoner public-URL media fetching, text-only requests, `video_frames`, and
  `mm_processor_kwargs` support in the exact published image.
- Exact Reasoner image formats, video containers/codecs, and request-level
  `fps`/`num_frames` constraints in the published image.
- Approved public wording for any Reasoner prompt format that requests or emits
  an explicit reasoning trace; do not treat legacy `<think>` examples as an API
  guarantee.
- Confirm that the published image retains the current no-audio API boundary.
- Confirm prompt upsampling in the published image, including the external
  OpenAI-compatible endpoint contract, supported template styles, timeout, and
  secret-injection expectations. Provider-specific native API compatibility is
  not established by current code.
- Approved third-party acknowledgement/notices artifact for the exact released
  image. The historical `acknowledgements-cosmos3-markdown.md` belongs to the
  previous Generator build and cannot establish the unified image's contents.
- Approved license/EULA links and whether the cookbook should link only to the
  model card and `/v1/license` or also include product-specific terms.

## Validation performed during discovery

- Parsed 22 relevant implementation and example Python files with the standard
  library AST parser; all parsed successfully.
- Loaded `profiles.json`, confirmed 39 rows (32 Generator, 7 Reasoner), and
  checked every row's positive VRAM floor and
  `n_gpus = nim_dp * nim_gp * nim_up * nim_tp` invariant.
- Confirmed every Reasoner row omits the performance-profile axis, fixes
  `nim_dp=nim_gp=nim_up=1`, and satisfies `n_gpus=nim_tp`.
- Confirmed the checked-in tests cover all Generator example payloads, all
  action modes, all transfer cases, Reasoner media ordering, streaming,
  Responses routes, route disabling, normalization, and error behavior.
- Re-audited the authoring-ready contract tables against the current field
  declarations, validators, runtime defaults, response models, and all ten
  first-party example/helper scripts. Confirmed that every cited source file
  exists in the reviewed NIM snapshot.
- Audited all 21 numbered sections (including section 12.1) of the current
  first-party `documentation.md` and assigned each to a planned page,
  correction, release gate, or explicit TBD. Separately inventoried all 12
  entries in `examples/`, including the helper, README, and package marker.
- Traced prompt upsampling through startup validation, request dispatch,
  external request construction, and tests. Confirmed it is optional,
  Generator-only, limited to T2V/I2V, uses a separate secret, and falls back to
  the original prompt on request-time failures.
- Compared first-party example media against existing cookbook assets by
  SHA-256. Confirmed byte-identical AV action inputs, two Reasoner media files,
  and five transfer prompts can be reused from their existing public-repository
  locations instead of copied from the NIM source.
- Audited all 16 previous official Generator user-guide/notices source files
  referenced by `docs/cosmos/index.rst` or the Generator source inventory and
  mapped 49 durable topic groups to a planned page, external authority, or
  explicit release TBD. Confirmed the runtime credential name is
  `NGC_API_KEY`; no reviewed source defines `NGC_TOKEN` as a container setting.
- Audited the historical Reasoner API page and its surrounding VLM launch,
  configuration, profile, observability, Helm, and support-matrix pages. Mapped
  36 durable Reasoner/VLM topic groups to current evidence, a planned page, or
  an explicit release-validation/TBD state. Confirmed that `video_frames`,
  `mm_processor_kwargs`, historical codec tables, and release 1.7.0 hardware
  rows cannot be carried forward as current facts.
- Audited all five current vLLM/vLLM-Omni notebooks named in the source map,
  including their request-construction cells, endpoint/transport behavior,
  response handling, task fields, assets, and client-side orchestration. Mapped
  20 Generator transport/field behaviors and five workflow families to current
  NIM equivalents or explicit exclusions. Confirmed that `/v1/videos*`,
  multipart `input_reference`, async polling, server-local `control_path`,
  `view_point`, request-level `guardrails`, text-to-image/sound, Reasoner
  `file://` media, and `mm_processor_kwargs` must not be copied literally.
- Audited the root README, shared Cosmos3 setup guide, and four nearby
  audiovisual/Reasoner/action/transfer READMEs for heading, setup, link, code
  fence, table, asset, output, and license-header conventions. Identified six
  inbound integration surfaces that would otherwise continue advertising
  separate legacy NIM images, narrower Generator capabilities, or unsupported
  Reasoner fields after the new guide is added.
- Attempted the repository-prescribed locked unit-test command for
  `test_data_models.py`, `test_examples.py`, `test_reasoner_inference.py`, and
  `test_profile_selection.py`. Test collection did not start because building
  internal `nim-sdk==0.12.5` requires the unavailable private Cargo registry
  `sw-nemollm-rust` in this host environment.
- Verified that the failed environment setup and all static checks left the NIM
  tracked worktree unchanged.

Do not report the unit tests as passing. Runtime/API validation remains pending
until the internal dependency environment or a published release image is
available.

## Authoring and validation checklist

Before drafting:

- Refresh all source commits.
- Review diffs to request models, examples, environment variables, profiles, and
  documentation since the snapshots above.
- Resolve or explicitly defer every item in the TBD ledger.
- Capture live OpenAPI under both a Generator and a Reasoner profile when a
  release image is available.
- Confirm each planned page has primary-source coverage in the capability map.
- Treat the previous Generator parity matrix as a minimum coverage gate; no row
  may be silently dropped.
- Treat the previous Reasoner/VLM parity matrix as the same kind of minimum
  coverage gate; unresolved release facts remain visible TBDs.
- Treat the current `documentation.md` and first-party example adaptation
  matrices as coverage gates too; every row must be implemented, corrected,
  intentionally excluded, or visibly deferred.
- Confirm whether the documentation change may update stale inbound README
  sections outside `cookbooks/cosmos3/nim`. If scope is path-only, preserve them
  but list the unresolved contradictions in the handoff/PR description.

While drafting:

- Put every fact in one canonical page and link to it elsewhere.
- Use stable descriptive headings rather than document-wide numbered sections.
- State runtime mode and endpoint beside every request example.
- Keep request bodies complete and directly runnable.
- Include expected status/output and output-file handling.
- Use dynamic `/v1/models` discovery where hard-coded served names are not part
  of the stable contract.
- Clearly distinguish defaults, recommended values, supported values, and tested
  values.
- Keep `NIM_PROMPT_UPSAMPLING_API_KEY` distinct from `NGC_API_KEY`, never print
  either secret, and describe only the implementation-proven OpenAI-compatible
  endpoint contract for prompt upsampling.
- Do not mention private local paths, internal NIMCraft processes, or unreleased
  profiles in public docs.
- Add the repository's OpenMDW-1.1 SPDX notice to every new Markdown and Python
  artifact, use labeled code fences, and avoid collapsed `<details>` sections in
  the focused guides.

Before publication:

- Validate Markdown links and anchors.
- If inbound integration edits are in scope, verify the root README, shared
  Cosmos3 README, and audiovisual/Reasoner/action/transfer READMEs point to the
  correct canonical new page and no longer make contradictory current-NIM
  claims.
- Parse every JSON request body.
- Syntax-check every Python example.
- Validate Generator payloads against the reviewed request model or live API.
- Exercise Reasoner request construction offline and, when available, against a
  live release image.
- Run at least one smoke request for every documented capability/profile mode
  on supported hardware, or clearly mark examples not runtime-validated.
- Compare published defaults/ranges against live OpenAPI and current code.
- Search for legacy image names, `/v1/videos`, `/v1/videos/sync`, obsolete model
  counts, unresolved placeholders, and contradictory limitations. Every
  remaining placeholder must map to this file's TBD ledger, be visibly labeled
  in public prose, and avoid implying a usable value; publication does not
  require inventing or prematurely resolving release-owned facts.
- Search Generator examples for vLLM-Omni-only transport or fields:
  `input_reference`, `extra_params`, `control_path`, `view_point`, request-level
  `guardrails`, async video polling, `text2image`, and sound-generation fields.
  Any occurrence must be explanatory comparison text, not a NIM request.
- Search Reasoner examples for local `file://` media and
  `mm_processor_kwargs`; neither is part of supported guidance without the
  release validation recorded in the TBD ledger.
- Confirm `SOURCES.md` remains ignored and untracked.
- Confirm `examples/outputs/` is ignored before running examples and that no
  generated payload, response, media, preview, or credential artifact is
  tracked.
- Confirm every new Markdown/Python artifact carries the current OpenMDW-1.1
  SPDX notice and every runnable code fence has an appropriate language label.
- Audit every row in the previous Generator parity matrix against the completed
  guides and record its final destination, authoritative link, or explicit TBD.
- Audit every row in the previous Reasoner/VLM parity matrix the same way,
  including the rows intentionally deferred for live release validation.
- Search for `NGC_TOKEN`; use it only as plain-language terminology or an
  explicitly mapped local alias. Runtime examples must pass `NGC_API_KEY`.
- If acknowledgement content is included, confirm it came from the exact
  released image/build and was approved for publication. Otherwise retain the
  explicit acknowledgement TBD and do not infer an inventory.
