<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Certified NIM documentation source map

> Source-provenance artifact maintained on the
> `egor/cosmos3_nim_docs` branch.
> This is an authoring reference, not an end-user guide.
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

The initial discovery phase established the evidence map and page boundaries
before public drafting. Authoring was subsequently authorized; the public
guides now use this file as their provenance and coverage ledger.

## Reviewed repository snapshots

Reviewed on 2026-07-27.

| Repository | Local path | Branch | Reviewed commit | Role |
| --- | --- | --- | --- | --- |
| Cosmos cookbook | `/Users/ekrivov/projects/cosmos` | `egor/nim_docs` | `eb5bf7940ec902bf44791126691b6242a7cd7b3b` | Target baseline before public authoring edits |
| Cosmos3 Certified NIM | `/Users/ekrivov/projects/cosmos-genai-nim` | `cosmos3` | `63578446d6c6eaeffc4b2a378f24bf1c9027494b` | Current product implementation and primary authority |
| NIM product documentation | `/Users/ekrivov/projects/documentation` | `main` | `9a81f6952ca0567b616ca7ce5c412950613e8dc7` | Previous Generator and Reasoner documentation |
| Cosmos framework | `/Users/ekrivov/projects/cosmos-framework` | `main` | `fbb5c9bf4b1298a09cabbe8d60389ef06ab60821` | Model terminology and non-NIM behavior only |

The runnable examples documented by this guide are maintained locally under
`cookbooks/cosmos3/nim/examples`. Their request shapes are validated against the
current API models, runtime behavior, tests, and live OpenAPI when available.
They use an independently maintained teaching structure while keeping
request fields synchronized with reviewed NIM fixtures and reusing canonical
cross-backend cookbook scenarios where the modalities match.

Before authoring or updating public docs, refresh the commit values and inspect
changes to the primary source files listed below.

### Incremental T2I source refresh

Reviewed on 2026-07-29:

| Repository | Branch | Reviewed commit | Change covered |
| --- | --- | --- | --- |
| Cosmos cookbook | `egor/cosmos3_nim_docs` | `60a7871` | Documentation baseline before T2I edits |
| Cosmos3 Certified NIM | `cosmos3` | `74064b2318222018af446b03701f8a8cbeaa28c3` | Merged T2I request/response contract, JPEG artifacts, prompt upsampling, visual guardrails, examples, and environment-variable cleanup |

The T2I refresh is incremental: the original snapshot below remains the
provenance for previously researched profiles and historical coverage. The
merged `cosmos3` state is authoritative for the new modality and the four
removed Generator execution variables.

T2I evidence at that commit:

- `serving_stack/data_models/generation.py` selects prompt-only T2I with
  `num_output_frames=1`, applies its defaults, and rejects conditioning inputs;
- `serving_stack/data_models/responses.py` requires exactly one of `b64_image`
  and `b64_video` and forbids action metadata on image responses;
- `serving_stack/workflow.py` encodes the one-frame visual result as JPEG;
- `serving_stack/prompt_upsampling.py` selects T2I templates and strips
  video-only duration/FPS fields; and
- `examples/t2i.py` supplies the canonical representative request fixture.

The same refresh removed `NIM_ATTENTION_BACKEND`, `NIM_ENABLE_CUDAGRAPH`,
`NIM_ENABLE_FULLGRAPH`, and `NIM_ENABLE_AUTOTUNE` from the Generator
configuration. `NIM_ENABLE_TORCH_COMPILE` remains supported and defaults to
`true`. The six variables intentionally excluded in the preceding
configuration review remain excluded and are not reintroduced by this update.

### Full source refresh for the documentation update

Reviewed through 2026-08-05:

| Repository | Branch | Reviewed commit | Change covered |
| --- | --- | --- | --- |
| Cosmos cookbook | `egor/cosmos3_nim_docs` | `6750fc8ce99ff33d855eb1a1ecb9b24c0a22a15d` | Merged documentation baseline including semi-final grouped hardware requirements |
| Cosmos3 Certified NIM | `cosmos3` | `90a77482335b87cfcd25bf9d61c65278acd3f5ce` | Shared model-variant profile axis, integrated-GPU unified-memory policy, semi-final profile catalog, synchronized Reasoner VRAM floors, Reasoner QA, explicit Generator modes, DFlash, system-memory admission, and earlier model/BYOC/Transfer contracts |

The current NIM commit is 70 commits beyond the incremental `74064b23` pin and
19 commits beyond the preceding `280bbea3` documentation pin.
The tracked source diff under `cosmos3/` changes 154 files. The source checkout
also contains an unrelated untracked `cosmos3/bugs/` directory; it was not used
as evidence. This section supersedes stale current-state conclusions in the
older snapshot sections below while retaining those sections as historical
provenance.

Current high-impact contracts:

- Generator `POST /v1/infer` now requires explicit `model_mode`. It renames
  `num_output_frames` to `num_frames`, `steps` to `num_inference_steps`, and
  top-level `image`/`video` to `input_reference`; Action mode moves from
  `action_params.mode` to the top level. Old fields receive HTTP 422 and
  responses are unchanged.
- `NIM_MODEL_PATH` replaces `NIM_FT_CHECKPOINT`. Generator accepts an absolute
  local checkpoint and retains profile-owned guardrails. Reasoner accepts an
  absolute local path or `hf://owner/repository[:revision]`, with optional
  `HF_TOKEN` and explicit download/offline policy.
- Generator and Reasoner profiles carry `model_variant` as their shared model
  selector axis. Generator variants are `nano`, `nano-droid`, `super`,
  `super-t2i`, `super-t2i-4step`, `super-i2v`, and `super-i2v-4step`;
  Reasoner variants are `nano` and `super`. `NIM_MODEL_VARIANT` selects the
  exact runtime checkpoint contract.
- Four-step T2I/I2V variants own `num_inference_steps=4`,
  `guidance_scale=1.0`, and scheduler flow shift. Clients omit those controls;
  specialist variants reject the
  wrong request mode.
- Nano-DROID is a policy-only Generator variant with a strict current-state
  observation, 32 action steps, an 8-wide output, and no visual response.
- Generator responses now permit image, video with optional action, or
  action-only output. T2I still forbids action metadata.
- Transfer has a startup-derived VRAM admission check separate from ordinary
  generation profile compatibility. `NIM_ALLOW_UNSAFE_TRANSFER=1` bypasses the
  check at OOM risk.
- Generator profile policy includes independent model, text-guard, and
  video-guard residency. Public operator guidance covers the profile-backed
  guardrail offload controls.
- Reasoner video pruning adds `NIM_VIDEO_PRUNING_METHOD=vidcom2|evs`, used when
  `NIM_VIDEO_PRUNING_RATE` is nonzero.
- Reasoner Chat defaults thinking off, accepts an explicit thinking/token
  budget and parsed-reasoning request, maps `developer` to `system`, enables
  Hermes-format automatic tool calls, and strictly validates
  `include_reasoning` and `top_logprobs`. Request/sampling errors use HTTP 400;
  media errors use HTTP 422.
- `NIM_USE_DFLASH=1` enables speculative decoding only for Nano Reasoner. Each
  generated Nano Reasoner profile carries the separate BF16 draft artifact;
  Generator and Super Reasoner are rejected at startup.
- Ten single-GPU Super BF16 model/layer-offload development rows require 150
  GiB of effective system memory. Startup checks the container cgroup limit
  before host physical memory.
- Generator and Reasoner FP8 compute-capability floors are both 8.9. The
  current Reasoner runtime and generated-profile VRAM floors are synchronized
  at 46 GiB for two-GPU Super BF16, 67 GiB for one-GPU Super FP8, and 73 GiB
  for one-GPU Super NVFP4. Grouped source-profile requirements are authorized
  for public hardware planning, with an explicit non-release caveat.
- `/v1/metadata` reports checkpoint source for either runtime and
  `model_variant` for Generator.
- Reasoner streaming is intentionally outside the public documentation scope;
  no public page, configuration table, or runnable example describes it.
- Benchmark-result processing changes are internal validation tooling and do
  not change the documented runtime contract.
- On integrated GPUs with unified host/device memory, profile selection reserves
  16 GiB for the host by default, compares profile floors with the remainder,
  and offers only resident-model/resident-guardrail Generator rows. The
  operator-facing `NIM_UNIFIED_MEMORY_HOST_RESERVE_GIB` adjusts the reserve;
  discrete-GPU selection is unchanged.

Running `local_nimcraft/make_profiles.py` from this source into an untracked
temporary output generated 122 development rows: 115 Generator and 7 Reasoner.
Every row carries the shared `model_variant` selector tag.
The Generator rows split across 18 `nano`, 7 `nano-droid`, 18 `super`, and 18
rows for each of the four specialist Super variants. Three Nano Reasoner rows
include the DFlash draft artifact, and ten Super BF16 offload rows carry the
150-GiB system-memory floor. The public support matrix groups these semi-final
requirements by user-facing model, precision, offload, GPU count, and floor; it
does not expose 122 development profile IDs or claim released support.

The current source tree no longer contains `cosmos3/documentation.md`, the
static `cosmos3/api_spec.yaml`, or a tracked generated `profiles.json`.
Implementation, tests, profile inputs/generator, and live release OpenAPI now
replace those former current-source references. The deleted guide and static
schema remain historical evidence only at the broad `63578446` snapshot.

Primary changed evidence:

- `api-update.md`
- `serving_stack/environment.py`
- `serving_stack/data_models/generation.py`
- `serving_stack/data_models/transfer.py`
- `serving_stack/reasoner/model_source.py`
- `serving_stack/reasoner/contracts.py`
- `serving_stack/reasoner/api.py`
- `serving_stack/reasoner/profile.py`
- `serving_stack/reasoner/runtime.py`
- `serving_stack/reasoner_inference.py`
- `serving_stack/data_models/actions.py`
- `serving_stack/data_models/responses.py`
- `serving_stack/generator_mapping.py`
- `serving_stack/nano_droid.py`
- `serving_stack/profile_selection/selection.py`
- `serving_stack/profile_selection/criteria.py`
- `serving_stack/profile_selection/hardware.py`
- `serving_stack/profile_selection/startup.py`
- `serving_stack/profile_selection/transfer_vram.py`
- `serving_stack/profile_selection/vram_profiles.yaml`
- `local_nimcraft/make_profiles.py`
- focused tests and examples for each contract

## Authority and conflict resolution

Use this order whenever sources disagree:

1. Current runtime implementation, request models, profile/configuration code,
   and tests in `cosmos-genai-nim/cosmos3`.
2. OpenAPI returned by the reviewed Certified NIM image at runtime, when the
   image is available for validation.
3. Historical `cosmos-genai-nim/cosmos3/documentation.md` from the reviewed
   `63578446` snapshot, only for durable organization or explanations.
4. Previous public Generator and Reasoner product documentation.
5. Current Cosmos cookbook documentation and approved assets.
6. Cosmos Framework documentation for model concepts only.

Rules:

- Treat implementation and validation tests as API truth.
- Validate local cookbook request bodies against the API contract rather than
  another repository's runner implementation.
- Use reviewed Cosmos3 NIM fixtures as representative request evidence, but do
  not copy their test-runner machinery or treat it as a public API contract.
- Treat the deleted historical `documentation.md` as a broad operational
  inventory, not as a current source or exact schema when it conflicts with
  code.
- Treat previous product docs as reusable explanations and information
  architecture, not as current names, defaults, limits, or support claims.
- Treat other backend materials as conceptual research only. Never copy their
  endpoints or request fields into Certified NIM docs without verification.
- Record unresolved release facts in the TBD ledger. Do not publish guesses.

## Historical resolved facts for source snapshot `63578446`

These facts record the original review and are retained for provenance. The
2026-08-03 full source refresh above supersedes them wherever the implementation
changed. Release-sensitive facts must still be rechecked against the published
image/manifest before publication.

### Runtime dispatch and defaults

- The exported manifest contains both `generator` and `reasoner` profiles.
- One selected profile starts one API/backend: Generator uses `pytriton` and
  Reasoner uses `vllm`.
- With no model-type selector, startup chooses `generator`.
- Unpinned selection softly preferred the Nano profile family and FP8.
- Generator additionally softly prefers `profile=latency` and then the offload
  order `none`, `model`, `layer`.
- Soft defaults are skipped rather than failing when they would empty the
  compatible candidate set.
- Generator and Reasoner shared model-family, precision, `n_gpus`, and `nim_*`
  selector axes. `profile=latency|throughput` was Generator-only and rejected
  for Reasoner.
- Runtime, precision, Generator performance, and offload controls acted as
  shorthands for selector tags. Explicit `NIM_MODEL_PROFILE` pinned one exact
  manifest profile.

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

- Generator exposes JSON `POST /v1/infer` and returns exactly one base64 visual
  output: JPEG for T2I or VP9/MP4 for video, with optional video action
  metadata.
- Current request-mode inference covers T2I, T2V, I2V, V2V, transfer, forward
  dynamics, policy, and inverse dynamics.
- Image-to-image and sound conditioning/output are not surfaced by this NIM
  source snapshot.
- Reasoner uses the OpenAI-compatible Chat Completions surface and inherited
  Responses routes.
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

### Authoring-ready Generator API contract

This inventory is intentionally concise. The public `api-reference.md` should
explain these fields in tables and link to task guides for combinations. It
should not reproduce Pydantic implementation details or error-message text.

All Generator capabilities use `POST /v1/infer`. Unknown fields are rejected.
The mode is inferred from the request shape:

| Mode | Required discriminator/input | Forbidden or mode-specific rule | Local cookbook example |
| --- | --- | --- | --- |
| T2I | Non-empty `prompt`; no conditioning inputs; `num_output_frames=1` | Image/video/Transfer/action/V2V controls are forbidden; I2I is unsupported | `cookbooks/cosmos3/nim/examples/t2i.py` |
| T2V | Non-empty `prompt`; no media; 25 or more frames | Video generation frame rules apply | `cookbooks/cosmos3/nim/examples/t2v.py` |
| I2V | `image` | `image` and `video` are mutually exclusive | `cookbooks/cosmos3/nim/examples/i2v.py` |
| V2V | `video` without `transfer` or `action_params` | V2V conditioning controls are valid only here | `cookbooks/cosmos3/nim/examples/v2v.py` |
| Transfer | Non-empty `transfer` | Cannot combine with `image`, `action_params`, or V2V conditioning controls | `cookbooks/cosmos3/nim/examples/transfer.py` |
| Forward dynamics | `image` plus `action_params.mode=forward_dynamics` | Requires an input action trajectory | `cookbooks/cosmos3/nim/examples/action.py` |
| Policy | `image` plus `action_params.mode=policy` | Produces rather than accepts an action trajectory | `cookbooks/cosmos3/nim/examples/action.py` |
| Inverse dynamics | `video` plus `action_params.mode=inverse_dynamics` | Produces rather than accepts an action trajectory | `cookbooks/cosmos3/nim/examples/action.py` |

Shared top-level request fields:

| Field | Contract in source snapshot |
| --- | --- |
| `prompt` | Optional string, at most 20,000 characters. Required only when no image, video, or transfer input establishes a request. Normalized to an empty string for media/action requests when omitted. |
| `negative_prompt` | Optional string, at most 20,000 characters. Omission becomes empty for T2I and selects the vendored structured OSS negative prompt for video modes; an explicit empty string disables it. |
| `image` | Base64, image data URL, or public HTTP(S) URL; at most 20,000,000 encoded characters. Empty/whitespace input is treated as absent. |
| `video` | Base64, video data URL, or public HTTP(S) URL; at most 100,000,000 encoded characters and 75 MB after decoding. Empty/whitespace input is treated as absent. |
| `seed` | Optional integer >= 0. Generated by the service when omitted. Public examples always set `0` for reproducibility. |
| `guidance_scale` | Finite JSON number in `[1.0, 7.0]`; T2I default `4.0`, ordinary video default `6.0`. |
| `steps` | JSON integer in `[1, 100]`; T2I default `50`, ordinary video default `35`. |
| `flow_shift` | Finite JSON number; T2I default `3.0`, ordinary video default `10.0`. No additional range constraint is present. |
| `resolution` | One of 18 keys across `256`, `480`, and `720` tiers, each with bare/`16_9`, `1_1`, `9_16`, `4_3`, and `3_4` spellings. Bare tiers mean 16:9. T2I defaults to `720_1_1`; video defaults to `720`. |
| `num_output_frames` | `1` selects T2I. Video uses the `4k+1` cadence, defaults to `189`, requires at least `25`, and caps output at `397`/`297`/`197` frames for the 256/480/720 tiers. |
| `fps` | Finite JSON number in `[1.0, 60.0]`; default `24.0`. Retained but not encoded for T2I; source recommends 10–30 for video quality. |
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

### Authoring-ready response and Reasoner contract

Generator success response contains exactly one visual media field:

| Field | Contract |
| --- | --- |
| `b64_image` | T2I-only raw base64 string containing a JPEG. |
| `b64_video` | Video-mode-only raw base64 string containing a VP9-encoded MP4. |
| `action` | Predicted action object for policy/inverse dynamics; otherwise `null`. It cannot be non-null on an image response. |

The inactive media field is omitted. An image response cannot include action
metadata.

Generator schema/media/guardrail validation generally returns HTTP 422;
unexpected internal failures return HTTP 500. Public docs should describe the
stable status semantics and example error envelope, not promise exact message
text.

Reasoner profiles expose OpenAI-compatible completion APIs rather than
`/v1/infer`. The authoring baseline is:

| Surface | Proven behavior and local cookbook coverage |
| --- | --- |
| `POST /v1/chat/completions` | Image and video messages; `cookbooks/cosmos3/nim/examples/reasoner.py`. |
| `POST /v1/completions` | Registered and passed through the same normalization/validation middleware; no local cookbook example is provided. |
| `POST /v1/responses` | Image input using `input_image` before `input_text`; `cookbooks/cosmos3/nim/examples/reasoner_responses.py`. |
| `GET /v1/responses/{response_id}` | Registered through NIMlib when Responses routes are enabled; requires stored state for meaningful retrieval. |
| `POST /v1/responses/{response_id}/cancel` | Registered through NIMlib when Responses routes are enabled; background/stored response behavior depends on storage configuration. |

Completion requests must include a non-empty `model`. The service injects
sampling defaults `temperature=0.7`, `top_k=20`, and `top_p=0.8` when omitted.
It validates temperature in `[0,2]`, top-p in `(0,1]`, and top-k as `-1` or an
integer >= 1. It normalizes OpenAI `response_format` and legacy guided-decoding
fields into vLLM structured outputs. The full inherited request/response schema
must come from the released image's live OpenAPI instead of being hand-copied.

Runtime routing, normalization, and tests establish these multimodal shapes:

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

### Local cookbook example design

The scripts under `cookbooks/cosmos3/nim/examples` are independently maintained
teaching examples. They keep request construction, the API call, status
handling, and primary output visible in each task script. Only strict local
media encoding, image/video decoding, and compact JSON-prompt serialization are
shared. Their contracts are checked against runtime code, request models,
tests, and live release evidence rather than another repository's runner
implementation. T2I, T2V, and I2V reuse the audiovisual cookbook's
`robot_draping`, `robot_kitchen`, and `car_driving` scenarios, respectively.
The five precomputed Transfer cases reuse the Transfer cookbook's prompt,
negative prompt, control, geometry, seed, and chunk-length choices. Action
reuses the three AV trajectories, two AV inverse videos, and one UMI trajectory
chunk while retaining NIM-owned request and response validation. Reasoner
reuses nine representative prompt-guide assets while replacing prompt-authored
thinking tags and free-form JSON extraction with NIM request controls and
structured output. It retains canonical case-specific sampling: `seed=0` for
the three image cases that use it, the trajectory temperature/top-p/penalty
recipe, and backend defaults for the other cases; all videos retain 4-FPS
sampling. API adapters and model-specific request contracts remain NIM-owned.

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
| `serving_stack/reasoner_inference.py` | Reasoner NIMlib/vLLM interface, request normalization, and route behavior |
| `serving_stack/environment.py` | Environment-variable parsing, profile selector shorthands, Reasoner tuning, BYOC, guardrails, logging, and media URL policy |
| `serving_stack/prompt_upsampling.py` | Optional Generator T2I/T2V/I2V prompt rewriting through an operator-supplied OpenAI-compatible endpoint |
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
| `serving_stack/reasoner_inference.py` | Enabled routes, engine arguments, model-name resolution, normalization, and errors |
| `serving_stack/environment.py` (`ReasonerEngineOptions`) | Prompt media limits and Reasoner-specific runtime variables |
| `serving_stack/tests/test_reasoner_inference.py` | Route inheritance, disable flags, normalization, and errors |

Known current Reasoner routes from runtime code and tests:

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/{response_id}`
- `POST /v1/responses/{response_id}/cancel`
- `GET /v1/models`
- NIM management and health endpoints exposed by the selected interface

The Responses storage/retrieve/background features depend on runtime flags. Do
not imply persistent response storage is on by default.

### Local cookbook runnable examples

| Source | Capability |
| --- | --- |
| `cookbooks/cosmos3/nim/examples/t2i.py` | General-purpose text-to-image |
| `cookbooks/cosmos3/nim/examples/t2i_4step.py` | Four-step specialist text-to-image with profile-owned sampling omitted |
| `cookbooks/cosmos3/nim/examples/t2v.py` | Text-to-video |
| `cookbooks/cosmos3/nim/examples/i2v.py` | General-purpose image-to-video using local media |
| `cookbooks/cosmos3/nim/examples/i2v_4step.py` | Four-step specialist image-to-video with profile-owned sampling omitted |
| `cookbooks/cosmos3/nim/examples/v2v.py` | Video-to-video using local media |
| `cookbooks/cosmos3/nim/examples/reasoner.py` | Nine canonical image/video Chat tasks, optional explicit reasoning, structured output validation, and saved artifacts |
| `cookbooks/cosmos3/nim/examples/reasoner_responses.py` | Reasoner Responses API |
| `cookbooks/cosmos3/nim/examples/action.py` | Forward dynamics, policy, and inverse dynamics |
| `cookbooks/cosmos3/nim/examples/transfer.py` | Precomputed and derived transfer controls |
| `cookbooks/cosmos3/nim/examples/common.py` | Strict media encoding, image/video decoding, and compact JSON-prompt serialization |

### Historical `documentation.md` remains a coverage floor

Every numbered source-guide section must have a deliberate destination or
correction. The implementation remains authoritative when the prose conflicts
with code or the current profile export.

| Source-guide section | Required treatment | Planned destination/status |
| --- | --- | --- |
| 1. Overview and prerequisites | Rewrite around the unified selected Generator/Reasoner runtime; preserve prerequisites categories but not unverified counts, versions, or stale Generator-only profile summaries | `README.md`, `prerequisites.md`; release identity/support values TBD |
| 2. Endpoints at a glance | Separate endpoints by active runtime and distinguish shared management routes from Generator/Reasoner inference | `api-reference.md`, `operations.md`; live OpenAPI capture pending |
| 3. T2V/I2V quickstart | Preserve minimal requests and MP4 decoding; use the portable Python encoder/decoder and add V2V/current modes elsewhere | `README.md`, `generation.md`, `examples/` |
| 4. Request reference | Keep common Generator fields compact; place complete nested and Reasoner contracts with their tasks; correct the stale 2,000-character heading | `api-reference.md`, `generation.md`, `action.md`, `transfer.md`, `reasoning.md` |
| 5. Frame-count math | Preserve 4k+1 explanation, per-tier caps, and action/transfer-specific frame rules | `generation.md`, `action.md`, `transfer.md` |
| 6. Image input requirements | Preserve raw base64/data URL/public URL concepts and media size/error behavior; prefer MIME-aware helpers | `generation.md`, `reasoning.md`, `support-matrix.md` |
| 7. Supported resolutions | Generate exact key-to-pixel mappings from `resolutions.py` and distinguish ordinary generation from action templates | `generation.md`, `action.md` |
| 8. Response format | Preserve `b64_video`, optional action metadata, decoding, output files, and playback caveat | `api-reference.md`, `generation.md`, `action.md`, task scripts |
| 9. Validation cheat sheet | Replace the T2V/I2V-only table with mode-complete current constraints on their canonical task pages | `generation.md`, `action.md`, `transfer.md`, `reasoning.md` |
| 10. Error envelope | Preserve stable status/type semantics without copying exact mutable messages; add Reasoner 400/422 behavior | `operations.md` |
| 11. Deployment | Preserve NGC login, cache, ports, Docker flags, single/multi-GPU concepts, throughput/latency, cleanup, and cold-start notes | `deployment.md`; image/profile values release-gated |
| 12. Environment variables | Re-audit every variable against `environment.py`; split canonical configuration from operational subsets | `configuration.md`, `operations.md` |
| 12.1 Prompt upsampling | Document the current optional, Generator-only T2I/T2V/I2V flow, secret handling, supported template styles, failure fallback, and non-applicable modes | `configuration.md`, `generation.md`, `operations.md` |
| 13. Profile selection | Preserve selectors, conflicts, pinning, soft defaults, layouts, and selection cascade from current code | `deployment.md`; released manifest recheck required |
| 14. Support matrix | Replace the stale prose grid with grouped current source-profile requirements and later reconcile it with the released manifest/tested matrix | `support-matrix.md`; semi-final hardware rows documented, released-row approval pending |
| 15. BYOC | Preserve historical layout/mount guidance, but supersede its Generator-only variable and boundary with current `NIM_MODEL_PATH` contracts | `bring-your-own-checkpoint.md`; released formats remain TBD |
| 16. Helm/Kubernetes | Preserve topic and operational categories but not placeholder chart identity or unverified values | `helm.md`; chart/values TBD |
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
- The current request path applies it to T2I, T2V, and I2V. V2V, action, and
  transfer requests are not upsampled.
- When enabled, current startup validation requires an endpoint URL, model, API
  key value, a valid `external_api|reasoner` template style, and the bundled
  template files. A Reasoner-profile container does not require this config.
- The endpoint is normalized to an OpenAI-compatible
  `/v1/chat/completions` URL and receives Bearer authorization. Public examples
  must not imply compatibility with a provider's native, non-OpenAI API.
- I2V sends the conditioning image to a vision-capable upsampler as a data URL;
  T2I and T2V send text only.
- The NIM strips the Reasoner-only `scene_imagination` scratch field. T2I pins
  resolution and aspect ratio and removes video-only duration/FPS fields.
  T2V/I2V pin resolution, aspect ratio, duration, and FPS from the original
  request so the upsampler cannot change output shape.
- Request-time endpoint, timeout, or response-parsing failures log a warning and
  fall back to the original prompt instead of failing generation.
- `NIM_PROMPT_UPSAMPLING_API_KEY` is a separate external-service secret from
  `NGC_API_KEY`. Neither value may appear in logs, examples, saved payloads, or
  source control.

Primary evidence:

- `serving_stack/environment.py` at `74064b23`
- `serving_stack/prompt_upsampling.py` at `74064b23`
- `serving_stack/generator_mapping.py` at `74064b23`
- `serving_stack/tests/test_prompt_upsampling.py` at `74064b23`
- `documentation.md:513-549`

### Local cookbook asset policy

Use assets already present in the cookbook repository and retain their existing
provenance and license history. Convert local media to data URLs at runtime
where required; do not copy private-source assets into the public cookbook.
For cross-backend generation examples, prefer the reviewed structured prompts,
negative prompts, and conditioning images under
`cookbooks/cosmos3/generator/audiovisual/assets/`. For Transfer comparisons,
reuse the matching prompt/control pair and shared negative prompt under
`cookbooks/cosmos3/generator/transfer/assets/`. For Action, reuse reviewed image,
video, and trajectory cases under `cookbooks/cosmos3/generator/action/assets/`
without importing unsupported embodiments or backend-only request fields. For
Reasoner, reuse reviewed media under `cookbooks/cosmos3/reasoner/assets/` but
adapt prompts that prescribe `<think>` tags to explicit Certified NIM thinking
controls. Serialize JSON prompt assets as compact strings because the Generator
API's `prompt` and `negative_prompt` fields are strings, not nested JSON
objects.

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
| Hardware prerequisites | CPU architecture, RAM, disk, shared memory, supported GPU architecture/count, homogeneity, and VRAM floors | `prerequisites.md`, `support-matrix.md`; semi-final GPU/profile floors documented, remaining host/release values TBD |
| Software prerequisites | Linux, driver, Docker, NVIDIA Container Toolkit, and setup verification with `nvidia-smi` | `prerequisites.md`; exact versions TBD |
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
| GPU selection | Cover `--gpus`, homogeneous devices, selector-visible GPU count, and released compatibility | `deployment.md`, `prerequisites.md`, `support-matrix.md` |
| IPC/shared memory | Explain `--shm-size` launch usage and `/dev/shm` media staging requirements | `prerequisites.md`, `deployment.md`; release values TBD |
| Environment-variable reference | Include shared NIM variables plus Generator- and Reasoner-specific variables with defaults, scope, and conflicts | `configuration.md`; operational subsets in `operations.md` |
| Model/profile selectors | Cover `NIM_MODEL_TYPE`, size, precision, Generator performance profile/offload, exact profile pinning, tag selectors, and parallelism aliases | `deployment.md` |
| Volumes | Cover persistent cache and read-only BYOC mounts separately | `deployment.md`, `bring-your-own-checkpoint.md` |
| Sampling reference | Preserve common Generator fields in the compact API reference and mode-specific defaults/ranges with their tasks | `api-reference.md`, `generation.md`, `reasoning.md`, `action.md`, `transfer.md` |
| Endpoint inventory | Separate shared management endpoints, Generator `/v1/infer`, and Reasoner OpenAI-compatible routes | `api-reference.md`, `operations.md` |
| Request and response examples | Include complete request bodies, expected status/shape, decoding, and saved artifacts | Every task guide plus `examples/` |
| Error handling | Cover stable 4xx/5xx semantics and mode-specific validation without promising exact error strings | `operations.md` and task guides |
| OpenAPI | Explain live `/openapi.json`; capture/validate it under both Generator and Reasoner profiles | `api-reference.md`; runtime capture pending |
| Resolution and frame-cap tables | Preserve exact key-to-WxH shapes and per-tier frame caps | `generation.md` |
| Model variants, precisions, and VRAM | Provide grouped source-profile requirements now, then reconcile against the release manifest and tested SKU inventory | `support-matrix.md`; source compute/VRAM floors documented, release approval pending |
| Parallelism/profile selection | Explain latency versus throughput, replicas/sharding, offload, selection cascade, and explicit pinning | `deployment.md` |
| Input/output codecs | State current VP9-in-MP4 output and validate all claimed image/video inputs against the release | `support-matrix.md`, `generation.md` |
| BYOC | Preserve mount pattern, expected checkpoint layout, profile cross-check, readiness, cache/ulimits, path rules, and metadata verification | `bring-your-own-checkpoint.md`; Generator-only unless release proves more |
| Helm prerequisites and chart selection | Cover GPU Operator/cluster needs and select the released chart version | `helm.md`; chart name/version TBD |
| Kubernetes NGC secrets | Cover both the `nvcr.io` image-pull secret and the generic secret whose key is `NGC_API_KEY` | `helm.md` |
| Helm values and GPU resources | Provide NIM-specific image, secret, cache, ports, environment, GPU count, probes, and service values | `helm.md`; validate against released chart |
| Kubernetes storage | Cover PVC/RWX implications and persistent model-cache tradeoffs; link to chart documentation for generic values | `helm.md` |
| Kubernetes monitoring | Cover monitoring concepts only if the released chart/runtime supports the documented settings | `helm.md`, `operations.md`; release validation pending |
| Helm launch, readiness, port-forward, and inference | Preserve a minimal end-to-end deployment verification flow | `helm.md` |
| Helm troubleshooting | Cover pending pods, GPU scheduling, storage mounts, and startup-probe failures | `operations.md` |
| Metrics endpoint and metric families | Document endpoint(s) and release-observed metrics; do not copy stale metric names without scraping the released image | `operations.md` |
| Prometheus and Grafana | Include a minimal scrape example and dashboard workflow; avoid pinning old tool versions unnecessarily | `operations.md` |
| Inspection endpoints | Cover health, metrics, metadata, models, manifest, version, and license per active runtime | `api-reference.md`, `operations.md` |
| Logging and distributed diagnostics | Cover service/backend log levels, JSON logs, NCCL/debug knobs, and performance costs | `operations.md` |
| Troubleshooting | Preserve prerequisite, Docker, profile, NGC download, readiness, BYOC, OOM, timeout, metrics, air-gap/cache, and playback cases; add current mode/API failures | `operations.md` |
| Release notes | Identify the documented release and link to authoritative release notes; summarize only changes relevant to these cookbooks | `release-notes.md`; content remains TBD |
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
| Chat Completions with curl | Provide one complete multimodal request against `/v1/chat/completions` | `reasoning.md` |
| OpenAI Python client | Use `base_url=<NIM_URL>/v1`, a non-secret placeholder client key, model discovery, and explicit `extra_body` handling | `reasoning.md`, `examples/reasoner.py` |
| Image by public URL | Retain only after release smoke testing of remote URL access and failure behavior | `reasoning.md`; live validation pending |
| Image by data URL/base64 | Runtime handling and tests support `image_url`; document MIME-aware data URLs and media-before-text ordering | `reasoning.md`, `examples/reasoner.py` |
| Accepted image formats | Historical JPG/JPEG/PNG claims are not a current release contract | `support-matrix.md`; exact release formats TBD |
| Video by public URL | Retain only after release smoke testing of download, timeout, and decode behavior | `reasoning.md`; live validation pending |
| Video by data URL/base64 | Runtime handling and tests support `video_url`; document payload-size implications | `reasoning.md`, `examples/reasoner.py` |
| Accepted video containers/codecs | Do not copy the historical MP4/MKV/FLV/3GP and H264/H265/VP9/FLV matrix; validate the released decoder path | `support-matrix.md`; exact matrix TBD |
| Pre-decoded `video_frames` input | Present only in the historical page; current runtime evidence does not establish it | Omit from runnable guidance until live verification; support TBD |
| Media before prompt text | Current prompt guidance and local cookbook requests place image/video content before text | `reasoning.md` |
| Request-level video sampling | The local cookbook example uses `media_io_kwargs.video.fps`; current startup default is 4 FPS | `reasoning.md` |
| `fps` versus `num_frames` constraints | Historical docs say the fields are mutually exclusive; confirm accepted shape, bounds, and precedence against the released API | `reasoning.md`; live validation pending |
| `mm_processor_kwargs` pixel budgets | Historical-only in reviewed sources; current runtime options do not expose it | Do not publish as supported until live validation; support/defaults TBD |
| Operator-level `NIM_MEDIA_IO_KWARGS` | Current Reasoner engine accepts the JSON object and the startup layer provides a complete 4-FPS `pynvvc` video default; make clear that an operator override replaces the complete object | `configuration.md`, `operations.md` |
| Per-prompt media limits | Current defaults are five images and one video, configurable with `NIM_MAX_IMAGES_PER_PROMPT` and `NIM_MAX_VIDEOS_PER_PROMPT` | `reasoning.md`, `configuration.md` |
| Sampling defaults and validation | Current normalization supplies temperature 0.7, top-k 20, and top-p 0.8; current validation defines their accepted ranges | `reasoning.md` |
| `extra_body` request extensions | Pass `media_io_kwargs`, `top_k`, structured-output fields, and `nvext` explicitly through the OpenAI client `extra_body` | `reasoning.md`, `examples/reasoner.py` |
| Structured/JSON outputs | Current middleware supports `response_format`, `structured_outputs`, and legacy guided-output normalization for completion routes | `reasoning.md`; add a current example |
| Responses API | Current NIM exposes create/retrieve/cancel routes through NIMlib; use the standard `input_image`/`input_text` request shape | `reasoning.md`, `examples/reasoner_responses.py` |
| Responses storage/background features | Storage is disabled by default and release behavior is still an explicit validation item | `reasoning.md`, `operations.md`; published-image validation pending |
| Reasoning prompt format and `<think>` output | The historical page and current cookbook prompt guide contain explicit reasoning traces; do not promise hidden chain-of-thought or reproduce it as a general API guarantee. Document task prompting and final-answer schemas with approved wording | `reasoning.md`; product/policy wording review required |
| Image/video captioning and VQA | Retain as foundational task examples using current request shapes and vendored assets | `reasoning.md` |
| Temporal localization, event timelines, timestamp, and interval queries | Preserve the current cookbook task taxonomy and use structured final-answer schemas where useful | `reasoning.md` |
| Synthetic-data critic / physical plausibility | Preserve as a task example without implying a separate endpoint or deterministic judgment | `reasoning.md` |
| Embodied, common-sense, planning, and situation-understanding prompts | Preserve representative examples and link to the detailed prompt guide rather than duplicating its full gallery | `reasoning.md` |
| 2D grounding and action trajectories | Preserve the normalized 0-1000 coordinate convention, output schema, and pixel-conversion explanation after model-release verification | `reasoning.md` |
| Text-only queries | Historical support is plausible but lacks reviewed released-image evidence | `reasoning.md`; release smoke test pending |
| Media and sampling errors | Current interface maps media-related failures to 422 and sampling/validation failures to 400; document stable semantics, not exact messages | `reasoning.md`, `operations.md` |
| Profiles, GPU/precision support, and KV-cache behavior | Replace the historical Reasoner 1.7.0 tables with synchronized current source floors and later distinguish released/tested configurations | `support-matrix.md`; seven semi-final source layouts documented, release approval pending |
| Environment-variable reference | Carry forward only variables present in the current runtime; include request logging, caching, sequence/token limits, media limits, video preprocessing, compilation, and attention controls | `configuration.md`, operational subsets in `operations.md` |
| Metrics, logging, Helm, and troubleshooting | Reuse the surrounding VLM guide's organization, but validate endpoints, chart values, metric names, and failure modes against this release | `helm.md`, `operations.md`; release validation pending |

Primary current evidence:

- `cosmos3/serving_stack/reasoner_inference.py:39-50,270-329,336-388,391-454,472-545,574-603`
- `cosmos3/serving_stack/environment.py:419-508`
- `cosmos3/serving_stack/profile_selection/startup.py:25-46`
- `cosmos3/serving_stack/patches/vllm/README.md:29-42`
- `cosmos3/serving_stack/tests/test_reasoner_inference.py:620-934`
- `cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md:6-99,101-659`
- `cookbooks/cosmos3/reasoner/run_with_nim.ipynb` task cells for canonical
  assets, media ordering, 4-FPS video sampling, and task-specific recipes

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
| `cookbooks/cosmos3/generator/audiovisual/README.md` | Generator workflow headings and quickstart/table style |
| `cookbooks/cosmos3/reasoner/README.md` | Reasoner task organization and guide style |
| `cookbooks/cosmos3/generator/transfer/README.md` | Transfer terminology, controls, and troubleshooting style |
| `cookbooks/cosmos3/generator/action/README.md` | Action capability organization and domain terminology |
| `cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md` | Prompt intent, expected task shapes, and coordinate conventions |
| `cookbooks/cosmos3/reasoner/run_with_nim.ipynb` | Canonical Reasoner assets, NIM media transport, and per-case sampling recipes |

### Prior cookbook research

Existing Cosmos3 cookbook material was consulted for terminology, task
taxonomy, presentation conventions, and approved asset reuse. Those external
examples are not reproduced or described here and are not API authorities.
All public request and response guidance is validated independently against the
Certified NIM runtime contract.

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
| Unqualified model-variant/count and hardware claims | Avoid | Label grouped current-source profile requirements as semi-final; reserve supported/tested claims for released manifest/model-card evidence |
| Backend-specific request examples copied between integrations | Avoid | Validate each local example against the Certified NIM contract |

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
- Save the primary generated media and predicted action where applicable. Keep
  Reasoner output visible in the terminal and avoid auxiliary artifact layers
  that obscure the request/response flow.
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
| NGC authentication, cache, ports, first launch | `documentation.md`, runtime configuration | Previous product quickstart | `README.md`, `deployment.md` |
| Catalog/model-card identity and release notes | Published NGC release | Previous introduction/release notes | `release-notes.md`, `README.md`; release URL/version TBD |
| Hardware and profile selection | `make_profiles.py`, profile selection, exported profiles | `documentation.md`, support matrices | `support-matrix.md`, `deployment.md` |
| Shared management endpoints | live OpenAPI, `api_spec.yaml`, NIM interface | `documentation.md` | `operations.md`; compact index in `api-reference.md` |
| Generator shared request/response envelope | `data_models/generation.py`, tests | Local cookbook examples | `api-reference.md` |
| T2I, T2V, I2V, V2V | `generation.py`, `responses.py`, Generator runtime/tests | Local cookbook examples | `generation.md` |
| Generator prompt upsampling | `prompt_upsampling.py`, `generator_inference.py`, `environment.py`, tests | `documentation.md` | `configuration.md`, `generation.md`, `operations.md` |
| Chat Completions | `reasoner_inference.py`, tests | Previous Reasoner API page and local cookbook example | `reasoning.md` |
| Responses API | `reasoner_inference.py`, Reasoner tests | Local cookbook example | `reasoning.md` |
| Reasoner media limits/preprocessing | `environment.py`, runtime tests | Previous Reasoner API page and prompt guide | `reasoning.md`, `configuration.md` |
| Forward dynamics | `actions.py`, runtime/tests | Local cookbook example and task documentation | `action.md` |
| Policy | `actions.py`, runtime/tests | Local cookbook example and task documentation | `action.md` |
| Inverse dynamics | `actions.py`, runtime/tests | Local cookbook example and task documentation | `action.md` |
| Transfer controls | `transfer.py`, runtime/tests | Local cookbook example and task documentation | `transfer.md` |
| Environment variables | `environment.py` and NIM framework contract | `documentation.md`, previous config pages | `configuration.md`, operational subsets in `operations.md` |
| BYOC | BYOC validation in `environment.py` | `documentation.md`, previous BYOC page | `bring-your-own-checkpoint.md` |
| Helm/Kubernetes | released chart contract when confirmed | previous Helm page, `documentation.md` | `helm.md` |
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
- `api-reference.md` owns only runtime routing and the common Generator
  envelope. Generation, Action, Transfer, and Reasoning own their detailed
  task contracts.
- Focused pages own prerequisites, configuration, support, Helm, and BYOC.
  Deployment owns the Docker launch workflow. Operations owns observation,
  generic errors, diagnostics, failure recovery, and production caveats.
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

- NGC API key and Docker login.
- Cache and volume permissions.
- Container image, ports, shared memory, and ulimits.
- Choosing `generator` or `reasoner`, model variant, precision, performance
  profile, GPU exposure, and explicit profile selection.
- Readiness and cold-start expectations.
- Links to focused requirements, configuration, support, Helm, and BYOC pages.

### `release-notes.md`

- Released version/date and image tag.
- Compatibility changes, known limitations, and upgrade guidance.
- Release-owned URLs and support statements; all remain TBD until approved.

### `prerequisites.md`

- CPU/GPU, RAM, disk, shared-memory, Linux, driver, Docker, and toolkit
  requirements.
- Network and NGC access requirements.
- Host and container-runtime verification.

### `configuration.md`

- Shared, Generator, Reasoner, and profile-selection environment variables.
- Defaults, scope, conflicts, and secret handling.
- Prompt-upsampling launch configuration and external-service secret.

### `support-matrix.md`

- Released model/precision/GPU/VRAM/profile and offload compatibility.
- Tested-versus-compatible distinctions.
- Released image/video formats and codecs.

### `helm.md`

- Cluster and GPU Operator prerequisites.
- Chart discovery, NGC secrets, values, storage, GPUs, probes, and services.
- Install/upgrade, rollout, port-forwarding, and verification.

### `bring-your-own-checkpoint.md`

- Shared `NIM_MODEL_PATH` contract with separate Generator and Reasoner source
  forms, materialization policy, layouts, and profile validation.
- Read-only local mounts, Reasoner `hf://` resolution, cache, secrets,
  readiness, verification, and common failures.
- Explicit rejection of obsolete Generator and historical Transfer variables.

### `api-reference.md`

- Endpoint matrix separated by runtime mode.
- Compact management-endpoint index linking to operations.
- Common Generator `/v1/infer` top-level request fields and strict JSON types.
- Common Generator response envelope.
- Links to canonical Generation, Action, Transfer, and Reasoning contracts.
- Live OpenAPI inspection.
- Model discovery and live `/openapi.json` usage.

This is the canonical location for field defaults, ranges, and constraints.
Workflow pages should link here instead of repeating full field tables.

### `generation.md`

- Common setup and output decoder.
- Text-to-image with one-frame selection and JPEG decoding.
- Text-to-video.
- Task-specific full-step and four-step Super T2I/I2V variant contracts.
- Image-to-video with local base64/data URL and public URL.
- Video-to-video and conditioning-frame controls.
- Optional prompt upsampling for T2I/T2V/I2V, including its mode boundary and
  original-prompt fallback behavior.
- Reproducibility, frame cadence, resolution, FPS, quality, and input/output
  media guidance.
- Capability-specific failure cases.

### `reasoning.md`

- Reasoner profile and served-model discovery.
- Chat Completions for text, image, and video.
- Responses API and its storage limitations.
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
- Policy input and video/action or specialist action-only result.
- Inverse dynamics input and video/action result.
- Nano-DROID strict observation, profile-owned fields, and `[32,8]` output.
- Shape, chunk-size, frame-count, action-space, and mode-specific validation.
- One complete runnable case for each general mode; Nano-DROID asset remains
  approval-gated.

### `transfer.md`

- Transfer concepts and supported control types.
- Precomputed edge, blur, depth, segmentation, and WSM controls.
- Server-derived edge and blur controls.
- Single-control and supported multi-control behavior.
- Transfer-specific VRAM admission and unsafe diagnostic override.
- Control video/media constraints and chunking fields.
- One compact example per distinct request shape, not per asset.

### `operations.md`

- Environment-variable reference not already explained in deployment.
- Health, readiness, model, version, metadata, manifest, license, and metrics.
- Logging and multi-GPU diagnostics.
- Guardrails, independent guardrail residency controls, and the consequences of
  disabling them.
- Specialist profile, action-only response, Transfer admission, and BYOC
  diagnostics.
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
- Until that release artifact is supplied, keep the public page explicitly TBD
  and do not infer or copy a component inventory.
- Keep the product EULA/license separate: link to repository/NGC terms and the
  running NIM's `/v1/license` endpoint from `README.md` and `operations.md`.

### `examples/`

Use plain Python, not notebooks. Keep request dictionaries directly editable and
avoid a large CLI abstraction.

Planned scripts:

- `common.py`
- `t2v.py`
- `i2v.py`
- `t2i_4step.py`
- `i2v_4step.py`
- `v2v.py`
- `reasoner.py`
- `reasoner_responses.py`
- `action.py`
- `transfer.py`

Each public guide also embeds one short curl or Python example so it remains
useful without opening a second file.

## Structural redistribution record

The 2026-07-27 split changes page ownership without dropping coverage. This
table supersedes earlier planning references that assigned all deployment and
API material to two large pages.

| Previous section | Canonical destination |
| --- | --- |
| `deployment.md` — Prerequisites | `prerequisites.md` |
| `deployment.md` — NGC key, image, cache, Docker launch/flags, readiness, selectors, shutdown | `deployment.md` |
| `deployment.md` — Configuration reference | `configuration.md` |
| `deployment.md` — Prompt upsampling configuration | `configuration.md`; workflow in `generation.md`, diagnostics in `operations.md` |
| `deployment.md` — Hardware/profile and offload compatibility | Selection mechanics in `deployment.md`; released rows in `support-matrix.md` |
| `deployment.md` — Bring your own checkpoint | `bring-your-own-checkpoint.md` |
| `deployment.md` — Kubernetes and Helm | `helm.md`; failures in `operations.md` |
| `api-reference.md` — Runtime and endpoint model | Compact routing table in `api-reference.md`; management semantics in `operations.md` |
| `api-reference.md` — Common Generator fields and strict types | `api-reference.md` |
| `api-reference.md` — Frame cadence, resolution keys, media representations | `generation.md`; released codecs in `support-matrix.md` |
| `api-reference.md` — Action fields and predicted-action response | `action.md` |
| `api-reference.md` — Transfer fields and defaults | `transfer.md` |
| `api-reference.md` — Reasoner APIs, sampling, and Responses | `reasoning.md` |
| `api-reference.md` — Generic errors | `operations.md` |
| `api-reference.md` — OpenAPI inspection | Compact command in `api-reference.md`; operational capture in `operations.md` |

The split also adds `release-notes.md` as the sole destination for published
version history, image tags, compatibility changes, known limitations, and
upgrade guidance. Its initial content remains release-dependent.

Historical Transfer BYOC variables (`NIM_EDGE_CHECKPOINT`,
`NIM_VIS_CHECKPOINT`, `NIM_DEPTH_CHECKPOINT`, and `NIM_SEG_CHECKPOINT`) remain
explicitly excluded because the current Certified NIM source does not expose
them. The current shared `NIM_MODEL_PATH` is documented separately for
Generator and Reasoner.

## Page-level authoring contracts

These contracts define what each public artifact owns, which sources establish
its content, and what it must not absorb from neighboring pages. They are the
handoff from research to drafting.

| Artifact | Canonical ownership | Primary current evidence | Release/TBD inputs | Non-goals |
| --- | --- | --- | --- | --- |
| `README.md` | Product scope, selected-runtime mental model, capability/endpoint index, minimum launch, first Generator/Reasoner requests, navigation | Runtime dispatch, profile generation/selection, metadata, and local cookbook examples | Final image/tag, product/model-card/release URLs, approved public naming | Exhaustive fields, full environment reference, profile internals, long troubleshooting |
| `release-notes.md` | Released versions, image tags, compatibility changes, known limitations, and upgrade notes | Approved published release inventory | Entire initial entry remains TBD until supplied | Inferred versioning, development-branch changelog |
| `prerequisites.md` | Host hardware/software, memory/storage/shared-memory, NGC access, and verification | Current source profile/hardware requirements, released support statement, previous prerequisite organization | CPU architecture, general RAM, disk/shared-memory, driver, Docker, and toolkit floors | Profile selector details, Docker launch tutorial |
| `deployment.md` | NGC key and login, cache/permissions, Docker flags, ports, selectors, readiness, and shutdown | Runtime/profile-selection code, current Makefile launch contract, and previous quickstart | Final image/tag and release URLs | Full environment tables, support matrix, Helm, BYOC contract |
| `configuration.md` | Shared, selection, Generator, Reasoner, and prompt-upsampling environment variables | `environment.py`, prompt-upsampling code/tests, framework contract | Released defaults and external endpoint compatibility | Complete launch commands, runtime troubleshooting |
| `support-matrix.md` | Model, precision, GPU, VRAM, profile, offload, and media/codec compatibility | Current profile generation/hardware policy; released manifest/test inventory | Released-row approval, tested SKU boundary, and media/codec matrix | Launch tutorial, profile IDs, or claims that every development row ships |
| `helm.md` | Cluster prerequisites, chart discovery, secrets, values, GPU resources, storage, probes, rollout, and verification | Released chart contract; previous Helm organization | Chart identity/version/schema and monitoring integration | Copied complete values catalogue, Docker deployment |
| `bring-your-own-checkpoint.md` | Generator and Reasoner checkpoint sources, layouts, local mounts/Hugging Face resolution, profile cross-check, cache, launch, verification, and failures | `environment.py`, `reasoner_model_source.py`, startup code, tests, previous BYOC page | Published formats and runtime boundary | Historical checkpoint variables, unsupported repository protocols |
| `api-reference.md` | Runtime routing, common Generator fields, strict JSON typing, common response, task links, and live OpenAPI | Generator request model/tests and routing code | Generator and Reasoner live OpenAPI | Detailed Action/Transfer/Reasoner tables, management semantics, generic errors |
| `generation.md` | T2I/T2V/I2V/V2V workflows, frame/resolution/media rules, conditioning, optional prompt upsampling, JPEG/MP4 decoding, reproducibility | `data_models/generation.py`, `data_models/responses.py`, `generator_mapping.py`, `prompt_upsampling.py`, tests, and local cookbook examples | Published capability set, live media/URL validation, prompt-upsampling smoke test, output playback | Action/transfer contracts, operator configuration tables |
| `reasoning.md` | Chat Completions, Responses, media ordering, task prompts, structured outputs, Reasoner-specific sampling/media guidance | `reasoner_inference.py`, Reasoner environment/startup, tests, local cookbook examples, and current prompt guide | Text-only/public-URL/media-format checks, Responses state features, approved reasoning-trace wording | Generic vLLM tuning reference, unsupported legacy `video_frames`/`mm_processor_kwargs`, hidden chain-of-thought promises |
| `action.md` | Complete action contract plus forward dynamics, policy, inverse dynamics, response, and validation | `data_models/actions.py`, runtime/tests, and the local cookbook example | Published-image/profile smoke tests and released domain/model boundary | General generation tutorial, framework/vLLM-Omni syntax |
| `transfer.md` | Complete transfer contract, control taxonomy, defaults, precomputed/derived forms, combinations, and validation | `data_models/transfer.py`, runtime/tests, and the local cookbook example | Published-image/profile smoke tests, exact supported combination matrix | Duplicate example for every asset, vLLM-Omni multipart syntax |
| `operations.md` | Health/readiness, management endpoints, generic errors, metrics/logs, guardrails, diagnostics, and troubleshooting | Runtime interface/environment/profile/guardrail code, tests, and previous operations docs | Live metrics, log/error samples, chart probes, release limitations | Basic launch tutorial, duplicate task-schema tables, secret values |
| `acknowledgements.md` | Third-party components and notices for the exact released image | Approved release/build acknowledgement inventory only | Entire content remains TBD until that artifact is supplied and approved | Product EULA copy, inferred dependency inventory, historical Generator notice reuse |
| `examples/common.py` | Strict local-media-to-data-URL conversion, image/video decoding, and compact JSON-prompt serialization | Current cookbook helper, reviewed audiovisual assets, and runtime media/prompt contracts | Final supported media types | Request dispatch, CLI framework, downloads, credentials |
| Task example scripts | One editable request with the API call and primary response handling visible in the same file | Current API models, runtime/tests, and local cookbook workflows | Live release smoke results and stable asset locations | Exhaustive parameter combinations, multi-level runner helpers, notebook-only execution |

### Canonical fact ownership

Use this table during drafting and review to resolve tempting duplication:

| Fact class | Canonical owner | Allowed repetition elsewhere |
| --- | --- | --- |
| Product identity and guide navigation | `README.md` | One-sentence context and links |
| Release versions, compatibility changes, and limitations | `release-notes.md` | Release-status link/summary |
| Host hardware/software and verification | `prerequisites.md` | Minimal pre-launch reminder |
| Image, NGC login, cache, Docker launch, selectors, and readiness | `deployment.md` | Minimal quickstart subset in `README.md` |
| Environment variables and prompt-upsampling launch configuration | `configuration.md` | Workflow-specific subset with link |
| Semi-final source-profile and released hardware/offload/media compatibility | `support-matrix.md` | One-row selection context; no copied matrix |
| Kubernetes/Helm deployment | `helm.md` | Troubleshooting symptoms in `operations.md` |
| Generator and Reasoner BYOC setup and contract | `bring-your-own-checkpoint.md` | Configuration row and operations symptoms |
| Runtime routing and common Generator envelope | `api-reference.md` | Task-specific request subsets |
| Frame/resolution/media rules | `generation.md` | Common field links from API reference |
| Detailed Action, Transfer, and Reasoner contracts | Corresponding task guide | Endpoint links and runnable subsets |
| Workflow order, task intent, representative payload, artifact handling | Corresponding task guide | One minimal first request in `README.md` |
| Reusable complete request/decoder implementation | `examples/` | Matching short snippets in guides |
| Health, management endpoints, generic errors, metrics, logs, guardrails, and diagnostics | `operations.md` | Readiness command and contextual links |
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
- Keep JSON bodies complete and valid. Use fenced blocks with language labels
  and no realistic-looking secret values. Put unresolved release values in
  prose or tables, never in runnable fenced blocks.
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
| `release-notes.md` | Artifact-dependent | Page purpose and initial TBD structure | Version/date, image/tag, compatibility changes, limitations, and upgrade notes |
| `prerequisites.md` | Ready with TBDs | Source GPU compute/count/VRAM floors, profile-specific system RAM, NGC/network needs, verification workflow | CPU architecture, general RAM, disk/shared-memory, driver, Docker, and toolkit values |
| `deployment.md` | Ready with TBDs | Docker/cache/credential mechanics, selectors, profile-selection concepts, readiness, shutdown | Final image/tag and release URLs |
| `configuration.md` | Source-ready with live gates | Current shared/Generator/Reasoner/prompt-upsampling variables and conflicts | Released defaults and external endpoint compatibility |
| `support-matrix.md` | Source-ready with release gates | Grouped semi-final model/precision/offload rows, GPU counts, compute gates, VRAM/system-memory/Transfer floors | Released-row approval, tested SKUs, and codecs |
| `helm.md` | Ready with TBDs | Required concepts, secret separation, storage/probe/rollout workflow | Chart identity/version/schema and monitoring values |
| `bring-your-own-checkpoint.md` | Source-ready with live gates | Current Generator boundary, layout, mount, cross-check, verification, failures | Released checkpoint formats and supported profile boundary |
| `api-reference.md` | Source-ready with live gates | Runtime routing, common Generator envelope, strict typing, response, task links | Live OpenAPI for both modes and generated route inventory |
| `generation.md` | Source-ready with live gates | T2I/T2V/I2V/V2V request construction, current fields/constraints, prompt-upsampling mode boundary and fallback, JPEG/MP4 base64 decoding, deterministic-seed guidance | Published-image capability and prompt-upsampling smoke results, remote-input behavior, playback/codec observations |
| `reasoning.md` | Ready with TBDs | Chat Completions, Responses create flow, data URLs, media ordering, sampling, structured outputs, task taxonomy | Text-only and public-URL smoke tests, exact media formats, Responses persistence features, approved reasoning-trace wording |
| `action.md` | Source-ready with live gates | Forward dynamics, policy, inverse dynamics, action shapes/domains, current representative payloads | Published-profile availability and one smoke result per documented mode/domain boundary |
| `transfer.md` | Source-ready with live gates | Precomputed and derived controls, current fields/validators, representative request shapes | Published-profile availability, exact supported combination matrix, smoke results |
| `operations.md` | Ready with TBDs | Health/readiness, management endpoints, stable error classes, current controls, prompt-upsampling fallback, troubleshooting | Live endpoint inventory, metrics scrape, log/failure samples, chart probes, release limitations |
| `acknowledgements.md` | Artifact-dependent | Page purpose and link placement only | Entire third-party inventory and notice text until the approved released-image artifact exists |
| `examples/` | Source-ready with live gates | Local request scripts, helpers, and cookbook assets/workflows | Final host-port convention, image-dependent model/profile selection, live response verification |

### Recommended authoring order

When the user authorizes drafting, use this dependency order:

1. Refresh the reviewed snapshots and reconcile any source drift.
2. Create only the agreed file scaffold and relative navigation targets.
3. Draft the compact `api-reference.md` for runtime routing and the common
   Generator envelope; assign task contracts to their corresponding guides.
4. Maintain local task scripts with complete editable request dictionaries and
   direct API calls; share only strict local-media encoding, response decoding,
   and compact JSON-prompt serialization in `examples/common.py`.
5. Draft `generation.md`, `reasoning.md`, `action.md`, and `transfer.md` as the
   canonical task contracts against current models and runnable scripts.
6. Draft `release-notes.md`, `prerequisites.md`, `configuration.md`,
   `support-matrix.md`, `helm.md`, `bring-your-own-checkpoint.md`, and the
   focused Docker `deployment.md`; keep release-owned values explicitly TBD.
7. Draft `operations.md` from current controls and historical coverage, marking
   live metrics, logs, probes, and release limitations as validation-gated.
8. Assemble `README.md` last so its guide index, capability matrix, and minimal
   examples summarize artifacts that actually exist.
9. Add acknowledgement content only when the approved artifact is supplied;
   otherwise retain the planned destination and explicit TBD.
10. Run the parity, link, syntax, secret, placeholder, and live-validation gates
    in the checklist below.

The scaffold restriction applied during the discovery-only phase. The scaffold
was created only after the user explicitly authorized public authoring.

### Source refresh procedure

Before the first public edit and again before publication:

1. Record `git rev-parse HEAD`, branch, and tracked worktree state for all four
   reviewed repositories.
2. Diff the current NIM commit against the snapshot recorded above, scoped to
   `cosmos3/serving_stack/data_models/`, `reasoner_inference.py`,
   `reasoner_model_source.py`, `environment.py`, Generator mapping/specialist
   contracts, profile selection/generation files, tests, and generated profile
   output. The deleted source guide/static schema are historical comparisons.
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
| The former static `api_spec.yaml` is deleted and never represented all dynamic routes | file history and current runtime routing | Generate/capture live OpenAPI separately for Generator, Reasoner, and relevant specialist profiles |
| Product/model parameter counts differ across sources | NIM source guide and public Cosmos README | Use approved public naming/size language; avoid counts until reconciled |
| Source examples default to port 18000 while public cookbook examples commonly use 8000 | Source task scripts, cookbook NIM pages | Standardize docs on `NIM_URL`, defaulting to `http://localhost:8000`; explain host-port remapping once |
| Generated profile export includes Generator BF16, FP8, and FP8 offload rows while the deleted `documentation.md` says the active Generator grid is BF16-only | current profile generator/YAML vs historical documentation | Use grouped current-source rows for semi-final planning, treat the deleted prose table as stale, and reconcile the final release manifest |
| Developer README says text generation lives in a separate Reasoner NIM while the current manifest and dispatcher include Reasoner profiles in the same image | `README.md:3` vs `profiles.json`, `inference.py`, `documentation.md:25` | Describe one-profile-at-a-time unified behavior only after confirming the published image; retain this as a release identity conflict |
| Local NIM identities and versions disagree | `nim-config.yaml`, Makefile, build-context `VERSION` | Never derive the public image/tag from local build defaults; obtain it from the published NIM/NIMCraft release |
| Hardware/profile tables are generated and release-sensitive | profile generator/export vs older documentation | Publish grouped semi-final source requirements with status labeling; reconcile exact rows against the released manifest |
| Historical prerequisites and the current source guide disagree on CPU/RAM/disk values | previous `prerequisites.rst` vs current profile gates | Publish only current source compute/profile floors; keep CPU, general RAM, disk, shared-memory, and software versions TBD until the released support statement resolves them |
| Helm URL and some launch values are placeholders in `documentation.md` | outstanding TBDs | Confirm released chart and image details before publication |
| Reasoner routes are dynamic and not fully represented by Generator OpenAPI | Reasoner NIMlib interface | Validate live OpenAPI separately under a Reasoner profile |
| Historical Reasoner docs claim `video_frames` and `mm_processor_kwargs`, but current runtime evidence does not establish them | old Reasoner API page vs current Reasoner options and tests | Keep both out of supported guidance until a published-image smoke test proves the exact request shapes |
| Historical Reasoner media support matrix belongs to release 1.7.0 | old VLM support matrix | Rebuild image/video format and codec claims from the released image; leave them TBD meanwhile |
| Historical and current prompt guides show explicit `<think>` traces | Reasoner API page and current prompt guide | Preserve task and output-format guidance, but obtain approved wording before documenting reasoning traces or chain-of-thought behavior |
| Other backends expose broader transports, fields, domains, and modalities than the Certified NIM request models | current NIM models and runtime tests | Validate every local request against the Certified NIM contract; never infer backend parity |
| Source-guide prompt-upsampling example names a provider-specific native endpoint, while the implementation normalizes and calls an OpenAI-compatible Chat Completions endpoint | `documentation.md` vs `prompt_upsampling.py` | Document the generic OpenAI-compatible contract; do not claim native provider compatibility without a published-image integration test |
| Existing inbound cookbook/root documentation duplicates separate legacy NIM launches and limitations | root README, shared Cosmos3 setup, audiovisual and Reasoner READMEs | Make the new guide canonical and, if scope permits, replace duplicated details with current summaries and links; otherwise report the remaining contradictions explicitly |
| Current source implements wider BYOC than the historical Generator guide | `NIM_MODEL_PATH`, Reasoner source resolver, startup, tests | Document separate Generator local and Reasoner local/`hf://` contracts; keep released checkpoint inventory TBD |
| Broader Cosmos3 supports audio/image generation beyond the currently established Certified NIM surface | model docs vs current NIM request/response contract | Document only capabilities proven by the Certified NIM runtime and release profile |

## Release-dependent TBD ledger

Do not close these from memory or legacy docs:

- Final NGC image repository and tag for the unified Certified NIM.
- Public product name and whether the release uses one unified image externally.
- Confirm which rows from the 122-row development profile generation are
  present in the published manifest, including specialist variants and
  guardrail-residency tags.
- Tested GPU/profile/support matrix and minimum driver/toolkit versions.
- Final Helm chart name, version, and values.
- Final NGC Catalog/model-card URL, authoritative release-notes URL, and release
  version used by these cookbooks.
- Confirm the published-image Generator and Reasoner BYOC boundary, accepted
  checkpoint layouts, Hugging Face source policy, and supported revisions.
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

## Validation performed during discovery and implementation

- Parsed the relevant implementation and local cookbook Python files with the
  standard library AST parser; all parsed successfully.
- Historical validation loaded the 39-row snapshot. The refresh through
  2026-08-04 ran `make_profiles.py` to a temporary untracked output and
  confirmed 122 current development rows (115 Generator, 7 Reasoner), all seven
  model variants, every `n_gpus = nim_dp * nim_gp * nim_up * nim_tp` invariant,
  three Nano Reasoner rows with the DFlash draft artifact, and ten Super BF16
  offload rows with the 150-GiB system-memory floor. Release support remains
  unvalidated.
- The hardware publication refresh at `5862d3a5` regenerated the same 122-row
  inventory and matched the 13 grouped Generator rows and seven Reasoner rows
  in `support-matrix.md` against source compute gates, GPU counts, per-device
  VRAM, effective system-memory floors, and Transfer thresholds. All five
  `test_profile_catalog.py` tests passed in an isolated `uv` environment.
- The same refresh asserted current source constants and controls for explicit
  Generator modes/field names, shared `NIM_MODEL_PATH`, model variants,
  Reasoner pruning, DFlash, reasoning/tool contracts and strict extension
  types, Transfer override, Nano-DROID `[32,8]`, and action-only responses. It
  compiled all then-current local examples and validated JSON fences, SPDX
  headers, local links/anchors, Markdown table structure, and whitespace.
- The `90a77482` refresh regenerated 122 rows, confirmed `model_variant` on
  every profile, preserved all grouped profile floors, passed 52 focused
  profile/benchmark/BYOC/Reasoner/Transfer source tests, statically verified the
  unified-memory reserve/residency policy, and compiled all eleven current local
  examples.
- Confirmed every Reasoner row omits the performance-profile axis, fixes
  `nim_dp=nim_gp=nim_up=1`, and satisfies `n_gpus=nim_tp`.
- Confirmed the checked-in tests cover Generator validation, action modes,
  transfer cases, Reasoner media handling, Responses routes, route disabling,
  normalization, and error behavior.
- Re-audited the authoring-ready contract tables against current field
  declarations, validators, runtime defaults, response models, and local
  cookbook requests.
- Audited all 21 numbered sections (including section 12.1) of the current
  first-party `documentation.md` and assigned each to a planned page,
  correction, release gate, or explicit TBD.
- Traced prompt upsampling through startup validation, request dispatch,
  external request construction, and tests. Confirmed it is optional,
  Generator-only, applies to T2I/T2V/I2V, uses a separate secret, and falls
  back to the original prompt on request-time failures.
- Confirmed that local cookbook examples reuse existing public-repository
  assets and preserve their established provenance.
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
- Reviewed surrounding cookbook material for terminology, task taxonomy,
  presentation conventions, and approved asset reuse without treating other
  backend examples as API authorities.
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
- Ran Python compilation, Ruff lint/format checks, embedded-JSON parsing,
  relative Markdown link checks, and `git diff --check` for the changed
  surfaces.
- After the structural split, rechecked every old deployment/API heading
  against the redistribution record. Recursively validated all local Markdown
  targets and anchors, all Markdown SPDX headers, nine JSON fences, five Python
  fences, 46 shell fences, two YAML fences, all cookbook example modules, and
  `git diff --check`.
- Confirmed `api-reference.md` no longer contains complete Action, Transfer, or
  Reasoner tables; their task pages now own those contracts. Confirmed
  prerequisites, configuration, support, Helm, and BYOC details no longer
  remain as duplicate sections in `deployment.md`.

Do not report the complete project suite or live runtime/API validation as
passing. Those remain pending until the internal dependency environment and a
published release image are available.

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
- Retain the historical deleted `documentation.md` coverage matrix as a parity
  floor; current behavior must come from implementation, tests, generated
  profiles, and live release OpenAPI.
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
- Search local cookbook Generator examples for vLLM-Omni-only transport or fields:
  `input_reference`, `extra_params`, `control_path`, `view_point`, request-level
  `guardrails`, async video polling, `text2image`, and sound-generation fields.
  Any occurrence must be explanatory comparison text, not a NIM request.
- Search local cookbook Reasoner examples for local `file://` media and
  `mm_processor_kwargs`; neither is part of supported guidance without the
  release validation recorded in the TBD ledger.
- Keep `SOURCES.md` as the tracked authoring provenance and coverage record;
  do not link it as an end-user runtime guide.
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
