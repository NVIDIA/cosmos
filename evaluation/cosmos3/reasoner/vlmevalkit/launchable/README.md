# Cosmos3 VANTAGE evaluation on Brev

A Jupyter front end for preparing data, checking an OpenAI-compatible video endpoint, running five real VQA questions, and exporting a full public VANTAGE submission. Notebook cells call `launchable.py`; dataset preparation, inference, and submission conversion stay in the selected upstream harness.

**Default: public data and the official public harness. Only `HF_TOKEN` is required as a launch secret.** No internal AWS, NGC, NVIDIA dataset-service, or API judge credentials are required for the eight VANTAGE tasks. A separately authenticated BYOM endpoint may require its own `COSMOS_API_KEY`.

Create a **read** token at [Hugging Face token settings](https://huggingface.co/settings/tokens) and supply it as Brev's secret `HF_TOKEN` launch parameter. Full tracking also requires accepting the [PhysicalAI-SmartSpaces access terms](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces). No internal NVIDIA key is needed for the public workflow.

This directory defines a launchable; it does not provision a VM or publish a Brev launchable. GPU serving, end-to-end GPU evaluation, runtime, and leaderboard parity are **not yet GPU-verified**. First successful predictions are not a benchmark score. Do not assume a sub-hour run or reference-score reproduction.

## Create the Brev launchable

Use these deployment fields in Brev:

| Field | Value |
| --- | --- |
| Environment | **VM mode**, Linux x86-64, Docker Engine with Compose v2 and NVIDIA Container Toolkit for local serving |
| Deployment source | This Cosmos repository checkout: `https://github.com/NVIDIA/cosmos.git`, on the branch/commit **containing this launchable**; record the exact deployment commit |
| Setup script command | `bash evaluation/cosmos3/reasoner/vlmevalkit/launchable/setup.sh` from the Cosmos checkout root |
| Managed Jupyter | **Disabled**; Compose starts Jupyter itself |
| Secret environment variable | `HF_TOKEN`, with access to the model/data and accepted applicable licenses |
| Application access | Authenticated/secure Brev link to port **8888**; do not publish the model endpoint |
| Conservative starting hardware | One RTX PRO 6000 with **96 GB VRAM**, at least **256 GB disk**; sizing is an unvalidated starting point, not a compatibility guarantee |

Do not select an upstream commit that lacks these files. Local edits are not automatically available to Brev: the deployment branch/commit must include them and be accessible to the VM. No branch is published by setup.

If Brev presents a setup-script text box, paste the setup command above, not just the filename. It must execute from the cloned Cosmos repository root.

Local serving defaults to `vllm/vllm-openai:cosmos3@sha256:db0bb920b0b54e82ea96a98659bbd21921f87d0dcfc86feffdafa2db3f08be55` (overridable with `VLLM_IMAGE`) and `nvidia/Cosmos3-Nano` at revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`. Both evaluator and serving containers target `linux/amd64`. Confirm the serving image is available to your VM and compatible with its GPU/driver. Disk must accommodate model weights, image layers, downloaded data, extracted media, caches, and results. Public data is roughly 42 GB, with roughly 16 GB more for SOT; headroom and actual resource needs depend on workload. Full SOT preparation also requires access to the SmartSpaces source and acceptance of its license; an HF token alone does not grant that access.

If running setup manually on the VM after injecting `HF_TOKEN` into its environment:

```bash
cd /path/to/cosmos/evaluation/cosmos3/reasoner/vlmevalkit/launchable
bash setup.sh
```

Setup builds the evaluator, starts Jupyter, and starts the local GPU endpoint unless `COSMOS_API_BASE` was supplied at deployment. It persists a generated Jupyter token in a permission-restricted `.env`; it does not write HF/model credentials there. Use the same environment variables when recreating services. Never paste secrets into the notebook, source files, shell history, or screenshots.

### Open Jupyter securely

On the VM, from the launchable directory:

```bash
docker compose exec vantage-eval jupyter server list
```

Treat the displayed login token as a secret. Open Brev's secure port-8888 link and authenticate with that token. The host binding is `127.0.0.1:8888`, not a public listener. If your access method requires a local tunnel, forward port 8888 through Brev/SSH rather than rebinding it to `0.0.0.0`. Open `/opt/vlmevalkit/launchable/Evaluate_Cosmos3.ipynb` in Jupyter. Jupyter's file root is `/`, allowing authenticated downloads from `/results`.

The evaluator has **no Docker socket**. Container lifecycle commands belong in the VM terminal, not notebook cells. The local model endpoint has no host port; only the Compose network can reach it.

## Notebook workflow

1. **Configure:** public source by default, a unique timestamped output directory, and inherited endpoint/model settings. A hidden `getpass` prompt is used only if `HF_TOKEN` was not inherited. Credentials remain in process memory; never save them in cells.
2. **Prepare VQA:** use the selected harness's official preparation. Downloads/preparation may be substantial even for a smoke run.
3. **Check endpoint:** wait for `/v1/models` and verify the configured served model.
4. **Smoke:** infer exactly five seeded `VANTAGE_VQA` questions. This tests actual data/media/model integration; it is not synthetic and is not a reference score.
5. **Inspect results:** the helper checks consumer-visible predictions, not just the native harness's exit status. The native harness can catch a dataset failure and still exit zero.
6. **Full public evaluation:** explicitly set `RUN_FULL = True` before running the full section. Its default is false, so **Run All does not start the full eight-task run**. It prepares all eight tasks and runs inference in a separate fresh directory.
7. **Export/download:** use the official public packager, then download the generated submission artifact via Jupyter. The notebook does not upload anything. Submit at [vantage-bench.org/submit](https://vantage-bench.org/submit); official scoring can take up to 24 hours.

A full submission covers all eight tasks: `VANTAGE_VQA`, `VANTAGE_EventVerification`, `VANTAGE_2DGrounding`, `VANTAGE_2DPointing`, `VANTAGE_Astro2D`, `VANTAGE_Temporal`, `VANTAGE_DVC`, and `VANTAGE_SOT`. Official pillars require complete task sets: semantic = VQA + EventVerification; spatial = Grounding + Pointing + Astro2D; temporal = Temporal + DVC; tracking = SOT. A five-question smoke or incomplete pillar is not a valid full submission. Export validation rejects incomplete inputs rather than inventing a submission schema.

### CLI equivalents

Run inside the evaluator (`docker compose exec vantage-eval bash` from the VM terminal). Work directories must be new; do not reuse smoke output for a full run.

```bash
cd /opt/vlmevalkit/launchable
python launchable.py prepare --source public --benchmarks VANTAGE_VQA
python launchable.py check --wait 1800
python launchable.py run --source public --benchmarks VANTAGE_VQA --mode infer --smoke --work-dir /results/public-smoke-unique
python launchable.py results --source public --work-dir /results/public-smoke-unique

python launchable.py prepare --source public --benchmarks all
python launchable.py run --source public --benchmarks all --mode infer --work-dir /results/public-full-unique
python launchable.py results --source public --work-dir /results/public-full-unique
python launchable.py export --source public --work-dir /results/public-full-unique
```

The normal public workflow is inference plus official submission export; leaderboard scoring is separate. Public `--mode all` maps to inference because public ground truth is held out; provenance records the requested and effective modes. Internal `--mode all` uses the NVIDIA harness's local evaluation path. `--dry-run` displays composed commands/configuration without running inference; it does not validate GPU/media compatibility. Predictions may be native TSV files, not spreadsheets. Treat predictions/media/logs as dataset-derived material and observe the source licenses when sharing.

## Reproducibility and source boundaries

| Component | Public mode | Internal mode |
| --- | --- | --- |
| Harness | [Clemson-Capstone/VANTAGE-Bench](https://github.com/Clemson-Capstone/VANTAGE-Bench), commit `ec1ebdebb955332a9d7fdd9e98f39627b2822ff5` | NVIDIA harness in this Cosmos checkout |
| Python/import root | `/opt/vantage-public/.venv/bin/python`, `/opt/vantage-public` | Base Python, `/opt/vlmevalkit` |
| Dataset | [nvidia/PhysicalAI-VANTAGE-Bench](https://huggingface.co/datasets/nvidia/PhysicalAI-VANTAGE-Bench), revision `ad5297f645ba90830478a4c6a72a3a7ab077a2f7`, plus official SOT source preparation | Official internal S3/cache preparation |
| Data root | `/data/public/LMUData` | `/data/internal/LMUData` |

The checkouts, Python environments, and data roots are intentionally separate: an internal TSV/cache must never be silently reused as public data and produce misleading scores. Both share only the persistent HF cache at `/cache/huggingface`. Public configs preserve the committed NVIDIA FPS/frame limits, pixel limits, and model sampling parameters where the official public classes accept them. Dataset classes are translated to the public implementations, Astro2D uses the public `Astro2D` dataset name, and native prompts, dataset construction, ID emission, and server-side scoring differ from the internal harness. Review the generated configs and run provenance for actual settings. No cross-harness score parity is claimed.

The image applies the tracked `public-harness.patch` to the pinned public source. It preserves TLS certificate verification and removes the `model_config.txt` diagnostic that exposed API credentials. It also aligns video serialization with this NVIDIA fork: lossless H.264 (`crf=0`), scalar FPS, no server-side resampling, and removal of the unsupported `do_resize` vLLM argument. Public dataset prompts and submission IDs remain upstream's; retain the patch with the source pin for reproduction.

On the host, `LAUNCHABLE_STATE_DIR` defaults to `state/` beside `setup.sh`. It contains `cache/`, `data/public/LMUData`, `data/internal/LMUData`, `results/`, and an optional `credentials/aws/`. `/results` survives evaluator restarts. Data/source pins do not freeze an externally served model; record the endpoint's model revision as well as the deployment commit and generated configuration.

## Bring your own model (BYOM)

At **initial deployment**, supply `COSMOS_API_BASE` as a full URL ending in `/v1/chat/completions`, `COSMOS_MODEL` as the exact served model ID, and, only if required, `COSMOS_API_KEY` as a secret. Setup skips the local GPU profile when this URL is set. The endpoint must support the harness's OpenAI-compatible image/video requests and be reachable from inside the evaluator container; `localhost` inside the container is not the VM host. Keep the deployment's network authentication in place.

The notebook preserves inherited endpoint/model settings. To change only the evaluation target for a session, use the optional BYOM cell **before** the health check. Prompt for any endpoint key rather than inserting it in notebook source. Do not print the environment or embed credentials in URLs. Changing notebook environment does not stop a previously launched local GPU container; use the VM lifecycle commands separately if needed.

## Optional internal mode

Public mode needs none of this. Internal mode requires authorized access to NVIDIA's `team-cosmos` S3 profile. **Configure its AWS credential-directory mount at initial deployment**, not by copying credentials into notebook cells. Set `AWS_CREDENTIALS_DIR` to an existing absolute host directory containing AWS `config` and `credentials` files with the `team-cosmos` profile, or populate the default `state/credentials/aws` securely before setup. Short-lived credentials/SSO sessions must be valid for the duration of preparation. Never commit these files.

Compose mounts that directory read-only at `/credentials/aws` and sets `AWS_CONFIG_FILE=/credentials/aws/config` and `AWS_SHARED_CREDENTIALS_FILE=/credentials/aws/credentials`. An empty default mount grants no access. No `NVDATASET_TENANTID`, `NGC_API_KEY`, or judge API key is needed for these eight VANTAGE tasks. If a different mount was not configured initially, reconfigure it on the VM and recreate the evaluator using setup; changing a notebook variable cannot create a host mount.

Set notebook `SOURCE = "internal"` before preparation/smoke, or use:

```bash
python launchable.py prepare --source internal --benchmarks VANTAGE_VQA
python launchable.py run --source internal --benchmarks VANTAGE_VQA --mode infer --smoke --work-dir /results/internal-smoke-unique
python launchable.py results --source internal --work-dir /results/internal-smoke-unique
python launchable.py export --source internal --work-dir /results/internal-smoke-unique

# Full internal run with authorized ground truth: infer and score all eight.
python launchable.py prepare --source internal --benchmarks all
python launchable.py run --source internal --benchmarks all --mode all --work-dir /results/internal-full-unique
python launchable.py results --source internal --work-dir /results/internal-full-unique
```

The internal export is an evidence bundle, **not** the public leaderboard submission format. Keep public and internal run directories separate. TAR and other non-VANTAGE benchmark families are outside this launchable's scope; do not add their credentials or claim their coverage.

## Logs, stop, restart

Run from the launchable directory on the **VM host**:

```bash
# Status and bounded recent logs (may contain model/data output; share carefully).
docker compose --profile local ps
docker compose logs --tail 100 vantage-eval
docker compose --profile local logs --tail 100 cosmos-endpoint

# Stop containers without removing caches, results, or volumes.
docker compose --profile local stop

# Restart existing containers with the same configuration.
# Local serving:
docker compose --profile local start cosmos-endpoint vantage-eval
# BYOM, instead:
docker compose start vantage-eval
```

For changed deployment environment/build configuration, rerun `bash setup.sh` with the intended secrets/environment rather than expecting `start` to recreate containers. Do not delete the state directory or use volume-removing commands to recover from a failed run. Check the run's log, fix the prerequisite, then use a new work directory. Stopping containers does not stop VM billing; use Brev's VM controls separately when appropriate.

## Validation status

These instructions are **not evidence of a GPU-validated launch**. No GPU hardware/runtime, full dataset preparation, full-eight inference, submission portal acceptance, or reference-score parity is certified here. Consult the accompanying implementation handoff for the specific local checks actually exercised. A successful image build or CPU dry run alone does not establish those claims.

Local checks exercised:

- Authenticated JupyterLab rendered this notebook; a Brev-style proxy Host header succeeded with a token and was rejected without one.
- Public preparation downloaded the pinned 2DPointing task and produced 1,005 input rows with linked media through the official preparation script.
- The `linux/amd64` evaluation image built successfully; its isolated public Python environment loaded all prepared pointing rows and resolved unique official submission IDs.
- A real repository video decoded and serialized as MP4 with normalized FPS and processor kwargs. This did not call a model.
- Public eight-task and internal VQA-smoke config generation, pinned public CLI argument parsing, Compose validation, and five artifact-safety regression tests passed.
- On an existing native Linux CPU host, a resource-limited notebook kernel invoked `launchable.py` and generated all eight public dry-run commands successfully.

No GPU VM was provisioned, and no predictions, official scores, or portal acceptance are claimed by those checks.

Development-platform caveat: the notebook kernel stalled under local ARM64 Docker's `linux/amd64` emulation, including in a clean upstream Jupyter image with no evaluation code. The same Jupyter stack started a kernel and executed code on native Linux x86-64. Use native Linux x86-64 for container-kernel validation, rather than downgrading dependencies to mask an emulation-specific failure.
