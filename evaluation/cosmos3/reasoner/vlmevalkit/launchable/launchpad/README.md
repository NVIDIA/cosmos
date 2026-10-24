# Private single-GPU LaunchPad setup

This Linux CLI reuses the parent launchable's pinned serving image/model command,
controller, manifest, internal dataset preparation and native VQA scorer. It does
not create or terminate an instance, expose a host model port, start a notebook,
or modify existing Jupyter/Traefik services. Use your provider's authorized SSH
connection. No instance locator or credentials belong in this source tree.

## Requirements

- Linux x86-64 host, NVIDIA driver/container runtime, Docker with Compose v2+,
  Python 3.9+, and one free GPU with sufficient memory for the pinned Nano model.
- This VLMEvalKit source tree, plus an existing `linux/amd64` evaluator image
  built from the parent `launchable/Dockerfile`. Setup does not rebuild images.
  Dependency input hashes must match the source tree; mismatches fail explicitly.
- For private internal data, authorized `team-cosmos` PDX S3 access. Public/HF
  model access is anonymous unless the actual model requires `HF_TOKEN`.

If a matching evaluator image does not already exist, build it explicitly from
VLMEvalKit (not during another launchable build):

```bash
docker build -f launchable/Dockerfile \
  --build-arg COSMOS_COMMIT="$(git rev-parse HEAD)" \
  -t cosmos3-vantage-eval:local .
```

The image supplies dependencies; the current source tree is bind-mounted
read-only into each evaluator/preparation container. Provenance records both the
base Git commit and a SHA-256 source-tree digest, alongside the exact dependency
image ID and model image/command. The image's baked commit is not passed off as
runtime source provenance. For an exported tree without `.git`, supply
`--commit FULL_BASE_REVISION`; the separate digest still identifies local changes.

The source root must be source-only: setup refuses `.env*`, runtime `state` or
`credentials` directories, `.aws`, `.ssh`, `.cache`, `.venv`, `.git`, key-like
files, and symlinks escaping the source tree. These files are never hashed or
mounted into inference. If an old launchable wrote `.env` or `state` under its
source, use a new clean export (for example a Git archive plus your reviewed
source changes), not the old directory. Do not delete or mutate retained secrets
or state to make an old source directory pass.

## Configure and start

Run from the VLMEvalKit root. Keep state outside the source tree, use a unique
project, and choose exactly one free GPU index or UUID:

```bash
LP=(python3 launchable/launchpad/launchpad.py \
  --state-dir "$HOME/cosmos-launchpad-state" \
  --project cosmos-launchpad --gpu 0 --image cosmos3-vantage-eval:local)
"${LP[@]}" up
"${LP[@]}" status
"${LP[@]}" check
```

`up` preserves weights, datasets and outputs, starts only the project's pinned
endpoint and waits for the existing controller's model readiness check. Running
it again resumes the same project without rebuilding. Neither the model API nor
an evaluator/notebook port is published. GPU selection overrides the parent's
single-device reservation without changing its serving flags. The private
inference network has Docker `internal: true`; the serving container also has a
download network for first-time model retrieval.

## Internal data: choose the route that actually has PDX access

Some LaunchPad hosts cannot reach `pdx.s8k.io:443`, even when their SSH and
Hugging Face access work. A connection timeout is not evidence of bad credentials.
Use the workstation-staging route below for such hosts. Direct remote preparation
requires a working network route to PDX; provisioning a profile does not add one.

### Direct preparation on a PDX-reachable host

On the local workstation with the working AWS CLI profile, run:

```bash
python3 launchable/launchpad/provision_profile.py \
  --host YOUR_SSH_HOST --user YOUR_SSH_USER \
  --remote-dir /absolute/state/path/credentials/team-cosmos
```

`--remote-dir` must be the chosen state directory followed by
`/credentials/team-cosmos`. If your SSH gateway needs an authenticated master
session, establish it normally first, then supply the same `--control-path`.
The script captures only `aws configure export-credentials --profile team-cosmos`
in private process memory and sends it over SSH stdin. It never copies `~/.aws`,
uses the workspace `.env`, prints values, or puts them in command arguments.
The remote directory is mode 0700; its two isolated files are mode 0600.

```bash
"${LP[@]}" probe
"${LP[@]}" prepare
```

`probe` performs one bounded VQA prefix listing and a one-byte ranged object
read. `prepare` calls the original internal VQA dataset constructor using that
profile. Preparation can be repeated against its existing native local dataset
layout. No S3 upload operation is invoked by either command. Interruptions during
a first download may require inspection of the native dataset's partial local
files; this wrapper does not delete data or claim atomic upstream downloads.

**Credential isolation:** only the one-shot preparation/probe service mounts the
isolated credentials. Inference/scoring have no credential mount, use `/dev/null`
for both AWS files, disable instance metadata credentials, and have no outbound
network. This matters because the upstream video cache attempts S3 cache uploads
on misses. Private inference can generate local cache entries but cannot
access/write S3. Do not mount a home directory or add AWS credentials to the
inference service; a read-only credentials mount is not a read-only S3 policy.

### Workstation preparation and data-only transfer

On a PDX-reachable workstation with AWS CLI, Docker, rsync and the matching
evaluator image, run from a source-only VLMEvalKit tree:

```bash
python3 launchable/launchpad/stage_internal.py \
  --local-state-dir "$HOME/private-cosmos-vqa-staging" \
  --image cosmos3-vantage-eval:local \
  --host YOUR_SSH_HOST --user YOUR_SSH_USER \
  --remote-state-dir /absolute/launchpad/state/path
```

The remote state path must be the same one passed to `launchpad.py`. Supply
`--control-path` when reusing an authenticated SSH gateway session. This helper
reads the official adapter's VQA S3 dispatch, uses native `aws s3 sync` with only
the authorized `team-cosmos` profile to download/resume its unchanged directory
layout, then invokes the original adapter's annotation-to-TSV method and original
controller preparation in a credential-free, network-disabled evaluator
container. It checks all referenced native video paths and retains its private
native preparation log under the private local state directory.

Only `data/internal/LMUData/datasets/VANTAGE_VQA/` is transferred by rsync. No
credentials, home directory, cache credential files, results or public data are
transferred. Local state is mode0700; transferred directories/files are
mode0700/0600. The helper has no `--delete`: rerunning the same command resumes
downloads/transfers and preserves existing remote results. No proxy, public
data service, S3 upload or network bypass is installed.

After transfer, on LaunchPad run `"${LP[@]}" prepare` to exercise the original
controller against its native local dataset layout, then run the smoke below.
This is local-cache preparation, not a claim that LaunchPad downloaded from PDX.
The remote `probe` still requires PDX connectivity and is expected to time out on
a host without that route; the staged route does not require remote credentials.

## Five-question internal smoke and native score

After preparation and endpoint readiness:

```bash
"${LP[@]}" smoke --run-name internal-vqa-smoke-001 --score
```

The original controller selects five seeded VQA questions and performs native
inference. Independent acceptance rejects empty/API/video-load failures, requires
exactly five unique predictions with the expected native sample IDs, and checks
that scoring labels match the prepared native dataset. `--score` invokes its
native local scorer only when all five answers actually exist, and verifies the
native metrics account for all five samples. Omit `--score` for inference-only.

Outputs remain private under `state/results/RUN_NAME`: native predictions,
logs, configs, native scoring files when requested, `run.json`,
`launchpad-acceptance.json`, and `launchpad-provenance.json`. Labels and model
responses must not be publicly exported. A five-question smoke is not a full
benchmark score or published-reference parity. Existing output directories are
refused; choose a fresh run name instead of overwriting or merging results.

## Stop and resume safely

```bash
"${LP[@]}" stop
"${LP[@]}" status
"${LP[@]}" up
"${LP[@]}" smoke --run-name internal-vqa-smoke-002 --score
```

`stop` targets only this project's endpoint container. It does not remove
containers, state, cached weights, data, predictions, unrelated services or the
instance. An allocated LaunchPad instance can continue accruing provider costs
when its containers are stopped. Keep the same project/state/GPU arguments on
resume. `status` and `stop` use the retained config and do not require an image
rebuild or source-provenance check. There is deliberately no delete/purge command.
