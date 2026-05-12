set dotenv-load

default:
  just --list

package_name := 'cosmos3'
module_name := 'cosmos3'
short_name := 'cosmos3'

cuda_name := env("CUDA_NAME",
  if arch() == "aarch64" {
    "cu130"
  } else {
    "cu128"
  }
)
test_max_gpus := env("TEST_MAX_GPUS",
  if arch() == "aarch64" {
    "4"
  } else {
    "8"
  }
)
test_max_processes := env("TEST_MAX_PROCESSES", "8")

# Unset environment variables that interfere with uv virtual environment.
export LD_LIBRARY_PATH := ''
# Avoid CUDA out-of-memory errors.
export PYTORCH_ALLOC_CONF := 'expandable_segments:True'

# Install the Python version specified in .python-version
_python-install:
  uv python install

# Setup the repository
setup: _python-install

_uv-sync *args:
  uv sync --all-extras --group={{cuda_name}} {{args}}

# Install the repository
install *args: setup (_uv-sync '--reinstall' args)

# Run a command in the package environment
run *args: _uv-sync
  uv run --no-sync {{args}}

# Setup pre-commit
_pre-commit-setup: setup
  uv tool install "pre-commit>=4.5.0"
  pre-commit install -c ci/.pre-commit-config-base.yaml

# Run pre-commit
pre-commit *args: _pre-commit-setup
  pre-commit run -a -c ci/.pre-commit-config-base.yaml
  pre-commit run -a {{args}} || pre-commit run -a {{args}}

# Run pyrefly
pyrefly *args: _uv-sync
  # TODO: Enable
  # uv run --no-sync pyrefly check -c pyrefly-src.toml --output-format=min-text {{args}}

# Run linting and formatting
lint: pre-commit

python_env_tmp := '''
venv_dir="$(mktemp -d)"
trap 'rm -rf "$venv_dir"' EXIT
uv venv -q "$venv_dir"
source "$venv_dir/bin/activate"
'''

_test-install-uv-sync cuda_name:
  #!/usr/bin/env bash
  set -euo pipefail
  {{python_env_tmp}}
  uv sync -q --active --all-extras --group={{cuda_name}}

_test-install-uv-pip cuda_name:
  #!/usr/bin/env bash
  set -euo pipefail
  {{python_env_tmp}}
  uv pip install -q -r pyproject.toml --all-extras --group={{cuda_name}}

_test-install-advanced cuda_name torch_name='torch210':
  #!/usr/bin/env bash
  set -euo pipefail
  {{python_env_tmp}}
  uv pip install -q "torch==2.10.0" "torchvision" --torch-backend={{cuda_name}}
  uv pip install -q "natten==0.21.6.dev6+{{cuda_name}}.{{torch_name}}" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/natten
  if [ "$(uname -m)" == "x86_64" ]; then
    uv pip install -q "flash-attn-3-nv==1.0.3+{{cuda_name}}.{{torch_name}}" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/flash-attn-3-nv
    uv pip install -q "flash-attn==2.7.4.post1+{{cuda_name}}.{{torch_name}}" -f https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0/flash-attn
  fi

[parallel]
_test-install cuda_name: \
  (_test-install-uv-sync cuda_name) \
  (_test-install-uv-pip cuda_name) \
  (_test-install-advanced cuda_name)

# Test installation methods in 'docs/setup.md'
[parallel]
test-install: \
  (_test-install 'cu128') \
  (_test-install 'cu130')

pytest_args := '-vv --exitfirst --instafail --durations=5 --force-regen'

# Run a single test
test-single name *args: _uv-sync
  uv run --no-sync pytest --capture=no --manual {{pytest_args}} {{args}} {{name}}

# Run GPU=0 tests
_test-gpu-0 *args: _uv-sync
  uv run --no-sync pytest --num-gpus=0 -n logical --maxprocesses={{test_max_processes}} --levels=0 {{pytest_args}} {{args}}

# Run GPU>0 tests
_test-gpu *args: _uv-sync
  #!/usr/bin/env bash
  set -euxo pipefail
  export TEST_MAX_GPUS={{test_max_gpus}}
  export TEST_MAX_PROCESSES={{test_max_processes}}
  AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
  for num_gpus in 1 $TEST_MAX_GPUS; do
    if [ $num_gpus -gt $AVAILABLE_GPUS ]; then
      break
    fi
    args="{{pytest_args}} {{args}}"
    if [ $num_gpus -ne 1 ]; then
      # Only run coverage for single-GPU tests.
      # All multi-GPU tests should have a corresponding single-GPU smoke test, which has full coverage.
      args="$args --no-cov"
    fi
    uv run --no-sync pytest --num-gpus=$num_gpus -n logical --levels=0 $args
  done

# Run tests
test *args: pyrefly (_test-gpu-0 args) (_test-gpu args)

coverage_args := '--cov-append --cov-report= --cov=' + module_name

# Initialize coverage
_coverage-init:
  rm -rf outputs/coverage

# Run tests with coverage
test-coverage *args: _coverage-init (test coverage_args args) (run 'coverage' 'xml')

# List tests
test-list *args: _uv-sync
  uv run --no-sync pytest --collect-only -q --manual {{args}}

# https://spdx.org/licenses/
allow_licenses := "MIT BSD-2-CLAUSE BSD-3-CLAUSE APACHE-2.0 ISC"
ignore_package_licenses := "nvidia-* cuda-* hf-xet certifi filelock matplotlib typing-extensions sentencepiece"

# Run licensecheck
_licensecheck *args:
  uvx licensecheck@2025.1.0 \
  --show-only-failing \
  --only-licenses {{allow_licenses}} \
  --ignore-packages {{ignore_package_licenses}} \
  --zero \
  {{args}}

# Run pip-licenses
_pip-licenses *args: _uv-sync
  uvx pip-licenses@5.5.0 \
  --python .venv/bin/python \
  --format=plain-vertical \
  --with-license-file \
  --no-license-path \
  --no-version \
  --with-urls \
  --output-file ATTRIBUTIONS.md \
  {{args}}
  pre-commit run --files ATTRIBUTIONS.md || true

# Update the license
license: _licensecheck _pip-licenses

# Run link-check
_link-check *args:
  pre-commit run -a --hook-stage manual link-check {{args}}

# Pre-release checks
release-check: license _link-check

# Release a new version
release pypi_token='dry-run' *args:
  ./bin/release.sh {{pypi_token}} {{args}}

# Build the docker container.
_docker-build *args:
   docker build {{args}} .

# Run the docker container
_docker build_args='' run_args='': (_docker-build build_args)
  #!/usr/bin/env bash
  set -euxo pipefail
  image_tag=$(docker build {{build_args}} -q .)
  docker run \
    -it \
    --runtime=nvidia \
    --ipc=host \
    --rm \
    -v .:/workspace \
    -v /workspace/.venv \
    -v /root/.cache:/root/.cache \
    -e HF_TOKEN="$HF_TOKEN" \
    {{run_args}} \
    $image_tag

docker_cu128_build_args := '--build-arg=CUDA_VERSION=12.8.1'
docker_cu130_build_args := '--build-arg=CUDA_VERSION=13.0.2'
docker_nightly_build_args := '-f docker/nightly.Dockerfile'

# Run the CUDA 12.8 docker container.
docker-cu128 *run_args: (_docker docker_cu128_build_args run_args)

# Run the CUDA 13.0 docker container.
docker-cu130 *run_args: (_docker docker_cu130_build_args run_args)

# Run the nightly docker container.
docker-nightly *run_args: (_docker docker_nightly_build_args run_args)

# Test the docker containers.
test-docker: \
  (_docker-build '-q' docker_cu128_build_args) \
  (_docker-build '-q' docker_cu130_build_args) \
  (_docker-build '-q' docker_nightly_build_args)
