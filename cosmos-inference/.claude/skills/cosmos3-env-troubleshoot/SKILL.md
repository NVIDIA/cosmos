---
name: cosmos3-env-troubleshoot
description: >
  Diagnose and fix Cosmos3 environment, installation, and runtime errors.
  Use when the user encounters an ImportError, ModuleNotFoundError, CUDA error,
  Docker error, checkpoint download failure, or any traceback during setup or inference.
---

# Cosmos3 Environment Troubleshooting

## When to use this skill

- Use when a user hits an error during installation, environment setup, or first run
- Use when a traceback mentions torch, CUDA, missing modules, or shared libraries
- Use when Docker or container setup fails
- Use when checkpoint downloads fail or HuggingFace auth errors appear

## Path convention

All paths below are relative to this file's location (`.agents/skills/cosmos3-env-troubleshoot/`).

## Step 1: Match against known errors

Check the error message against the table below. Each row links to the canonical fix in the docs.

| Error signature                                                               | Cause                                | Fix location                                                                                           |
| ----------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `ImportError: cannot import name '_functionalization' from 'torch._C'`        | NGC container library conflict       | `../../../docs/setup.md` § PyTorch Import Issue — run `export LD_LIBRARY_PATH=''`                      |
| `ModuleNotFoundError: No module named 'cosmos3'`                              | Package not installed                | `../../../docs/setup.md` § Dependency Issue — run `uv sync --all-extras --group=cu130 --reinstall`     |
| `ModuleNotFoundError: No module named <other>`                                | Dependency missing                   | `../../../docs/setup.md` § Dependency Issue — reinstall venv                                           |
| `fatal error: Python.h: No such file or directory`                            | Broken Python / uv install           | `../../../docs/setup.md` § Python Issue — reinstall uv + venv from scratch                             |
| `OSError: <lib>: cannot open shared object file`                              | CUDA version mismatch                | `../../../docs/setup.md` § CUDA Issue — install matching `cuda-toolkit-<major>`                        |
| `docker: Error response from daemon: unknown or invalid runtime name: nvidia` | Docker nvidia runtime not configured | `../../../docs/setup.md` § Docker Container — run `sudo nvidia-ctk runtime configure --runtime=docker` |
| HuggingFace 401 / download failures                                           | Auth or license not accepted         | `../../../docs/setup.md` § Downloading Checkpoints — check `HF_TOKEN`, accept license agreement        |

## Step 2: If no documented fix matches, try common remediation

Run these diagnostic commands to collect information, then attempt fixes in order:

### Diagnostic commands

```shell
# System
uname -a
cat /etc/os-release | head -5

# Python
python --version
which python

# CUDA
nvidia-smi
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.version.cuda}')"

# Package
uv pip list | head -20
```

### Remediation ladder (try in order)

1. **Clear library path**: `export LD_LIBRARY_PATH=''`
2. **Reinstall venv**: `uv sync --all-extras --group=cu130 --reinstall`
3. **Reinstall uv + venv from scratch**:

   ```shell
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install --reinstall
   rm -rf .venv
   uv sync --all-extras --group=cu130 --reinstall
   source .venv/bin/activate
   ```

4. **Check CUDA version alignment**: the major CUDA version from `nvidia-smi` must match `torch.version.cuda`
5. **Try Docker**: if the host environment is too broken, fall back to the Docker container (see `../../../docs/setup.md`)

## Step 3: If still unresolved, generate a bug report

If none of the above resolves the issue, collect environment information and present the user with a pre-filled bug report they can submit as a GitHub issue.

Fill in the template below by running the diagnostic commands and inserting the results:

````markdown
## Environment

- **OS**: <output of `uname -a`>
- **Python**: <output of `python --version`>
- **CUDA (system)**: <output of `nvidia-smi` — first line with driver/CUDA version>
- **CUDA (torch)**: <output of `python -c "import torch; print(torch.version.cuda)">`>
- **torch version**: <output of `python -c "import torch; print(torch.__version__)">`>
- **cosmos3 version**: <output of `python -c "import cosmos3; print(cosmos3.__version__)"` or "not installed">
- **Installation method**: <uv sync / uv pip / Docker / NGC container>

## Error

```
<full traceback>
```

## What was tried

1. <list each remediation step attempted and its result>

## Additional context

<any other relevant details — multi-GPU setup, custom CUDA install, etc.>
````
