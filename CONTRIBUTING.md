# Contributing to NVIDIA Cosmos

Thank you for your interest in contributing to NVIDIA Cosmos. This guide covers how to propose changes, add new applications, and maintain the quality bar we hold for community-facing content.

## Code of Conduct

This project adheres to the [NVIDIA Open Source Code of Conduct](https://github.com/NVIDIA/cosmos/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by filing an issue or contacting [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).

---

## How to Contribute

### Reporting Issues

Open an issue on [GitHub Issues](https://github.com/NVIDIA/cosmos/issues) with:

- A clear, descriptive title
- Steps to reproduce (if applicable)
- Expected vs. actual behavior
- Environment details: OS, CUDA version, GPU model, Python version, `uv` version
- Relevant logs or error messages

### Contribution Workflow

1. **Fork** the repository and create a branch from `main`:

   ```bash
   git checkout -b cookbook/descriptive-name   # or docs/, fix/, benchmark/
   ```

2. **Make your changes** following the guidelines below.

3. **Test your changes.** Run your notebook end-to-end on the target GPU. Verify existing cookbooks are unaffected.

4. **Commit** with a clear message:

   ```bash
   git commit -m "Add worker-safety Reasoner application with vLLM backend"
   ```

5. **Push and open a Pull Request** against `main`.

### Pull Request Guidelines

- Provide a clear description of what your PR does and why
- Reference related issues (e.g., `Fixes #123`)
- One logical change per PR
- Ensure your branch is up to date with `main`
- Respond to review feedback promptly

---

## Cookbook Structure

The `cookbooks/` directory is organized by **model generation → tower → capability**. NVIDIA-shipped examples and finetune scripts live under each tower directory. Community and external contributions go under the **`applications/`** directory, which mirrors the same tower structure.

```
cookbooks/
└── cosmos3/
    ├── README.md                          # Shared setup (all backends)
    ├── cosmos3-model-architecture.png
    │
    ├── reasoner/                          # Reasoner Tower (NVIDIA-shipped)
    │   ├── README.md
    │   ├── assets/
    │   ├── run_with_vllm.ipynb
    │   ├── run_with_nim.ipynb
    │   ├── run_with_cosmos_framework.ipynb
    │   └── finetune/
    │
    ├── generator/                         # Generator Tower (NVIDIA-shipped)
    │   ├── audiovisual/                   # T2I, T2V, I2V, audio
    │   │   ├── README.md
    │   │   └── finetune/
    │   ├── action/                        # Policy, FDM, IDM
    │   │   ├── README.md
    │   │   └── finetune/
    │   └── transfer/                      # Video-to-video transfer
    │       └── README.md
    │
    └── applications/                      # ← Community & external contributions
        ├── README.md                      # Applications overview + index
        │
        ├── reasoner/                      # Reasoner applications
        │   ├── README.md
        │   └── <your-application>/        # ← Your contribution goes here
        │       ├── README.md
        │       ├── run_<task>_with_<backend>.ipynb
        │       └── assets/
        │
        └── generator/                     # Generator applications
            ├── README.md
            ├── audiovisual/               # T2I, T2V, I2V, audio applications
            │   └── <your-application>/
            ├── action/                    # Policy, FDM, IDM applications
            │   └── <your-application>/
            └── transfer/                  # Video transfer applications
                └── <your-application>/
```

### Where Does My Application Go?

| Your application does... | Place it under |
|--------------------------|---------------|
| Image/video understanding, VLM, reasoning, grounding | `cookbooks/cosmos3/applications/reasoner/<your-app>/` |
| Text-to-image, text-to-video, image-to-video, audio | `cookbooks/cosmos3/applications/generator/audiovisual/<your-app>/` |
| Robotics policy, forward/inverse dynamics | `cookbooks/cosmos3/applications/generator/action/<your-app>/` |
| Video-to-video style transfer, edge-guided generation | `cookbooks/cosmos3/applications/generator/transfer/<your-app>/` |

If your application spans multiple towers (e.g., Reasoner analysis → Generator synthesis), place it under the primary tower and document the cross-tower dependency in your README.

---

## Application Quality Requirements

Every application merged into this repo must meet these requirements. Reviewers will check each item.

### 1. Open-Access Data Only

- All datasets must be **publicly downloadable** without NVIDIA-internal credentials
- Acceptable sources: HuggingFace Hub (public or gated with free access), public URLs, synthetic data generated in the notebook
- If working with partners, request a **small public subset** for the application example
- Include the dataset license in your README

**Not acceptable:** Internal S3 buckets, VPN-only URLs, private NFS mounts, datasets requiring paid partner agreements

### 2. Results / Expected Output

Every application must include a **Results** section showing what a successful run looks like:

- **Inference applications:** Sample generated images/videos, text outputs, or action trajectories saved to `assets/`
- **Post-training applications:** Training loss curves, before/after comparison, evaluation metrics
- **Timing benchmarks:** Wall-clock time on the target GPU (e.g., "Cosmos3-Nano T2V: 45s on 1× A100")

This lets developers validate their own runs against a known-good baseline.

### 3. Canonical Setup (No Hidden Dependencies)

- **Do not duplicate setup instructions.** Link to the shared [`cookbooks/cosmos3/README.md`](cookbooks/cosmos3/README.md) for backend installation (Cosmos Framework, Diffusers, vLLM, NIM)
- Your README should only document **application-specific** dependencies beyond the shared setup
- All dependencies must be installable via `uv pip install` or `apt-get` — no manual builds
- Pin specific versions of critical packages when they affect reproducibility

### 4. One-Click Runnable

- Each notebook should run **top-to-bottom without manual intervention**
- Use environment variables for configurable paths (`HF_TOKEN`, `COSMOS3_MEDIA_ROOT`, etc.)
- Default to the smallest model size (Cosmos3-Nano) so the widest set of GPUs can run it
- If an application requires a running server (vLLM, NIM), provide the exact launch command in the README and automate the health check in the notebook

### 5. Naming Convention

Follow the existing pattern:

```
run_<task>_with_<backend>.ipynb
```

Examples:
- `run_with_vllm.ipynb` — generic Reasoner inference via vLLM
- `run_fd_with_cosmos_framework.ipynb` — forward dynamics via Cosmos Framework
- `run_video_transfer_with_cosmos_framework.ipynb` — video transfer via Cosmos Framework

For markdown-only guides (no notebook): `run_<task>_with_<backend>.md`

### 6. Author Attribution

Every application must credit its authors to increase visibility and recognition:

- **README:** Include an author block immediately after the title (see [README template](#application-readme-template))
- **Notebook:** Include an author block in the first markdown cell, right after the SPDX header and title

Use this format:

```markdown
> **Authors:** [Full Name](https://linkedin.com/in/handle), [Full Name](https://linkedin.com/in/handle)
> **Organization:** [Your Organization](https://your-org.com/)
```

This is required for all new contributions and encouraged for existing cookbooks.

---

## Application README Template

Each application directory needs a `README.md`. Use this structure:

```markdown
# [Application Title]

> **Authors:** [Your Name](https://linkedin.com/in/your-handle)
> **Organization:** [Your Organization](https://your-org.com/)

One-paragraph description of what this application demonstrates and why it matters.

## What You'll Build

- Bullet list of concrete outputs (e.g., "Generate a 480p video from a text prompt")

## Prerequisites

- Link to [shared setup](../../README.md#backend-name) for backend installation
- Any additional application-specific requirements

## Backends

| Backend | Notebook | GPU Requirement |
|---------|----------|----------------|
| vLLM    | [`run_with_vllm.ipynb`](run_with_vllm.ipynb) | 1× A100 (80 GB) |
| NIM     | [`run_with_nim.ipynb`](run_with_nim.ipynb) | 1× A100 (80 GB) |

## Quick Start

Minimal steps to go from clone to first result:

    1. Set up the backend (link)
    2. Run the notebook
    3. Check your outputs in `assets/`

## Results / Expected Output

Sample outputs, metrics, and timing benchmarks from a successful run.

## Dataset

| Name | Source | License | Size |
|------|--------|---------|------|
| Dataset Name | [HuggingFace link](...) | Apache 2.0 | ~2 GB |
```

---

## Contribution Areas

We welcome contributions in these areas:

| Area | Examples |
|------|---------|
| **New applications** | Domain-specific recipes (robotics, AV, healthcare, manufacturing, smart infrastructure) |
| **New backends** | Additional serving/inference backends for existing applications |
| **Documentation** | README improvements, prompt guides, architecture explanations |
| **Bug fixes** | Notebook fixes, broken links, version compatibility issues |
| **Benchmarks** | Inference timing across GPU configurations (A100, H100, L40S, RTX 4090) |
| **Post-training recipes** | SFT, LoRA, domain adaptation examples with open datasets |

### What We Won't Merge

- Applications that depend on internal/proprietary datasets
- Notebooks that require manual mid-run intervention
- Changes that break existing cookbook functionality
- Generated binary files (model weights, large media) — use HuggingFace/external links instead

---

## Development Setup

### Prerequisites

- Python 3.10 or later
- CUDA 12.8 or 13.x (see [Troubleshooting](README.md#troubleshooting))
- An NVIDIA GPU with sufficient VRAM for your target workflow
- `uv` >= 0.11.3 ([astral.sh/uv](https://astral.sh/uv))
- `git-lfs` installed (`apt-get install git-lfs`)

### Getting Started

```bash
git clone https://github.com/NVIDIA/cosmos.git
cd cosmos
```

Follow [cookbooks/cosmos3/README.md](cookbooks/cosmos3/README.md) to set up the backend(s) your application uses.

### Testing Your Application

Before submitting:

1. **Clean run:** Restart your kernel and run all cells top-to-bottom
2. **Minimal GPU:** Test on the smallest supported GPU configuration
3. **No secrets:** Verify no API keys, tokens, or internal paths are committed
4. **Output cells:** Clear large output cells but keep the Results section outputs
5. **File sizes:** Ensure no single file exceeds 10 MB (use git-lfs for larger assets or link externally)

---

## License

By contributing to this project, you agree that your contributions will be licensed under the [OpenMDW-1.1 License](LICENSE). All contributions must comply with the terms of this license.

## Questions?

If you have questions about contributing, open an issue or reach out at [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
