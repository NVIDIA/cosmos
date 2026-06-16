# Contributing to NVIDIA Cosmos

Thank you for your interest in contributing to NVIDIA Cosmos. This guide covers how to propose changes, add new cookbooks, and maintain the quality bar we hold for community-facing content.

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
   git commit -m "Add worker-safety Reasoner cookbook with vLLM backend"
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

The `cookbooks/` directory is organized by **model generation → tower → capability**. Each cookbook is a self-contained directory with a README, one or more runnable notebooks, and supporting assets.

```
cookbooks/
└── cosmos3/
    ├── README.md                          # Shared setup (all backends)
    ├── cosmos3-model-architecture.png
    │
    ├── reasoner/                          # Reasoner Tower
    │   ├── README.md                      # Reasoner overview + backend table
    │   ├── basic_examples/                # Shipped starter cookbooks
    │   │   ├── reasoner_prompt_guide.md
    │   │   ├── run_with_vllm.ipynb
    │   │   ├── run_with_nim.ipynb
    │   │   ├── run_with_cosmos_framework.ipynb
    │   │   └── assets/
    │   └── <your-cookbook>/                # ← Community contributions go here
    │       ├── README.md
    │       ├── run_<task>_with_<backend>.ipynb
    │       └── assets/
    │
    ├── generator/
    │   ├── audiovisual/                   # Generator: T2I, T2V, I2V, audio
    │   │   ├── README.md
    │   │   ├── basic_examples/            # Shipped starter cookbooks
    │   │   │   ├── run_with_diffusers.ipynb
    │   │   │   ├── run_with_vllm_omni.ipynb
    │   │   │   ├── run_with_cosmos_framework.ipynb
    │   │   │   └── assets/
    │   │   └── <your-cookbook>/            # ← Community contributions go here
    │   │
    │   ├── action/                        # Generator: policy, FDM, IDM
    │   │   ├── README.md
    │   │   ├── basic_examples/            # Shipped starter cookbooks
    │   │   │   ├── run_fd_with_cosmos_framework.ipynb
    │   │   │   ├── run_fd_with_vllm.ipynb
    │   │   │   ├── run_id_with_cosmos_framework.ipynb
    │   │   │   ├── run_id_with_vllm.ipynb
    │   │   │   ├── run_policy_with_cosmos_framework.md
    │   │   │   └── assets/
    │   │   └── <your-cookbook>/            # ← Community contributions go here
    │   │
    │   └── transfer/                      # Generator: video-to-video transfer
    │       ├── README.md
    │       ├── basic_examples/            # Shipped starter cookbooks
    │       │   ├── run_video_transfer_with_cosmos_framework.ipynb
    │       │   ├── preview_helpers.py
    │       │   ├── specs/
    │       │   └── assets/
    │       └── <your-cookbook>/            # ← Community contributions go here
    │
    └── end2end/                            # Multi-tower or application workflows
        ├── README.md
        └── <your-cookbook>/                # ← Community contributions go here
            ├── README.md
            ├── run_<task>_with_<backend>.md
            └── assets/
```

### Where Does My Cookbook Go?

| Your cookbook does... | Place it under |
|----------------------|---------------|
| Image/video understanding, VLM, reasoning, grounding | `cookbooks/cosmos3/reasoner/` |
| Text-to-image, text-to-video, image-to-video, audio | `cookbooks/cosmos3/generator/audiovisual/` |
| Robotics policy, forward/inverse dynamics | `cookbooks/cosmos3/generator/action/` |
| Video-to-video style transfer, edge-guided generation | `cookbooks/cosmos3/generator/transfer/` |
| Multi-tower application workflows or external robot/simulation stacks | `cookbooks/cosmos3/end2end/` |

If your cookbook spans multiple towers (e.g., Reasoner analysis → Generator synthesis), create a new directory under `cookbooks/cosmos3/` with a clear name (e.g., `cookbooks/cosmos3/end2end/`).

---

## Cookbook Quality Requirements

Every cookbook merged into this repo must meet these requirements. Reviewers will check each item.

### 1. Open-Access Data Only

- All datasets must be **publicly downloadable** without NVIDIA-internal credentials
- Acceptable sources: HuggingFace Hub (public or gated with free access), public URLs, synthetic data generated in the notebook
- If working with partners, request a **small public subset** for the cookbook example
- Include the dataset license in your README

**Not acceptable:** Internal S3 buckets, VPN-only URLs, private NFS mounts, datasets requiring paid partner agreements

### 2. Results / Expected Output

Every cookbook must include a **Results** section showing what a successful run looks like:

- **Inference cookbooks:** Sample generated images/videos, text outputs, or action trajectories saved to `assets/`
- **Post-training cookbooks:** Training loss curves, before/after comparison, evaluation metrics
- **Timing benchmarks:** Wall-clock time on the target GPU (e.g., "Cosmos3-Nano T2V: 45s on 1× A100")

This lets developers validate their own runs against a known-good baseline.

### 3. Canonical Setup (No Hidden Dependencies)

- **Do not duplicate setup instructions.** Link to the shared [`cookbooks/cosmos3/README.md`](cookbooks/cosmos3/README.md) for backend installation (Cosmos Framework, Diffusers, vLLM, NIM)
- Your README should only document **cookbook-specific** dependencies beyond the shared setup
- All dependencies must be installable via `uv pip install` or `apt-get` — no manual builds
- Pin specific versions of critical packages when they affect reproducibility

### 4. One-Click Runnable

- Each notebook should run **top-to-bottom without manual intervention**
- Use environment variables for configurable paths (`HF_TOKEN`, `COSMOS3_MEDIA_ROOT`, etc.)
- Default to the smallest model size (Cosmos3-Nano) so the widest set of GPUs can run it
- If a cookbook requires a running server (vLLM, NIM), provide the exact launch command in the README and automate the health check in the notebook

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

---

## Cookbook README Template

Each cookbook directory needs a `README.md`. Use this structure:

```markdown
# [Cookbook Title]

One-paragraph description of what this cookbook demonstrates and why it matters.

## What You'll Build

- Bullet list of concrete outputs (e.g., "Generate a 480p video from a text prompt")

## Prerequisites

- Link to [shared setup](../README.md#backend-name) for backend installation
- Any additional cookbook-specific requirements

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
| **New cookbooks** | Domain-specific applications (robotics, AV, healthcare, manufacturing) |
| **New backends** | Additional serving/inference backends for existing cookbooks |
| **Documentation** | README improvements, prompt guides, architecture explanations |
| **Bug fixes** | Notebook fixes, broken links, version compatibility issues |
| **Benchmarks** | Inference timing across GPU configurations (A100, H100, L40S, RTX 4090) |
| **Post-training recipes** | SFT, LoRA, domain adaptation examples with open datasets |

### What We Won't Merge

- Cookbooks that depend on internal/proprietary datasets
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

Follow [cookbooks/cosmos3/README.md](cookbooks/cosmos3/README.md) to set up the backend(s) your cookbook uses.

### Testing Your Cookbook

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
