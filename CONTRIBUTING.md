# Contributing to NVIDIA Cosmos

Thank you for your interest in contributing to NVIDIA Cosmos. This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the [NVIDIA Open Source Code of Conduct](https://github.com/NVIDIA/cosmos/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by filing an issue or contacting [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).

## How to Contribute

### Reporting Issues

If you encounter a bug or have a feature request, please open an issue on the [GitHub Issues](https://github.com/NVIDIA/cosmos/issues) page. When filing an issue, include:

- A clear and descriptive title
- Steps to reproduce the problem (if applicable)
- Expected behavior vs. actual behavior
- Your environment details (OS, CUDA version, GPU model, Python version)
- Relevant logs or error messages

### Submitting Changes

1. **Fork the repository** and create a new branch from `main`:

   ```shell
   git checkout -b your-branch-name
   ```

2. **Make your changes.** Ensure your changes follow the project conventions and do not introduce regressions.

3. **Test your changes.** Verify that existing cookbooks and examples still work correctly with your modifications.

4. **Commit your changes** with a clear, descriptive commit message:

   ```shell
   git commit -m "Brief description of the change"
   ```

5. **Push to your fork** and open a Pull Request against the `main` branch of the upstream repository.

### Pull Request Guidelines

- Provide a clear description of what your PR does and why
- Reference any related issues (e.g., `Fixes #123`)
- Keep PRs focused: one logical change per PR
- Ensure your branch is up to date with `main` before submitting
- Be responsive to review feedback

## Development Setup

### Prerequisites

- Python 3.10 or later
- CUDA 12.8 or 13.x (see [Troubleshooting](README.md#troubleshooting) for version matching)
- An NVIDIA GPU with sufficient VRAM for your target workflow
- `uv` >= 0.11.3 (install from [astral.sh/uv](https://astral.sh/uv))

### Getting Started

1. Clone the repository:

   ```shell
   git clone https://github.com/NVIDIA/cosmos.git
   cd cosmos
   ```

2. Set up your environment following the instructions in the [README](README.md).

3. Explore the [cookbooks](cookbooks/) for end-to-end examples of Generator and Reasoner workflows.

## Contribution Areas

We welcome contributions in the following areas:

- **Cookbooks and examples:** New notebooks demonstrating Cosmos 3 capabilities
- **Documentation:** Improvements to README, cookbook READMEs, or inline documentation
- **Bug fixes:** Fixes for issues in existing code or documentation
- **Benchmarks:** Additional inference benchmark results across different hardware configurations

## License

By contributing to this project, you agree that your contributions will be licensed under the [OpenMDW-1.1 License](LICENSE). All contributions must comply with the terms of this license.

## Questions?

If you have questions about contributing, feel free to open an issue or reach out at [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
