<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Maintain the Cosmos3 Certified NIM documentation

Open this `maintainer` directory in a compatible coding assistant when reviewing
or editing the public NIM documentation. The nested [AGENTS.md](AGENTS.md) and
maintainer skill provide the evidence, ownership, editing, and validation
workflow. The public documentation and examples are one directory above this
folder.

## Maintainer workflow

1. Start the assistant with `cookbooks/cosmos3/nim/maintainer` as its working
   directory.
2. Ask it to load the `cosmos3-nim-docs-maintainer` skill.
3. Describe the documentation change and provide any approved image or release
   evidence.
4. Review the resulting changes under `..` and the reported static or live
   validation separately.

Run documentation commands from the public documentation root:

```bash
cd ..
uv lock --check
uv run --locked python -m compileall -q examples
git diff --check
```

For customer usage, deployment, and troubleshooting assistance, open the
[public NIM documentation directory](../README.md) instead.
