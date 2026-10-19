<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM license and notices

The governing download terms for the Cosmos3 NIM are:

- Use of the container is governed by the
  [NVIDIA Software License](https://github.com/NVIDIA/cuda-python/blob/main/cuda_bindings/LICENSE)
  and the
  [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/).
- Use of the models is governed by
  [OpenMDW-1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1).

The cookbook source is covered by the repository [LICENSE](../../../LICENSE).
Read the terms in the image you deploy before you use it.

## Where to find the terms

The running NIM exposes bundled product license information at `/v1/license`:

```bash
curl -fsS "$NIM_URL/v1/license"
```

The release image also includes these files. Inspect them in the exact image
you deploy so the terms match your deployment:

| Path in the image | Contents |
| --- | --- |
| `/opt/nim/LICENSE` | NIM terms |
| `/opt/nim/MODEL_LICENSE` | Model terms |
| `/opt/nim/NOTICE` | Product notices and contact information |
| `/opt/nim/licenses/` | Included component licenses and notices |
| `/usr/share/cosmos3/oss-source/` | Package-modification script, source manifest, reproduction instructions, and checksums for the removal of unused Berkeley DB adapters |

## Acknowledgements

The licenses and notices for the Python packages installed in each image are
listed in [Acknowledgements](acknowledgements/README.md).
