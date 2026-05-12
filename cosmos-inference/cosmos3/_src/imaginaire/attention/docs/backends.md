# Imaginaire Attention Subpackage Docs > Backends

The goal is to support as many stable and reliable backends as possible, both for feature coverage,
and for delivering the best performance.

## NATTEN

[NATTEN](https://natten.org) ships standard Attention kernels in addition to sparse /
multi-dimensional kernels.

Minimum version required: `0.21.5.dev3`.

### Feature coverage

| Feat/Backend | Ampere/RTX         | Hopper | Blackwell          |
| ------------ | ------------------ | ------ | ------------------ |
| Causal mask  | :white_check_mark: |        | :white_check_mark: |
| Varlen       | :white_check_mark: |        | :white_check_mark: |
| GQA/MQA      |                    |        | :white_check_mark: |
| MLA          | :white_check_mark: |        |                    |

This backend supports torch compile.

## Flash Attention v2

Flash Attention v2 (original C++ kernels) are available under the `flash2` backend.
Requires the `flash_attn` package.

Minimum version required: `2.7.0`.
Maximum version supported: `2.7.4`.

This backend supports torch compile.

### Feature coverage

| Feat/Backend | Ampere/RTX         |
| ------------ | ------------------ |
| Causal mask  | :white_check_mark: |
| Varlen       | :white_check_mark: |
| GQA/MQA      | :white_check_mark: |
| MLA          |                    |

## Flash Attention v3

Flash Attention v3 (original C++ kernels) are available under the `flash3` backend.
Requires the `flash_attn_3` package.

Version required: `3.0.0.b*`.

### Feature coverage

| Feat/Backend | Ampere/RTX         |
| ------------ | ------------------ |
| Causal mask  | :white_check_mark: |
| Varlen       | :white_check_mark: |
| GQA/MQA      | :white_check_mark: |
| MLA          |                    |

MLA is technically supported, but disabled due to an API bug in the backward pass.

Torch compile is NOT yet supported for this backend.
