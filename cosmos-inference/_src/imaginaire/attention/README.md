# Imaginaire Attention Subpackage

A subpackage within cosmos3._src.imaginaire that integrates only the best and most reliable
solutions, and provides simple APIs to end-users.

For more information, please refer to the [docs](docs/).

## Basic API

```python
from cosmos3._src.imaginaire.attention import attention

output = attention(
    query=query,
    key=key,
    value=value,
)
```

- **Optional** `scale`: attention (softmax/dot product) scale. Defaults to `head_dim ** -0.5`.
- **Optional** `return_lse`: returns logsumexp if `True`
- **Optional** `backend`: explicitly set backend instead of automatically selecting the best compatible

## Tensor layouts

Imaginaire Attention only supports one tensor memory layout:
heads-last torch contiguous (`torch.contiguous_format`).

With this layout, input tensors `query`, `key`, and `value` are represented as rank-4 tensors, with
dimension 0 representing batch, dimension 1 representing sequence length, dimension 2 representing
attention heads, and dimension 3 representing head dimension.
This layout is also consistent with the `contiguous_format` memory layout in PyTorch, meaning the
right-most dimension (head dimension) is the major dimension (has stride 1), and tokens from
different heads are interleaved.

```python
def verify_heads_last_contig_tensor(x: Tensor):
    assert x.shape[0] == batch
    assert x.shape[1] == seqlen
    assert x.shape[2] == heads
    assert x.shape[3] == head_dim

    assert x.stride(3) == 1
    assert x.stride(2) == head_dim
    assert x.stride(1) == heads * head_dim
    assert x.stride(0) == heads * head_dim * seqlen
```
