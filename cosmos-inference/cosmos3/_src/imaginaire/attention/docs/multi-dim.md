# Imaginaire Attention Subpackage Docs > Features > Multi-Dimensional Attention

Multi-Dimensional Attention is the primary API for handling various complex masks and sparsity
patterns, such as the spatio-temporal mask, and sliding window attention.

## Basic API

```python
from cosmos3._src.imaginaire.attention import multi_dimensional_attention

output = multi_dimensional_attention(
    query=query,
    key=key,
    value=value,
)
```

Sparsity parameters:

- **Optional** `window_size`: allows reducing the attention span by limiting each token's context to
    a local sliding window. References:
  - [Image Transformer](https://arxiv.org/abs/1802.05751)
  - [Stand-alone self-attention](https://arxiv.org/abs/1906.05909)
  - [Neighborhood attention transformer](https://arxiv.org/abs/2204.07143)
- **Optional** `dilation`: introduces gaps between the tokens within a sliding window, capturing
    global context without more computation.
    Reference: [Dilated neighborhood attention transformer](https://arxiv.org/abs/2209.15001)

Other masking parameters:

- **Optional** `stride`: introduces delays into the sliding window, for **potential** efficiency
    gains. Reference: [Generalized Neighborhood Attention](https://arxiv.org/abs/2504.16922).
- **Optional** `is_causal`: allows causally masking individual dimensions. This parameter can
    implement the spatio-temporal mask (causal masking across temporal dimension, bi-directional
    along space).

All sparsity / masking parameters can be specified **per dimension**.
The key feature of `multi_dimensional_attention` over the standard `attention` API is supporting
multi-dimensional layouts of tokens (i.e. multi-dimensional feature maps).

This means `query`, `key` and `value` are not necessarily 4-D tensors; they can be 4-D, 5-D, or 6-D,
representing 1-D, 2-D, and 3-D token layouts (see [Tensor layouts](#tensor-layouts)).

- **Optional** `scale`: attention (softmax/dot product) scale. Defaults to `head_dim ** -0.5`.
- **Optional** `return_lse`: returns logsumexp if `True`
- **Optional** `backend`: explicitly set backend instead of automatically selecting the best compatible

## Tensor layouts

In addition to requiring the [contiguous heads-last tensor layout](../README.md#tensor-layouts),
Multi-Dimensional Attention also requires the "sequence length" dimension to be unrolled / unfolded
back into its original representation:

```python
# 1-D case: language, audio
batch, X, heads, head_dim = query_1d.shape
#      _
#      ^
#      |
#      |-----> token layout shape

# 2-D case: images
batch, X, Y, heads, head_dim = query_2d.shape
#      ____
#        ^
#        |
#        |-----> token layout shape

# 3-D case: videos / 3-D images
batch, X, Y, Z, heads, head_dim = query_3d.shape
#      _______
#         ^
#         |
#         |------> token layout shape
```

Multi-Dimensional Attention also requires the shapes of `query`, `key` and `value` to match along
those dimensions, henceforth called the **token layout shape**:

```python
assert query_1d.shape[1:2] == key_1d.shape[1:2] == value_1d.shape[1:2]

assert query_2d.shape[1:3] == key_2d.shape[1:3] == value_2d.shape[1:3]

assert query_3d.shape[1:4] == key_3d.shape[1:4] == value_3d.shape[1:4]
```

This is because of the large number of sparsity / masking features (and their combinations)
supported, which is mainly possible by making the assumption that query and context coordinate
spaces are the same, eliminating the requirement for a mapping between the two.

Problems with a different query and key/value token layout shape may be supported in the future.

## Backends

The only backend supporting multi-dimensional attention for now is `natten`.

## Spatio-Temporal Attention

Spatio-Temporal attention (causal masking across the time dimension, and no masking / bi-directional
across spatial dimensions) is a special case of Multi-Dimensional Attention.
You can either implement it by marking `is_causal` as expected in `multi_dimensional_attention`, or
directly use `spatio_temporal_attention`:

```python
from cosmos3._src.imaginaire.attention import spatio_temporal_attention

output = spatio_temporal_attention(
    query=query,
    key=key,
    value=value,
)
```
