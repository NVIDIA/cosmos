# Imaginaire Attention Subpackage Docs > Features

## Causal mask

Causal masking requires explicit indication of causal mask type.
For example, simply passing `is_causal=True` will fail:

```python
output = attention(
    query=query,
    key=key,
    value=value,
    is_causal=True
)
```

Result:

```
ValueError: Argument causal_type must be specified when is_causal=True.
```

There are currently two types of causal masking that are supported, and many popular backends tend
to support only one. It's therefore critical to to choose the correct one for your application.

```python
from cosmos3._src.imaginaire.attention.masks import CausalType

# Causal type choices:
#   - CausalType.TopLeft
#   - CausalType.BottomRight

output = attention(
    query=query,
    key=key,
    value=value,
    is_causal=True,
    causal_type=CausalType.TopLeft,
)
```

### Top-left causal mask

Q sequence length = KV sequence length = 5

|     | K1       | K2       | K3       | K4       | K5       |
| --- | -------- | -------- | -------- | -------- | -------- |
| Q1  | &#x2713; | &#x2717; | &#x2717; | &#x2717; | &#x2717; |
| Q2  | &#x2713; | &#x2713; | &#x2717; | &#x2717; | &#x2717; |
| Q3  | &#x2713; | &#x2713; | &#x2713; | &#x2717; | &#x2717; |
| Q4  | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2717; |
| Q5  | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2713; |

Q sequence length = 2, KV sequence length = 5

|     | K1       | K2       | K3       | K4       | K5       |
| --- | -------- | -------- | -------- | -------- | -------- |
| Q1  | &#x2713; | &#x2717; | &#x2717; | &#x2717; | &#x2717; |
| Q2  | &#x2713; | &#x2713; | &#x2717; | &#x2717; | &#x2717; |

Q sequence length = 5, KV sequence length = 2

|     | K1       | K2       |
| --- | -------- | -------- |
| Q1  | &#x2713; | &#x2717; |
| Q2  | &#x2713; | &#x2713; |
| Q3  | &#x2713; | &#x2713; |
| Q4  | &#x2713; | &#x2713; |
| Q5  | &#x2713; | &#x2713; |

### Bottom-right causal mask

Q sequence length = KV sequence length = 5

|     | K1       | K2       | K3       | K4       | K5       |
| --- | -------- | -------- | -------- | -------- | -------- |
| Q1  | &#x2713; | &#x2717; | &#x2717; | &#x2717; | &#x2717; |
| Q2  | &#x2713; | &#x2713; | &#x2717; | &#x2717; | &#x2717; |
| Q3  | &#x2713; | &#x2713; | &#x2713; | &#x2717; | &#x2717; |
| Q4  | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2717; |
| Q5  | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2713; |

(identical to top-left in this special case)

Q sequence length = 2, KV sequence length = 5

|     | K1       | K2       | K3       | K4       | K5       |
| --- | -------- | -------- | -------- | -------- | -------- |
| Q1  | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2717; |
| Q2  | &#x2713; | &#x2713; | &#x2713; | &#x2713; | &#x2713; |

Q sequence length = 5, KV sequence length = 2

|     | K1       | K2       |
| --- | -------- | -------- |
| Q1  | &#x2717; | &#x2717; |
| Q2  | &#x2717; | &#x2717; |
| Q3  | &#x2717; | &#x2717; |
| Q4  | &#x2713; | &#x2717; |
| Q5  | &#x2713; | &#x2713; |

## GQA/MQA

Simply pass `key` and `value` without repeating attention heads.

**NOTE**: `key`/`value` heads must evenly divide `query` heads.

**NOTE**: the behavior is similar to `repeat_interleave`, not `repeat`.

## Variable length

**(Less efficient option)** Pass sequence lengths directly:

```python
output = attention(
    query=query,
    key=key,
    value=value,
    seqlens_Q=torch.tensor(sequence_length_list_Q, device=query.device),
    seqlens_KV=torch.tensor(sequence_length_list_KV, device=query.device),
)
```

This will manually compute the maximum sequence lengths, and cumulative sums (with the additional
padding).

**(More efficient option)** Compute cumulative sequence lengths and maximums once, and reuse it:

```python
from cosmos3._src.imaginaire.attention.varlen import generate_varlen_parameters


# they correspond to.
(
    cumulative_seqlen_Q,
    cumulative_seqlen_KV,
    max_seqlen_Q,
    max_seqlen_KV,
) = generate_varlen_parameters(query, key, value, seqlens_Q, seqlens_KV)

# in all attention layers that follow:
output = attention(
    query=query,
    key=key,
    value=value,
    cumulative_seqlen_Q=cumulative_seqlen_Q,
    cumulative_seqlen_KV=cumulative_seqlen_KV,
    max_seqlen_Q=max_seqlen_Q,
    max_seqlen_KV=max_seqlen_KV,
)
```
