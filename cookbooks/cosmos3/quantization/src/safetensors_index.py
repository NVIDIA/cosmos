# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Build a whole-model ``model.safetensors.index.json`` for a diffusers/HF checkpoint.

The transformers and diffusers loaders resolve weights for a composite checkpoint
through a root-level ``model.safetensors.index.json``: a ``weight_map`` that maps
each tensor key to the (root-relative) safetensors file that holds it, plus a
``metadata.total_size`` byte count. When that index is missing — or points at
files that no longer exist — ``from_pretrained`` fails to locate the shards.

This module rebuilds a correct index by reading only the safetensors *headers*
(no tensors are materialized), so it is cheap to run over multi-GB checkpoints.

Typical use::

    index = SafetensorsIndex()
    index.update_dir(root / "transformer", "transformer")
    index.update(root / "vision_encoder/model.safetensors", "vision_encoder/model.safetensors")
    index.write(root / "model.safetensors.index.json")
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

_METADATA_KEY = "__metadata__"


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Read a safetensors file header without loading any tensor data.

    The safetensors format begins with a little-endian ``u64`` header length,
    followed by that many bytes of JSON describing each tensor
    (``{dtype, shape, data_offsets: [begin, end]}``). Returns the parsed header
    dict and the summed tensor byte count (``end - begin`` over all tensors),
    which matches HuggingFace's ``metadata.total_size`` convention.
    """
    path = Path(path)
    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        (header_len,) = struct.unpack("<Q", length_bytes)
        header = json.loads(f.read(header_len))

    total_bytes = 0
    for key, meta in header.items():
        if key == _METADATA_KEY:
            continue
        begin, end = meta["data_offsets"]
        total_bytes += end - begin
    return header, total_bytes


class SafetensorsIndex:
    """Accumulator for a HuggingFace-style ``*.safetensors.index.json``.

    Feed it safetensors files with :meth:`update` / :meth:`update_dir`; each call
    reads the file header and records ``weight_map[tensor_key] = relative_path``
    while accumulating ``metadata.total_size``. Serialize with :meth:`to_json` or
    persist with :meth:`write`.
    """

    def __init__(self) -> None:
        self.metadata: dict[str, int] = {"total_size": 0}
        self.weight_map: dict[str, str] = {}

    def update(self, file_path: Path, rel_name: str) -> "SafetensorsIndex":
        """Index one safetensors file, mapping each of its tensors to ``rel_name``.

        ``rel_name`` is stored verbatim in the ``weight_map`` and should be the
        file's path relative to the checkpoint root (e.g.
        ``"transformer/diffusion_pytorch_model-00001-of-00002.safetensors"``).
        """
        header, total_bytes = read_safetensors_header(file_path)
        for key in header:
            if key == _METADATA_KEY:
                continue
            prev = self.weight_map.get(key)
            if prev is not None and prev != rel_name:
                raise ValueError(
                    f"Duplicate tensor key {key!r} in both {prev!r} and {rel_name!r}; "
                    "the checkpoint has colliding weight names across files and cannot "
                    "share a single root index without prefixing."
                )
            self.weight_map[key] = rel_name
        self.metadata["total_size"] += total_bytes
        return self

    def update_dir(self, dir_path: Path, prefix: str) -> "SafetensorsIndex":
        """Index every ``*.safetensors`` shard in ``dir_path`` under ``prefix``.

        Each shard is mapped to ``f"{prefix}/{shard.name}"`` (or just ``shard.name``
        when ``prefix`` is empty). Component index sidecars such as
        ``diffusion_pytorch_model.safetensors.index.json`` are ignored — only the
        actual weight files are recorded. Raises if the directory has no shards.
        """
        dir_path = Path(dir_path)
        shards = sorted(dir_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found in {dir_path}")
        for shard in shards:
            rel_name = f"{prefix}/{shard.name}" if prefix else shard.name
            self.update(shard, rel_name)
        return self

    def to_dict(self) -> dict:
        return {"metadata": self.metadata, "weight_map": self.weight_map}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: Path, indent: int = 2) -> Path:
        path = Path(path)
        path.write_text(self.to_json(indent=indent))
        return path


DEFAULT_COMPONENTS = ("transformer", "vision_encoder")


def build_root_index(
    root_dir: Path,
    components: tuple[str, ...] | list[str] = DEFAULT_COMPONENTS,
) -> SafetensorsIndex:
    """Build a root ``SafetensorsIndex`` for the given checkpoint components.

    For each ``component`` name, indexes ``root_dir/component/*.safetensors`` under
    that component prefix. Components that are absent or contain no shards are
    skipped (so a text-only checkpoint without ``vision_encoder/`` still works).
    Raises if none of the requested components yielded any weights.
    """
    root_dir = Path(root_dir)
    index = SafetensorsIndex()
    indexed_any = False
    for component in components:
        comp_dir = root_dir / component
        if not comp_dir.is_dir():
            continue
        if not any(comp_dir.glob("*.safetensors")):
            continue
        index.update_dir(comp_dir, component)
        indexed_any = True
    if not indexed_any:
        raise FileNotFoundError(
            f"None of the components {list(components)} under {root_dir} contain "
            "*.safetensors weights; nothing to index."
        )
    return index
