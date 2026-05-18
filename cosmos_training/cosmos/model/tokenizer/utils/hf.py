# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Helpers for consistent HuggingFace cache-only loading in tokenizer jobs."""

from __future__ import annotations

import atexit
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

NEMOTRON_VISION_TOKEN_REMAP: dict[str, str] = {
    "<SPECIAL_20>": "<|vision_start|>",
    "<SPECIAL_21>": "<|vision_end|>",
    "<SPECIAL_22>": "<|image_pad|>",
}
_PATCHED_NEMOTRON_SNAPSHOT_CACHE: dict[tuple[str, str], str] = {}
_PATCHED_NEMOTRON_SNAPSHOT_DIRS: set[Path] = set()


def _cleanup_patched_nemotron_snapshots() -> None:
    """Delete any process-local writable Nemotron tokenizer snapshots."""
    for snapshot_dir in list(_PATCHED_NEMOTRON_SNAPSHOT_DIRS):
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    _PATCHED_NEMOTRON_SNAPSHOT_DIRS.clear()
    _PATCHED_NEMOTRON_SNAPSHOT_CACHE.clear()


atexit.register(_cleanup_patched_nemotron_snapshots)


def _iter_snapshot_candidates(model_root: Path) -> list[Path]:
    """Return snapshot candidates in cache-preferred order for one HF model root."""
    snapshots_root = model_root / "snapshots"
    refs_root = model_root / "refs"

    candidate_names: list[str] = []
    seen_names: set[str] = set()

    preferred_ref_paths = [refs_root / "main", refs_root / "master"]
    for ref_path in preferred_ref_paths:
        if not ref_path.is_file():
            continue
        ref_value = ref_path.read_text(encoding="utf-8").strip()
        if ref_value and ref_value not in seen_names:
            candidate_names.append(ref_value)
            seen_names.add(ref_value)

    if refs_root.is_dir():
        for ref_path in sorted(path for path in refs_root.rglob("*") if path.is_file()):
            ref_value = ref_path.read_text(encoding="utf-8").strip()
            if ref_value and ref_value not in seen_names:
                candidate_names.append(ref_value)
                seen_names.add(ref_value)

    if snapshots_root.is_dir():
        for snapshot_path in sorted(path for path in snapshots_root.iterdir() if path.is_dir()):
            snapshot_name = snapshot_path.name
            if snapshot_name not in seen_names:
                candidate_names.append(snapshot_name)
                seen_names.add(snapshot_name)

    return [snapshots_root / candidate_name for candidate_name in candidate_names]


def resolve_hf_snapshot_path(
    model_name: str,
    cache_dir: str | None,
    required_files: tuple[str, ...] = ("tokenizer_config.json",),
) -> str | None:
    """Find a cached HuggingFace snapshot directory for one model.

    Args:
        model_name: HuggingFace model identifier such as ``Qwen/Qwen3-0.6B``.
        cache_dir: Root HF cache directory, typically ``HF_HOME``.
        required_files: Files that must exist inside the snapshot directory.

    Returns:
        The first matching snapshot directory, or ``None`` if not found.
    """
    if cache_dir is None:
        return None

    safe_name = model_name.replace("/", "--")
    for hub_prefix in ("hub", ""):
        model_root = (
            Path(cache_dir) / hub_prefix / f"models--{safe_name}"
            if hub_prefix
            else Path(cache_dir) / f"models--{safe_name}"
        )
        snapshots_root = model_root / "snapshots"
        if not snapshots_root.is_dir():
            continue

        for snapshot_path in _iter_snapshot_candidates(model_root):
            if not snapshot_path.is_dir():
                continue
            if all((snapshot_path / filename).exists() for filename in required_files):
                return str(snapshot_path)
    return None


def load_auto_tokenizer_from_cache(
    model_name: str,
    cache_dir: str | None,
    required_files: tuple[str, ...] = ("tokenizer_config.json",),
    **kwargs: Any,
) -> Any:
    """Load a tokenizer from the local HF cache when available.

    When ``cache_dir`` is set but no local snapshot exists, this still enforces
    ``local_files_only=True`` so tokenizer workers fail fast instead of trying to
    reach the public HuggingFace API from restricted clusters.
    """
    from transformers import AutoTokenizer

    local_snapshot = resolve_hf_snapshot_path(model_name, cache_dir, required_files=required_files)
    if local_snapshot is not None:
        return AutoTokenizer.from_pretrained(local_snapshot, local_files_only=True, **kwargs)
    return AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=cache_dir is not None,
        **kwargs,
    )


def _patch_nemotron_tokenizer_snapshot_in_place(snapshot_dir: Path) -> None:
    """Rename reserved Nemotron placeholder tokens to tokenizer-visible vision tokens."""
    tokenizer_json_path = snapshot_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        with tokenizer_json_path.open(encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        for entry in tokenizer_data.get("added_tokens", []):
            content = entry.get("content")
            if content in NEMOTRON_VISION_TOKEN_REMAP:
                entry["content"] = NEMOTRON_VISION_TOKEN_REMAP[content]

        vocab = tokenizer_data.get("model", {}).get("vocab", {})
        for old_name, new_name in NEMOTRON_VISION_TOKEN_REMAP.items():
            if old_name in vocab:
                vocab[new_name] = vocab.pop(old_name)

        with tokenizer_json_path.open("w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f)

    tokenizer_config_path = snapshot_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        with tokenizer_config_path.open(encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        for entry in tokenizer_config.get("added_tokens_decoder", {}).values():
            content = entry.get("content")
            if content in NEMOTRON_VISION_TOKEN_REMAP:
                entry["content"] = NEMOTRON_VISION_TOKEN_REMAP[content]

        with tokenizer_config_path.open("w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f)


def prepare_nemotron_tokenizer_snapshot(
    model_name: str,
    cache_dir: str | None,
    required_files: tuple[str, ...] = ("tokenizer_config.json", "tokenizer.json"),
) -> str | None:
    """Return a writable copied snapshot with Nemotron vision tokens remapped."""
    local_snapshot = resolve_hf_snapshot_path(model_name, cache_dir, required_files=required_files)
    if local_snapshot is None:
        return None

    source_snapshot = Path(local_snapshot)
    cache_key = (model_name, str(source_snapshot.resolve()))
    cached_snapshot = _PATCHED_NEMOTRON_SNAPSHOT_CACHE.get(cache_key)
    if cached_snapshot is not None and Path(cached_snapshot).is_dir():
        return cached_snapshot

    patched_root = Path(
        tempfile.mkdtemp(
            prefix=(f"imaginaire4_nemotron_tokenizer_{model_name.replace('/', '--')}_{source_snapshot.name}_"),
            dir=tempfile.gettempdir(),
        )
    )
    shutil.copytree(source_snapshot, patched_root, dirs_exist_ok=True)
    _patch_nemotron_tokenizer_snapshot_in_place(patched_root)
    patched_snapshot = str(patched_root)
    _PATCHED_NEMOTRON_SNAPSHOT_CACHE[cache_key] = patched_snapshot
    _PATCHED_NEMOTRON_SNAPSHOT_DIRS.add(patched_root)
    return patched_snapshot
