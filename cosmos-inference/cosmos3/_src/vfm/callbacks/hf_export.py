# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HFExportCallback: export VLM DCP checkpoints to HuggingFace safetensors format.

Design notes
------------
- Hooks into ``on_save_checkpoint`` (called by DistributedCheckpointer.save() before I/O).
- All ranks participate in the weight-gather phase (DTensor.full_tensor() all-gathers).
- Rank 0 accumulates CPU tensors, writes shards, and uploads — other ranks exit early.
- File I/O and upload run in a background thread on rank 0 to avoid blocking training.
- Worker exceptions are stored in ``_worker_exception`` and re-raised on the next
  checkpoint or at train end, so failures are never silently swallowed.
- Controlled entirely via ``config.checkpoint.hf_export`` (HFExportConfig).

Phase 2+ note
-------------
Weight parameters are iterated via ``model.model._model.named_parameters()`` where
``model.model`` is the ``HFModel`` wrapper and ``model.model._model`` is the underlying
HuggingFace transformer.  Parameter names are already HF-native — no weight_mapper
remapping is required.
"""

import json
import os
import shutil
import threading
from typing import Any

import torch

from cosmos3._src.imaginaire.utils import log
from cosmos3._src.imaginaire.utils.callback import Callback
from cosmos3._src.imaginaire.utils.distributed import is_rank0

try:
    from safetensors.torch import save_file as _safetensors_save_file
except ImportError:
    _safetensors_save_file = None

try:
    from transformers import AutoTokenizer, GenerationConfig
except ImportError:
    AutoTokenizer = None
    GenerationConfig = None

# Map string dtype names (as stored in ParallelismConfig.precision) to torch dtypes.
_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def _upload_folder_to_s3(local_folder: str, bucket: str, s3_prefix: str, credential_path: str) -> None:
    """Upload every file under *local_folder* to ``s3://{bucket}/{s3_prefix}/...``.

    Uses the i4 ``easy_io`` S3 backend (Boto3Backend), which reads credentials from
    *credential_path*.  Files are uploaded as streaming transfers via boto3's
    ``upload_file()`` — the full shard is never loaded into memory.
    """
    from cosmos3._src.imaginaire.utils.easy_io import easy_io

    backend = easy_io.get_file_backend(
        backend_args={
            "backend": "s3",
            "s3_credential_path": credential_path,
            "path_mapping": None,
        }
    )
    for root, _, files in os.walk(local_folder):
        for fname in sorted(files):
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_folder)
            s3_path = f"s3://{bucket}/{s3_prefix}/{rel}"
            # Pass the local path string so Boto3Backend uses upload_file() —
            # a streaming transfer that avoids reading the whole shard into memory.
            backend.put(local_path, s3_path)
            log.info(f"[HFExportCallback] Uploaded {local_path} → {s3_path}")


class HFExportCallback(Callback):
    """Export VLM weights to HuggingFace-compatible safetensors after each DCP checkpoint.

    Enabled / configured via ``config.checkpoint.hf_export`` (HFExportConfig).  Disabled
    by default — add this callback and set ``hf_export.enabled = True`` to activate.

    Exports written to::

        {job.path_local}/hf_exports/iter_{iteration:09d}/
            00000.safetensors
            ...
            model.safetensors.index.json
            config.json
            tokenizer.json   (if tokenizer can be loaded from model_name_or_path)

    Optionally uploads to:
    - S3 (``hf_export.upload_to_object_store``)
    - HuggingFace Hub (``hf_export.hf_repo_id``)

    Args:
        dtype: Export weight dtype (e.g. ``"bfloat16"``).  Use
            ``"${model.config.policy.parallelism.precision}"`` in the Hydra callback config to
            inherit from the training precision.
    """

    # HuggingFace convention: max 4 GB per shard file.
    _MAX_SHARD_BYTES: int = 4 * 1024**3

    def __init__(self, dtype: str = "bfloat16") -> None:
        self._export_dtype: torch.dtype | None = _DTYPE_MAP.get(dtype)
        self._current_iteration: int = 0
        self._export_thread: threading.Thread | None = None
        # Stores any exception raised inside the background worker so it can be
        # re-raised on the main thread at the next checkpoint or train end.
        self._worker_exception: BaseException | None = None

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_save_checkpoint_start(self, model: Any, iteration: int = 0) -> None:
        self._current_iteration = iteration

    def on_save_checkpoint(self, model: Any, state_dict: dict[str, Any]) -> None:
        hf_cfg = self.config.checkpoint.hf_export
        if not hf_cfg.enabled:
            return

        iteration = self._current_iteration
        if iteration % hf_cfg.export_every_n != 0:
            return

        # Deferred import to avoid circular dependency at module load time.
        from cosmos3._src.vfm.models.vlm_model import VLMModel

        if not isinstance(model, VLMModel):
            # The legacy vlm/train.py path passes model_parts: list[nn.Module] (raw HF
            # models without the VLMModel attribute structure).  HF export requires the
            # VLMModel wrapper, which is only available via the unified scripts/train.py path.
            if isinstance(model, list):
                log.warning(
                    "[HFExportCallback] Received model_parts (list) instead of VLMModel. "
                    "HF export requires the unified training path (scripts/train.py). Skipping."
                )
            else:
                log.warning(
                    "[HFExportCallback] model is not VLMModel (got %s); skipping HF export.",
                    type(model).__name__,
                )
            return

        if _safetensors_save_file is None:
            raise ImportError("safetensors is required for HFExportCallback. Install it with: pip install safetensors")

        output_dir = os.path.join(self.config.job.path_local, "hf_exports", f"iter_{iteration:09d}")

        # ----------------------------------------------------------------
        # Phase 1 (all ranks): gather sharded parameters into CPU chunks.
        # full_tensor() is a collective operation — all ranks must participate.
        # ----------------------------------------------------------------
        cpu_chunks, manifest, total_size = self._gather_weights(model)

        # ----------------------------------------------------------------
        # Phase 2 (rank 0, background thread): file I/O + optional upload.
        # ----------------------------------------------------------------
        if not is_rank0():
            return

        # Block on any still-running export from the previous checkpoint and
        # propagate any worker exception before starting a new export.
        if self._export_thread is not None and self._export_thread.is_alive():
            log.warning(
                "[HFExportCallback] Previous export thread still running; waiting before starting export for iter %d.",
                iteration,
            )
            self._export_thread.join()

        if self._worker_exception is not None:
            exc = self._worker_exception
            self._worker_exception = None
            raise RuntimeError(f"[HFExportCallback] Previous export failed with: {exc}") from exc

        self._export_thread = threading.Thread(
            target=self._save_and_upload,
            args=(cpu_chunks, manifest, total_size, model.hf_config, model.model_name_or_path, output_dir, iteration),
            daemon=True,
        )
        self._export_thread.start()

    def on_train_end(self, model: Any, iteration: int = 0) -> None:
        """Wait for the final export thread so the process does not exit prematurely."""
        if self._export_thread is not None and self._export_thread.is_alive():
            log.info("[HFExportCallback] Waiting for export thread to finish...")
            self._export_thread.join()
            log.info("[HFExportCallback] Export thread done.")

        if self._worker_exception is not None:
            exc = self._worker_exception
            self._worker_exception = None
            raise RuntimeError(f"[HFExportCallback] Export thread failed with: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_weights(self, model: Any) -> tuple[list[dict[str, torch.Tensor]], dict[str, str], int]:
        """Iterate model parameters, all-gather DTensor shards, and build CPU chunks.

        Must be called on **all ranks**. Only rank 0 populates the returned
        ``cpu_chunks`` and ``manifest``; other ranks return empty structures but still
        participate in the distributed all-gathers.

        Returns:
            cpu_chunks:  List of ``{weight_name: cpu_tensor}`` dicts, one per shard file.
            manifest:    Mapping of ``weight_name → shard_filename``.
            total_size:  Total byte count of all exported tensors (for the index JSON).
        """
        cpu_chunks: list[dict[str, torch.Tensor]] = []
        manifest: dict[str, str] = {}
        current_chunk: dict[str, torch.Tensor] = {}
        current_chunk_bytes: int = 0
        total_size: int = 0
        file_idx: int = 0

        for name, param in model.model._model.named_parameters():
            # Phase 2+: HFModel initialises _model via AutoModelForImageTextToText /
            # AutoModelForCausalLM, so parameter names are HF-native and match the
            # safetensors checkpoint keys loaded by _load_vlm_weights().
            #
            # MoE note: Qwen3VLMoeTextExpertsGroupedMm stores expert weights in HF-native
            # grouped layout — gate_up_proj: [E, H, 2F], down_proj: [E, F, H] — matching
            # the checkpoint format exactly.  No transposition or per-expert fan-out is
            # needed.  (The legacy Phase 0 path stored tensors in a transposed internal
            # format [E, 2F, H] under the name "gate_and_up_projs" and required
            # weight_mapper.policy_map_local_key_for_export_tensor() to transpose back on
            # export.  Phase 2 uses HFModel and has no such internal reformat.)
            #
            # torch.compile and gradient-checkpointing wrappers inject prefixes into
            # named_parameters() output.  Strip them so exported keys are HF-native,
            # matching what HFModel._load_vlm_weights() does for the in-memory state dict.
            name = name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")

            # Gather across FSDP / TP / CP ranks (collective — all ranks must call).
            if isinstance(param, torch.distributed.tensor.DTensor):
                param = param.full_tensor()
            param = param.detach()
            if self._export_dtype is not None:
                param = param.to(dtype=self._export_dtype)

            tensor_bytes = param.element_size() * param.numel()

            # Flush the current chunk when the shard size limit would be exceeded.
            # current_chunk_bytes is tracked on ALL ranks so shard boundaries are
            # consistent (the shard_name written into manifest must agree everywhere).
            if current_chunk_bytes + tensor_bytes > self._MAX_SHARD_BYTES and current_chunk_bytes > 0:
                if is_rank0():
                    cpu_chunks.append(current_chunk)
                    current_chunk = {}
                file_idx += 1
                current_chunk_bytes = 0

            shard_name = f"{file_idx:05d}.safetensors"
            if is_rank0():
                current_chunk[name] = param.cpu()
                manifest[name] = shard_name
                total_size += tensor_bytes
            current_chunk_bytes += tensor_bytes

        # Flush the final (possibly partial) chunk.
        if current_chunk_bytes > 0 and is_rank0() and current_chunk:
            cpu_chunks.append(current_chunk)

        return cpu_chunks, manifest, total_size

    def _save_and_upload(
        self,
        cpu_chunks: list[dict[str, torch.Tensor]],
        manifest: dict[str, str],
        total_size: int,
        hf_config: Any,
        model_name_or_path: str,
        output_dir: str,
        iteration: int,
    ) -> None:
        """Write safetensors shards, HF config, tokenizer; upload to S3 / HF Hub.

        Runs on rank 0 inside a background thread.  Any exception is stored in
        ``self._worker_exception`` so the main thread can re-raise it.
        """
        try:
            self._do_save_and_upload(
                cpu_chunks, manifest, total_size, hf_config, model_name_or_path, output_dir, iteration
            )
        except Exception as exc:
            log.error(
                "[HFExportCallback] Export worker for iter %d raised an exception: %s",
                iteration,
                exc,
                exc_info=True,
            )
            self._worker_exception = exc

    def _do_save_and_upload(
        self,
        cpu_chunks: list[dict[str, torch.Tensor]],
        manifest: dict[str, str],
        total_size: int,
        hf_config: Any,
        model_name_or_path: str,
        output_dir: str,
        iteration: int,
    ) -> None:
        """Core export logic (called from the background thread via ``_save_and_upload``).

        Error handling is tiered:
        - Steps 1-4 (shards, index JSON, HF config, source-model file copy): any exception
          propagates to the outer ``_save_and_upload`` wrapper so the main thread is notified.
          A failed file copy leaves the checkpoint unusable for trust_remote_code models, so
          it is treated as a hard failure like the shard writes.
        - Steps 5-7 (tokenizer, generation_config, S3 upload, HF Hub upload): failures are
          treated as soft warnings.  The tokenizer and generation config are best-effort; upload
          failures do not invalidate the local safetensors export, so an outage must not abort
          training.
        """
        hf_cfg = self.config.checkpoint.hf_export
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"[HFExportCallback] Writing iter {iteration} export to {output_dir}")

        # 1. Safetensors shards — one file per chunk (ordered by file_idx).
        #    Each chunk is cleared after writing so its tensors can be GC'd
        #    incrementally rather than being held until the whole loop completes.
        for i in range(len(cpu_chunks)):
            chunk = cpu_chunks[i]
            shard_path = os.path.join(output_dir, f"{i:05d}.safetensors")
            _safetensors_save_file(chunk, shard_path)
            log.info(f"[HFExportCallback] Wrote {shard_path}")
            cpu_chunks[i] = {}  # release tensor references for GC

        # 2. model.safetensors.index.json
        #    total_size is pre-computed in _gather_weights to avoid needing chunks here.
        index_json = {"metadata": {"total_size": total_size}, "weight_map": manifest}
        index_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(index_path, "w") as fh:
            json.dump(index_json, fh, indent=4)

        # 3. HuggingFace model config.
        hf_config.save_pretrained(output_dir)

        # 4. Copy missing .py/.json files for trust_remote_code models.
        #    Only applicable when model_name_or_path is a local directory.
        #    The full directory layout is preserved so nested packages referenced by
        #    auto_map are included (mirroring convert_checkpoint.py's copytree approach).
        #    Files already present in the export dir (e.g., config.json written by
        #    hf_config.save_pretrained) are never overwritten.
        #    HARD failure: a broken copy leaves the checkpoint unloadable, so any I/O error
        #    propagates to the background-worker wrapper (same as shard writes).
        if model_name_or_path and os.path.isdir(model_name_or_path):
            real_src = os.path.realpath(model_name_or_path)
            real_out = os.path.realpath(output_dir)
            copied = []
            for root, dirs, files in os.walk(real_src):
                real_root = os.path.realpath(root)
                # Prune any subtree that is, leads to, or is inside output_dir.
                # This prevents recursing into previously written export dirs when
                # output_dir (or a parent of it) lives inside model_name_or_path.
                dirs[:] = [
                    d
                    for d in dirs
                    if not (
                        (p := os.path.realpath(os.path.join(real_root, d))) == real_out
                        or p.startswith(real_out + os.sep)
                        or real_out.startswith(p + os.sep)
                    )
                ]
                if real_root == real_out or real_root.startswith(real_out + os.sep):
                    continue
                rel_dir = os.path.relpath(real_root, real_src)
                for fname in files:
                    if not (fname.endswith(".py") or fname.endswith(".json")):
                        continue
                    src = os.path.join(real_root, fname)
                    dst_dir = output_dir if rel_dir == "." else os.path.join(output_dir, rel_dir)
                    dst = os.path.join(dst_dir, fname)
                    if not os.path.exists(dst):
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.copy2(src, dst)
                        copied.append(os.path.join(rel_dir, fname) if rel_dir != "." else fname)
            if copied:
                log.info(f"[HFExportCallback] Copied missing files from source model: {copied}")

        # 5. Tokenizer (best-effort — may fail for custom / gated models).
        if AutoTokenizer is not None and model_name_or_path:
            try:
                tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
                tok.save_pretrained(output_dir)
            except Exception as exc:
                log.warning(f"[HFExportCallback] Tokenizer save skipped: {exc}")

        # 6. Generation config (best-effort — not all models expose one).
        if GenerationConfig is not None and model_name_or_path:
            try:
                gen_cfg = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
                gen_cfg.save_pretrained(output_dir)
            except Exception as exc:
                log.warning(f"[HFExportCallback] generation_config save skipped: {exc}")

        # 7. S3 upload — soft failure: local export is intact regardless of upload outcome.
        obj_store = hf_cfg.upload_to_object_store
        if obj_store.enabled:
            s3_prefix = f"{self.config.job.path}/hf_exports/iter_{iteration:09d}"
            try:
                _upload_folder_to_s3(output_dir, obj_store.bucket, s3_prefix, obj_store.credentials)
                log.info(f"[HFExportCallback] S3 upload done: s3://{obj_store.bucket}/{s3_prefix}")
            except Exception as exc:
                # Intentionally soft: an upload outage must not crash training.
                log.warning(f"[HFExportCallback] S3 upload failed (local export intact): {exc}")

        # 8. HuggingFace Hub upload — soft failure: see comment above.
        if hf_cfg.hf_repo_id:
            self._upload_to_hf_hub(output_dir, hf_cfg.hf_repo_id)

        log.info(f"[HFExportCallback] Export complete for iter {iteration}.")

    @staticmethod
    def _upload_to_hf_hub(output_dir: str, repo_id: str, max_retries: int = 3) -> None:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            log.warning("[HFExportCallback] huggingface_hub not installed; skipping HF Hub upload.")
            return

        api = HfApi()
        for attempt in range(1, max_retries + 1):
            try:
                api.create_repo(repo_id=repo_id, exist_ok=True)
                break
            except Exception as exc:
                log.warning(f"[HFExportCallback] create_repo attempt {attempt}/{max_retries} failed: {exc}")
                if attempt == max_retries:
                    log.warning(
                        f"[HFExportCallback] Could not create HF Hub repo '{repo_id}' after "
                        f"{max_retries} attempts; skipping upload."
                    )
                    return

        for attempt in range(1, max_retries + 1):
            try:
                api.upload_folder(
                    folder_path=output_dir,
                    repo_id=repo_id,
                    commit_message=f"Upload checkpoint from {os.path.basename(output_dir)}",
                )
                log.info(f"[HFExportCallback] Uploaded to HF Hub: {repo_id}")
                return
            except Exception as exc:
                log.warning(f"[HFExportCallback] HF Hub upload attempt {attempt}/{max_retries} failed: {exc}")

        log.warning(f"[HFExportCallback] All {max_retries} HF Hub upload attempts failed for {repo_id}.")
