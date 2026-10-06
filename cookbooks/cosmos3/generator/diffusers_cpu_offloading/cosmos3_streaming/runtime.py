from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .attention.loader import load_attention_backend
from .block_views import (
    MEMORY_SERIAL_MLP_CHUNK_TOKENS,
    GenerationStack,
    PrefixKV,
    RotaryPair,
    UnderstandingStack,
)
from .config import StreamingConfig
from .memory_trace import CudaMemoryTrace
from .metrics import RuntimeStats
from .offloading import offload_groups, remove_group_hooks

_GIB = 2**30
_AUTO_PREFETCH_MIN_VRAM_BYTES = 11 * _GIB
_AUTO_PREFETCH_UNCONDITIONAL_VRAM_BYTES = 14 * _GIB
_AUTO_PREFETCH_LARGE_BUFFER_MIN_VRAM_BYTES = 20 * _GIB
# Three 480p Cosmos3-Nano transfer branches occupy about 490 MiB in their
# persistent generation hidden states. The 640 MiB ceiling leaves the 12 GiB
# card enough room for the additional decoder block held by prefetch, while
# keeping 720p on the lower-peak synchronous path.
_AUTO_PREFETCH_MAX_GENERATION_STATE_BYTES = 640 * 2**20
_AUTO_BRANCH_STAGING_MAX_VRAM_BYTES = 14 * _GIB
_AUTO_BRANCH_STAGING_MIN_STATE_BYTES = _AUTO_PREFETCH_MAX_GENERATION_STATE_BYTES
_TRANSFER_BRANCH_NAMES = {"cond_full", "cond_no_control", "uncond_full"}


@dataclass(slots=True)
class _BranchCache:
    und_len: int
    generation_rotary: RotaryPair
    prefixes: list[PrefixKV]


@dataclass(slots=True)
class _PreparedGeneration:
    hidden: torch.Tensor
    original_latent_shapes: list[tuple[int, int, int]]


@dataclass(slots=True)
class _ModalityPredictions:
    vision: list[torch.Tensor]
    sound: list[torch.Tensor] | None
    action: list[torch.Tensor] | None


@dataclass(frozen=True, slots=True)
class _GenerationPrefetchPolicy:
    enabled: bool
    reason: str
    branch_count: int
    generation_tokens: int
    estimated_state_bytes: int


@dataclass(frozen=True, slots=True)
class _GenerationBranchStagingPolicy:
    enabled: bool
    reason: str
    branch_count: int
    estimated_state_bytes: int


class StreamingCosmosRuntime:
    """Request-scoped Cosmos 3 understanding cache and split executor.

    Supports the coupled vision, sound, and action denoising paths. Transfer
    chunk scheduling is integrated by the modular-pipeline adapter.
    """

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.offload_device = torch.device(config.offload_device)
        self.backend = load_attention_backend(config.attention_backend, **config.attention_backend_kwargs)
        self.stats = RuntimeStats()
        self.release_cache_before_safety = False
        self.detected_vram_bytes = 0
        self.effective_use_stream = config.use_stream
        self.generation_prefetch_reason = "unresolved"
        self.generation_workload_tokens = 0
        self.estimated_generation_state_bytes = 0
        self.generation_branch_staging = False
        self.generation_branch_staging_reason = "unresolved"
        self.vae_decode_mode = "unresolved"
        self.memory_trace = (
            CudaMemoryTrace(
                config.memory_trace_path,
                self.device,
                synchronize=config.memory_trace_synchronize,
            )
            if config.memory_trace_path is not None
            else None
        )

        self._active = False
        self._transformer: torch.nn.Module | None = None
        self._understanding_stack: UnderstandingStack | None = None
        self._generation_stack: GenerationStack | None = None
        self._branch_cache: dict[str, _BranchCache] = {}
        self._cache_signature: tuple[Any, ...] | None = None
        self._understanding_signature: tuple[Any, ...] | None = None
        self._precision_signature: tuple[bool, bool] | None = None
        self._inference_tensor_versions: dict[int, tuple[torch.Tensor, torch.Tensor, int]] = {}
        self._generation_modules_resident = False
        self._decode_prepared = False
        self._installed_generation_use_stream: bool | None = None
        self._request_prefetch_signature: tuple[Any, ...] | None = None
        self._request_branch_staging_signature: tuple[Any, ...] | None = None

    def begin_request(self) -> None:
        if self._active:
            raise RuntimeError("StreamingCosmosRuntime is not reentrant")
        self._active = True
        self._precision_signature = None
        self._decode_prepared = False
        self._branch_cache.clear()
        self._inference_tensor_versions.clear()
        self._cache_signature = None
        self._understanding_signature = None
        self._request_prefetch_signature = None
        self._request_branch_staging_signature = None
        self.stats = RuntimeStats(
            understanding_weight_bytes=self.stats.understanding_weight_bytes,
            generation_weight_bytes=self.stats.generation_weight_bytes,
        )
        if self.memory_trace is not None:
            try:
                self.memory_trace.begin()
            except BaseException:
                self._active = False
                raise
            self.trace_checkpoint(
                "memory_policy",
                configured_profile=self.config.memory_profile,
                detected_vram_bytes=self.detected_vram_bytes,
                vae_decode_mode=self.vae_decode_mode,
                generation_use_stream=self.effective_use_stream,
            )

    def resolve_memory_policy(self, total_vram_bytes: int) -> None:
        """Resolve backend-specific internals from the intent-level profile."""

        self.detected_vram_bytes = total_vram_bytes
        uses_large_buffers = bool(getattr(self.backend, "retains_large_device_buffers", False))
        sub_14gb_auto = self.config.memory_profile == "auto" and total_vram_bytes < 14 * 2**30
        constrained_auto = (
            self.config.memory_profile == "auto" and total_vram_bytes < _AUTO_PREFETCH_LARGE_BUFFER_MIN_VRAM_BYTES
        )
        self.effective_use_stream = self.config.use_stream and not (
            sub_14gb_auto or (uses_large_buffers and constrained_auto)
        )

    def end_request(self, *, failed: bool = False) -> None:
        try:
            self._branch_cache.clear()
            self._inference_tensor_versions.clear()
            self._cache_signature = None
            self._understanding_signature = None
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)
            for stack in (self._understanding_stack, self._generation_stack):
                if stack is not None and self.config.enable_group_offload:
                    offload_groups(stack)
                    if failed:
                        remove_group_hooks(stack)
            self._offload_generation_modules()
            if failed and self._transformer is not None:
                self._transformer.to(self.offload_device)
                # A failed first forward may leave a partial lazy-prefetch
                # trace. Rebuild views/hooks on retry instead of reusing it.
                self._transformer = None
                self._understanding_stack = None
                self._generation_stack = None
                self._installed_generation_use_stream = None
            release_buffers = getattr(self.backend, "release_buffers", None)
            if release_buffers is not None and not self._decode_prepared:
                release_buffers()
            if self.device.type == "cuda" and torch.cuda.is_available() and self.config.empty_cuda_cache_on_end:
                torch.cuda.empty_cache()
        finally:
            try:
                if self.memory_trace is not None:
                    self.memory_trace.end()
            finally:
                self._active = False

    def prepare_for_decode(self) -> None:
        """Release denoising-only state before VAE decode begins."""

        if self._decode_prepared:
            return
        self.trace_checkpoint("denoising_resources.release.start")
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self._branch_cache.clear()
        self._inference_tensor_versions.clear()
        self._cache_signature = None
        self._understanding_signature = None
        self._offload_generation_modules()
        release_buffers = getattr(self.backend, "release_buffers", None)
        if release_buffers is not None:
            release_buffers()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._decode_prepared = True
        self.trace_checkpoint("denoising_resources.release.end")

    @torch.no_grad()
    def predict_modalities(
        self,
        transformer: torch.nn.Module,
        branches: dict[str, dict[str, Any]],
    ) -> dict[str, _ModalityPredictions]:
        """Predict per-branch vision, sound, and action velocities."""

        if not self._active:
            raise RuntimeError("begin_request() must be called before predict_modalities()")
        if not branches:
            raise ValueError("at least one prediction branch is required")
        # Transfer decodes every autoregressive chunk. A new denoising phase
        # after that decode must be allowed to release its own caches again.
        if self._decode_prepared:
            self._decode_prepared = False

        self._bind_transformer(transformer, branches)
        self._invalidate_changed_precision(transformer)
        self.stats.denoising_calls += 1
        denoising_call = self.stats.denoising_calls - 1
        self.trace_checkpoint("transformer.start", step=denoising_call)
        self.stats.estimated_block_h2d_bytes += self.stats.generation_weight_bytes
        signature = self._static_signature(branches)
        if not self._branch_cache:
            self._prefill_understanding(branches)
            self._cache_signature = signature
        elif signature != self._cache_signature:
            understanding_signature = self._static_understanding_signature(branches)
            if understanding_signature == self._understanding_signature and set(branches) == set(self._branch_cache):
                self._refresh_generation_rotaries(branches)
            else:
                self._branch_cache.clear()
                self._prefill_understanding(branches)
            self._cache_signature = signature

        self._onload_generation_modules()
        branch_policy = self._resolve_generation_branch_staging_policy(transformer, branches)
        self._record_generation_branch_staging_policy(branch_policy)
        prepared: dict[str, _PreparedGeneration] = {}
        for name, kwargs in branches.items():
            item = self._prepare_generation(kwargs)
            if branch_policy.enabled:
                item.hidden = self._stage_hidden_on_host(item.hidden)
            prepared[name] = item
        if branch_policy.enabled and self.device.type == "cuda":
            # Preparation uses differently sized temporary tensors. Return those
            # cached segments before the first large MLP allocation so the
            # serial branch working set starts from a compact allocator state.
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        branch_names = list(branches)
        first_name = branch_names[0]
        second_name = branch_names[1] if len(branch_names) > 1 else None
        extra_names = branch_names[2:]
        first = prepared[first_name]
        second = None if second_name is None else prepared[second_name]
        first_cache = self._branch_cache[first_name]
        second_cache = None if second_name is None else self._branch_cache[second_name]
        assert self._generation_stack is not None
        if extra_names:
            first_hidden, second_hidden, extra_hiddens = self._generation_stack(
                first.hidden,
                first_cache.generation_rotary,
                first_cache.prefixes,
                None if second is None else second.hidden,
                None if second_cache is None else second_cache.generation_rotary,
                None if second_cache is None else second_cache.prefixes,
                [prepared[name].hidden for name in extra_names],
                [self._branch_cache[name].generation_rotary for name in extra_names],
                [self._branch_cache[name].prefixes for name in extra_names],
                stage_branches=branch_policy.enabled,
                execution_device=self.device,
                offload_device=self.offload_device,
            )
        else:
            first_hidden, second_hidden = self._generation_stack(
                first.hidden,
                first_cache.generation_rotary,
                first_cache.prefixes,
                None if second is None else second.hidden,
                None if second_cache is None else second_cache.generation_rotary,
                None if second_cache is None else second_cache.prefixes,
                stage_branches=branch_policy.enabled,
                execution_device=self.device,
                offload_device=self.offload_device,
            )
            extra_hiddens = []

        outputs = {
            first_name: self._decode_predictions(first_hidden, branches[first_name], first.original_latent_shapes)
        }
        if second_name is not None:
            if second_hidden is None or second is None:
                raise RuntimeError("generation stack did not return its second branch")
            outputs[second_name] = self._decode_predictions(
                second_hidden,
                branches[second_name],
                second.original_latent_shapes,
            )
        for name, hidden in zip(extra_names, extra_hiddens):
            outputs[name] = self._decode_predictions(
                hidden,
                branches[name],
                prepared[name].original_latent_shapes,
            )
        self.trace_checkpoint("transformer.end", step=denoising_call)
        return outputs

    def predict_vision(
        self,
        transformer: torch.nn.Module,
        branches: dict[str, dict[str, Any]],
    ) -> dict[str, list[torch.Tensor]]:
        """Compatibility wrapper for callers that only consume vision."""

        return {name: prediction.vision for name, prediction in self.predict_modalities(transformer, branches).items()}

    def _invalidate_changed_precision(self, transformer: torch.nn.Module) -> None:
        from diffusers.pipelines.cosmos import mixed_precision as mixed

        native = getattr(transformer, mixed._RUNTIME_ATTRIBUTE, None)
        signature = (
            (native.use_high_precision("reasoner"), native.use_high_precision("generation"))
            if native is not None and native.active
            else None
        )
        if signature != self._precision_signature:
            # Native dispatch may change understanding projections at a step
            # boundary. K/V computed with the previous precision cannot be reused.
            self._branch_cache.clear()
            self._cache_signature = None
            self._understanding_signature = None
            self._precision_signature = signature

    def _bind_transformer(
        self,
        transformer: torch.nn.Module,
        branches: dict[str, dict[str, Any]],
    ) -> None:
        policy = self._resolve_generation_prefetch_policy(transformer, branches)
        if self._transformer is transformer:
            self._use_installed_prefetch_policy(policy)
            return
        if self._transformer is not None:
            raise RuntimeError("a StreamingCosmosRuntime cannot be rebound to another transformer")

        self._transformer = transformer
        block_callback = self._trace_block if self.config.memory_trace_blocks else None
        self._understanding_stack = UnderstandingStack(transformer.layers, self.backend, block_callback)
        self._generation_stack = GenerationStack(transformer.layers, self.backend, block_callback)
        self.stats.understanding_weight_bytes = self._parameter_bytes(self._understanding_stack)
        self.stats.generation_weight_bytes = self._parameter_bytes(self._generation_stack)
        self.effective_use_stream = policy.enabled
        self._installed_generation_use_stream = policy.enabled if self.config.enable_group_offload else False
        self._record_generation_prefetch_policy(policy, actual_enabled=self.effective_use_stream)

        if self.config.enable_group_offload:
            from diffusers.hooks import apply_group_offloading

            common = {
                "onload_device": self.device,
                "offload_device": self.offload_device,
                "offload_type": "block_level",
                "num_blocks_per_group": 1,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            }
            # Understanding is invoked once, so Diffusers' lazy streamed trace
            # cannot benefit it. Synchronous block offload avoids a useless trace.
            apply_group_offloading(
                self._understanding_stack,
                use_stream=False,
                record_stream=False,
                **common,
            )
            self._install_generation_offload(self.effective_use_stream)
        else:
            self._understanding_stack.to(self.device)
            self._generation_stack.to(self.device)

    def _resolve_generation_prefetch_policy(
        self,
        transformer: torch.nn.Module,
        branches: dict[str, dict[str, Any]],
    ) -> _GenerationPrefetchPolicy:
        """Select async generation-block prefetch from the actual request shape.

        VAE tiling has a different lifetime and is resolved independently by
        the modular-pipeline adapter. This policy only budgets the extra model
        block that Diffusers keeps on the GPU while the current block computes.
        """

        branch_count = self._prefetch_branch_count(branches)
        generation_tokens = self._generation_token_count(branches, branch_count)
        # The official FP8 checkpoint keeps the persistent hidden state in
        # BF16 and quantizes Linear inputs internally. Using FP8 weight size
        # here would under-budget the activation state and could select an
        # unsafe prefetch policy on a 12 GiB card.
        dtype_bytes = self._generation_activation_element_size(transformer, branches)
        hidden_size = int(transformer.config.hidden_size)
        estimated_state_bytes = generation_tokens * hidden_size * dtype_bytes

        common = {
            "branch_count": branch_count,
            "generation_tokens": generation_tokens,
            "estimated_state_bytes": estimated_state_bytes,
        }
        if not self.config.enable_group_offload:
            return _GenerationPrefetchPolicy(False, "group_offload_disabled", **common)
        if not self.config.use_stream:
            return _GenerationPrefetchPolicy(False, "streaming_disabled", **common)
        if self.config.memory_profile == "low_memory":
            return _GenerationPrefetchPolicy(False, "low_memory_profile", **common)
        if self.config.memory_profile == "performance":
            return _GenerationPrefetchPolicy(True, "performance_profile", **common)

        uses_large_buffers = bool(getattr(self.backend, "retains_large_device_buffers", False))
        if uses_large_buffers and self.detected_vram_bytes < _AUTO_PREFETCH_LARGE_BUFFER_MIN_VRAM_BYTES:
            return _GenerationPrefetchPolicy(False, "attention_backend_reserve", **common)
        if self.detected_vram_bytes == 0:
            return _GenerationPrefetchPolicy(self.config.use_stream, "vram_unknown", **common)
        if self.detected_vram_bytes >= _AUTO_PREFETCH_UNCONDITIONAL_VRAM_BYTES:
            return _GenerationPrefetchPolicy(True, "vram_headroom", **common)
        if self.detected_vram_bytes < _AUTO_PREFETCH_MIN_VRAM_BYTES:
            return _GenerationPrefetchPolicy(False, "insufficient_vram", **common)
        if estimated_state_bytes <= _AUTO_PREFETCH_MAX_GENERATION_STATE_BYTES:
            return _GenerationPrefetchPolicy(True, "workload_headroom", **common)
        return _GenerationPrefetchPolicy(False, "workload_too_large", **common)

    def _resolve_generation_branch_staging_policy(
        self,
        transformer: torch.nn.Module,
        branches: dict[str, dict[str, Any]],
    ) -> _GenerationBranchStagingPolicy:
        """Select host staging for Transfer branch hidden states.

        Transfer can evaluate as many as three guidance branches. The decoder
        block already computes them serially while its weights are resident,
        but the normal path retains every branch hidden state on CUDA. On a
        12 GiB card at 720p those persistent BF16 states consume enough memory
        to prevent the generation MLP from allocating its temporary output.
        """

        branch_count = self._prefetch_branch_count(branches)
        generation_tokens = self._generation_token_count(branches, branch_count)
        dtype_bytes = self._generation_activation_element_size(transformer, branches)
        estimated_state_bytes = generation_tokens * int(transformer.config.hidden_size) * dtype_bytes
        common = {"branch_count": branch_count, "estimated_state_bytes": estimated_state_bytes}

        is_transfer = bool(_TRANSFER_BRANCH_NAMES.intersection(branches))
        if not is_transfer:
            return _GenerationBranchStagingPolicy(False, "not_transfer", **common)
        if len(branches) <= 1:
            return _GenerationBranchStagingPolicy(False, "single_active_branch", **common)
        if self.config.memory_profile == "performance":
            return _GenerationBranchStagingPolicy(False, "performance_profile", **common)
        if self.config.memory_profile == "low_memory":
            return _GenerationBranchStagingPolicy(True, "low_memory_profile", **common)
        if self.detected_vram_bytes == 0:
            return _GenerationBranchStagingPolicy(False, "vram_unknown", **common)
        if self.detected_vram_bytes >= _AUTO_BRANCH_STAGING_MAX_VRAM_BYTES:
            return _GenerationBranchStagingPolicy(False, "vram_headroom", **common)
        if estimated_state_bytes <= _AUTO_BRANCH_STAGING_MIN_STATE_BYTES:
            return _GenerationBranchStagingPolicy(False, "workload_headroom", **common)
        return _GenerationBranchStagingPolicy(True, "workload_pressure", **common)

    def _record_generation_branch_staging_policy(self, policy: _GenerationBranchStagingPolicy) -> None:
        signature = (policy.enabled, policy.reason, policy.branch_count, policy.estimated_state_bytes)
        if signature == self._request_branch_staging_signature:
            return
        self._request_branch_staging_signature = signature
        self.generation_branch_staging = policy.enabled
        self.generation_branch_staging_reason = policy.reason
        self.trace_checkpoint(
            "generation_branch_staging_policy",
            enabled=policy.enabled,
            reason=policy.reason,
            branch_count=policy.branch_count,
            estimated_generation_state_bytes=policy.estimated_state_bytes,
            mlp_chunk_tokens=MEMORY_SERIAL_MLP_CHUNK_TOKENS if policy.enabled else None,
            detected_vram_bytes=self.detected_vram_bytes,
        )

    @staticmethod
    def _prefetch_branch_count(branches: dict[str, dict[str, Any]]) -> int:
        # Transfer guidance intervals can change which branches execute during
        # a request. Budget all three branches up front so a later interval does
        # not invalidate hooks installed at the first denoising step.
        if _TRANSFER_BRANCH_NAMES.intersection(branches):
            return 3
        return len(branches)

    @staticmethod
    def _generation_token_count(branches: dict[str, dict[str, Any]], branch_count: int) -> int:
        lengths = [max(0, int(kwargs["sequence_length"]) - int(kwargs["und_len"])) for kwargs in branches.values()]
        if not lengths:
            return 0
        if branch_count == len(lengths):
            return sum(lengths)
        return max(lengths) * branch_count

    @staticmethod
    def _generation_activation_element_size(
        transformer: torch.nn.Module,
        branches: dict[str, dict[str, Any]],
    ) -> int:
        for kwargs in branches.values():
            for field in ("vision_tokens", "sound_tokens", "action_tokens"):
                tensors = kwargs.get(field)
                if tensors:
                    return tensors[0].element_size()
        config_dtype = getattr(transformer.config, "dtype", "bfloat16")
        if isinstance(config_dtype, torch.dtype):
            dtype = config_dtype
        else:
            dtype = getattr(torch, str(config_dtype).removeprefix("torch."), torch.bfloat16)
        return torch.tensor([], dtype=dtype).element_size()

    def _use_installed_prefetch_policy(self, policy: _GenerationPrefetchPolicy) -> None:
        installed = self._installed_generation_use_stream
        if installed is None:
            raise RuntimeError("generation offload hooks were not initialized")
        if installed != policy.enabled and self.config.enable_group_offload:
            # No generation forward is active here. Drain the previous stream
            # before replacing its hooks and pinned CPU backing stores. A new
            # async policy gets a fresh lazy execution trace on its first call.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            stack = self._require_generation_stack()
            offload_groups(stack)
            remove_group_hooks(stack)
            self._install_generation_offload(policy.enabled)
            self._installed_generation_use_stream = policy.enabled
        self.effective_use_stream = policy.enabled
        self._record_generation_prefetch_policy(policy, actual_enabled=policy.enabled)

    def _install_generation_offload(self, use_stream: bool) -> None:
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(
            self._require_generation_stack(),
            onload_device=self.device,
            offload_device=self.offload_device,
            offload_type="block_level",
            num_blocks_per_group=1,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            use_stream=use_stream,
            record_stream=self.config.record_stream and use_stream,
        )

    def _require_generation_stack(self) -> GenerationStack:
        if self._generation_stack is None:
            raise RuntimeError("generation stack is not initialized")
        return self._generation_stack

    def _record_generation_prefetch_policy(
        self,
        policy: _GenerationPrefetchPolicy,
        *,
        actual_enabled: bool,
    ) -> None:
        signature = (
            actual_enabled,
            policy.reason,
            policy.branch_count,
            policy.generation_tokens,
            policy.estimated_state_bytes,
        )
        if signature == self._request_prefetch_signature:
            return
        self._request_prefetch_signature = signature
        self.generation_prefetch_reason = policy.reason
        self.generation_workload_tokens = policy.generation_tokens
        self.estimated_generation_state_bytes = policy.estimated_state_bytes
        self.trace_checkpoint(
            "generation_prefetch_policy",
            generation_use_stream=actual_enabled,
            reason=policy.reason,
            branch_count=policy.branch_count,
            generation_tokens=policy.generation_tokens,
            estimated_generation_state_bytes=policy.estimated_state_bytes,
            max_auto_generation_state_bytes=_AUTO_PREFETCH_MAX_GENERATION_STATE_BYTES,
            detected_vram_bytes=self.detected_vram_bytes,
        )

    def _tensor_signature(self, tensor: torch.Tensor) -> tuple[Any, ...]:
        if not torch.is_inference(tensor):
            return (id(tensor), tensor._version, tuple(tensor.shape))
        # Inference tensors have no version counter. Keep a request-local
        # snapshot so in-place edits still invalidate cached conditioning.
        # Only this fallback needs a content comparison (and potentially a sync).
        previous = self._inference_tensor_versions.get(id(tensor))
        revision = 0 if previous is None else previous[2]
        if previous is None or not torch.equal(tensor, previous[1]):
            revision += 1
            with torch.inference_mode(False):
                snapshot = tensor.detach().clone()
            self._inference_tensor_versions[id(tensor)] = (tensor, snapshot, revision)
        return (id(tensor), revision, tuple(tensor.shape))

    def _static_signature(self, branches: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
        """Cheap identity signature for modular pipeline fields that must stay static.

        The official Cosmos modular workflow reuses these tensor objects through
        the denoising loop. Avoiding a content hash prevents a GPU-to-CPU sync on
        every diffusion step.
        """

        signature: list[Any] = []
        for name, kwargs in branches.items():
            signature.extend((name, int(kwargs["und_len"]), int(kwargs["sequence_length"])))
            for field in ("input_ids", "text_indexes", "position_ids"):
                tensor = kwargs[field]
                signature.append((field, self._tensor_signature(tensor)))
        return tuple(signature)

    def _static_understanding_signature(self, branches: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
        signature: list[Any] = []
        for name, kwargs in branches.items():
            signature.extend((name, int(kwargs["und_len"])))
            for field in ("input_ids", "text_indexes"):
                tensor = kwargs[field]
                signature.append((field, self._tensor_signature(tensor)))
            signature.append(self._understanding_positions_key(kwargs))
        return tuple(signature)

    @staticmethod
    def _understanding_positions_key(kwargs: dict[str, Any]) -> tuple[Any, ...]:
        # Compare prefix contents, not the full position tensor's version:
        # Transfer may change generation positions without changing its prompt.
        # Called only during prefill or when the cheap static signature changes,
        # never on the ordinary unchanged-tensor denoising path.
        prefix = kwargs["position_ids"][..., : int(kwargs["und_len"])]
        return (tuple(prefix.shape), tuple(prefix.reshape(-1).tolist()))

    def _prefill_understanding(self, branches: dict[str, dict[str, Any]]) -> None:
        transformer = self._require_transformer()
        understanding_stack = self._require_understanding_stack()
        self.stats.understanding_prefills += 1
        understanding_groups: dict[tuple[Any, ...], list[str]] = {}
        for name, kwargs in branches.items():
            key = self._understanding_branch_key(kwargs)
            understanding_groups.setdefault(key, []).append(name)
        self.stats.estimated_block_h2d_bytes += self.stats.understanding_weight_bytes * (
            (len(understanding_groups) + 1) // 2
        )
        self.trace_checkpoint("understanding_prefill.start")

        embed_tokens = transformer.embed_tokens
        embed_tokens.to(self.device)
        try:
            group_items = list(understanding_groups.items())
            prefixes_by_key: dict[tuple[Any, ...], list[PrefixKV]] = {}
            for group_index in range(0, len(group_items), 2):
                first_key, first_names = group_items[group_index]
                first_kwargs = branches[first_names[0]]
                first_hidden = self._prepare_understanding_hidden(first_kwargs)
                first_rotary, _ = self._compute_rotary(first_kwargs)

                second_key = None
                second_hidden = None
                second_rotary = None
                if group_index + 1 < len(group_items):
                    second_key, second_names = group_items[group_index + 1]
                    second_kwargs = branches[second_names[0]]
                    second_hidden = self._prepare_understanding_hidden(second_kwargs)
                    second_rotary, _ = self._compute_rotary(second_kwargs)

                first_prefixes, second_prefixes = understanding_stack(
                    first_hidden,
                    first_rotary,
                    second_hidden,
                    second_rotary,
                )
                prefixes_by_key[first_key] = first_prefixes
                if second_key is not None:
                    if second_prefixes is None:
                        raise RuntimeError("understanding stack did not produce its second branch cache")
                    prefixes_by_key[second_key] = second_prefixes
        finally:
            embed_tokens.to(self.offload_device)

        for name, kwargs in branches.items():
            _, generation_rotary = self._compute_rotary(kwargs)
            self._branch_cache[name] = _BranchCache(
                und_len=int(kwargs["und_len"]),
                generation_rotary=generation_rotary,
                prefixes=prefixes_by_key[self._understanding_branch_key(kwargs)],
            )
        self._understanding_signature = self._static_understanding_signature(branches)
        self.trace_checkpoint("understanding_prefill.end")

    def _understanding_branch_key(self, kwargs: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(kwargs["und_len"]),
            self._tensor_signature(kwargs["input_ids"]),
            self._tensor_signature(kwargs["text_indexes"]),
            self._understanding_positions_key(kwargs),
        )

    def _refresh_generation_rotaries(self, branches: dict[str, dict[str, Any]]) -> None:
        for name, kwargs in branches.items():
            _, generation_rotary = self._compute_rotary(kwargs)
            cache = self._branch_cache[name]
            cache.generation_rotary = generation_rotary

    def trace_checkpoint(self, event: str, **fields: Any) -> None:
        if self.memory_trace is not None:
            self.memory_trace.checkpoint(event, **fields)

    def _trace_block(self, tower: str, block: int) -> None:
        self.trace_checkpoint(
            f"{tower}.block.end",
            step=max(self.stats.denoising_calls - 1, 0),
            block=block,
        )

    def _prepare_understanding_hidden(self, kwargs: dict[str, Any]) -> torch.Tensor:
        transformer = self._require_transformer()
        input_ids = kwargs["input_ids"].to(self.device)
        text_indexes = kwargs["text_indexes"].to(self.device)
        und_len = int(kwargs["und_len"])

        embeddings = transformer.embed_tokens(input_ids)
        hidden = embeddings.new_zeros((und_len, transformer.config.hidden_size))
        if text_indexes.numel() and (text_indexes.min() < 0 or text_indexes.max() >= und_len):
            raise ValueError("text indexes must address only the understanding prefix")
        hidden[text_indexes] = embeddings
        return hidden

    def _compute_rotary(self, kwargs: dict[str, Any]) -> tuple[RotaryPair, RotaryPair]:
        transformer = self._require_transformer()
        position_ids = kwargs["position_ids"].to(self.device)
        und_len = int(kwargs["und_len"])

        # Keep this shape handling identical to Cosmos3OmniTransformer.forward.
        rotary_position_ids = position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1)
        parameter = next(transformer.embed_tokens.parameters())
        cos, sin = transformer.rotary_emb(
            position_ids=rotary_position_ids,
            device=self.device,
            dtype=parameter.dtype,
        )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        return (cos[:und_len], sin[:und_len]), (cos[und_len:], sin[und_len:])

    def _prepare_generation(self, kwargs: dict[str, Any]) -> _PreparedGeneration:
        transformer = self._require_transformer()
        vision_tokens = [tensor.to(device=self.device, dtype=transformer.dtype) for tensor in kwargs["vision_tokens"]]
        packed, original_shapes = transformer._patchify_and_pack_latents(vision_tokens)
        packed = transformer.proj_in(packed)

        timesteps = kwargs["vision_timesteps"].to(self.device) * transformer.config.timestep_scale
        time_dtype = next(transformer.time_embedder.parameters()).dtype
        timestep_embeddings = transformer.time_embedder(transformer.time_proj(timesteps).to(time_dtype))
        timestep_embeddings = timestep_embeddings.to(packed.dtype)
        packed = transformer._apply_timestep_embeds_to_noisy_tokens(
            packed_tokens=packed,
            packed_timestep_embeds=timestep_embeddings,
            noisy_frame_indexes=kwargs["vision_noisy_frame_indexes"],
            token_shapes=kwargs["vision_token_shapes"],
        )

        und_len = int(kwargs["und_len"])
        generation_length = int(kwargs["sequence_length"]) - und_len
        hidden = packed.new_zeros((generation_length, transformer.config.hidden_size))
        local_indexes = kwargs["vision_sequence_indexes"].to(self.device) - und_len
        if local_indexes.numel() and (local_indexes.min() < 0 or local_indexes.max() >= generation_length):
            raise ValueError("vision indexes must address only the generation suffix")
        hidden[local_indexes] = packed

        has_sound = "sound_tokens" in kwargs
        if has_sound:
            required_sound = {
                "sound_token_shapes",
                "sound_sequence_indexes",
                "sound_timesteps",
                "sound_noisy_frame_indexes",
            }
            missing_sound = sorted(required_sound - kwargs.keys())
            if missing_sound:
                raise KeyError(f"missing Cosmos sound transformer inputs: {missing_sound}")
            if not hasattr(transformer, "audio_proj_in"):
                raise RuntimeError("the loaded Cosmos transformer does not include the sound projection heads")

            sound_tokens = [
                tensor.to(device=self.device, dtype=transformer.dtype) for tensor in kwargs["sound_tokens"]
            ]
            packed_sound = transformer._pack_sound_latents(sound_tokens, kwargs["sound_token_shapes"])
            packed_sound = transformer.audio_proj_in(packed_sound)
            packed_sound = packed_sound + transformer.audio_modality_embed.to(
                device=self.device, dtype=packed_sound.dtype
            )
            sound_timesteps = kwargs["sound_timesteps"].to(self.device) * transformer.config.timestep_scale
            sound_timestep_embeddings = transformer.time_embedder(
                transformer.time_proj(sound_timesteps).to(time_dtype)
            ).to(packed_sound.dtype)
            packed_sound = transformer._apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_sound,
                packed_timestep_embeds=sound_timestep_embeddings,
                noisy_frame_indexes=kwargs["sound_noisy_frame_indexes"],
                token_shapes=kwargs["sound_token_shapes"],
            )
            sound_indexes = kwargs["sound_sequence_indexes"].to(self.device) - und_len
            if sound_indexes.numel() and (sound_indexes.min() < 0 or sound_indexes.max() >= generation_length):
                raise ValueError("sound indexes must address only the generation suffix")
            hidden[sound_indexes] = packed_sound

        has_action = "action_tokens" in kwargs
        if has_action:
            required_action = {
                "action_token_shapes",
                "action_sequence_indexes",
                "action_mse_loss_indexes",
                "action_timesteps",
                "action_noisy_frame_indexes",
                "action_domain_ids",
            }
            missing_action = sorted(required_action - kwargs.keys())
            if missing_action:
                raise KeyError(f"missing Cosmos action transformer inputs: {missing_action}")
            if not hasattr(transformer, "action_proj_in"):
                raise RuntimeError("the loaded Cosmos transformer does not include the action projection heads")

            action_tokens = [
                tensor.to(device=self.device, dtype=transformer.dtype) for tensor in kwargs["action_tokens"]
            ]
            packed_action, action_domain_ids = transformer._pack_action_latents(
                action_tokens,
                kwargs["action_token_shapes"],
                kwargs["action_domain_ids"],
            )
            action_domain_ids = action_domain_ids.to(self.device)
            packed_action = transformer.action_proj_in(packed_action, action_domain_ids)
            packed_action = packed_action + transformer.action_modality_embed.to(
                device=self.device, dtype=packed_action.dtype
            )
            if kwargs["action_mse_loss_indexes"].numel() > 0:
                action_timesteps = kwargs["action_timesteps"].to(self.device) * transformer.config.timestep_scale
                action_timestep_embeddings = transformer.time_embedder(
                    transformer.time_proj(action_timesteps).to(time_dtype)
                ).to(packed_action.dtype)
                packed_action = transformer._apply_timestep_embeds_to_noisy_tokens(
                    packed_tokens=packed_action,
                    packed_timestep_embeds=action_timestep_embeddings,
                    noisy_frame_indexes=kwargs["action_noisy_frame_indexes"],
                    token_shapes=kwargs["action_token_shapes"],
                )
            action_indexes = kwargs["action_sequence_indexes"].to(self.device) - und_len
            if action_indexes.numel() and (action_indexes.min() < 0 or action_indexes.max() >= generation_length):
                raise ValueError("action indexes must address only the generation suffix")
            hidden[action_indexes] = packed_action
        return _PreparedGeneration(hidden=hidden, original_latent_shapes=original_shapes)

    def _stage_hidden_on_host(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.device == self.offload_device:
            return hidden
        if self.offload_device.type == "cpu" and self.device.type == "cuda":
            staged = torch.empty_like(hidden, device=self.offload_device, pin_memory=True)
            staged.copy_(hidden, non_blocking=True)
            return staged
        return hidden.to(self.offload_device)

    @staticmethod
    def _parameter_bytes(module: torch.nn.Module) -> int:
        unique: dict[int, torch.nn.Parameter] = {id(parameter): parameter for parameter in module.parameters()}
        return sum(parameter.numel() * parameter.element_size() for parameter in unique.values())

    def _decode_predictions(
        self,
        hidden: torch.Tensor,
        kwargs: dict[str, Any],
        original_latent_shapes: list[tuple[int, int, int]],
    ) -> _ModalityPredictions:
        transformer = self._require_transformer()
        if hidden.device != self.device:
            hidden = hidden.to(
                self.device,
                non_blocking=hidden.device.type == "cpu" and hidden.is_pinned(),
            )
        hidden = transformer.norm_moe_gen(hidden)
        indexes = kwargs["vision_mse_loss_indexes"].to(self.device) - int(kwargs["und_len"])
        packed_predictions = transformer.proj_out(hidden[indexes])
        vision = transformer._unpatchify_and_unpack_latents(
            packed_predictions,
            token_shapes_vision=kwargs["vision_token_shapes"],
            noisy_frame_indexes_vision=kwargs["vision_noisy_frame_indexes"],
            original_latent_shapes=original_latent_shapes,
        )
        sound = None
        if "sound_tokens" in kwargs:
            indexes = kwargs["sound_mse_loss_indexes"].to(self.device) - int(kwargs["und_len"])
            packed_sound_predictions = transformer.audio_proj_out(hidden[indexes])
            sound = transformer._unpack_sound_latents(
                packed_sound_predictions,
                kwargs["sound_token_shapes"],
                kwargs["sound_noisy_frame_indexes"],
            )
        action = None
        if "action_tokens" in kwargs:
            per_noisy_domain_ids = [
                domain_id.reshape(1).expand(len(noisy_indexes))
                for domain_id, noisy_indexes in zip(
                    kwargs["action_domain_ids"], kwargs["action_noisy_frame_indexes"]
                )
            ]
            per_noisy_domain_ids = torch.cat(per_noisy_domain_ids).to(self.device)
            indexes = kwargs["action_mse_loss_indexes"].to(self.device) - int(kwargs["und_len"])
            packed_action_predictions = transformer.action_proj_out(hidden[indexes], per_noisy_domain_ids)
            action = transformer._unpack_action_latents(
                packed_action_predictions,
                kwargs["action_token_shapes"],
                kwargs["action_noisy_frame_indexes"],
            )
        return _ModalityPredictions(vision=vision, sound=sound, action=action)

    def _onload_generation_modules(self) -> None:
        if self._generation_modules_resident:
            return
        transformer = self._require_transformer()
        for module in self._generation_modules(transformer):
            module.to(self.device)
        self._generation_modules_resident = True

    def _offload_generation_modules(self) -> None:
        # A module.to() can fail partway through onload before the resident
        # flag is set. Always visit the small generation heads during cleanup.
        if self._transformer is None:
            return
        for module in self._generation_modules(self._transformer):
            module.to(self.offload_device)
        self._generation_modules_resident = False

    @staticmethod
    def _generation_modules(transformer: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
        modules = (
            transformer.proj_in,
            transformer.time_proj,
            transformer.time_embedder,
            transformer.norm_moe_gen,
            transformer.proj_out,
        )
        if hasattr(transformer, "audio_proj_in"):
            modules += (transformer.audio_proj_in, transformer.audio_proj_out)
        if hasattr(transformer, "action_proj_in"):
            modules += (transformer.action_proj_in, transformer.action_proj_out)
        return modules

    def _require_transformer(self) -> torch.nn.Module:
        if self._transformer is None:
            raise RuntimeError("runtime has not been bound to a transformer")
        return self._transformer

    def _require_understanding_stack(self) -> UnderstandingStack:
        if self._understanding_stack is None:
            raise RuntimeError("runtime has not created the understanding stack")
        return self._understanding_stack
