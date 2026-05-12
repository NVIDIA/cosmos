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

"""Abstract interfaces for persistent memory in the MoT transformer stack.

``MemoryState`` is a *mutable* Python object that lives **outside** the
``torch.compile`` boundary.  It is responsible for reading cached tensors
(``read_for_layer``) and writing new tensors back (``write_for_layer``).

``MemoryValue`` is a *read-only* tensor container that is safe to pass
**into** a compiled decoder layer.  Concrete implementations are plain
dataclasses whose fields are tensors (or None).  No methods on
``MemoryValue`` should mutate state.

``KVToStore`` is a type alias for the 4-tuple of tensors
``(gen_k, gen_v, und_k, und_v)`` returned by each compiled layer so
the caller can write them back into the cache outside the compile boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

# (gen_k, gen_v, und_k, und_v) returned by each compiled layer for the caller
# to write back into the cache outside the torch.compile boundary.
KVToStore = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class MemoryValue(ABC):
    """Read-only tensor container safe to pass into ``torch.compile``.

    Concrete subclasses (e.g. ``ARMemoryValue``, ``KVTrainMemoryValue``)
    are plain dataclasses of tensors.  No methods on this class should
    mutate state or perform non-trivial computation.
    """

    @property
    def supports_context_parallel_attention(self) -> bool:
        """Whether this memory value is compatible with context-parallel attention.

        Overridden by ``KVTrainMemoryValue`` to return ``False``.  Used by
        ``ContextParallelDispatch`` to reject an unsupported combination
        without importing the concrete subclass.
        """
        return True


class MemoryState(ABC):
    """Mutable persistent memory that lives outside ``torch.compile``.

    The outer loop in ``_impl_forward`` calls ``read_for_layer`` before
    each decoder layer and ``write_for_layer`` after.  The ``MemoryState``
    object itself is **never** passed into a compiled region.
    """

    @abstractmethod
    def init(self, hidden_states: dict, device: torch.device) -> None:
        """Initialization per training step.

        Called once before any transformer layers are processed.

        Args:
            hidden_states: The packed sequence (``FactoredSequencePack``).
            device: Target device for any new tensors.
        """

    @abstractmethod
    def read_for_layer(self, layer_idx: int) -> MemoryValue:
        """Produce a read-only tensor snapshot for *layer_idx*.

        Used to retrieve KV values from the cache.
        The returned ``MemoryValue`` is passed into the compiled decoder
        layer as ``memory_value``.
        """

    @abstractmethod
    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        """Store the K/V tensors produced by *layer_idx* back into the cache.

        Called outside the ``torch.compile`` boundary.
        """

    @abstractmethod
    def is_gen_only(self) -> bool:
        """Return ``True`` when only the generation pathway should run.

        When ``True``, the decoder layer assumes that the text caption has
        already been processed and cached in the MemoryState object.
        Used for autoregressive frame-by-frame generation of video.
        """

    @property
    def uses_rolling_kv_cache(self) -> bool:
        """Whether this memory uses the rolling KV-cache / compile-safe path.

        When ``True``, the network skips NATTEN metadata computation because
        temporal causality is handled inside three-way attention instead.
        """
        return False
