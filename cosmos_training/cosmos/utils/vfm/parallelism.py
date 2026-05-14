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
"""Unified ParallelDims for Cosmos3 VFM and VLM (multi-mesh, overlay design).

Topology
--------
- ``dp_replicate * dp_shard == world_size`` — the dp dims partition all ranks.
- ``cp`` (context parallel) and ``cfgp`` (CFG parallel) are *overlay* axes:
  they do NOT consume FSDP rank slots.  ``cfgp * cp`` must divide
  ``world_size`` so the overlay grid is well-formed, but the same rank may
  appear in both a dp group AND a cp/cfgp group.

Three meshes are built (any subset, depending on which axes are >1):

================  ===========================================================
Mesh              Shape / dims
================  ===========================================================
``dp_mesh``       2-D ``(dp_replicate, dp_shard)`` for FSDP/HSDP
``cp_mesh``       1-D, size ``cp``    (context parallelism)
``cfgp_mesh``     1-D, size ``cfgp``  (CFG parallelism, inference-only)
================  ===========================================================

Use cases
---------
- VLM training      — ``dp_shard`` (+ optional ``dp_replicate``); cp=cfgp=1.
- VFM training      — ``dp_shard`` (+ optional ``dp_replicate``) + optional cp.
- VFM inference     — ``dp_shard`` + cfgp/cp overlays; replicate forced to 1.

FSDP wrapping for VLM ``HFModel`` instances lives in
``projects.cosmos3.vfm.models.parallelize_vlm``; MoT wrapping lives in
``projects.cosmos3.vfm.models.mot.parallelize_unified_mot``.  Both consume
``ParallelDims`` from this module.
"""

import math
from dataclasses import dataclass, field

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from cosmos.utils import log

_MAX_CP = 8


@dataclass
class ParallelDims:
    """Unified multi-mesh parallel dimensions descriptor.

    Construct, then call :meth:`build_meshes` to allocate the underlying
    DeviceMeshes.  ``cp`` and ``cfgp`` are overlay axes that share rank slots
    with dp; the invariant ``dp_replicate * dp_shard == world_size`` always
    holds, regardless of cp/cfgp.

    Args:
        world_size:            Total number of ranks (typically WORLD_SIZE env var).
        dp_shard:              FSDP shard size.  Pass ``-1`` to auto-infer to
                               ``world_size`` (overlay semantics: cp/cfgp do NOT
                               consume the dp budget).
        dp_replicate:          HSDP replicate size.  Pass ``-1`` to auto-infer to
                               ``world_size // dp_shard``.
        cp:                    Context parallel size in ``[1, 8]``.  Overlay axis.
        cfgp:                  CFG parallel size in ``(1, 2)``.  Overlay axis,
                               inference-only (rejected at construction time
                               unless ``enable_inference_mode`` is True). Also is
                               used for only VFM to parallelize the conditional and
                               unconditional guidance.
        enable_inference_mode: Selects inference-time semantics — ``cfgp`` may
                               be >1 and ``dp_enabled`` ignores ``dp_replicate``
                               (matches the legacy VFM inference path).
    """

    world_size: int
    dp_shard: int = -1
    dp_replicate: int = -1
    cp: int = 1
    cfgp: int = 1
    enable_inference_mode: bool = False
    _meshes: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        # --- overlay range checks (run before division below) ---
        if self.cp < 1 or self.cp > _MAX_CP:
            raise ValueError(f"CP (Context Parallelism) must be in [1, {_MAX_CP}]. got {self.cp}")
        if self.cfgp not in (1, 2):
            raise ValueError(f"CFGP (CFG Parallelism) must be 1 or 2. got {self.cfgp}")
        if not self.enable_inference_mode and self.cfgp > 1:
            raise ValueError(
                f"CFG (Guidance Parallelism) must be 1 when enable_inference_mode is False. got {self.cfgp}"
            )

        # --- dp_shard auto-infer / clamp ---
        # Overlay semantics: cp/cfgp do NOT consume FSDP rank slots, so the full
        # world is available to dp_shard.  Without auto-inference here, the
        # pre-unification call form derives a negative dp_replicate, flips
        # dp_enabled off, and silently disables FSDP for
        # data_parallel_shard_degree=-1 runs (e.g. test_smoke.py).
        if self.dp_shard <= 0:
            self.dp_shard = self.world_size
            log.info(f"dp_shard auto-inferred to world_size = {self.world_size}")
        elif self.dp_shard > self.world_size:
            # Clamp + warn rather than fail-fast: a mis-sized launch (e.g. an 8-way
            # FSDP config on a 4-GPU smoke) will silently run a different topology
            # than requested. Emit a loud warning so the regression is visible in
            # logs; future work should fail-fast at the call site instead.
            log.warning(
                f"dp_shard ({self.dp_shard}) > world_size ({self.world_size}); clamping dp_shard to world_size."
            )
            self.dp_shard = self.world_size

        # --- dp_replicate auto-infer ---
        assert self.dp_replicate == -1 or self.dp_replicate >= 1, "dp_replicate must be -1 or >=1."
        if self.dp_replicate < 0:
            log.info(
                "dp_replicate is set to -1, will be automatically determined based on "
                f"world_size {self.world_size} // dp_shard {self.dp_shard}."
            )
            self.dp_replicate = self.world_size // self.dp_shard
            log.info(f"dp_replicate is set to {self.dp_replicate}.")

        # --- partition checks ---
        rest = self.world_size // (self.cfgp * self.cp)
        if rest * self.cfgp * self.cp != self.world_size:
            raise ValueError(
                f"Invalid parallel dims: rest({rest}) * cfgp({self.cfgp}) * cp({self.cp}) "
                f"!= WORLD_SIZE({self.world_size})"
            )
        if self.dp_replicate * self.dp_shard != self.world_size:
            raise ValueError(
                f"Invalid parallel dims: dp_replicate({self.dp_replicate}) * "
                f"dp_shard({self.dp_shard}) != WORLD_SIZE({self.world_size})"
            )

    # --- mesh construction --------------------------------------------------

    def _build_mesh(self, device_type: str, dims: list[int], names: list[str]) -> "DeviceMesh":
        if len(dims) != len(names):
            raise ValueError("Dimensions and names must have the same length.")
        if any(d <= 0 for d in dims):
            raise ValueError(f"All mesh dimensions must be > 0. got dims: {dims}, names: {names}.")
        if math.prod(dims) != self.world_size:
            raise ValueError(f"Invalid parallel dims: prod({dims}) != WORLD_SIZE({self.world_size})")

        log.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        return init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))

    def build_meshes(self, device_type: str = "cuda") -> None:
        """Build the dp / cp / cfgp meshes.

        cp + cfgp are bundled into a single 3-D overlay mesh
        ``(rest, cfgp, cp)`` so they share the same backing process group;
        the dp mesh is a separate 2-D ``(dp_replicate, dp_shard)``.

        After this call, :attr:`mesh` may contain any subset of the keys
        ``'dp'``, ``'dp_shard'``, ``'dp_replicate'``, ``'cp'``, ``'cfgp'``
        depending on which axes are enabled.
        """
        self._meshes = {}

        if self.cfgp_enabled or self.cp_enabled:
            overlay_mesh = self._build_mesh(
                device_type,
                dims=[self.world_size // (self.cfgp * self.cp), self.cfgp, self.cp],
                names=["rest", "cfgp", "cp"],
            )
            if self.cfgp_enabled:
                self._meshes["cfgp"] = overlay_mesh["cfgp"]
            if self.cp_enabled:
                self._meshes["cp"] = overlay_mesh["cp"]

        if self.dp_enabled:
            self._meshes["dp"] = self._build_mesh(
                device_type,
                dims=[self.dp_replicate, self.dp_shard],
                names=["dp_replicate", "dp_shard"],
            )
            if self.dp_shard_enabled:
                self._meshes["dp_shard"] = self._meshes["dp"]["dp_shard"]
            if self.dp_replicate_enabled:
                self._meshes["dp_replicate"] = self._meshes["dp"]["dp_replicate"]

    # --- mesh accessors -----------------------------------------------------

    @property
    def mesh(self) -> dict:
        """Read-only view of all built meshes.

        Empty until :meth:`build_meshes` is called.  After that, may contain
        any subset of ``'dp'``, ``'dp_shard'``, ``'dp_replicate'``, ``'cp'``,
        ``'cfgp'`` depending on which axes are enabled.  Prefer the named
        accessors (:attr:`dp_mesh`, :attr:`dp_shard_mesh`, …) over keying
        into this dict directly.
        """
        return self._meshes

    @property
    def dp_mesh(self) -> "DeviceMesh | None":
        """2-D ``(dp_replicate, dp_shard)`` mesh, or None if dp is not enabled."""
        return self._meshes.get("dp")

    @property
    def dp_shard_mesh(self) -> "DeviceMesh | None":
        """1-D ``dp_shard`` mesh, or None if dp_shard is not enabled."""
        return self._meshes.get("dp_shard")

    @property
    def dp_replicate_mesh(self) -> "DeviceMesh | None":
        """1-D ``dp_replicate`` mesh, or None if dp_replicate is not enabled."""
        return self._meshes.get("dp_replicate")

    @property
    def cp_mesh(self) -> "DeviceMesh | None":
        return self._meshes.get("cp")

    @property
    def cfgp_mesh(self) -> "DeviceMesh | None":
        return self._meshes.get("cfgp")

    # --- boolean flags ------------------------------------------------------

    @property
    def dp_enabled(self) -> bool:
        if self.enable_inference_mode:
            return self.dp_shard > 1
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_shard_enabled(self) -> bool:
        return self.dp_shard > 1

    @property
    def dp_replicate_enabled(self) -> bool:
        return self.dp_replicate > 1

    @property
    def cp_enabled(self) -> bool:
        return self.cp > 1

    @property
    def cfgp_enabled(self) -> bool:
        return self.cfgp > 1

    # --- rank/size helpers --------------------------------------------------

    @property
    def cp_rank(self) -> int:
        return self._meshes["cp"].get_local_rank() if self.cp_enabled else 0

    @property
    def cp_size(self) -> int:
        return self._meshes["cp"].size() if self.cp_enabled else 1

    @property
    def cfgp_rank(self) -> int:
        return self._meshes["cfgp"].get_local_rank() if self.cfgp_enabled else 0

    @property
    def cfgp_size(self) -> int:
        return self._meshes["cfgp"].size() if self.cfgp_enabled else 1
