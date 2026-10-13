# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Topology helpers for Cosmos3 action forward-dynamics rollouts.

The helpers are intentionally dependency-light.  The core path accepts binary
or labeled masks represented as Python lists, NumPy arrays, or objects with a
``tolist()`` method.  Optional persistent-homology enrichment is attempted only
when a backend is selected and ``ripser`` is importable.

Determinism is a property of this module, not an accident of it.  For a given
input the report is byte-identical across runs, processes, and the insertion
order of the caller's label mapping: labels are evaluated in sorted order,
specialist ranking ties break on ``specialist_id``, and the optional
persistent-homology backend is either explicitly absent or explicitly required
rather than silently environment-dependent.
"""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

Mask2D = Any
BoolGrid = list[list[bool]]
Connectivity = Literal[4, 8]
ReferenceMode = Literal["first_generated", "previous", "conditioning"]
PersistentHomologyBackend = Literal["none", "auto", "ripser"]

SCHEMA_VERSION = "topology_metrics.v1"


@dataclass(frozen=True)
class PersistentHomologyConfig:
    """Optional persistent-homology enrichment.

    ``backend`` is the only switch, so a configured run cannot silently do
    nothing:

    - ``"none"``   -- skip entirely (default), status ``"disabled"``.
    - ``"auto"``   -- use ``ripser`` when importable, otherwise report status
      ``"unavailable"`` with a warning.  Output depends on the environment.
    - ``"ripser"`` -- require ``ripser``; raise ``ImportError`` when missing so
      a run cannot quietly produce different numbers on a different machine.
    """

    backend: PersistentHomologyBackend = "none"
    max_dim: int = 1
    sample_limit: int = 2000
    include_diagrams: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ("none", "auto", "ripser"):
            raise ValueError(f"backend must be one of 'none', 'auto', 'ripser', got {self.backend!r}")
        if self.max_dim < 0:
            raise ValueError("max_dim must be >= 0")
        if self.sample_limit < 1:
            raise ValueError("sample_limit must be >= 1")


@dataclass(frozen=True)
class TopologyConfig:
    """Evaluation settings.

    ``foreground_connectivity`` and ``background_connectivity`` must be
    complementary (8/4 or 4/8).  This is the digital Jordan curve condition: with
    a matching pair such as 8/8, the diagonal ring ``.#. / #.# / .#.`` reports
    zero holes even though it plainly encloses one, and ``components - holes``
    stops being an Euler characteristic.
    """

    foreground_connectivity: Connectivity = 8
    background_connectivity: Connectivity = 4
    min_component_area_px: int = 16
    frame_stride: int = 1
    generated_frame_start: int = 1
    reference: ReferenceMode = "first_generated"
    stable_delta_threshold: int = 0
    persistent_homology: PersistentHomologyConfig = field(default_factory=PersistentHomologyConfig)

    def __post_init__(self) -> None:
        for name in ("foreground_connectivity", "background_connectivity"):
            value = getattr(self, name)
            if value not in (4, 8):
                raise ValueError(f"{name} must be 4 or 8, got {value!r}")
        if self.foreground_connectivity + self.background_connectivity != 12:
            raise ValueError(
                "foreground_connectivity and background_connectivity must be complementary "
                f"(8/4 or 4/8), got {self.foreground_connectivity}/{self.background_connectivity}"
            )
        if self.min_component_area_px < 1:
            raise ValueError("min_component_area_px must be >= 1")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if self.generated_frame_start < 0:
            raise ValueError("generated_frame_start must be >= 0")
        if self.stable_delta_threshold < 0:
            raise ValueError("stable_delta_threshold must be >= 0")


@dataclass(frozen=True)
class RolloutSpec:
    video_id: str
    domain_name: str | None = None
    fps: float | None = None
    action_chunk_size: int | None = None
    chunk_count: int | None = None
    conditioning_frames_per_chunk: int = 1

    def __post_init__(self) -> None:
        # A non-positive fps would make the finite-difference dt clamp to 1e-9 and
        # inflate every topology speed by nine orders of magnitude.
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be > 0 when provided, got {self.fps!r}")


@dataclass(frozen=True)
class ComponentStats:
    components: int
    area_px: int
    area_ratio: float
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None


@dataclass(frozen=True)
class PersistentHomologySummary:
    """Persistent-homology bar counts.

    ``h0_bars``/``h1_bars`` are diagram cardinalities, not Betti numbers: a Rips
    H0 diagram carries one bar per input point, so ``h0_bars`` always equals
    ``sampled_points``.  The combinatorial Betti proxies this module reports are
    ``FrameTopology.components`` and ``FrameTopology.holes``.
    """

    status: Literal["disabled", "unavailable", "empty", "computed", "failed"]
    h0_bars: int | None = None
    h1_bars: int | None = None
    h1_total_persistence: float | None = None
    h1_max_persistence: float | None = None
    sampled_points: int = 0
    warning: str | None = None
    diagrams: list[list[list[float]]] | None = None


@dataclass
class FrameTopology:
    video_id: str
    domain_name: str | None
    label: str
    frame_index: int
    generated_index: int
    chunk_index: int | None
    timestamp_s: float | None
    width: int
    height: int
    area_px: int
    area_ratio: float
    components: int
    holes: int
    euler_characteristic: int
    centroid_x: float | None
    centroid_y: float | None
    bbox_x0: int | None
    bbox_y0: int | None
    bbox_x1: int | None
    bbox_y1: int | None
    topology_delta_prev: int = 0
    topology_delta_ref: int = 0
    ph_status: str = "disabled"
    ph_h0_bars: int | None = None
    ph_h1_bars: int | None = None
    ph_h1_total_persistence: float | None = None
    ph_h1_max_persistence: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StabilitySummary:
    frame_count: int
    label_count: int
    component_change_count: int
    hole_change_count: int
    mean_topology_delta_ref: float
    max_topology_delta_ref: int
    boundary_jump_count: int
    stable_frame_ratio: float
    stability_score: float


@dataclass(frozen=True)
class TopologyStateVector:
    """Compact topology state used for inference/routing extensions."""

    components: float
    holes: float
    euler_characteristic: float
    area_ratio: float
    centroid_x_norm: float
    centroid_y_norm: float
    topology_delta_prev: float
    topology_delta_ref: float


@dataclass(frozen=True)
class FiniteDifferenceSample:
    """Finite-difference topology dynamics for one frame transition."""

    label: str
    frame_index: int
    generated_index: int
    chunk_index: int | None
    dt_s: float
    topology_speed: float
    topology_acceleration: float
    component_velocity: float
    hole_velocity: float
    euler_velocity: float
    area_velocity: float
    centroid_velocity: float
    local_change_ratio: float


@dataclass(frozen=True)
class TopologyConvergenceGate:
    """Thresholds for theorem-style rollout convergence checks."""

    min_stability_score: float = 0.80
    min_betti_stability: float = 0.75
    max_mean_delta_ref: float = 1.0
    max_boundary_jump_count: int = 0
    max_topology_speed: float = 2.0
    betti_window: int = 4


@dataclass(frozen=True)
class TopologyConvergenceResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]
    messages: tuple[str, ...]


@dataclass(frozen=True)
class SpecialistSignal:
    """Candidate robot/model specialist signal for topology-aware routing."""

    specialist_id: str
    capability_tags: tuple[str, ...]
    state: TopologyStateVector
    reliability: float = 1.0
    cost: float = 0.0
    utilization: int = 0


@dataclass(frozen=True)
class SpecialistRanking:
    specialist_id: str
    score: float
    topology_distance: float
    reliability: float
    cost: float
    utilization: int
    matched_tags: tuple[str, ...]


@dataclass
class TopologyReport:
    schema_version: str
    run: RolloutSpec
    config: TopologyConfig
    frames: list[FrameTopology]
    summary: StabilitySummary
    warnings: list[str] = field(default_factory=list)


_NEIGHBORS: dict[int, tuple[tuple[int, int], ...]] = {
    4: ((0, -1), (-1, 0), (1, 0), (0, 1)),
    8: ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)),
}


def _mask_to_bool_grid(mask: Mask2D) -> BoolGrid:
    if hasattr(mask, "tolist"):
        mask = mask.tolist()
    rows = list(mask)
    if not rows:
        raise ValueError("mask must have at least one row")

    grid: BoolGrid = []
    width: int | None = None
    for row in rows:
        row_values = list(row)
        if width is None:
            width = len(row_values)
            if width == 0:
                raise ValueError("mask rows must not be empty")
        elif len(row_values) != width:
            raise ValueError("mask rows must all have the same width")
        grid.append([bool(value) for value in row_values])
    return grid


def _region_stats(pixels: Sequence[tuple[int, int]]) -> tuple[int, tuple[int, int, int, int], tuple[float, float]]:
    area = len(pixels)
    xs = [x for x, _y in pixels]
    ys = [y for _x, y in pixels]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    centroid = (sum(xs) / area, sum(ys) / area)
    return area, bbox, centroid


def _regions(grid: BoolGrid, target: bool, connectivity: Connectivity) -> list[list[tuple[int, int]]]:
    height = len(grid)
    width = len(grid[0])
    offsets = _NEIGHBORS[connectivity]
    seen = [[False] * width for _y in range(height)]
    regions: list[list[tuple[int, int]]] = []

    for y in range(height):
        for x in range(width):
            if seen[y][x] or grid[y][x] != target:
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y][x] = True
            pixels: list[tuple[int, int]] = []

            while queue:
                px, py = queue.popleft()
                pixels.append((px, py))
                for dx, dy in offsets:
                    nx = px + dx
                    ny = py + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if seen[ny][nx] or grid[ny][nx] != target:
                        continue
                    seen[ny][nx] = True
                    queue.append((nx, ny))

            regions.append(pixels)

    return regions


def _filtered_components(grid: BoolGrid, config: TopologyConfig) -> tuple[ComponentStats, BoolGrid]:
    """Return foreground stats plus the grid holding only the kept components.

    Everything downstream -- holes, Euler characteristic, persistent homology --
    reads the returned grid, so every metric describes the same foreground set.
    """

    height = len(grid)
    width = len(grid[0])
    kept_regions = [
        region
        for region in _regions(grid, True, config.foreground_connectivity)
        if len(region) >= config.min_component_area_px
    ]

    kept_grid: BoolGrid = [[False] * width for _y in range(height)]
    for region in kept_regions:
        for x, y in region:
            kept_grid[y][x] = True

    if not kept_regions:
        return ComponentStats(components=0, area_px=0, area_ratio=0.0, bbox=None, centroid=None), kept_grid

    pixels = [point for region in kept_regions for point in region]
    area_px, bbox, centroid = _region_stats(pixels)
    return (
        ComponentStats(
            components=len(kept_regions),
            area_px=area_px,
            area_ratio=area_px / float(width * height),
            bbox=bbox,
            centroid=centroid,
        ),
        kept_grid,
    )


def _enclosed_background(grid: BoolGrid, connectivity: Connectivity) -> int:
    height = len(grid)
    width = len(grid[0])
    holes = 0
    for region in _regions(grid, False, connectivity):
        touches_border = any(x == 0 or y == 0 or x == width - 1 or y == height - 1 for x, y in region)
        if not touches_border:
            holes += 1
    return holes


def label_components(mask: Mask2D, config: TopologyConfig = TopologyConfig()) -> ComponentStats:
    """Return foreground connected-component statistics for a binary mask."""

    return _filtered_components(_mask_to_bool_grid(mask), config)[0]


def count_holes(mask: Mask2D, config: TopologyConfig = TopologyConfig()) -> int:
    """Count background regions enclosed by the area-filtered foreground.

    The area filter is applied first so that ``components - holes`` is the Euler
    characteristic of one set.  A ring below ``min_component_area_px`` is dropped
    whole and contributes neither a component nor a hole.
    """

    _stats, kept_grid = _filtered_components(_mask_to_bool_grid(mask), config)
    return _enclosed_background(kept_grid, config.background_connectivity)


def _foreground_points(grid: BoolGrid) -> list[tuple[int, int]]:
    return [(x, y) for y, row in enumerate(grid) for x, value in enumerate(row) if value]


def _persistent_homology_from_grid(grid: BoolGrid, config: TopologyConfig) -> PersistentHomologySummary:
    ph_config = config.persistent_homology
    if ph_config.backend == "none":
        return PersistentHomologySummary(status="disabled")

    points = _foreground_points(grid)
    if not points:
        return PersistentHomologySummary(status="empty", sampled_points=0)

    if len(points) > ph_config.sample_limit:
        stride = max(1, len(points) // ph_config.sample_limit)
        points = points[::stride][: ph_config.sample_limit]

    try:
        import numpy as np
        from ripser import ripser
    except ImportError as exc:
        if ph_config.backend == "ripser":
            raise ImportError(
                "persistent_homology backend 'ripser' was requested but ripser is not installed; "
                "install ripser or use backend='auto' to degrade instead of failing"
            ) from exc
        return PersistentHomologySummary(
            status="unavailable",
            sampled_points=len(points),
            warning="persistent homology requested but ripser is not installed",
        )

    try:
        # ripser needs an (n, 2) array; a list of pixel tuples is rejected.
        result = ripser(np.asarray(points, dtype=float), maxdim=ph_config.max_dim)
    except Exception as exc:  # pragma: no cover - defensive around optional dependency
        return PersistentHomologySummary(
            status="failed",
            sampled_points=len(points),
            warning=f"ripser failed: {exc}",
        )

    diagrams = result.get("dgms", [])
    h0 = diagrams[0] if len(diagrams) > 0 else []
    h1 = diagrams[1] if len(diagrams) > 1 else []
    # Infinite bars carry no finite lifetime, so they are excluded from the
    # persistence sums rather than treated as arbitrarily long.
    h1_lifetimes = [float(death - birth) for birth, death in h1 if death != float("inf") and death >= birth]
    encoded_diagrams = None
    if ph_config.include_diagrams:
        encoded_diagrams = [[[float(birth), float(death)] for birth, death in diagram] for diagram in diagrams]

    return PersistentHomologySummary(
        status="computed",
        h0_bars=len(h0),
        h1_bars=len(h1),
        h1_total_persistence=sum(h1_lifetimes),
        h1_max_persistence=max(h1_lifetimes, default=0.0),
        sampled_points=len(points),
        diagrams=encoded_diagrams,
    )


def compute_persistent_homology(
    mask: Mask2D,
    config: TopologyConfig = TopologyConfig(),
) -> PersistentHomologySummary:
    """Compute the optional PH summary for the area-filtered foreground."""

    _stats, kept_grid = _filtered_components(_mask_to_bool_grid(mask), config)
    return _persistent_homology_from_grid(kept_grid, config)


def _chunk_index(generated_index: int, action_chunk_size: int | None) -> int | None:
    if action_chunk_size is None or action_chunk_size <= 0:
        return None
    return generated_index // action_chunk_size


def _frame_topology_from_grid(
    grid: BoolGrid,
    *,
    frame_index: int,
    generated_index: int,
    timestamp_s: float | None,
    label: str,
    rollout: RolloutSpec,
    config: TopologyConfig,
) -> FrameTopology:
    height = len(grid)
    width = len(grid[0])
    components, kept_grid = _filtered_components(grid, config)
    holes = _enclosed_background(kept_grid, config.background_connectivity)
    ph = _persistent_homology_from_grid(kept_grid, config)

    warnings: list[str] = []
    if components.components == 0:
        warnings.append("empty_foreground")
    if ph.warning:
        warnings.append(ph.warning)

    bbox = components.bbox
    centroid = components.centroid
    return FrameTopology(
        video_id=rollout.video_id,
        domain_name=rollout.domain_name,
        label=label,
        frame_index=frame_index,
        generated_index=generated_index,
        chunk_index=_chunk_index(generated_index, rollout.action_chunk_size),
        timestamp_s=timestamp_s,
        width=width,
        height=height,
        area_px=components.area_px,
        area_ratio=components.area_ratio,
        components=components.components,
        holes=holes,
        euler_characteristic=components.components - holes,
        centroid_x=None if centroid is None else centroid[0],
        centroid_y=None if centroid is None else centroid[1],
        bbox_x0=None if bbox is None else bbox[0],
        bbox_y0=None if bbox is None else bbox[1],
        bbox_x1=None if bbox is None else bbox[2],
        bbox_y1=None if bbox is None else bbox[3],
        ph_status=ph.status,
        ph_h0_bars=ph.h0_bars,
        ph_h1_bars=ph.h1_bars,
        ph_h1_total_persistence=ph.h1_total_persistence,
        ph_h1_max_persistence=ph.h1_max_persistence,
        warnings=warnings,
    )


def compute_frame_topology(
    mask: Mask2D,
    *,
    frame_index: int,
    generated_index: int,
    timestamp_s: float | None,
    label: str,
    rollout: RolloutSpec,
    config: TopologyConfig = TopologyConfig(),
) -> FrameTopology:
    """Compute topology metrics for one label/mask/frame."""

    return _frame_topology_from_grid(
        _mask_to_bool_grid(mask),
        frame_index=frame_index,
        generated_index=generated_index,
        timestamp_s=timestamp_s,
        label=label,
        rollout=rollout,
        config=config,
    )


def _normalise_labeled_masks(
    masks: Mapping[str, Sequence[Mask2D]] | Sequence[Mask2D],
) -> dict[str, list[BoolGrid]]:
    """Convert every mask once, validate shapes, and fix a canonical label order.

    Labels are visited in sorted order so the report does not inherit the
    caller's mapping insertion order.
    """

    if isinstance(masks, Mapping):
        if not masks:
            raise ValueError("masks mapping must not be empty")
        raw = {label: masks[label] for label in sorted(masks)}
    else:
        raw = {"foreground": masks}

    grids: dict[str, list[BoolGrid]] = {}
    expected_shape: tuple[int, int] | None = None
    expected_len: int | None = None
    for label, sequence in raw.items():
        if not sequence:
            raise ValueError(f"mask sequence for label {label!r} must not be empty")
        if expected_len is None:
            expected_len = len(sequence)
        elif len(sequence) != expected_len:
            raise ValueError("all labels must contain the same number of frames")

        label_grids: list[BoolGrid] = []
        for mask in sequence:
            grid = _mask_to_bool_grid(mask)
            shape = (len(grid), len(grid[0]))
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError("all masks must have the same height and width")
            label_grids.append(grid)
        grids[label] = label_grids
    return grids


def compute_sequence_topology(
    masks: Mapping[str, Sequence[Mask2D]] | Sequence[Mask2D],
    rollout: RolloutSpec,
    config: TopologyConfig = TopologyConfig(),
) -> TopologyReport:
    """Compute topology drift and stability over a rollout mask sequence."""

    labeled_grids = _normalise_labeled_masks(masks)

    frames: list[FrameTopology] = []
    warnings: list[str] = []
    if config.reference == "conditioning" and config.generated_frame_start == 0:
        warnings.append(
            "reference='conditioning' needs generated_frame_start > 0; "
            "falling back to the first generated frame"
        )

    for label, sequence in labeled_grids.items():
        label_frames: list[FrameTopology] = []
        for frame_index in range(config.generated_frame_start, len(sequence), config.frame_stride):
            label_frames.append(
                _frame_topology_from_grid(
                    sequence[frame_index],
                    frame_index=frame_index,
                    generated_index=frame_index - config.generated_frame_start,
                    timestamp_s=None if rollout.fps is None else frame_index / float(rollout.fps),
                    label=label,
                    rollout=rollout,
                    config=config,
                )
            )
        if not label_frames:
            warnings.append(f"label {label!r} had no frames after generated_frame_start/stride filtering")

        conditioning_frame = None
        if config.reference == "conditioning" and config.generated_frame_start > 0:
            conditioning_index = config.generated_frame_start - 1
            conditioning_frame = _frame_topology_from_grid(
                sequence[conditioning_index],
                frame_index=conditioning_index,
                generated_index=-1,
                timestamp_s=None if rollout.fps is None else conditioning_index / float(rollout.fps),
                label=label,
                rollout=rollout,
                config=config,
            )
        _apply_drift(label_frames, config.reference, conditioning_frame)
        frames.extend(label_frames)

    summary = summarize_stability(frames, config)
    if frames and all(frame.components == 0 for frame in frames):
        warnings.append("all evaluated frames have empty foreground masks")

    return TopologyReport(
        schema_version=SCHEMA_VERSION,
        run=rollout,
        config=config,
        frames=frames,
        summary=summary,
        warnings=warnings,
    )


def evaluate_fd_rollout(
    masks: Mapping[str, Sequence[Mask2D]] | Sequence[Mask2D],
    rollout: RolloutSpec,
    config: TopologyConfig = TopologyConfig(),
) -> TopologyReport:
    """Alias for notebook readability."""

    return compute_sequence_topology(masks=masks, rollout=rollout, config=config)


def _topology_delta(current: FrameTopology, other: FrameTopology) -> int:
    return abs(current.components - other.components) + abs(current.holes - other.holes)


def _apply_drift(
    frames: list[FrameTopology],
    reference: ReferenceMode,
    conditioning_frame: FrameTopology | None = None,
) -> None:
    if not frames:
        return

    ref_frame = frames[0]
    for index, frame in enumerate(frames):
        prev_frame = frames[index - 1] if index > 0 else frame
        frame.topology_delta_prev = _topology_delta(frame, prev_frame)
        if reference == "previous":
            compare_to = prev_frame
        elif reference == "conditioning" and conditioning_frame is not None:
            compare_to = conditioning_frame
        else:
            compare_to = ref_frame
        frame.topology_delta_ref = _topology_delta(frame, compare_to)


def _group_by_label(frames: Sequence[FrameTopology]) -> list[tuple[str, list[FrameTopology]]]:
    """Group frames by label in sorted label order, each group sorted by frame index."""

    by_label: dict[str, list[FrameTopology]] = {}
    for frame in frames:
        by_label.setdefault(frame.label, []).append(frame)
    return [
        (label, sorted(by_label[label], key=lambda item: item.frame_index)) for label in sorted(by_label)
    ]


def summarize_stability(
    frames: Sequence[FrameTopology],
    config: TopologyConfig = TopologyConfig(),
) -> StabilitySummary:
    """Summarize topology drift across all labels and frames."""

    if not frames:
        return StabilitySummary(
            frame_count=0,
            label_count=0,
            component_change_count=0,
            hole_change_count=0,
            mean_topology_delta_ref=0.0,
            max_topology_delta_ref=0,
            boundary_jump_count=0,
            stable_frame_ratio=1.0,
            stability_score=1.0,
        )

    component_change_count = 0
    hole_change_count = 0
    boundary_jump_count = 0
    stable_count = sum(
        1
        for frame in frames
        if frame.topology_delta_prev <= config.stable_delta_threshold
        and frame.topology_delta_ref <= config.stable_delta_threshold
    )

    grouped = _group_by_label(frames)
    for _label, sorted_frames in grouped:
        for prev, current in zip(sorted_frames, sorted_frames[1:]):
            if current.components != prev.components:
                component_change_count += 1
            if current.holes != prev.holes:
                hole_change_count += 1
            if (
                current.chunk_index is not None
                and prev.chunk_index is not None
                and current.chunk_index != prev.chunk_index
                and current.topology_delta_prev > config.stable_delta_threshold
            ):
                boundary_jump_count += 1

    deltas = [frame.topology_delta_ref for frame in frames]
    mean_delta = sum(deltas) / len(deltas)
    stable_frame_ratio = stable_count / len(frames)
    # ponytail: hand-tuned penalties, not a calibrated score. Fit the constants
    # against labelled rollouts before treating the value as comparable across runs.
    drift_penalty = min(1.0, mean_delta / 4.0)
    boundary_penalty = min(0.25, boundary_jump_count / len(frames))
    stability_score = max(0.0, min(1.0, stable_frame_ratio * (1.0 - drift_penalty) - boundary_penalty))

    return StabilitySummary(
        frame_count=len(frames),
        label_count=len(grouped),
        component_change_count=component_change_count,
        hole_change_count=hole_change_count,
        mean_topology_delta_ref=mean_delta,
        max_topology_delta_ref=max(deltas),
        boundary_jump_count=boundary_jump_count,
        stable_frame_ratio=stable_frame_ratio,
        stability_score=stability_score,
    )


def frame_to_state_vector(frame: FrameTopology) -> TopologyStateVector:
    """Convert a frame report into a normalized state vector.

    This is the bridge from evaluation to topological inference.  It is small on
    purpose: downstream ranking and neighborhood checks should not need to know
    the whole frame schema.
    """

    centroid_x_norm = 0.0 if frame.centroid_x is None else frame.centroid_x / max(1, frame.width - 1)
    centroid_y_norm = 0.0 if frame.centroid_y is None else frame.centroid_y / max(1, frame.height - 1)
    return TopologyStateVector(
        components=float(frame.components),
        holes=float(frame.holes),
        euler_characteristic=float(frame.euler_characteristic),
        area_ratio=float(frame.area_ratio),
        centroid_x_norm=float(centroid_x_norm),
        centroid_y_norm=float(centroid_y_norm),
        topology_delta_prev=float(frame.topology_delta_prev),
        topology_delta_ref=float(frame.topology_delta_ref),
    )


def topology_state_distance(left: TopologyStateVector, right: TopologyStateVector) -> float:
    """Weighted L1 distance for sparse topology-neighborhood reasoning.

    All weights are positive, so this is a genuine metric: symmetric, zero only
    on identical states, and obeying the triangle inequality.  ``euler_characteristic``
    is ``components - holes``, so a component-only change costs ``1.0 + 0.5`` and
    a component/hole change that leaves Euler invariant costs ``1.0 + 1.0``.
    """

    return (
        1.0 * abs(left.components - right.components)
        + 1.0 * abs(left.holes - right.holes)
        + 0.5 * abs(left.euler_characteristic - right.euler_characteristic)
        + 0.5 * abs(left.area_ratio - right.area_ratio)
        + 0.25 * abs(left.centroid_x_norm - right.centroid_x_norm)
        + 0.25 * abs(left.centroid_y_norm - right.centroid_y_norm)
        + 0.25 * abs(left.topology_delta_prev - right.topology_delta_prev)
        + 0.25 * abs(left.topology_delta_ref - right.topology_delta_ref)
    )


def _state_subtract(left: TopologyStateVector, right: TopologyStateVector, scale: float) -> TopologyStateVector:
    return TopologyStateVector(
        components=(left.components - right.components) * scale,
        holes=(left.holes - right.holes) * scale,
        euler_characteristic=(left.euler_characteristic - right.euler_characteristic) * scale,
        area_ratio=(left.area_ratio - right.area_ratio) * scale,
        centroid_x_norm=(left.centroid_x_norm - right.centroid_x_norm) * scale,
        centroid_y_norm=(left.centroid_y_norm - right.centroid_y_norm) * scale,
        topology_delta_prev=(left.topology_delta_prev - right.topology_delta_prev) * scale,
        topology_delta_ref=(left.topology_delta_ref - right.topology_delta_ref) * scale,
    )


def compute_fdtd_rollout_trace(frames: Sequence[FrameTopology]) -> list[FiniteDifferenceSample]:
    """Compute finite-difference topology dynamics over a rollout.

    The trace is an FDTD-inspired diagnostic over topology state, not a physical
    PDE solver.  Per label it takes the backward difference of the state vector,
    so ``topology_speed`` is the weighted-L1 norm of the state velocity and
    ``topology_acceleration`` the norm of its second difference.  Samples are
    emitted in sorted label order, then frame order, independent of how the
    caller ordered ``frames``.
    """

    samples: list[FiniteDifferenceSample] = []
    for _label, ordered in _group_by_label(frames):
        previous_velocity: TopologyStateVector | None = None
        for prev, current in zip(ordered, ordered[1:]):
            prev_state = frame_to_state_vector(prev)
            current_state = frame_to_state_vector(current)
            if prev.timestamp_s is not None and current.timestamp_s is not None:
                dt = max(current.timestamp_s - prev.timestamp_s, 1e-9)
            else:
                dt = max(current.frame_index - prev.frame_index, 1)

            local_change = topology_state_distance(current_state, prev_state)
            velocity = _state_subtract(current_state, prev_state, 1.0 / dt)
            if previous_velocity is None:
                topology_acceleration = 0.0
            else:
                topology_acceleration = topology_state_distance(velocity, previous_velocity) / dt
            centroid_velocity = (velocity.centroid_x_norm**2 + velocity.centroid_y_norm**2) ** 0.5

            samples.append(
                FiniteDifferenceSample(
                    label=current.label,
                    frame_index=current.frame_index,
                    generated_index=current.generated_index,
                    chunk_index=current.chunk_index,
                    dt_s=float(dt),
                    topology_speed=float(local_change / dt),
                    topology_acceleration=float(topology_acceleration),
                    component_velocity=float(velocity.components),
                    hole_velocity=float(velocity.holes),
                    euler_velocity=float(velocity.euler_characteristic),
                    area_velocity=float(velocity.area_ratio),
                    centroid_velocity=float(centroid_velocity),
                    local_change_ratio=float(local_change),
                )
            )
            previous_velocity = velocity

    return samples


def _betti_stability(frames: Sequence[FrameTopology], window: int) -> tuple[float, int]:
    """Return the stable-window fraction and the number of windows it came from."""

    if window < 2:
        raise ValueError("window must be >= 2")

    stable_windows = 0
    total_windows = 0
    for _label, label_frames in _group_by_label(frames):
        for start in range(0, len(label_frames) - window + 1):
            segment = label_frames[start : start + window]
            total_windows += 1
            if len({(frame.components, frame.holes) for frame in segment}) == 1:
                stable_windows += 1

    if total_windows == 0:
        return 1.0, 0
    return stable_windows / total_windows, total_windows


def betti_stability_score(frames: Sequence[FrameTopology], window: int = 4) -> float:
    """Return how often component/hole proxies remain stable over a sliding window.

    A rollout shorter than ``window`` yields no windows and scores 1.0 by
    convention.  Use :func:`evaluate_topological_convergence`, which reports
    ``betti_windows`` alongside the score, when that distinction matters.
    """

    return _betti_stability(frames, window)[0]


def evaluate_topological_convergence(
    report: TopologyReport,
    gate: TopologyConvergenceGate = TopologyConvergenceGate(),
) -> TopologyConvergenceResult:
    """Evaluate theorem-style topology convergence gates for a rollout report."""

    trace = compute_fdtd_rollout_trace(report.frames)
    max_speed = max((sample.topology_speed for sample in trace), default=0.0)
    betti_score, betti_windows = _betti_stability(report.frames, gate.betti_window)
    checks = {
        "stability_score": report.summary.stability_score >= gate.min_stability_score,
        "betti_stability": betti_score >= gate.min_betti_stability,
        "mean_delta_ref": report.summary.mean_topology_delta_ref <= gate.max_mean_delta_ref,
        "boundary_jump_count": report.summary.boundary_jump_count <= gate.max_boundary_jump_count,
        "topology_speed": max_speed <= gate.max_topology_speed,
    }
    metrics = {
        "stability_score": report.summary.stability_score,
        "betti_stability": betti_score,
        # betti_windows == 0 means the rollout was shorter than gate.betti_window
        # and the Betti check passed vacuously.
        "betti_windows": float(betti_windows),
        "mean_delta_ref": report.summary.mean_topology_delta_ref,
        "boundary_jump_count": float(report.summary.boundary_jump_count),
        "max_topology_speed": max_speed,
    }
    messages = tuple(name for name, ok in checks.items() if not ok)
    return TopologyConvergenceResult(
        passed=all(checks.values()),
        checks=checks,
        metrics=metrics,
        messages=messages,
    )


def rank_topology_specialists(
    query: TopologyStateVector,
    specialists: Sequence[SpecialistSignal],
    *,
    task_tags: Sequence[str] = (),
    top_k: int = 5,
    similarity_weight: float = 1.0,
    reliability_weight: float = 0.5,
    cost_weight: float = 0.2,
    freshness_weight: float = 0.1,
) -> list[SpecialistRanking]:
    """Rank robot/model specialists by topology similarity and routing signals.

    Ties break on ``specialist_id`` so the returned order does not depend on the
    order the candidates were supplied in.
    """

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    requested = set(task_tags)
    rankings: list[SpecialistRanking] = []
    for specialist in specialists:
        distance = topology_state_distance(query, specialist.state)
        matched = tuple(sorted(requested.intersection(specialist.capability_tags)))
        tag_score = len(matched) / len(requested) if requested else 1.0
        freshness = 1.0 / (1.0 + max(0, specialist.utilization))
        score = (
            similarity_weight * (tag_score - distance)
            + reliability_weight * specialist.reliability
            - cost_weight * specialist.cost
            + freshness_weight * freshness
        )
        rankings.append(
            SpecialistRanking(
                specialist_id=specialist.specialist_id,
                score=score,
                topology_distance=distance,
                reliability=specialist.reliability,
                cost=specialist.cost,
                utilization=specialist.utilization,
                matched_tags=matched,
            )
        )

    rankings.sort(key=lambda item: (-item.score, item.specialist_id))
    return rankings[:top_k]


def report_to_dict(report: TopologyReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "run": asdict(report.run),
        "config": asdict(report.config),
        "frames": [asdict(frame) for frame in report.frames],
        "summary": asdict(report.summary),
        "warnings": list(report.warnings),
    }


def write_topology_json(report: TopologyReport, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report_to_dict(report), indent=2, sort_keys=True) + "\n")
    return out


CSV_FIELDS = (
    "video_id",
    "domain_name",
    "label",
    "frame_index",
    "generated_index",
    "chunk_index",
    "timestamp_s",
    "width",
    "height",
    "area_px",
    "area_ratio",
    "components",
    "holes",
    "euler_characteristic",
    "topology_delta_prev",
    "topology_delta_ref",
    "centroid_x",
    "centroid_y",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "ph_status",
    "ph_h0_bars",
    "ph_h1_bars",
    "ph_h1_total_persistence",
    "ph_h1_max_persistence",
    "warnings",
)


def write_topology_csv(report: TopologyReport, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for frame in report.frames:
            row = asdict(frame)
            row["warnings"] = ";".join(frame.warnings)
            writer.writerow({field_name: row.get(field_name) for field_name in CSV_FIELDS})
    return out


def threshold_frame_to_mask(frame: Any, threshold: float = 32.0) -> BoolGrid:
    """Convert a grayscale/RGB frame-like object into a binary mask.

    This is a convenience adapter for notebooks.  Production evaluations should
    prefer task-specific masks, labels, or keypoint-derived point clouds.
    """

    if hasattr(frame, "tolist"):
        frame = frame.tolist()

    mask: BoolGrid = []
    for row in frame:
        mask_row: list[bool] = []
        for value in row:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                channels = [float(channel) for channel in value[:3]]
                luminance = sum(channels) / len(channels)
            else:
                luminance = float(value)
            mask_row.append(luminance >= threshold)
        mask.append(mask_row)
    return _mask_to_bool_grid(mask)


def frames_to_luminance_masks(frames: Iterable[Any], threshold: float = 32.0) -> list[BoolGrid]:
    return [threshold_frame_to_mask(frame, threshold=threshold) for frame in frames]
