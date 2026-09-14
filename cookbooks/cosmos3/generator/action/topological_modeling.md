# Topological Modeling for Cosmos Action Forward Dynamics

This note describes an optional modeling and evaluation layer for Cosmos3 action
forward dynamics. It does not change model inference. It adds a way to inspect
whether generated robotics rollouts preserve simple topology over time.

## Modeling Rationale

Forward dynamics predicts future observations from a start image and action
trajectory. For robotics rollouts, useful structure often lives in relationships:
object separation, contact regions, gripper/object coupling, holes, occlusions,
and chunk-boundary continuity. Pixel error alone does not expose these failures
cleanly.

The contribution treats generated rollout masks or point-cloud proxies as
field-like state samples:

- finite-difference and FDTD thinking motivate checking how state evolves across
  adjacent frames and chunks;
- topology metrics summarize whether visible structure splits, merges, opens, or
  collapses;
- topological inference concepts motivate stable manifold state, sparse
  neighborhood reasoning, and convergence checks;
- swarm-routing concepts motivate future multi-agent robot/model specialists
  selected by topology-aware state similarity.

The implementation is fresh Cosmos-native code. No external research prototype
code is imported or vendored.

## Implemented Core

`topology_helpers.py` evaluates masks for generated FD rollouts and emits
deterministic JSON/CSV summaries.

The mask-first API is deliberate. It keeps the core metric independent of any
specific segmentation model and makes the evaluator safe for CPU-only tests.
Notebook users can provide task-specific binary masks, labeled masks, or simple
luminance-threshold masks for quick inspection.

Frame-level metrics include:

- connected foreground components;
- enclosed background holes as a loop proxy;
- Euler characteristic `components - holes`;
- foreground area, bounding box, and centroid;
- drift from the previous frame;
- drift from the first generated frame;
- optional persistent-homology summaries when a backend is selected.

### Digital-topology preconditions

Two conditions make `components - holes` an Euler characteristic rather than a
pair of unrelated counts, and the config enforces both:

1. **Complementary connectivity.** `foreground_connectivity` and
   `background_connectivity` must be 8/4 or 4/8. This is the digital Jordan
   curve condition. Under 8/8 the diagonal ring

   ```
   .#.
   #.#
   .#.
   ```

   reports zero holes because the enclosed pixel reaches the border diagonally;
   under 4/4 it splits into four components and reports one hole, giving
   `chi = 3`. Only the complementary pairs return `chi = 0`. A matching pair
   raises `ValueError`.

2. **One filtered foreground.** `min_component_area_px` is applied before holes
   are counted, so both terms describe the same set. A ring below the threshold
   is discarded whole and contributes neither a component nor a hole; counting
   holes on the unfiltered mask would report `chi = -1` for an empty foreground.

Sequence-level metrics include:

- component and hole change counts;
- mean and maximum topology drift;
- chunk-boundary jump count;
- stable frame ratio;
- heuristic stability score in `[0, 1]`.

Finite-difference and convergence metrics include:

- `compute_fdtd_rollout_trace`, an FDTD-inspired trace over topology state
  velocity, acceleration, and local change ratio;
- `TopologyConvergenceGate`, a theorem-style threshold bundle for stable
  rollouts;
- `evaluate_topological_convergence`, a pass/fail certificate over stability,
  Betti-stability, drift, boundary jumps, and finite-difference speed.

## FD/FDTD Modeling

Finite-difference modeling gives the evaluator its local-time bias. A rollout is
not inspected as disconnected images; it is inspected as a sequence whose visible
structure should evolve smoothly unless the action implies a topological event.

This is especially relevant for autoregressive robotics chunks. A chunk boundary
that splits one object mask into two components, opens a previously closed
contact loop, or collapses a stable foreground component can be flagged even when
the generated video still looks plausible at a glance.

The helper computes finite differences over compact topology state vectors. This
does not claim to solve a physical PDE. It gives the action FD cookbook a
deterministic analogue of velocity and acceleration over rollout topology, so
large local topology jumps become measurable.

## Topology Metrics

The first metric layer is intentionally simple:

1. build connected components over foreground masks;
2. count enclosed background regions as hole proxies;
3. compare component and hole counts over time;
4. summarize stability per rollout.

This gives a deterministic baseline that can be reviewed without GPU access,
model weights, or external topology packages. Persistent homology is treated as
an optional enrichment, not a hard dependency.

`PersistentHomologyConfig.backend` is the only switch: `"none"` skips the pass,
`"auto"` uses `ripser` when importable and reports status `"unavailable"`
otherwise, and `"ripser"` raises `ImportError` when the package is missing. The
strict mode exists because an enrichment that silently disappears on one machine
turns a comparison between two runs into a comparison between two environments.

The PH summary reports `h0_bars` and `h1_bars`, which are diagram
cardinalities, not Betti numbers -- a Rips H0 diagram carries one bar per input
point, so `h0_bars` always equals `sampled_points`. The Betti proxies this
module stands behind are the combinatorial `components` and `holes`.

## Determinism

The report is a comparison artifact, so it is fixed byte-for-byte by the input
alone:

- labels are evaluated in sorted order, so the frame list, CSV row order, and
  finite-difference trace do not inherit the caller's mapping insertion order;
- `rank_topology_specialists` breaks score ties on `specialist_id` rather than
  on the order candidates were passed in;
- persistent homology is either explicitly absent or explicitly required, never
  environment-dependent by default;
- `RolloutSpec` rejects a non-positive `fps`, which would otherwise clamp the
  finite-difference `dt` to `1e-9` and inflate every topology speed by nine
  orders of magnitude.

## Topological Inference Extension

The same report schema supports an inference-oriented state vector today:

- `frame_to_state_vector` compresses a frame report into component, hole,
  Euler, area, centroid, and drift features;
- `topology_state_distance` compares state vectors with a weighted L1 distance.
  All weights are positive, so it is a metric: symmetric, zero only on identical
  states, and obeying the triangle inequality. `euler_characteristic` is a
  derived coordinate, so a component-only change costs `1.0 + 0.5` and a
  component/hole change that leaves Euler invariant costs `1.0 + 1.0`;
- `betti_stability_score` checks whether component/hole proxies stabilize over a
  sliding window;
- `evaluate_topological_convergence` produces a small convergence certificate
  from stability, Betti-stability, drift, and finite-difference speed checks. It
  reports `betti_windows` alongside the score: a rollout shorter than
  `betti_window` yields no windows and scores 1.0 vacuously, and the count is
  what distinguishes that from a genuinely stable rollout.

These helpers are not a new inference engine. They are small interfaces that can
support later inference-oriented analysis:

- represent rollout frames as points on a manifold;
- use sparse neighborhood queries to compare the current state to prior states;
- monitor Betti-stability over a sliding window;
- detect drift when topology changes faster than action-conditioned dynamics
  justify;
- gate stronger claims behind reproducible tests or formal checks.

Those ideas remain extension points. The current helper exposes metrics and
schemas, not a new inference engine.

## Swarm Robotics Extension

For multi-agent robotics, topology summaries can become routing features. A
future swarm evaluator can build on the included `rank_topology_specialists`
helper, which applies a sparse top-k rule over:

- topology-aware state similarity;
- worker reliability;
- estimated cost;
- utilization balance;
- local drift risk.

This would let a multi-robot or multi-model system route hard local dynamics
problems to specialists without changing the base Cosmos3 generation interface.

## Recommended Use

1. Generate DROID or UMI action FD outputs using the existing notebooks.
2. Produce task-specific masks for objects, grippers, or contact regions.
3. Call `evaluate_fd_rollout` with a `RolloutSpec`.
4. Write `topology_metrics.json` and, when useful, `topology_metrics.csv` beside
   the generated rollout output.
5. Inspect drift spikes at autoregressive chunk boundaries.

The metric is diagnostic. It should not be reported as generation improvement
unless paired with a controlled validation protocol.
