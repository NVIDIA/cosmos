# Asset Specifications

Every visual placeholder in the prototype, with exact production specs. Owner column is TBD — assign per asset.

## Brand (`assets/brand/`)

| File | Spec |
|---|---|
| `cosmos-logo.png` | ✅ In place — the existing Cosmos logo (274×152) copied from the current repo root. Used by the README today. |
| `cosmos-wordmark-{light,dark}.png` | Upgrade path: dark/light lockups via `<picture>` + `prefers-color-scheme`, PNG, transparent, ~3× render size for retina, NVIDIA green (#76B900) accent. Swap into the README hero when produced. |

## Demo grid (`assets/demos/`) — the six hero assets ✅ in place

All six are sourced from the [Cosmos 3 research page](https://research.nvidia.com/labs/cosmos-lab/cosmos3/), converted to GIF (480px wide, 12 fps, first 4 s, ≤ 5 MB — GitHub READMEs won't autoplay video). Source MP4s kept under `demos/mp4/` for future `<video>` upgrades or re-cuts.

| File | Source | Shows |
|---|---|---|
| `transfer_worldscenario.gif` | Internal collage (rows 1–4, cols 1/2/5), middle column removed, 640×360 @15 fps | World-scenario transfer: control layout (top) → generated drive (bottom) |
| `physics_newton_cradle.gif` | Internal eval sample `eval_newton_0016_right.mp4`, 3× speed, 640px @16 fps | Physics-aware generation: Newton's cradle momentum transfer |
| `policy_screwdriver.gif` | `robot-policy/20260520_01_…_trim_4x.mp4` | Real robot policy execution |
| `driving_sim_falling_rocks.gif` | `audio-visual-gen/falling_rocks2.mp4` | Driving simulation: hazard scenario — "Falling Rocks" sign appears, rocks fall, car stops |
| `fd_egocentric_repair_poses.gif` | `forward-dynamics/egocentric-011{,-camera-trajectory,-hand-pose}.mp4` (composited) | Forward dynamics rollout with input camera + hand pose overlays |
| `reasoner_driving_hazard.gif` | Provided GIF `cosmos-world-reasoning-3s.gif` (2026-09-17), re-encoded 600px @14 fps, 128 colors; no MP4 source on hand | World Reasoner: dashcam hazard anticipation — rolling ball, reasoning overlay legible |

Re-cut guidance if replacing: obvious motion in frame 1, no watermark text, captions live in the README.

## Diagrams

| File | Spec |
|---|---|
| `assets/cosmos3-architecture.png` | Existing architecture diagram migrates from `cookbooks/cosmos3/`; produce dark-mode variant, referenced from `docs/reference/models.md`. |
| `assets/platform-map.png` | *(Recommended, not yet placed)* Supabase-style one-glance diagram: model family → this repo (framework, cookbooks, recipes) → ecosystem (Curator, Evaluator, Diffusers/vLLM/NIM). Would slot between "What is Cosmos?" and "Find your path". |

## Repo-level

| Item | Spec |
|---|---|
| Social preview (repo settings) | 1280×640 PNG: wordmark + one hero frame from `t2v_warehouse`. This is what every share card on X/LinkedIn/Slack shows. |
| Favicon-scale mark | Square Cosmos mark for HF org page and docs site if/when one exists. |
