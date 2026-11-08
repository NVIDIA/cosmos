# Benchmark anomaly review

[Back to the inference benchmark index](../inference_benchmarks.md)

The Cosmos3 Edge, Nano, and Super benchmark tables, including FP8 Generator and Reasoner results, contain unresolved performance anomalies and coverage gaps. Agreement with a source measurement establishes transcription accuracy; it does not validate the workload or explain surprising performance. See [measurement settings](benchmark_methodology.md) before comparing results.

## Review scope and checks

The review scanned 9,633 published numeric cells: 5,504 Generator and 4,129 Reasoner values, including eager Transformers measurements. Generator runtime comparisons require the same model, effective precision, modality, embodiment/control, resolution, GPU, and GPU count. Reasoner comparisons also match published input/output lengths, video FPS, concurrency, and metric. These dimensions cannot establish parity where prompts, token pruning, action payloads, timing boundaries, or software environments differ.

The scan found 1,559 cross-runtime metric pairs differing by more than 2×. Of these, 230 lack a comparison caveat identified by the scan and need workload-level investigation. These are pairs, not independent experiments or confirmed bugs. FP8 NIM copies on base-model and FP8 pages are deduplicated for comparisons. Request counts are excluded from runtime performance comparisons.

GPU comparisons use ten plausible device/platform orderings as diagnostic hypotheses, not a universal GPU ranking. They flag 450 metric comparisons with a reversal above 5%, including 75 above 2×. There are also 22 cases where the next higher resolution is over 10% faster, and six serving-throughput scaling flags above 1.8× or below 0.5× between adjacent tested high-concurrency settings. A throughput increase when concurrency doubles can be legitimate; the flags select cases for inspection, not automatic rejection. Related latency/throughput metrics can flag the same run more than once. Repeated-run variance is unavailable for many measurements, so small differences do not establish regressions.

## Documentation corrections

- **Corrected six Super FP8 PyTorch transfer cells.** Workflow labels had incorrectly carried across source-table sections, assigning policy timings to edge-controlled transfer. The corrected DGX Spark values are 445.80 / 3,247.40 / 12,642.00 s at 256p / 480p / 720p; DGX Station is 26.00 / 169.60 / 625.50 s.
- **Withheld 45 Super FP8 enterprise vLLM Reasoner values.** Rows labelled output length 1 have output-token rates approximately 100 times request rates, along with conflicting workload/concurrency labels. Other source rows have fractional request counts or repeated unrelated values. The entire enterprise block is withheld rather than relabelled from ratios. Corrected request-level results are required.
- **Added 193 omitted numeric table cells.** These comprise 100 NIM 2.0.0 cells across the Nano and Nano FP8 pages (50 distinct configurations displayed twice) and 93 additional enterprise Generator cells. The latter include correctly identified native video-to-video and Diffusers vision, audiovisual, transfer, and action workloads.
- **Removed unsupported comparison claims.** Workloads are not universally identical across engines; GPU count does not specify tensor/context/CFG/Ring/Ulysses layout. NIM profiles and GPU-count selections must be matched explicitly.

All previously published numeric cells from before this PR remain unchanged. After withholding the inconsistent new block, the scan finds no nonpositive/nonfinite values, fractional published request counts, TTFT above request latency, or output/request-throughput identities inconsistent with the requested output length beyond source rounding and the review tolerance. Passing these checks does not validate every measurement.

## Largest findings requiring investigation

All Generator times below are seconds. Values remain published unless the workload labels or measurements are demonstrably inconsistent.

| Area | Observation | Interpretation and next check |
|---|---|---|
| Super BF16 runtime gap | RTX PRO 6000 Blackwell ×8, t2i, 256p: PyTorch **93.18**, vLLM-Omni **3.11** (**29.96×**), SGLang **2.01** (**46.36×** versus PyTorch). | Values match the available sources. Confirm denoising steps, warmup, sharding, kernel path, and payload; SGLang's frame/step/build metadata is incomplete. These are not established engine speedups. |
| FP8 video-to-video runtime gap | Nano B300, 1 GPU, 720p: Diffusers **118.73**, vLLM-Omni **15.44** (**7.69×**). Super B300 is **384.10** versus **55.19** (**6.96×**). | The gap persists across devices. Verify frame/step budgets, precision recipe, attention path, and timing boundary with a matched workload. |
| Nano BF16 GPU reversal | Diffusers robot forward dynamics, 1 GPU, 256p: B300 **22.65**, B200 **2.00** (**11.33×**). Super is **22.21** versus **3.00** (**7.40×**). | B300 was rerun with PyTorch 2.12.0 / CUDA 13 in a different environment. Reproduce both GPUs with the same payload, build, attention backend, and step budget. |
| Nano FP8 platform reversal | vLLM-Omni t2i, 1 GPU, 480p: GB300 **12.27**, B300 **5.11** (**2.40×**). GB300 then takes **6.89** at 720p. | Profile warmup, host behavior, kernels, steps, and dispatch thresholds. Do not infer a universal GPU order from this small workload. |
| Super FP8 resolution reversal | GB300 vLLM-Omni t2i, 1 GPU: **20.95** at 480p versus **12.36** at 720p. | Source agreement leaves the reversal unexplained; obtain a matched rerun. |
| Other resolution reversals | Nano BF16 SGLang B300 t2i: **3.12** at 480p versus **1.80** at 720p. Edge BF16 PyTorch B300 t2i: **3.14** at 256p versus **1.93** at 480p. | Confirm resolution labels, denoising budgets, and kernel/compilation behavior before comparing tiers. |
| Edge FP8 image regression | The matched RTX PRO 6000 campaign reports FP8 about **4× slower** than BF16; RTX PRO 5000 is **5.79** versus **1.69** (**3.43×**). | These use matched campaign controls, not the historical BF16 table. The failed attention paths and cuDNN fallback need profiling. FP8 does not guarantee a latency improvement. |
| Edge FP8 serving discontinuity | DGX Spark, input 50 / output 100 / video 1 FPS: output throughput rises from **85.44** to **170.26 tok/s** between concurrency 128 and 256 after a near plateau from 64 to 128; request latency falls from **129,088.36** to **121,437.58 ms**. | This reproduced the earlier jump but remains unexplained. Inspect scheduling, batch composition, completion counts, and run duration; source reproduction alone does not explain it. |

Nano FP8 NIM shows another high-concurrency change: RTX PRO 6000, input 100 / output 1 / video 4 FPS, rises from 3.94 to 8.37 requests/s between concurrency 128 and 256 after a near plateau, while latency falls from 26,336.40 to 24,627.21 ms. Inspect scheduling and run duration before treating it as a stable capacity increase.

## Source caveats retained with the tables

- **Timing and workload boundaries:** Edge FP8 mixes explicitly identified server inference and client request timing. BF16 vLLM-Omni transfer is end-to-end wall time. Other tables use generation time; some Diffusers modality measurements report the best run. NIM image profiles use 35 steps versus 50 in the recorded vLLM-Omni profile. Native PyTorch FP8 uses a different quantization recipe. These distinctions prevent an unqualified runtime ranking.
- **Compilation and initialization failures:** Native BF16 B300 runs hit CUTLASS initialization failures; Super H100 SXM runs encountered NCCL failures. FP8 multi-control on datacenter Blackwell failed TorchInductor dynamic-shape compilation. Edge FP8 FlashInfer failed in the measured container, FlashAttention failed on Blackwell, and T3000 emulation required `MAX_JOBS=2` to avoid compilation memory exhaustion. Identical warmup excludes the approximately six-second cold FP8 transition cost.
- **Memory and timeouts:** Some Super H100/H20 FP8 single-GPU workloads ran out of memory, and H20 SGLang runs timed out. Super TensorRT-LLM H200 SXM 720p video/audiovisual workloads and some multi-control splits failed; Diffusers H200 TP1/CP8 720p video/audio also failed. A separately reported seven-second Super vLLM-Omni H200 ×8 i2av 720p value was explicitly invalidated by out-of-memory failure and is excluded. Failures apply to the recorded runtime and split.
- **Action and transfer payloads:** Edge FP8 embedded policy uses a 736×544 conditioning canvas, with inconsistent cross-campaign chunk descriptions. SGLang inverse-camera labels require embodiment confirmation; unidentified generic action samples are not assigned to AV/robot rows. Some blur retries use 4:3 versus 16:9 source assets. These are not controlled payload comparisons.
- **Reasoner video processing:** Edge NIM uses 60% video-token pruning (`vidcom2`), unlike the separately measured vLLM input path. H.264/NVDEC and VP9/CPU decoding remain separate. Embedded frame dropping and visual-token budgets can differ; FPS alone does not identify equal input work. Empty single-token replies are disclosed, and successful-request metrics do not establish zero errors. Orin's Reasoner weight-only path does not imply Generator FP8 support.
- **Withheld enterprise timings:** A Nano FP8 campaign has contradictory runtime identification and also reports thermal throttling and DGX Station t2i/v2v regressions. It remains unassigned rather than being merged into an engine's table.

## Coverage still missing

The selected-source inventory has no remaining validated Generator value at the selected 1-or-8 GPU count that is simply omitted from its corresponding blank cell. This is a limited inventory check, not a claim of complete report coverage. Separate campaigns and incomplete exports below still need reconciliation. Blank cells also include reserved combinations, failures, and measurements at other GPU counts; they must not all be counted as missing successful experiments.

| Coverage gap | Work needed before adding results |
|---|---|
| AMD Edge/Nano/Super campaigns, including MI300X, MI325X, and MI355X where measured | Reconcile newer runs, dependency pins, precision, and generation-versus-diffusion timing. These are within the model scope; their absence is not evidence of lack of support. |
| Regional Nano BF16 primary and transfer campaigns | Reconcile L20, RTX PRO 5000 variants, RTX 4090 48GB, RTX 5880 Ada, RTX 6000D, and DGX Spark measurements. Preserve 48GB audiovisual memory failures and the reported RTX 6000D performance reversal. Some transfer runs use locked clocks/power and 121-frame / 30-FPS controls or 101-frame / 10-FPS WSM; they cannot inherit the NIM 25-frame profile. |
| Separate certified and earlier NIM Reasoner profiles | The published Nano/Super NIM tables use distinct 10-second VSS workloads. Other profiles use different video/token-pruning settings. NVDEC allocations failed at memory utilization 0.93; two reruns succeeded at 0.85. Reconcile complete successful grids before publishing. |
| Edge Policy-DROID NIM | Identify the dedicated BF16 policy model and separate four-step/guidance-3 from 30-step/guidance-1 results. Do not merge with base Edge FP8 policy or normalize only by step count. |
| Datacenter/GB300 FP8 Reasoner and incomplete BF16/transfer exports | Obtain complete, validated tables and workload labels. Super's inconsistent enterprise vLLM block remains withheld. |
| NIM 2.0.0 Nano transfer blur | Two single-GPU measurements use 736×544, which is not the existing 480p tier. They remain excluded without changing the current resolution format. |
| Two-GPU alternatives | Intentionally excluded by the current 1-or-8 selection; no new GPU-count format is introduced. Previously published counts remain intact. |

Of 3,382 pre-existing Generator cells, 3,216 match available source values at display precision and 166 still lack an exact complete source match. Most unresolved cells are BF16 vLLM-Omni transfer or newer primary results with incomplete accessible exports; others need workload-label or source-version confirmation. Five pre-existing Nano BF16 Reasoner cells retain a 511-versus-512 request-population discrepancy. These values are preserved pending resolution; see [coverage and validation notes](benchmark_update_notes.md).

Before treating the tables as a controlled comparison, obtain corrected serving exports, reconcile the remaining campaigns, and rerun the largest unexplained gaps with identical public payloads, precision recipes, steps, parallel splits, warmup/repetition policy, clocks, and software versions. Publish the reproducible setup and outcome with each resolved anomaly.
