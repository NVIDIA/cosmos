# Benchmark measurement settings

[Back to the inference benchmark index](../inference_benchmarks.md)

The benchmark pages publish measured latency and throughput with the reproducible settings retained for each workload. Measurements come from several runtime versions and campaigns; equal model, GPU, and resolution labels alone do not establish an identical workload. Missing settings are identified below rather than inferred from an example recipe.

## Reading a Generator table

Every value is seconds. Each row has one **GPUs** count that applies to every resolution in that row. Newly added Generator results use the current 1-or-8 GPU selection; two-GPU measurements are excluded, and previously published GPU counts are preserved. Keep that count fixed when comparing cells. Engines can require different counts on the same hardware. An empty cell has no validated result at that count; an out-of-memory failure, timeout, or unresolved measurement can also leave a cell empty.

The 256p tier is 320×192, 480p is 832×480, and 720p is 1280×720 where those dimensions were recorded. Nano and Super primary video workloads use 189 frames at 24 FPS. The historical Edge PyTorch and vLLM-Omni video sweep uses 121 frames. Action rollout lengths, action chunks, and transfer inputs are separate workload dimensions; an action row must not be treated as a primary-video run at the same resolution.

Generation time includes the runtime's measured generation path, including decoding where recorded; diffusion time covers denoising only. Client request time and server inference time have additional, different boundaries. The vLLM-Omni BF16 transfer rows use end-to-end wall time, converted from milliseconds to seconds. The Edge FP8 page explicitly reports request/inference time and must not be read as generation-only latency. Historical embedded Edge end-to-end timings also remain separate from datacenter generation timings.

## Generator runtime settings

| Runtime | Recorded configuration and selection |
|---|---|
| PyTorch, BF16 | Native OSS generation; latency automatic-sharding preset; CUDA graphs disabled for audiovisual and action measurements. Primary results are included where validated. |
| PyTorch, FP8 | Native generation, with a quantization recipe that differs from the mixed-step FP8 recipe used by other engines. The marked primary rows are not directly comparable with those engines. |
| vLLM-Omni, BF16 | Nano/Super primary and audiovisual runs use the 0.20 integration. Action runs use the `vllm/vllm-omni:cosmos3` image and public action samples. Forward-AV values aggregate `av_forward`, `av_left`, and `av_right`; inverse-AV values aggregate `av_inverse_0` and `av_inverse_1`. No forced `TORCH_SDPA` override was set. |
| vLLM-Omni, FP8 Nano/Super | Generation-time metric. `FLASHINFER_ATTN` is the main attention path; RTX PRO 6000 uses `CUDNN_ATTN`. Some multi-GPU configurations use a cuDNN fallback after FlashInfer incompatibility or memory failures. No action sample is assigned to an embodiment when the measurement does not identify it. |
| Diffusers, BF16 | Primary single-GPU, additional-modality, transfer, and multi-GPU workloads were measured separately. The additional single-GPU modality sweep reports best-run generation time. The multi-GPU sweep uses a different denoising-step budget, so its single-GPU values are excluded. At a published multi-GPU count, the fastest measured tensor/context-parallel split is selected. |
| Diffusers, FP8 | Generation time; 480p and 720p measurements. Both single- and multi-GPU results exist. Recorded eight-GPU splits include TP1/CP8 on H200 and TP2/CP4 on H100 or H20. |
| SGLang, BF16 | Average generation time. At a fixed GPU count, the fastest measured TP/CFG/Ring/Ulysses split is selected. Super needs multiple GPUs on several smaller-memory devices; H200 single-GPU results are also absent. |
| SGLang, FP8 | Average generation time at 480p and 720p. Recorded eight-GPU splits include TP1/CFG2/SP4/Ring1/Ulysses4 on H200 and TP2/CFG2/SP2/Ring1/Ulysses2 on H100 or H20. `GPUs = TP × CFG × SP`, with `SP = Ring × Ulysses`. |
| TensorRT-LLM | Average generation time. The fastest measured TP/CFG/Ulysses split is selected at the row's GPU count. The measured hardware set excludes H100 NVL and H200 NVL, and this sweep has no action or FP8 results. |
| NIM Generator | FP8 latency profiles, also shown on the corresponding FP8 pages. The refreshed primary workload uses 35 steps, 189 video frames at 24 FPS, concurrency 1, one warmup request, and 10 measured image requests or 3 measured video requests. Generation-time measurements are distinct from client request latency. |

The FP8 mixed-step recipe retains the first and last three denoising steps in higher precision. Precision labels describe weights and the recorded compute recipe, not necessarily the KV cache, every layer, or every denoising step. On hardware without native FP8 compute, a weight-only path can be used; the Edge FP8 Reasoner page identifies that case.

NIM 2.0.0 Nano measurements on DGX Spark and RTX PRO 5000 are separate from the other displayed NIM latency profiles. They use FP8, 189-frame video at 24 FPS with 35 steps and guidance 6, and 61-frame AV actions at 10 FPS with 30 steps and guidance 1. Transfer uses 25 frames and 50 steps: ordinary controls use 30 FPS and guidance 3; WSM uses 10 FPS and guidance 1. Hint guidance is 1.5 for edge/depth, 2 for segmentation, and 3 for WSM. Blur uses 736×544 and is excluded from the existing resolution columns.

## Reasoner workload and timing

The vLLM tables report AIPerf serving metrics. Concurrency counts simultaneous client requests; it is not tensor parallelism. TTFT is time to first output token, request latency is end-to-end request time, request throughput counts completed requests per second, and output throughput counts generated output tokens per second. Request count identifies the population used by the measurement. Some additional platform tables retain rounded source values; trailing zeros do not add measurement accuracy.

The FP8 video-serving workload uses 50 text input tokens plus one 374×374, 30-second synthetic video at 1 or 2 FPS, and requests 1 or 100 output tokens. The Edge embedded vLLM page supplies memory-utilization settings, runtime versions, kernel overrides, and common serving arguments. It distinguishes native Thor FP8 computation from Orin weight-only computation and identifies T3000 emulation.

NIM measurements are separate sections. The Edge NIM 2.1.0 H.264 workload uses NVDEC; its VP9 workload is labelled separately and uses CPU decoding through OpenCV. Edge NIM uses video-token pruning 0.6 with `vidcom2`, GPU memory utilization 0.7, and a maximum batched-token budget of 24,300. Its FPS label alone does not imply the same input work as vLLM. Embedded Thor can also drop the last video frame and reduce its visual-token budget at higher FPS. Nano/Super NIM measurements use two 10-second video workloads with pruning and caches disabled; their input/output token counts and frame rates are stated with the tables. Eager Transformers measurements are a third, separate timing protocol.

Some single-token serving runs returned empty replies. Where successful-request metrics are retained, the text explicitly says so; those metrics do not establish a zero-error serving rate. Failed NIM Edge runs are excluded. Never combine successful-request throughput with a different request population.

## Reproduction limits

Exact prompts, seeds, action payloads, runtime builds, and warmup/repetition counts were not retained with every historical measurement. The newer SGLang and Nano/Super FP8 vLLM-Omni tables do not provide a complete pinned software environment. Multi-GPU split selection can vary by workload and resolution. These gaps prevent an exact replay of every table cell, and measurements must not be presented as a controlled cross-runtime ranking.

Use the public [audiovisual cookbook](https://nvda.ws/4zd0fw5), [action cookbook](https://nvda.ws/4dUeBsJ), and linked model recipes for representative requests. Their sample-specific frame counts and action chunks do not establish the exact benchmark payload. No missing setting or timing is reconstructed from a recipe, a speedup summary, or a different model.

## Publication criteria

Only results that pass source and measurement validation are published. Measurements awaiting validation remain blank, including affected comparison sets when their workload parity or results are unresolved. No value is inferred or adjusted to match an expected runtime, precision, GPU, or resolution ranking.
