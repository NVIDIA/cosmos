#!/usr/bin/env bash
set -euo pipefail

# Run VLMEvalKit benchmark against a custom inference endpoint.
# Supports running a single dataset or all datasets with --all.

# dataset_name:class_name
ALL_DATASETS=(
    "MMMU_DEV_VAL:MMMUDataset"
    "MathVista_MINI:MathVista"
    "MathVision:MathVision"
    "MMBench_DEV_EN:ImageMCQDataset"
    "MMBench_DEV_EN_V11:ImageMCQDataset"
    "HallusionBench:ImageYORNDataset"
    "InfoVQA_VAL:ImageVQADataset"
    "DocVQA_VAL:ImageVQADataset"
    "AI2D_TEST:ImageMCQDataset"
    "CountBenchQA:CountBenchQA"
    "CosmosERQA:CosmosERQA"
    "OCRBench_v2:OCRBench_v2"
)

usage() {
    cat <<'EOF'
Usage: ./run_endpoint_bench.sh --endpoint URL --model MODEL (--dataset DATASET | --all) [OPTIONS]

Required:
  --endpoint URL        Full chat completions URL (e.g. https://...lepton.run/v1/chat/completions)
  --model MODEL         Model name served at the endpoint (e.g. Qwen3-VL-8B-Instruct)
  --dataset DATASET     Dataset name (e.g. CountBenchQA)
  --all                 Run all built-in datasets

Optional:
  --dataset-class CLASS Dataset class name (e.g. ImageVQADataset; default: auto-detect or same as dataset)
  --temperature FLOAT   Sampling temperature (default: 0)
  --max-tokens INT      Max tokens to generate (default: 16384)
  --presence-penalty FLOAT  Penalize repeated tokens to reduce repetitive reasoning (default: not set)
  --retry INT           Number of retries (default: 10)
  --timeout INT         Request timeout in seconds (default: 300)
  --enable-thinking     Enable model thinking/reasoning (default: disabled)
  --work-dir DIR        Output directory (default: ./outputs)
  --api-nproc INT       Parallel API calls (default: 4)
  --nframe INT          Number of frames for dataset-side video extraction (override dataset default)
  --fps FLOAT           FPS for dataset-side video extraction (override dataset default)
  --model-nframes INT   Number of frames for model-side video processing (for VIDEO_LLM models)
  --model-fps FLOAT     FPS for model-side video processing (for VIDEO_LLM models)
  --model-max-frames INT  Max frames cap for model-side video processing
  --run-name NAME       Custom name for the model entry in config (default: derived from model name)
  --keep-config         Do not delete the temp config file after run

Environment:
  COSMOS_API_KEY        API key for the inference endpoint (required)
  OPENAI_API_KEY        API key for LLM judge (required by some datasets)
  OPENAI_API_BASE       Custom base URL for LLM judge (optional)
  NGC_API_KEY           NGC API key for downloading datasets from DSS (required by some datasets)
  NVDATASET_TENANTID    Tenant ID for NVIDIA Data Services (required by some datasets)
EOF
    exit 1
}

# Run a single benchmark. Args: dataset_name, class_name
run_one() {
    local dataset="$1"
    local dataset_class="$2"

    # Resolve paths relative to this script (which lives alongside run.py)
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

    local config_file
    config_file=$(mktemp "${script_dir}/tmp_config_XXXXXX.json")

    # Build the data section with optional video fields
    local data_json="{\"class\": \"${dataset_class}\", \"dataset\": \"${dataset}\""
    if [[ -n "$NFRAME" ]]; then
        data_json="${data_json}, \"nframe\": ${NFRAME}"
    fi
    if [[ -n "$FPS" ]]; then
        data_json="${data_json}, \"fps\": ${FPS}"
    fi
    data_json="${data_json}}"

    # Build model-side video processing kwargs (independent of dataset config)
    local model_video_kwargs=""
    if [[ -n "$MODEL_NFRAMES" ]]; then
        model_video_kwargs="${model_video_kwargs}, \"nframes\": ${MODEL_NFRAMES}"
    fi
    if [[ -n "$MODEL_FPS" ]]; then
        model_video_kwargs="${model_video_kwargs}, \"fps\": ${MODEL_FPS}"
    fi
    if [[ -n "$MODEL_MAX_FRAMES" ]]; then
        model_video_kwargs="${model_video_kwargs}, \"max_frames\": ${MODEL_MAX_FRAMES}"
    fi

    cat > "$config_file" <<CONF
{
  "model": {
    "${RUN_NAME}": {
      "class": "CosmosReason2",
      "model": "${MODEL}",
      "api_base": "${ENDPOINT}",
      "temperature": ${TEMPERATURE},
      "max_tokens": ${MAX_TOKENS},
      "presence_penalty": ${PRESENCE_PENALTY},
      "retry": ${RETRY},
      "timeout": ${TIMEOUT},
      "chat_template_kwargs": {"enable_thinking": ${ENABLE_THINKING}},
      "verbose": false${model_video_kwargs}
    }
  },
  "data": {
    "${dataset}": ${data_json}
  }
}
CONF

    echo "=== Endpoint Benchmark ==="
    echo "Endpoint : $ENDPOINT"
    echo "Model    : $MODEL"
    echo "Dataset  : $dataset (class: $dataset_class)"
    echo "Run Name : $RUN_NAME"
    echo "Config   : $config_file"
    echo "Work Dir : $WORK_DIR"
    echo "Thinking : $ENABLE_THINKING"
    echo "=========================="

    local rc=0
    python "${script_dir}/run.py" \
        --config "$config_file" \
        --work-dir "$WORK_DIR" \
        --api-nproc "$API_NPROC" \
        --save-eval-results || rc=$?

    if [[ "$KEEP_CONFIG" == "false" && -f "$config_file" ]]; then
        rm -f "$config_file"
    fi
    return $rc
}

# Defaults
TEMPERATURE=0
MAX_TOKENS=16384
PRESENCE_PENALTY=null
RETRY=10
TIMEOUT=300
ENABLE_THINKING=false
WORK_DIR="./outputs"
API_NPROC=4
RUN_NAME=""
KEEP_CONFIG=false
NFRAME=""
FPS=""
MODEL_NFRAMES=""
MODEL_FPS=""
MODEL_MAX_FRAMES=""
ENDPOINT=""
MODEL=""
DATASET=""
DATASET_CLASS=""
RUN_ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --endpoint)        ENDPOINT="$2"; shift 2 ;;
        --model)           MODEL="$2"; shift 2 ;;
        --dataset)         DATASET="$2"; shift 2 ;;
        --dataset-class)   DATASET_CLASS="$2"; shift 2 ;;
        --all)             RUN_ALL=true; shift ;;
        --temperature)     TEMPERATURE="$2"; shift 2 ;;
        --max-tokens)      MAX_TOKENS="$2"; shift 2 ;;
        --presence-penalty) PRESENCE_PENALTY="$2"; shift 2 ;;
        --retry)           RETRY="$2"; shift 2 ;;
        --timeout)         TIMEOUT="$2"; shift 2 ;;
        --enable-thinking) ENABLE_THINKING=true; shift ;;
        --nframe)          NFRAME="$2"; shift 2 ;;
        --fps)             FPS="$2"; shift 2 ;;
        --model-nframes)   MODEL_NFRAMES="$2"; shift 2 ;;
        --model-fps)       MODEL_FPS="$2"; shift 2 ;;
        --model-max-frames) MODEL_MAX_FRAMES="$2"; shift 2 ;;
        --work-dir)        WORK_DIR="$2"; shift 2 ;;
        --api-nproc)       API_NPROC="$2"; shift 2 ;;
        --run-name)        RUN_NAME="$2"; shift 2 ;;
        --keep-config)     KEEP_CONFIG=true; shift ;;
        -h|--help)         usage ;;
        *) echo "Error: Unknown option: $1"; usage ;;
    esac
done

# Validate required args
if [[ -z "$ENDPOINT" || -z "$MODEL" ]]; then
    echo "Error: --endpoint and --model are required."
    usage
fi

if [[ "$RUN_ALL" == "false" && -z "$DATASET" ]]; then
    echo "Error: either --dataset or --all is required."
    usage
fi

if [[ -n "$NFRAME" && -n "$FPS" ]]; then
    echo "Error: --nframe and --fps cannot be set at the same time."
    exit 1
fi

if [[ -n "$MODEL_NFRAMES" && -n "$MODEL_FPS" ]]; then
    echo "Error: --model-nframes and --model-fps cannot be set at the same time."
    exit 1
fi

if [[ -z "${COSMOS_API_KEY:-}" ]]; then
    echo "Error: COSMOS_API_KEY environment variable is not set."
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Warning: OPENAI_API_KEY is not set. Some datasets require an LLM judge and will fail without it."
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo "Warning: NGC_API_KEY is not set. Some datasets require it to download data from DSS."
fi

if [[ -z "${NVDATASET_TENANTID:-}" ]]; then
    echo "Warning: NVDATASET_TENANTID is not set. Some datasets require it to download data from DSS."
fi

# Derive run name from model if not provided
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')
fi

# Single dataset mode: resolve class from ALL_DATASETS, --dataset-class, or fallback to dataset name
if [[ "$RUN_ALL" == "false" ]]; then
    if [[ -z "$DATASET_CLASS" ]]; then
        # Try to find the class from the built-in list
        for entry in "${ALL_DATASETS[@]}"; do
            if [[ "${entry%%:*}" == "$DATASET" ]]; then
                DATASET_CLASS="${entry##*:}"
                break
            fi
        done
        # Fallback: use dataset name as class name
        if [[ -z "$DATASET_CLASS" ]]; then
            DATASET_CLASS="$DATASET"
        fi
    fi
    run_one "$DATASET" "$DATASET_CLASS"
    exit $?
fi

# --all mode: loop through all datasets
PASSED=()
FAILED=()

for entry in "${ALL_DATASETS[@]}"; do
    dataset="${entry%%:*}"
    dataset_class="${entry##*:}"

    echo ""
    echo "########################################"
    echo "# [$((${#PASSED[@]} + ${#FAILED[@]} + 1))/${#ALL_DATASETS[@]}] $dataset"
    echo "########################################"
    echo ""

    if run_one "$dataset" "$dataset_class"; then
        PASSED+=("$dataset")
    else
        echo "*** FAILED: $dataset ***"
        FAILED+=("$dataset")
    fi
done

echo ""
echo "========================================"
echo "Summary: ${#PASSED[@]} passed, ${#FAILED[@]} failed out of ${#ALL_DATASETS[@]} total"
echo "========================================"
if [[ ${#PASSED[@]} -gt 0 ]]; then
    echo "Passed:"
    for d in "${PASSED[@]}"; do echo "  - $d"; done
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed:"
    for d in "${FAILED[@]}"; do echo "  - $d"; done
    exit 1
fi
