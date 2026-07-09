# Quickstart

Before running the evaluation script, you need to **configure** the VLMs and set the model_paths properly.

After that, you can use a single script `run.py` to inference and evaluate multiple VLMs and benchmarks at a same time.

## Step 0. Installation & Setup essential keys

**Installation.**

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

**Setup Keys.**

To infer with API models (GPT-4v, Gemini-Pro-V, etc.) or use LLM APIs as the **judge or choice extractor**, you need to first setup API keys. VLMEvalKit will use an judge **LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**
- You can place the required keys in `$VLMEvalKit/.env` or directly set them as the environment variable. If you choose to create a `.env` file, its content will look like:

  ```bash
  # The .env file, place it under $VLMEvalKit
  # API Keys of Proprietary VLMs
  # QwenVL APIs
  DASHSCOPE_API_KEY=
  # Gemini w. Google Cloud Backends
  GOOGLE_API_KEY=
  # OpenAI API
  OPENAI_API_KEY=
  OPENAI_API_BASE=
  # StepAI API
  STEPAI_API_KEY=
  # REKA API
  REKA_API_KEY=
  # GLMV API
  GLMV_API_KEY=
  # CongRong API
  CW_API_BASE=
  CW_API_KEY=
  # SenseNova API
  SENSENOVA_API_KEY=
  # Hunyuan-Vision API
  HUNYUAN_SECRET_KEY=
  HUNYUAN_SECRET_ID=
  # LMDeploy API
  LMDEPLOY_API_BASE=
  # MiniMax API
  MINIMAX_API_KEY=
  # You can also set a proxy for calling api models during the evaluation stage
  EVAL_PROXY=
  ```

- Fill the blanks with your API keys (if necessary). Those API keys will be automatically loaded when doing the inference and evaluation.
## Step 1. Configuration

**VLM Configuration**: All VLMs are configured in `vlmeval/config.py`. Few legacy VLMs (like MiniGPT-4, LLaVA-v1-7B) requires additional configuration (configuring the code / model_weight root in the config file). During evaluation, you should use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM. Make sure you can successfully infer with the VLM before starting the evaluation with the following command `vlmutil check {MODEL_NAME}`.

Note: For the Qwen-VL series models (Qwen-VL, Qwen2-VL, Qwen2.5-VL), the upper and lower bounds of the number of pixels specified in vlmeval/config.py are as follows:

```
min_pixels=1280 * 28 * 28,
max_pixels=16384 * 28 * 28,
```
Where 1280 is the maximum value recommended by Qwen for balancing performance, computational resources, and memory, and 16384 is the theoretical maximum value for model input. This setting has a positive effect on some vision tasks that require high resolution (such as document understanding). However, there is no practical basis for considering this setting. If you need to align with the official settings, you can remove these two values, or set them to the following values from the official Qwen demo:

```
min_pixels=256 * 28 * 28,
max_pixels=1280 * 28 * 28,
```

## Step 2. Evaluation

**New!!!**  We integrated a new config system to enable more flexible evaluation settings. Check the [Document](/docs/en/ConfigSystem.md) or run `python run.py --help` for more details 🔥🔥🔥

We use `run.py` for evaluation. To use the script, you can use `$VLMEvalKit/run.py` or create a soft-link of the script (to use the script anywhere):

**Arguments**

- `--data (list[str])`: Set the dataset names that are supported in VLMEvalKit (names can be found in the codebase README).
- `--model (list[str])`: Set the VLM names that are supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`).
- `--mode (str, default to 'all', choices are ['all', 'infer'])`: When `mode` set to "all", will perform both inference and evaluation; when set to "infer", will only perform the inference.
- `--api-nproc (int, default to 4)`: The number of threads for OpenAI API calling.
- `--work-dir (str, default to '.')`: The directory to save evaluation results.

**Command for Evaluating Image Benchmarks **

You can run the script with `python` or `torchrun`:

```bash
# When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior).
# That is recommended for evaluating very large VLMs (like IDEFICS-80B-Instruct).

# IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG, Inference and Evalution
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose
# IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG, Inference only
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose --mode infer

# When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference.
# However, that is only suitable for VLMs that consume small amounts of GPU memory.

# IDEFICS-9B-Instruct, Qwen-VL-Chat, mPLUG-Owl2 on MMBench_DEV_EN, MME, and SEEDBench_IMG. On a node with 8 GPU. Inference and Evaluation.
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose
# Qwen-VL-Chat on MME. On a node with 2 GPU. Inference and Evaluation.
torchrun --nproc-per-node=2 run.py --data MME --model qwen_chat --verbose
```

**Command for Evaluating Video Benchmarks**

```bash
# When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior).
# That is recommended for evaluating very large VLMs (like IDEFICS-80B-Instruct).

# IDEFICS2-8B on MMBench-Video, with 8 frames as inputs and vanilla evaluation. On a node with 8 GPUs. MMBench_Video_8frame_nopack is a defined dataset setting in `vlmeval/dataset/video_dataset_config.py`.
torchrun --nproc-per-node=8 run.py --data MMBench_Video_8frame_nopack --model idefics2_8
# GPT-4o (API model) on MMBench-Video, with 1 frame per second as inputs and pack evaluation (all questions of a video in a single query).
python run.py --data MMBench_Video_1fps_pack --model GPT4o
```

The evaluation results will be printed as logs, besides. **Result Files** will also be generated in the directory `$YOUR_WORKING_DIRECTORY/{model_name}`. Files ending with `.csv` contain the evaluated metrics.

### Frequently Asked Questions

#### Constructing Input Prompt: The `build_prompt()` Function
If you find that the model's output does not match the expected results when evaluating a specific benchmark, it could be due to the model not constructing the input prompt correctly.

In VLMEvalKit, each `dataset` class includes a function named `build_prompt()`, which is responsible for formatting input questions. Different benchmarks can either customize their own `build_prompt()` function or use the default implementation.

For instance, when handling the default [Multiple-Choice QA](https://github.com/open-compass/VLMEvalKit/blob/43af13e052de6805a8b08cd04aed5e0d74f82ff5/vlmeval/dataset/image_mcq.py#L164), the `ImageMCQDataset.build_prompt()` method combines elements such as `hint`, `question`, and `options` (if present in the dataset) into a complete question format, as shown below:

```
HINT
QUESTION
Options:
A. Option A
B. Option B
···
Please select the correct answer from the options above.
```

Additionally, since different models may have varying evaluation requirements, VLMEvalKit also supports customizing the prompt construction method at the model level through `model.build_prompt()`. For an example, you can refer to [InternVL](https://github.com/open-compass/VLMEvalKit/blob/43af13e052de6805a8b08cd04aed5e0d74f82ff5/vlmeval/vlm/internvl_chat.py#L324).

**Note: If both `model.build_prompt()` and `dataset.build_prompt()` are defined, `model.build_prompt()` will take precedence over `dataset.build_prompt()`, effectively overriding it.**

Some models, such as Qwen2VL and InternVL, define extensive prompt-building methods for various types of benchmarks. To provide more flexibility in adapting to different benchmarks, VLMEvalKit allows users to customize the `model.use_custom_prompt()` function within the model. By adding or modifying the `use_custom_prompt()` function, you can decide which benchmarks should utilize the model's custom prompt logic. Below is an example:

```python
def use_custom_prompt(self, dataset: str) -> bool:
    from vlmeval.dataset import DATASET_TYPE, DATASET_MODALITY
    dataset_type = DATASET_TYPE(dataset, default=None)
    if not self._use_custom_prompt:
        return False
    if listinstr(['MMVet'], dataset):
        return True
    if dataset_type == 'MCQ':
        return True
    if DATASET_MODALITY(dataset) == 'VIDEO':
        return False
    return False
```
Only when the `use_custom_prompt()` function returns `True` will VLMEvalKit call the model's `build_prompt()` function for the current benchmark.
With this approach, you can flexibly control which benchmarks use the model's custom prompt logic based on your specific needs, thereby better adapting to different models and tasks.

#### Model Splitting

Currently, VLMEvalKit automatically supports GPU resource allocation and model splitting between processes on the same machine. This feature is supported when the inference backend is `lmdeploy` or `transformers`, with the following behaviors:

- When launching with `python` command, the model is by default allocated to all available GPUs. If you want to specify which GPUs to use, you can use `CUDA_VISIBLE_DEVICES` environment variable.
- When starting with `torchrun` command, each model instance will be allocated to `N_GPU // N_PROC` GPUs, where `N_PROC` is the number of processes specified by the `--nproc-per-node` parameter in the torchrun command. The value of `N_GPU` is determined as follows:
  - If `CUDA_VISIBLE_DEVICES` environment variable is not set, `N_GPU` will be the total number of available GPUs.
  - If `CUDA_VISIBLE_DEVICES` environment variable is set, `N_GPU` will be the number of GPUs specified by the `CUDA_VISIBLE_DEVICES` environment variable, and only the specified GPUs will be utilized.
Below are specific examples of running evaluation tasks on a machine equipped with 8 GPUs:

```bash
<!-- Launch two model instances in data parallel, each instance using 4 GPUs -->
torchrun --nproc-per-node=2 run.py --data MMBench_DEV_EN --model InternVL3-78B
<!-- Launch one model instance, using all 8 GPUs -->
python run.py --data MMBench_DEV_EN --model InternVL3-78B
<!-- Launch three model instances, each instance using 2 GPUs, GPU 0 and 7 are not used -->
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc-per-node=3 run.py --data MMBench_DEV_EN --model InternVL3-38B
```

PS: The feature is not compatible with `vllm` backend. When you evaluate a model with `vllm` backend, please use `python` to launch, and all visible GPU devices will be used.

#### Performance Discrepancies

Model performance may vary across different environments. As a result, you might observe discrepancies between your evaluation results and those listed on the official VLMEvalKit leaderboard. These differences could be attributed to variations in versions of libraries such as `transformers`, `cuda`, and `torch`.

Besides, if you encounter unexpected performance, we recommend first reviewing the local generation records (`{model}_{dataset}.xlsx`) or the evaluation records (`{model}_{dataset}_{judge_model}.xlsx`). This may help you better understand the evaluation outcomes and identify potential issues.

## Deploy a local language model as the judge / choice extractor
The default setting mentioned above uses OpenAI's GPT as the judge LLM. However, you can also deploy a local judge LLM with [LMDeploy](https://github.com/InternLM/lmdeploy).

First install:
```
pip install lmdeploy openai
```

And then deploy a local judge LLM with the single line of code. LMDeploy will automatically download the model from Huggingface. Assuming we use internlm2-chat-1_8b as the judge, port 23333, and the key sk-123456 (the key must start with "sk-" and follow with any number you like):
```
lmdeploy serve api_server internlm/internlm2-chat-1_8b --server-port 23333
```

You need to get the model name registered by LMDeploy with the following python code:
```
from openai import OpenAI
client = OpenAI(
    api_key='sk-123456',
    base_url="http://0.0.0.0:23333/v1"
)
model_name = client.models.list().data[0].id
```

Now set some environment variables to tell VLMEvalKit how to use the local judge LLM. As mentioned above, you can also set them in `$VLMEvalKit/.env` file:
```
OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
LOCAL_LLM=<model_name you get>
```

Finally, you can run the commands in step 2 to evaluate your VLM with the local judge LLM.

Note that

- If you hope to deploy the judge LLM in a single GPU and evaluate your VLM on other GPUs because of limited GPU memory, try `CUDA_VISIBLE_DEVICES=x` like
```
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server internlm/internlm2-chat-1_8b --server-port 23333
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc-per-node=3 run.py --data HallusionBench  --model qwen_chat --verbose
```
- If the local judge LLM is not good enough in following the instructions, the evaluation may fail. Please report such failures (e.g., by issues).
- It's possible to deploy the judge LLM in different ways, e.g., use a private LLM (not from HuggingFace) or use a quantized LLM. Please refer to the [LMDeploy doc](https://lmdeploy.readthedocs.io/en/latest/serving/api_server.html). You can use any other deployment framework if they support OpenAI API.


### Using LMDeploy to Accelerate Evaluation and Inference

You can refer this [doc](/docs/en/EvalByLMDeploy.md)
