# JSONL Dataset

This guide describes the JSONL dataset format.

Prerequisites:

- [Training](./training.md)

## Inference

Run inference on a single sample:

```shell
export DATASET_PATH=$(uvx hf@latest download --repo-type dataset nvidia/bridge-v2-subset-synthetic-captions --revision 46468e12ac0dd36901e9e3240d4fc7620942b5d7 --quiet)/sft_dataset_bridge

torchrun --nproc-per-node=8 -m cosmos3.scripts.inference \
    --parallelism-preset=latency \
    -i "$DATASET_PATH/val/inference_prompt*/episode_049683_clip000.json" \
    -o outputs/train_inference \
    --checkpoint-path Cosmos3-Nano \
    --seed=0
```

- The ground truth video is in `${DATASET_PATH}/val/videos/`.
- The input image for I2V is in `${DATASET_PATH}/val/images/`.
- The input 5-frame video clip for V2V is in `${DATASET_PATH}/val/videos_5frames/`.

### Result Comparison

Each example below uses the following layout:

- Row 1 (T2V): ground truth video (left), before SFT (middle), after 500 iterations of SFT (right).
- Row 2 (I2V): input image (left), before SFT (middle), after 500 iterations of SFT (right).
- Row 3 (V2V): 5-frame input clip (left), before SFT (middle), after 500 iterations of SFT (right).

**episode_049683_clip000**

<details><summary><b>Input prompt</b></summary>

> A robotic arm with articulated joints and a gripping mechanism is positioned centrally on a wooden kitchen countertop, manipulating a small silver metal object while also interacting with scattered black coffee beans. The arm moves the metal object slightly, adjusting its position before shifting focus to the coffee beans, scattering and repositioning them with precision. The countertop is surrounded by kitchen elements, including a stove on the right, a microwave on the left, and two canned goods labeled "Tomato Juice" and "Baking Soda" in the background. The scene is illuminated by bright, even indoor lighting, casting minimal shadows, and the camera remains static throughout, offering a top-down perspective that emphasizes the robotic arm's movements. The composition centers on the robotic arm and its interaction with the metal object and coffee beans, with a shallow depth of field keeping the focus sharp on these elements while softly blurring the background. The overall atmosphere is technical and functional, highlighting the precision and control of the robotic manipulation within a domestic kitchen setting.

</details>

<video src="https://github.com/user-attachments/assets/d85ccc4f-7eea-46e0-afda-955ab9e19cbe" controls width="100%"></video>

**episode_009171_clip000**

<details><summary><b>Input prompt</b></summary>

> A robotic arm with a black and metallic gripper, accented with blue near its base, extends over a white rectangular tray filled with scattered brown almonds, methodically picking up and placing each almond in a precise line across the tray's surface. The arm moves with deliberate, controlled motion, shifting its position to reach different almonds while maintaining a top-down perspective that captures the entire workspace. The background reveals an indoor setting with a wooden table and various kitchen items, including a metal bowl and utensils, subtly visible behind the tray. The lighting is bright and evenly distributed, casting minimal shadows and highlighting the contrast between the white tray, the brown almonds, and the metallic sheen of the robotic arm. The camera remains static throughout, offering a wide-angle view that emphasizes the robotic arm's precision and the systematic rearrangement of the almonds, creating a clean, minimalist aesthetic that underscores the technical nature of the task. The scene unfolds as a continuous, uninterrupted sequence, showcasing the robotic arm's efficiency in organizing the almonds without any cuts or transitions.

</details>

<video src="https://github.com/user-attachments/assets/50623895-c49d-4dbb-a0c0-5281aaac5c23" controls width="100%"></video>

## Format

Example sample:

```json
{
    "uuid": "episode_000015_clip000",
    "duration": 17.4,
    "width": 256,
    "height": 256,
    "vision_path": "videos/episode_000015_clip000.mp4",
    "t2w_windows": [
        {
            "start_frame": 0,
            "end_frame": 86,
            "temporal_interval": 1,
            "caption": "A black robotic arm, featuring articulated joints and a metallic finish, extends over a white tray placed on a wooden table, manipulating small black objects that resemble beads or marbles. The arm moves with precision, grasping clusters of these objects, lifting them, and relocating them across the tray\u2019s surface in a methodical manner, often shifting them from one side to another. The background reveals an indoor workspace with visible equipment, illuminated by bright, even lighting that casts minimal shadows, emphasizing the technical nature of the scene. The camera remains static throughout, offering a medium shot that centers on the robotic arm and tray, with a slightly angled top-down perspective that highlights the contrast between the black objects, white tray, and wooden table. The robotic arm\u2019s movements are continuous and deliberate, showcasing its ability to handle and reposition the objects with accuracy, while the scene maintains a minimalist and functional aesthetic throughout."
        }
    ]
}
```

## Video Captioning

If you have video sources and would like to synthesize caption annotations to build video–text pairs for training, follow this section for data preprocessing. The script sends each video directly to a Vision-Language Model (VLM), which analyzes the visual content and produces a dense narrative caption following a two-phase process (scene analysis → narrative rewrite) — the same format expected by the Cosmos3 training pipeline.

The captioning prompt template is available at [`cosmos3/defaults/video_captioner.txt`](../cosmos3/defaults/video_captioner.txt).

### Server setup

The captioning script passes video files to vLLM via `video_url` content parts using `file://` paths, so the server must be able to read files from the local filesystem. We recommend [Qwen/Qwen3-VL-8B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) as the VLM. Start the server — this may take a couple of minutes:

```shell
uvx --with nvidia-cuda-runtime-cu12 \
    vllm@0.19.0 serve Qwen/Qwen3-VL-8B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --allowed-local-media-path /
```

The server is ready when you see `Application startup complete.`

### Run Video Captioning

Caption a single video:

```shell
python -m cosmos3.scripts.caption_from_video \
    --video /path/to/video.mp4 -o outputs/captions \
    --server http://localhost:8000/v1
```

Caption all `.mp4` files in a directory:

```shell
python -m cosmos3.scripts.caption_from_video \
    --video /path/to/videos/ -o outputs/captions \
    --server http://localhost:8000/v1
```

Caption videos listed in a JSONL manifest (each line must have a `vision_path` field pointing to a video):

```shell
python -m cosmos3.scripts.caption_from_video \
    -i samples.jsonl -o outputs/captions \
    --server http://localhost:8000/v1
```

Options:

| Flag                     | Default  | Description                      |
| ------------------------ | -------- | -------------------------------- |
| `--max_workers`          | `16`     | Concurrent API requests          |
| `--prompt_template_path` | built-in | Path to a custom prompt template |
| `--debug`                | `False`  | Save raw API responses           |

Each video produces an output directory containing `caption.txt` (the plain-text caption) and `sample_args.json` (metadata).

### Create Dataset

After generating the captions, you will have videos and captions stored in the following file structure:

```
path/to/dataset/
└── captions/
└── videos/
```

To create a video dataset JSONL file for post-training, run the following command:

```
python -m cosmos3.scripts.captions_to_sft_jsonl \
    --captions-dir outputs/sft_dataset/train/captions \
    --videos-dir outputs/sft_dataset/train/videos \
    -o outputs/sft_dataset/train/video_dataset_file.jsonl
```

It will create a dataset JSONL file containing captions and their corresponding paths to video files.
