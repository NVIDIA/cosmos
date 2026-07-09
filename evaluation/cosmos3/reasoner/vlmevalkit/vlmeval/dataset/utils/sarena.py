import asyncio
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import nest_asyncio
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

try:
    from ...utils.mp_util import cpu_count
except ImportError:
    from os import cpu_count

# Import metrics
try:
    from .SArena.metrics import InternSVGMetrics, MetricsConfig
    # Try importing Video Metrics
    from .SArena.video.video_metrics import InternSVGVideoMetrics, VideoMetricsConfig
except ImportError:
    # Fallback or placeholder if relative import fails in standalone mode
    class MetricsConfig:
        pass

    class InternSVGMetrics:
        pass

    class InternSVGVideoMetrics:
        pass

    class VideoMetricsConfig:
        pass


try:
    from ...smp import LMUDataRoot, dump, load
except ImportError:
    # Fallback to pandas if smp is not available
    def load(file_path):
        if file_path.endswith(".xlsx"):
            return pd.read_excel(file_path)
        if file_path.endswith(".pkl"):
            return pd.read_pickle(file_path)
        return pd.read_csv(file_path)

    def dump(data, file_path):
        pass

    def LMUDataRoot():
        return os.path.expanduser("~/LMUData")


# ================= Configuration =================

SARENA_CONFIGS = {
    "SArena": {
        "root": os.path.join(LMUDataRoot(), "SArenaSrcData"),
        "url": "https://huggingface.co/datasets/JoeLeelyf/SArena-VLMEvalKit/resolve/main/SArena.zip",
        "zip_md5": "884c4a5eff55325e22c5f0aae21b13e7",
        "zip_name": "SArena.zip",
        "has_animation": True,
    },
    "SArena_MINI": {
        "root": os.path.join(LMUDataRoot(), "SArena_MINI_SrcData"),
        "url": "https://huggingface.co/datasets/JoeLeelyf/SArena-VLMEvalKit/resolve/main/SArena_MINI.zip",
        "zip_md5": "d49fe7241d16b54aa33a09e90d95af96",
        "zip_name": "SArena_MINI.zip",
        "has_animation": False,
    },
}

TOKENIZER_PATH = "OpenGVLab/InternVL3-8B"

# Chromium args for Pyppeteer
CHROME_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-software-rasterizer",
]

# ================= Task Definitions =================
# Tasks for full SArena (includes Animation)
TASK_CONFIGS_FULL = [
    # --- Icon ---
    ("SArena-Icon", "Understanding", "Icon/understanding/sarena_un.jsonl", False),
    ("SArena-Icon", "Generation-T2SVG", "Icon/generation/text2svg.jsonl", False),
    ("SArena-Icon", "Generation-I2SVG", "Icon/generation/img2svg.jsonl", True),
    ("SArena-Icon", "Edit-Color-Complex", "Icon/edit/color_complex.jsonl", False),
    ("SArena-Icon", "Edit-Color-Simple", "Icon/edit/color_simple.jsonl", False),
    ("SArena-Icon", "Edit-Crop", "Icon/edit/crop.jsonl", False),
    ("SArena-Icon", "Edit-Flip", "Icon/edit/flip.jsonl", False),
    ("SArena-Icon", "Edit-Opacity", "Icon/edit/opacity.jsonl", False),
    ("SArena-Icon", "Edit-Outline", "Icon/edit/outline.jsonl", False),
    ("SArena-Icon", "Edit-Rotate", "Icon/edit/rotate.jsonl", False),
    ("SArena-Icon", "Edit-Scale", "Icon/edit/scale.jsonl", False),
    (
        "SArena-Icon",
        "Edit-StyleTransform",
        "Icon/edit/styletransform_openmoji.jsonl",
        False,
    ),
    ("SArena-Icon", "Edit-Translate", "Icon/edit/translate.jsonl", False),
    # --- Illustration ---
    ("SArena-Illustration", "Generation-T2SVG", "illustration/text2svg.jsonl", False),
    ("SArena-Illustration", "Generation-I2SVG", "illustration/img2svg.jsonl", True),
    # --- Chemistry ---
    ("SArena-Chemistry", "Generation-T2SVG", "chemistry/text2svg.jsonl", False),
    ("SArena-Chemistry", "Generation-I2SVG", "chemistry/img2svg.jsonl", True),
    # --- Animation ---
    ("SArena-Animation", "Generation-T2SANI", "animation/text2sani.jsonl", False),
    ("SArena-Animation", "Generation-V2SANI", "animation/video2sani.jsonl", True),
]

# Tasks for MINI subset (no Animation)
TASK_CONFIGS_MINI = [
    # --- Icon ---
    ("SArena-Icon", "Understanding", "Icon/understanding/sarena_un.jsonl", False),
    ("SArena-Icon", "Generation-T2SVG", "Icon/generation/text2svg.jsonl", False),
    ("SArena-Icon", "Generation-I2SVG", "Icon/generation/img2svg.jsonl", True),
    ("SArena-Icon", "Edit-Color-Complex", "Icon/edit/color_complex.jsonl", False),
    ("SArena-Icon", "Edit-Color-Simple", "Icon/edit/color_simple.jsonl", False),
    ("SArena-Icon", "Edit-Crop", "Icon/edit/crop.jsonl", False),
    ("SArena-Icon", "Edit-Flip", "Icon/edit/flip.jsonl", False),
    ("SArena-Icon", "Edit-Opacity", "Icon/edit/opacity.jsonl", False),
    ("SArena-Icon", "Edit-Outline", "Icon/edit/outline.jsonl", False),
    ("SArena-Icon", "Edit-Rotate", "Icon/edit/rotate.jsonl", False),
    ("SArena-Icon", "Edit-Scale", "Icon/edit/scale.jsonl", False),
    ("SArena-Icon", "Edit-StyleTransform", "Icon/edit/styletransform_openmoji.jsonl", False),
    ("SArena-Icon", "Edit-Translate", "Icon/edit/translate.jsonl", False),
    # --- Illustration ---
    ("SArena-Illustration", "Generation-T2SVG", "illustration/text2svg.jsonl", False),
    ("SArena-Illustration", "Generation-I2SVG", "illustration/img2svg.jsonl", True),
    # --- Chemistry ---
    ("SArena-Chemistry", "Generation-T2SVG", "chemistry/text2svg.jsonl", False),
    ("SArena-Chemistry", "Generation-I2SVG", "chemistry/img2svg.jsonl", True),
]


def get_task_configs(dataset):
    """Return appropriate task configs based on dataset name."""
    if "MINI" in dataset:
        return TASK_CONFIGS_MINI
    return TASK_CONFIGS_FULL


EDIT_TASK_MAP = {
    "Edit-Color-Complex": "color_complex",
    "Edit-Color-Simple": "color_simple",
    "Edit-Crop": "crop",
    "Edit-Flip": "flip",
    "Edit-Rotate": "rotate",
    "Edit-Scale": "scale",
    "Edit-Translate": "translate",
    "Edit-Opacity": "opacity",
    "Edit-StyleTransform": "styletransform_openmoji",
    "Edit-Outline": "outline",
}

# ================= Data Prep Utilities =================


def _read_file_safe(file_path):
    """Safely read file contents with proper resource management."""
    if not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def check_md5(file_path, expected_md5):
    if expected_md5 is None:
        return True
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5


def download_and_extract_sarena(dataset="SArena"):
    config = SARENA_CONFIGS[dataset]
    sarena_root = config["root"]
    zip_path = os.path.join(sarena_root, config["zip_name"])

    if not os.path.exists(sarena_root):
        os.makedirs(sarena_root)

    need_download = not os.path.exists(zip_path)
    if not need_download and config["zip_md5"]:
        if not check_md5(zip_path, config["zip_md5"]):
            need_download = True

    if need_download:
        print(f"Downloading SArena data from {config['url']}...")
        try:
            response = requests.get(config["url"], stream=True)
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Download failed: {e}")
            return

    if not os.path.exists(os.path.join(sarena_root, "Icon")):
        print(f"Extracting {config['zip_name']}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(sarena_root)


def load_gt_captions(caption_path):
    caption_map = {}
    if not os.path.exists(caption_path):
        return caption_map

    print(f"  Loading captions from {caption_path}...")
    try:
        with open(caption_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    image_name = item.get("image", "")
                    if not image_name:
                        continue

                    uid_str = image_name.split(".")[0]
                    if uid_str.isdigit():
                        uid = int(uid_str)
                        captions_list = item.get("caption", [])
                        if isinstance(captions_list, list) and len(captions_list) > 0:
                            text = captions_list[0].get("generated_text", "").strip()
                            caption_map[uid] = text
                except Exception:
                    continue
    except Exception as e:
        print(f"  Error reading caption file: {e}")

    return caption_map


# ================= Rendering Utilities (Image) =================


def extract_svg_code(prediction_text):
    if not isinstance(prediction_text, str):
        return ""
    pattern = r"```svg(.*?)```"
    matches = re.findall(pattern, prediction_text, re.DOTALL)
    if matches:
        return matches[-1].strip()

    start = prediction_text.find("<svg")
    end = prediction_text.rfind("</svg>")
    if start != -1 and end != -1:
        return prediction_text[start: end + 6]

    if "<?xml" in prediction_text and "</svg>" in prediction_text:
        start = prediction_text.find("<?xml")
        end = prediction_text.rfind("</svg>")
        return prediction_text[start: end + 6]

    return prediction_text.strip()


def _raster_worker_safe(args):
    """Worker function for Static Image Rendering (CairoSVG)"""
    svg_path, output_path, width, height = args
    import signal
    import xml.etree.ElementTree as ET

    import cairosvg

    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Rendering timed out")

    if not os.path.exists(svg_path) or os.path.getsize(svg_path) == 0:
        Image.new("RGB", (width, height), (0, 0, 0)).save(output_path)
        return

    try:
        ET.parse(svg_path)
    except ET.ParseError:
        with open(output_path + ".err", "w") as f:
            f.write(f"Invalid SVG XML: {svg_path}")
        Image.new("RGB", (width, height), (0, 0, 0)).save(output_path)
        return

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    try:
        cairosvg.svg2png(
            url=svg_path,
            write_to=output_path,
            background_color="white",
            output_width=width,
            output_height=height,
        )
    except Exception as e:
        with open(output_path + ".err", "w") as f:
            f.write(str(e))
        Image.new("RGB", (width, height), (0, 0, 0)).save(output_path)
    finally:
        signal.alarm(0)


# ================= Rendering Utilities (Video) =================


async def svg_to_mp4(
    svg_file_path: str, output_path: str, width: int = 448, height: int = 448
):
    """
    Renders an SVG animation to MP4 using Pyppeteer and MoviePy.
    """
    from moviepy import ImageSequenceClip
    from pyppeteer import launch

    DURATION_SECONDS = 5
    FPS = 30
    VIEWPORT = {"width": width, "height": height}

    if os.path.exists(output_path):
        return

    num_frames = int(DURATION_SECONDS * FPS)
    tmp_profile_dir = tempfile.mkdtemp(prefix="pptr_profile_")

    try:
        with tempfile.TemporaryDirectory() as frame_dir:
            browser = await launch(
                headless=True,
                dumpio=False,  # Reduce noise
                args=CHROME_ARGS + [f"--user-data-dir={tmp_profile_dir}"],
            )
            page = await browser.newPage()
            await page.setViewport(VIEWPORT)

            with open(svg_file_path, "r", encoding="utf-8") as f:
                svg_content = f.read()

            html = f"""
<!doctype html><html><head><meta charset="utf-8" />
<style>
html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #ffffff; overflow: hidden; }}
#stage {{ width: 100vw; height: 100vh; display: block; }}
#stage > svg {{ width: 100%; height: 100%; display: block; }}
</style></head><body><div id="stage">{svg_content}</div></body></html>
            """
            await page.setContent(html)

            # Helper to center and scale SVG
            await page.evaluate(
                """
                (Width, Height) => {
                    const svg = document.querySelector('svg');
                    if (!svg) return;
                    const hasViewBox = svg.hasAttribute('viewBox');
                    if (!hasViewBox) {
                        const wAttr = parseFloat((svg.getAttribute('width')  || '').replace('px','')) || Width;
                        const hAttr = parseFloat((svg.getAttribute('height') || '').replace('px','')) || Height;
                        svg.setAttribute('viewBox', `0 0 ${wAttr} ${hAttr}`);
                    }
                    svg.removeAttribute('x'); svg.removeAttribute('y');
                    svg.setAttribute('width', '100%'); svg.setAttribute('height', '100%');
                    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

                    // Add white bg
                    if (!svg.querySelector("rect[data-bg='1']")) {
                        const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
                        bg.setAttribute("width", "100%"); bg.setAttribute("height", "100%");
                        bg.setAttribute("fill", "#ffffff"); bg.setAttribute("data-bg", "1");
                        svg.insertBefore(bg, svg.firstChild);
                    }
                }
                """,
                width,
                height,
            )

            frame_paths = []
            for i in range(num_frames):
                t = i / FPS
                frame_path = os.path.join(frame_dir, f"frame_{i:05d}.png")

                # Pause and Seek
                await page.evaluate(
                    f"""(() => {{
                    const root = document.querySelector('svg');
                    if (root && typeof root.pauseAnimations === 'function') root.pauseAnimations();
                    if (root && typeof root.setCurrentTime === 'function') root.setCurrentTime({t});
                }})"""
                )

                await page.screenshot({"path": frame_path, "omitBackground": False})
                frame_paths.append(frame_path)

            await browser.close()

            if frame_paths:
                frame_paths.sort()
                clip = ImageSequenceClip(frame_paths, fps=FPS)
                clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio=False,
                    preset="medium",
                    logger=None,
                    ffmpeg_params=["-pix_fmt", "yuv420p"],
                )
                clip.close()
    except Exception as e:
        print(f"Error rendering video {svg_file_path}: {e}")
        # Create a dummy video on failure to prevent crash in metrics
        pass
    finally:
        shutil.rmtree(tmp_profile_dir, ignore_errors=True)


def _video_render_wrapper(args):
    """Multiprocessing wrapper for async video rendering"""
    svg_path, out_path, w, h = args
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)
    try:
        loop.run_until_complete(svg_to_mp4(svg_path, out_path, w, h))
    except Exception as e:
        print(f"Failed to render {svg_path}: {e}")
    finally:
        loop.close()


# ================= Metrics Processors =================


def evaluate_understanding(gt_file, predictions_map):
    total = 0
    correct = 0
    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    with open(gt_file, "r") as f:
        for line in f:
            item = json.loads(line)
            cid = item["id"]
            subject = item.get("Subject", "Unknown_Subject")
            gt_ans = item["conversations"][1]["value"].strip()[0]

            pred = predictions_map.get(cid, "")
            if not pred and isinstance(cid, int):
                pred = predictions_map.get(str(cid), "")

            is_correct = False
            pred_choice = re.search(r"[A-D]", pred.strip())
            if pred_choice and pred_choice.group(0) == gt_ans:
                is_correct = True
                correct += 1

            total += 1
            subject_stats[subject]["total"] += 1
            if is_correct:
                subject_stats[subject]["correct"] += 1

    overall_acc = correct / total if total > 0 else 0
    per_subject_results = {
        subj: (stats["correct"] / stats["total"] if stats["total"] > 0 else 0)
        for subj, stats in subject_stats.items()
    }
    return overall_acc, per_subject_results


def evaluate_image_generation(
    bench_name, task_type, gt_dir, task_temp_dir, tokenizer_path=None
):
    gt_img_dir = os.path.join(gt_dir, "images")
    gt_svg_dir = os.path.join(gt_dir, "svg")

    if task_type == "text2svg":
        config = MetricsConfig(
            use_FID=True,
            use_FID_C=True,
            use_CLIP_Score_T2I=True,
            use_CLIP_Score_I2I=True,
            use_token_length=True,
        )
    else:
        config = MetricsConfig(
            use_DINO_Score=True,
            use_LPIPS=True,
            use_SSIM=True,
            use_PSNR=True,
            use_token_length=True,
        )

    calculator = InternSVGMetrics(config, tokenizer_path)

    temp_jsonl = os.path.join(task_temp_dir, "test.jsonl")
    gt_imgs, pred_imgs, gt_svgs, pred_svgs, captions = [], [], [], [], []

    with open(temp_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            uid = data["id"]
            p_img = os.path.join(task_temp_dir, "images", f"{uid}.png")
            p_svg = os.path.join(task_temp_dir, "svg", f"{uid}.svg")
            g_img = os.path.join(gt_img_dir, f"{uid}.png")
            g_svg = os.path.join(gt_svg_dir, f"{uid}.svg")

            pred_imgs.append(
                Image.open(p_img).copy()
                if os.path.exists(p_img)
                else Image.new("RGB", (448, 448))
            )
            gt_imgs.append(
                Image.open(g_img).copy()
                if os.path.exists(g_img)
                else Image.new("RGB", (448, 448))
            )

            pred_svgs.append(_read_file_safe(p_svg))
            gt_svgs.append(_read_file_safe(g_svg))
            captions.append(data.get("caption", ""))

    batch = {
        "gt_im": gt_imgs,
        "pred_im": pred_imgs,
        "gt_svg": gt_svgs,
        "pred_svg": pred_svgs,
        "caption": captions,
    }
    return calculator.calculate_metrics(batch)


def evaluate_video_generation(task_l2, gt_root, task_temp_dir, tokenizer_path):
    """
    Evaluates video generation (T2SANI / V2SANI)
    """
    gt_video_dir = os.path.join(gt_root, "video")
    gt_svg_dir = os.path.join(gt_root, "svg")
    # Assuming 'overall' exists for FVD reference
    overall_video_dir = os.path.join(gt_root, "overall")

    if "T2SANI" in task_l2:
        config = VideoMetricsConfig(
            use_FVD=True,
            use_ViCLIP_T2V=True,
            use_ViCLIP_V2V=True,
            use_DINO_Video=False,
            use_SSIM_Video=False,
            use_LPIPS_Video=False,
            use_PSNR_Video=False,
            use_token_length=True,
        )
    else:
        config = VideoMetricsConfig(
            use_FVD=False,
            use_ViCLIP_T2V=False,
            use_ViCLIP_V2V=False,
            use_DINO_Video=True,
            use_SSIM_Video=True,
            use_LPIPS_Video=True,
            use_PSNR_Video=True,
            use_token_length=True,
        )

    calculator = InternSVGVideoMetrics(config, tokenizer_path)

    temp_jsonl = os.path.join(task_temp_dir, "test.jsonl")

    pred_videos, gt_videos, caption = [], [], []
    gt_svgs, pred_svgs, overall_video, pred_video_fvd = [], [], [], []

    with open(temp_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            uid = data["id"]

            # Paths
            p_vid = os.path.join(task_temp_dir, "video", f"{uid}.mp4")
            p_svg = os.path.join(task_temp_dir, "svg", f"{uid}.svg")
            p_vid_128 = os.path.join(task_temp_dir, "video_128", f"{uid}.mp4")

            g_vid = os.path.join(gt_video_dir, f"{uid}.mp4")
            g_svg = os.path.join(gt_svg_dir, f"{uid}.svg")

            # Validations
            if os.path.exists(p_vid):
                pred_videos.append(p_vid)
                pred_svgs.append(_read_file_safe(p_svg))
            else:
                pred_videos.append(
                    "example/pure_black.mp4"
                )  # Should handle missing logic inside metric if needed
                pred_svgs.append("")

            # FVD specific: Needs 128x128 dir
            if os.path.exists(p_vid_128):
                pred_video_fvd.append(os.path.dirname(p_vid_128))
            else:
                pred_video_fvd.append(
                    os.path.dirname(p_vid_128)
                )  # Metric might handle empty or check logic

            overall_video.append(overall_video_dir)
            gt_videos.append(g_vid)
            gt_svgs.append(_read_file_safe(g_svg))

            # Caption extraction from instruction if present
            # Assuming task_records saved 'caption' or raw instruction
            cap_text = data.get("caption", "")
            if not cap_text and "conversations" in data:
                try:
                    q = data["conversations"][0]["value"]
                    if "Instruction: " in q:
                        cap_text = q.split("Instruction: ")[1]
                    else:
                        cap_text = q
                except Exception:
                    pass
            caption.append(cap_text)

    batch = {
        "gt_video": gt_videos,
        "pred_video": pred_videos,
        "caption": caption,
        "gt_svg": gt_svgs,
        "pred_svg": pred_svgs,
        "overall_video": overall_video,
        "pred_video_fvd": pred_video_fvd,
    }

    calculator.calculate_metrics(batch)
    return calculator.summarize_metrics()


def process_editing_batch(gt_root, task_temp_dir, task_name, calculator):
    gt_img_dir = os.path.join(gt_root, "data", task_name, "images")
    gt_svg_dir = os.path.join(gt_root, "data", task_name, "svg")
    temp_jsonl = os.path.join(task_temp_dir, "test.jsonl")

    gt_imgs, pred_imgs, gt_svgs, pred_svgs = [], [], [], []
    with open(temp_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            uid = data["id"]
            p_img = os.path.join(task_temp_dir, "images", f"{uid}.png")
            p_svg = os.path.join(task_temp_dir, "svg", f"{uid}.svg")
            g_img = os.path.join(gt_img_dir, f"{uid}.png")
            g_svg = os.path.join(gt_svg_dir, f"{uid}.svg")

            pred_imgs.append(
                Image.open(p_img).copy()
                if os.path.exists(p_img)
                else Image.new("RGB", (448, 448))
            )
            gt_imgs.append(
                Image.open(g_img).copy()
                if os.path.exists(g_img)
                else Image.new("RGB", (448, 448))
            )

            pred_svgs.append(_read_file_safe(p_svg))
            gt_svgs.append(_read_file_safe(g_svg))

    batch = {
        "gt_im": gt_imgs,
        "pred_im": pred_imgs,
        "gt_svg": gt_svgs,
        "pred_svg": pred_svgs,
    }
    return calculator.calculate_metrics(batch)


# ================= Helper: Load Predictions =================


def load_all_predictions_df(eval_file):
    print(f"Loading predictions from {eval_file}...")
    if eval_file.endswith(".xlsx"):
        df = pd.read_excel(eval_file)
    elif eval_file.endswith(".pkl"):
        df = pd.read_pickle(eval_file)
    elif eval_file.endswith(".csv"):
        df = pd.read_csv(eval_file)
    else:
        df = load(eval_file)
        if isinstance(df, list):
            df = pd.DataFrame(df)

    df.columns = [c.strip() for c in df.columns]

    if "prediction" not in df.columns:
        for col in ["answer", "output", "response"]:
            if col in df.columns:
                df.rename(columns={col: "prediction"}, inplace=True)
                break

    if "l2-category" not in df.columns and "category" in df.columns:
        df["l2-category"] = df["category"]

    return df


def get_task_predictions_map(df, l1_category, l2_prefix):
    mask_l2 = (
        df["l2-category"]
        .astype(str)
        .str.contains(l2_prefix, case=False, regex=False, na=False)
    )
    possible_l1_cols = ["l2-category"]
    if "category" in df.columns:
        possible_l1_cols.append("category")
    if "dataset" in df.columns:
        possible_l1_cols.append("dataset")

    mask_l1 = pd.Series(False, index=df.index)
    for col in possible_l1_cols:
        mask_l1 |= (
            df[col]
            .astype(str)
            .str.contains(l1_category, case=False, regex=False, na=False)
        )

    subset = df[mask_l2 & mask_l1].copy()
    print(
        f"  Filtering for L1='{l1_category}' & L2='{l2_prefix}' -> Found {len(subset)} rows."
    )

    pred_map = {}
    for _, row in subset.iterrows():
        l2_val = str(row["l2-category"])
        prediction = row["prediction"]
        try:
            if "_" in l2_val:
                local_id_str = l2_val.split("_")[-1]
                if local_id_str.isdigit():
                    pred_map[int(local_id_str)] = prediction
                    continue
            if "id" in row and str(row["id"]).isdigit():
                pred_map[int(row["id"])] = prediction
            elif "index" in row and str(row["index"]).isdigit():
                pred_map[int(row["index"])] = prediction
        except Exception:
            pass
    return pred_map


def convert_np_to_native(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# ================= Main Function =================


def evaluate_sarena(eval_file, tokenizer_path=TOKENIZER_PATH, dataset="SArena"):
    config = SARENA_CONFIGS[dataset]
    sarena_root = config["root"]

    download_and_extract_sarena(dataset)
    all_preds_df = load_all_predictions_df(eval_file)
    eval_dir = os.path.dirname(os.path.abspath(eval_file))
    temp_root = os.path.join(eval_dir, f"{dataset.lower()}_svg_temp")
    if not os.path.exists(temp_root):
        os.makedirs(temp_root)
    print(f"Render output directory: {temp_root}")

    final_json_results = defaultdict(lambda: defaultdict(dict))

    # Initialize Image Edit Calculator
    edit_config = MetricsConfig(
        use_DINO_Score=True,
        use_LPIPS=True,
        use_SSIM=True,
        use_PSNR=True,
        use_token_length=True,
    )
    try:
        edit_calculator = InternSVGMetrics(edit_config, tokenizer_path)
    except Exception:
        edit_calculator = None
        print("Warning: Could not initialize Image Metrics Calculator.")

    has_processed_edit = False

    task_configs = get_task_configs(dataset)
    for category, l2_prefix, rel_path, _ in task_configs:
        src_path = os.path.join(sarena_root, rel_path)
        if not os.path.exists(src_path):
            print(f"[Warning] File not found: {rel_path}. Skipping.")
            continue

        task_key = f"{category}_{l2_prefix}"
        print(f"Processing Task: {task_key}")

        current_task_preds = get_task_predictions_map(
            all_preds_df, category, l2_prefix
        )

        # --- Load GT Captions/Instructions ---
        caption_map = {}
        target_caption_file = None
        if "Generation" in l2_prefix and "Animation" not in category:
            if "Icon" in category:
                target_caption_file = os.path.join(
                    sarena_root, "Icon/generation/caption.jsonl"
                )
            elif "Illustration" in category:
                target_caption_file = os.path.join(
                    sarena_root, "illustration/caption.jsonl"
                )
            elif "Chemistry" in category:
                target_caption_file = os.path.join(
                    sarena_root, "chemistry/caption.jsonl"
                )
            if target_caption_file and os.path.exists(target_caption_file):
                caption_map = load_gt_captions(target_caption_file)

        # --- Setup Task Directories ---
        task_dir = os.path.join(temp_root, task_key.replace(" ", "_"))
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
        os.makedirs(os.path.join(task_dir, "svg"), exist_ok=True)

        is_anim = "Animation" in category
        if is_anim:
            os.makedirs(os.path.join(task_dir, "video"), exist_ok=True)
            os.makedirs(os.path.join(task_dir, "video_128"), exist_ok=True)
        else:
            os.makedirs(os.path.join(task_dir, "images"), exist_ok=True)

        with open(src_path, "r", encoding="utf-8") as f:
            src_lines = f.readlines()

        render_queue = []
        predictions_map_un = {}
        task_records = []

        for line in src_lines:
            src_item = json.loads(line)
            orig_id = src_item["id"]
            prediction = current_task_preds.get(orig_id, "")

            if "Understanding" in l2_prefix:
                predictions_map_un[orig_id] = str(prediction)
            else:
                svg_code = extract_svg_code(str(prediction))
                svg_path = os.path.join(task_dir, "svg", f"{orig_id}.svg")
                with open(svg_path, "w", encoding="utf-8") as sf:
                    sf.write(svg_code)

                if is_anim:
                    vid_path = os.path.join(task_dir, "video", f"{orig_id}.mp4")
                    vid_path_128 = os.path.join(
                        task_dir, "video_128", f"{orig_id}.mp4"
                    )
                    render_queue.append(
                        {
                            "svg_path": svg_path,
                            "out_path": vid_path,
                            "w": 448,
                            "h": 448,
                            "type": "video",
                        }
                    )
                    render_queue.append(
                        {
                            "svg_path": svg_path,
                            "out_path": vid_path_128,
                            "w": 128,
                            "h": 128,
                            "type": "video",
                        }
                    )
                else:
                    img_path = os.path.join(task_dir, "images", f"{orig_id}.png")
                    render_queue.append(
                        {
                            "svg_path": svg_path,
                            "out_path": img_path,
                            "w": 448,
                            "h": 448,
                            "type": "image",
                        }
                    )

            meta = {"id": orig_id}
            # For Animation, get instruction as caption
            if is_anim and "conversations" in src_item:
                q = src_item["conversations"][0]["value"]
                meta["caption"] = (
                    q.split("Instruction: ")[1] if "Instruction: " in q else q
                )
            else:
                meta["caption"] = caption_map.get(orig_id, "")
            task_records.append(meta)

        with open(os.path.join(task_dir, "test.jsonl"), "w") as tf:
            for r in task_records:
                tf.write(json.dumps(r) + "\n")

        # --- 1. Evaluation: Understanding ---
        if "Understanding" in l2_prefix:
            acc, sub_acc = evaluate_understanding(src_path, predictions_map_un)
            final_json_results[category]["Understanding"]["Overall"] = acc
            for subj, val in sub_acc.items():
                final_json_results[category]["Understanding"][subj] = val
            print(f"  Accuracy: {acc:.2%}")
            continue

        # --- 2. Rendering ---
        if render_queue:
            video_tasks = [x for x in render_queue if x["type"] == "video"]
            image_tasks = [x for x in render_queue if x["type"] == "image"]

            if image_tasks:
                print(f"  Rasterizing {len(image_tasks)} Images...")
                pool_args = [
                    (t["svg_path"], t["out_path"], t["w"], t["h"])
                    for t in image_tasks
                ]
                with multiprocessing.Pool(
                    processes=min(cpu_count(), 32), maxtasksperchild=50
                ) as pool:
                    for _ in tqdm(
                        pool.imap_unordered(_raster_worker_safe, pool_args),
                        total=len(pool_args),
                        desc="Images",
                    ):
                        pass

            if video_tasks:
                print(f"  Rendering {len(video_tasks)} Videos (Async/MP)...")
                pool_args = [
                    (t["svg_path"], t["out_path"], t["w"], t["h"])
                    for t in video_tasks
                ]
                # Use ProcessPoolExecutor to handle async loops in subprocesses
                with ProcessPoolExecutor(max_workers=min(cpu_count(), 16)) as executor:
                    futures = [
                        executor.submit(_video_render_wrapper, arg)
                        for arg in pool_args
                    ]
                    for f in tqdm(
                        as_completed(futures), total=len(futures), desc="Videos"
                    ):
                        try:
                            f.result()
                        except Exception as e:
                            print(f"  Video rendering error: {e}")

        # --- 3. Evaluation: Generation & Editing ---
        try:
            if is_anim:
                # Animation Metrics
                gt_root = os.path.join(sarena_root, "animation")
                metrics_res = evaluate_video_generation(
                    l2_prefix, gt_root, task_dir, tokenizer_path
                )
                final_json_results[category][l2_prefix] = metrics_res
                print(f"  Video Scores: {metrics_res}")

            elif "Icon" in category and "Edit" in l2_prefix and edit_calculator:
                # Icon Editing Accumulation
                task_name = EDIT_TASK_MAP.get(l2_prefix, "unknown")
                gt_root = os.path.join(sarena_root, "Icon/edit")
                process_editing_batch(gt_root, task_dir, task_name, edit_calculator)
                has_processed_edit = True
                print(f"  [Accumulated] Editing metrics for {task_name}")

            elif "Generation" in l2_prefix:
                # Image Generation
                bench_name = category.split("-")[1]
                task_type = "text2svg" if "T2SVG" in l2_prefix else "img2svg"

                if bench_name == "Icon":
                    gt_root = os.path.join(sarena_root, "Icon/generation")
                elif bench_name == "Illustration":
                    gt_root = os.path.join(sarena_root, "illustration")
                elif bench_name == "Chemistry":
                    gt_root = os.path.join(sarena_root, "chemistry")

                metrics_res = evaluate_image_generation(
                    bench_name, task_type, gt_root, task_dir, tokenizer_path
                )
                clean_task_name = "T2SVG" if "T2SVG" in l2_prefix else "I2SVG"
                final_json_results[category][clean_task_name] = metrics_res
                print(f"  Scores: {metrics_res}")

        except Exception as e:
            print(f"  Error calculating metrics for {task_key}: {e}")
            import traceback

            traceback.print_exc()
            final_json_results[category][l2_prefix]["ERROR"] = str(e)

    # 4. Finalize Editing Metrics
    if has_processed_edit and edit_calculator:
        print("\nCalculating Overall Editing Metrics...")
        overall_edit_score = edit_calculator.summarize_metrics()
        final_json_results["SArena-Icon"]["Editing"] = overall_edit_score
        print(f"Editing Overall Scores: {overall_edit_score}")

    # --- Save Results ---
    def recursive_convert(d):
        if isinstance(d, defaultdict) or isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        return convert_np_to_native(d)

    final_output_dict = recursive_convert(final_json_results)
    json_output_path = eval_file.replace(".xlsx", "_results.json").replace(
        ".pkl", "_results.json"
    )
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_output_dict, f, indent=4, ensure_ascii=False)

    print(
        f"\n{'=' * 50}\nSArena Evaluation Complete\nResults saved to: {json_output_path}\n{'=' * 50}"
    )
    return final_output_dict
