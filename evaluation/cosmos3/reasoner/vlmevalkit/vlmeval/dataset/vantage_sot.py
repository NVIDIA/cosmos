"""
VANTAGE_SOT — Single-Object Tracking benchmark for VLMs.

The VLM receives N frames as individual images with the target's GT bbox
shown only in frame 0 (initialization). It must localize the same object
in all subsequent frames by outputting a single JSON response.

Metrics:
  success_auc   — area under P(IoU > t) curve, t in [0, 1]  (VOT standard; primary)
  mean_iou      — average IoU across eval frames
  precision@0.5 — fraction of frames with IoU >= 0.5
  freeze_rate   — fraction of predictions identical to the previous frame
  null_rate     — fraction of frames where model output null instead of a bbox

Data format (prepared_data_dir):
  Each sequence is a subdirectory containing:
    gt.json        — metadata and ground-truth bboxes
    frames/f00.png, f01.png, ...  — extracted frames
    frames/crop.png               — target crop from frame 0
    frames/f00_ann.png            — frame 0 with GT bbox drawn
"""

import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLIP_FRAMES = 8    # 8 frames: enough temporal signal, models attend carefully
DEFAULT_FRAME_STRIDE = 15  # sample every 15th source frame (0.5s at 30fps);
                            # 8 frames spans ~3.5s of real motion

FREEZE_IOU_THRESHOLD = 0.95  # bbox is "frozen" if IoU with prev frame >= this

# S3 stage for the HF-release-with-annotations VANTAGE-bench SOT subset.
# Same prepared 8-frame layout regardless of which DATASET_CONFIGS preset is
# selected — variants change only the inference-time slicing, not the source.
S3_BUCKET = 'cosmos_understanding'
S3_PREFIX = 'benchmark/vantage_benchmark_hf_release_annotations/tracking/MetropolisSOT_benchmark_8f'


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SOT_PROMPT_INTRO_QWEN = """\
You are a visual object tracker. Track a specific {object_type} across {n_frames} video frames.

The TARGET is shown in the crop image above AND highlighted with a GREEN RECTANGLE in Frame 0.
Its initial bounding box is {init_bbox} (format: [x1, y1, x2, y2], coordinates in 0-1000 space \
where 0=left/top edge, 1000=right/bottom edge).

Frames 1 to {last_frame} show the scene without any markings — locate the same {object_type} in each.

Tracking rules:
- Output a bounding box for EVERY frame from 1 to {last_frame}
- If the object moves, your bbox MUST reflect its new position — do NOT copy the Frame 0 bbox \
  to every frame; that is freezing, not tracking
- If the object is partially occluded or briefly unclear, estimate its position based on its \
  last known location and direction of movement
- Only output null if the object has completely exited the frame boundaries with no visible trace
- Track precisely: observe how the object's position changes between consecutive frames

First, reason through the motion step by step:
- Look at frames 1 to {last_frame} and describe how the {object_type} moves (direction, speed, any occlusion)
- Use this reasoning to determine the precise bbox for each frame

Then output ONLY a JSON object with a key for EVERY frame from 1 to {last_frame} (no other text after it):
{{
  "frame_1": [x1, y1, x2, y2],
  "frame_2": [x1, y1, x2, y2],
  "frame_3": [x1, y1, x2, y2],
  ...
  "frame_{last_frame}": [x1, y1, x2, y2]
}}
You MUST include all {last_frame} frames — do not skip any frame index.

FORMAT REMINDER (your response will be parsed programmatically):
- Output a single JSON object {{...}}, never a list [...].
- Each value must be a 4-tuple [x1, y1, x2, y2], never a 2-tuple [x, y].
- Do not add "label" fields. The dict keys are only "frame_1".."frame_{last_frame}".
"""

# Gemini/Gemma convention: yxyx tuple order. Frame template + init_bbox both
# emitted in [y1, x1, y2, x2]. Parser swaps yxyx→xyxy at evaluate time
# (per-frame keys carry no family marker, so dispatch is by model_family).
SOT_PROMPT_INTRO_GEMINI = """\
You are a visual object tracker. Track a specific {object_type} across {n_frames} video frames.

The TARGET is shown in the crop image above AND highlighted with a GREEN RECTANGLE in Frame 0.
Its initial bounding box is {init_bbox} (format: [y1, x1, y2, x2], coordinates in 0-1000 space \
where 0=top/left edge, 1000=bottom/right edge).

Frames 1 to {last_frame} show the scene without any markings — locate the same {object_type} in each.

Tracking rules:
- Output a bounding box for EVERY frame from 1 to {last_frame}
- If the object moves, your bbox MUST reflect its new position — do NOT copy the Frame 0 bbox \
  to every frame; that is freezing, not tracking
- If the object is partially occluded or briefly unclear, estimate its position based on its \
  last known location and direction of movement
- Only output null if the object has completely exited the frame boundaries with no visible trace
- Track precisely: observe how the object's position changes between consecutive frames

First, reason through the motion step by step:
- Look at frames 1 to {last_frame} and describe how the {object_type} moves (direction, speed, any occlusion)
- Use this reasoning to determine the precise bbox for each frame

Then output ONLY a JSON object with a key for EVERY frame from 1 to {last_frame} (no other text after it):
{{
  "frame_1": [y1, x1, y2, x2],
  "frame_2": [y1, x1, y2, x2],
  "frame_3": [y1, x1, y2, x2],
  ...
  "frame_{last_frame}": [y1, x1, y2, x2]
}}
You MUST include all {last_frame} frames — do not skip any frame index.
"""

_SOT_INTRO_BY_FAMILY = {
    'cr':     SOT_PROMPT_INTRO_QWEN,
    'qwen3':  SOT_PROMPT_INTRO_QWEN,
    'gemini': SOT_PROMPT_INTRO_GEMINI,
    'gemma4': SOT_PROMPT_INTRO_GEMINI,
}


def build_sot_prompt(
    n_frames: int,
    init_bbox: List[float],
    object_type: str = "object",
    model_family: str = "cr",
) -> str:
    is_gemini = model_family in ('gemini', 'gemma4')
    if is_gemini:
        # init bbox in yxyx
        init_bbox_str = "[{}, {}, {}, {}]".format(
            round(init_bbox[1]), round(init_bbox[0]),
            round(init_bbox[3]), round(init_bbox[2]),
        )
    else:
        init_bbox_str = "[{}, {}, {}, {}]".format(
            round(init_bbox[0]), round(init_bbox[1]),
            round(init_bbox[2]), round(init_bbox[3]),
        )
    intro = _SOT_INTRO_BY_FAMILY.get(model_family, SOT_PROMPT_INTRO_QWEN)
    return intro.format(
        n_frames=n_frames,
        last_frame=n_frames - 1,
        init_bbox=init_bbox_str,
        object_type=object_type,
    )


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _draw_init_bbox(frame_np: np.ndarray, bbox_1000: List[float]) -> np.ndarray:
    """Draw the init bbox on frame 0 so the model has a visual anchor."""
    import cv2
    img = frame_np.copy()
    h, w = img.shape[:2]
    x1 = int(bbox_1000[0] / 1000 * w)
    y1 = int(bbox_1000[1] / 1000 * h)
    x2 = int(bbox_1000[2] / 1000 * w)
    y2 = int(bbox_1000[3] / 1000 * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, 'TARGET', (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return img


def extract_target_crop(
    frame_np: np.ndarray,
    bbox_1000: List[float],
    out_path: str,
    padding: float = 0.15,
) -> bool:
    """
    Save a cropped image of the target object from frame 0.

    bbox_1000: [x1, y1, x2, y2] in 0-1000 normalized coords.
    padding: fractional padding added around the bbox (15% by default).
    Returns True on success.
    """
    try:
        import PIL.Image
        h, w = frame_np.shape[:2]
        x1 = bbox_1000[0] / 1000 * w
        y1 = bbox_1000[1] / 1000 * h
        x2 = bbox_1000[2] / 1000 * w
        y2 = bbox_1000[3] / 1000 * h
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * padding, bh * padding
        cx1 = max(0, int(x1 - px))
        cy1 = max(0, int(y1 - py))
        cx2 = min(w, int(x2 + px))
        cy2 = min(h, int(y2 + py))
        if cx2 <= cx1 or cy2 <= cy1:
            return False
        crop = PIL.Image.fromarray(frame_np[cy1:cy2, cx1:cx2])
        crop.save(out_path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_FRAME_TUP_RE = re.compile(
    r'"frame[_]?(\d+)"\s*:\s*\[\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)


def _regex_extract_frame_bboxes(
    text: str,
    n_frames: int,
    swap_yx: bool,
) -> Dict[int, List[float]]:
    """Fallback for list-of-dicts wrappers and truncated JSON. First match per
    frame index wins."""
    result: Dict[int, List[float]] = {}
    for m in _FRAME_TUP_RE.finditer(str(text)):
        idx = int(m.group(1))
        if idx < 1 or idx >= n_frames:
            continue
        if idx in result:
            continue
        bbox = [float(m.group(i + 2)) for i in range(4)]
        if swap_yx:
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
        bbox = [max(0.0, min(b, 1000.0)) for b in bbox]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        result[idx] = bbox
    return result


def parse_sot_response(
    text: str,
    n_frames: int,
    model_family: str = "cr",
) -> Dict[int, Optional[List[float]]]:
    """
    Parse VLM response into {frame_idx: bbox_or_None}.

    Frame indices are 1-based in the response (frame_1 ... frame_{n-1})
    since frame_0 is the initialization frame.

    Per-frame keys (`frame_N`) carry no axis-order marker, so for the Gemini
    family the per-frame tuple is interpreted as yxyx and swapped to xyxy
    before the downstream clamp/reorder runs.

    Returns dict for frames 1..n_frames-1. Missing frames → None.
    """
    swap_yx = model_family in ('gemini', 'gemma4')
    if not text or pd.isna(text):
        return {}

    text = str(text).strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object anywhere in the response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if not isinstance(parsed, dict):
        return dict(_regex_extract_frame_bboxes(text, n_frames, swap_yx))

    result = {}
    for key, val in parsed.items():
        # Accept "frame_1", "frame1", "1"
        m = re.search(r'(\d+)', str(key))
        if not m:
            continue
        idx = int(m.group(1))
        if idx < 1 or idx >= n_frames:
            continue

        if val is None:
            result[idx] = None
            continue

        if not isinstance(val, (list, tuple)) or len(val) < 4:
            result[idx] = None
            continue

        try:
            bbox = [float(v) for v in val[:4]]
        except (ValueError, TypeError):
            result[idx] = None
            continue

        if swap_yx:
            # yxyx → xyxy
            bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]

        # Clamp to valid range
        bbox = [
            max(0.0, min(bbox[0], 1000.0)),
            max(0.0, min(bbox[1], 1000.0)),
            max(0.0, min(bbox[2], 1000.0)),
            max(0.0, min(bbox[3], 1000.0)),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            result[idx] = None
            continue

        result[idx] = bbox

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_2d(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


ROBOT_TYPES = {"NovaCarter", "Transporter", "AgilityDigit", "FourierGR1T2", "Forklift"}


def compute_seq_tags(
    init_bbox: List[float],
    gt_bboxes: Dict,
    object_type: str,
    n_other_tracks: int = 0,
) -> List[str]:
    """Compute difficulty/category tags for a sequence from its GT data."""
    tags = []

    # Size tag based on init bbox area in 0-1000 space
    w = (init_bbox[2] - init_bbox[0]) / 1000
    h = (init_bbox[3] - init_bbox[1]) / 1000
    area = w * h
    if area < 0.007:
        tags.append("small")
    elif area > 0.02:
        tags.append("large")
    else:
        tags.append("medium")

    # Motion tag: total centre displacement across frames (0-1000 space)
    centres = []
    for bbox in gt_bboxes.values():
        if bbox is not None:
            centres.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
    if len(centres) > 1:
        dx = max(c[0] for c in centres) - min(c[0] for c in centres)
        dy = max(c[1] for c in centres) - min(c[1] for c in centres)
        motion = (dx ** 2 + dy ** 2) ** 0.5
        if motion < 10:
            tags.append("stationary")
        elif motion > 80:
            tags.append("fast_motion")
        else:
            tags.append("moderate_motion")
    else:
        tags.append("stationary")

    # Occlusion: any null GT frames
    if any(v is None for v in gt_bboxes.values()):
        tags.append("occluded")

    # Object category
    tags.append("robot" if object_type in ROBOT_TYPES or object_type == "Robot" else "person")

    # Crowded: many other objects sharing the same camera window
    if n_other_tracks >= 5:
        tags.append("crowded")

    return tags


def compute_sot_metrics(
    gt_bboxes: Dict[int, Optional[List[float]]],  # {frame_id: bbox or None} — None = absent
    pred_bboxes: Dict[int, Optional[List[float]]],  # {clip_frame_idx: bbox_or_None}
    frame_ids: List[int],                  # ordered list mapping clip_idx → frame_id
    occluded: Dict[int, bool],             # {frame_id: bool}
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Compute per-frame IoU, mean IoU, precision, freeze rate,
    and visibility-split IoU (visible vs occluded frames).

    Frame 0 is the init frame — excluded from all metrics.

    gt_bbox = None means the object is absent in that frame:
      - pred = None  → both agree object gone, skip (correct, no penalty)
      - pred = bbox  → false detection, IoU = 0
    """
    ious = []
    visible_ious = []
    occluded_ious = []
    null_count = 0
    false_det_count = 0
    freeze_count = 0
    prev_pred = None

    # frame_ids[0] is init — start from index 1
    for clip_idx in range(1, len(frame_ids)):
        fid = frame_ids[clip_idx]
        gt_bbox = gt_bboxes.get(fid)
        pred_bbox = pred_bboxes.get(clip_idx)

        if gt_bbox is None:
            # Object absent in GT
            if pred_bbox is not None:
                # False detection — penalize
                false_det_count += 1
                ious.append(0.0)
                visible_ious.append(0.0)
            # pred is also None → both agree, skip
            continue

        if pred_bbox is None:
            # Object present in GT but model said not visible
            null_count += 1
            iou = 0.0
        else:
            iou = iou_2d(gt_bbox, pred_bbox)

            # Freeze detection: pred nearly identical to previous pred (diagnostic only)
            if prev_pred is not None and iou_2d(pred_bbox, prev_pred) >= FREEZE_IOU_THRESHOLD:
                freeze_count += 1

            prev_pred = pred_bbox

        ious.append(iou)
        is_occluded = occluded.get(fid, False)
        if is_occluded:
            occluded_ious.append(iou)
        else:
            visible_ious.append(iou)

    n_eval = len(ious)
    if n_eval == 0:
        return {
            'mean_iou': 0.0, 'success_auc': 0.0,
            'precision': 0.0, 'precision_25': 0.0, 'precision_75': 0.0,
            'freeze_rate': 0.0, 'null_rate': 0.0, 'false_det_rate': 0.0,
            'visible_iou': 0.0, 'occluded_iou': 0.0,
            'n_eval_frames': 0,
        }

    n_predicted = n_eval - null_count  # frames where model gave a bbox
    freeze_denom = max(1, n_predicted - 1)

    # false_det_rate: fraction of absent-GT frames where model predicted a bbox
    n_absent = sum(1 for fid in frame_ids[1:] if gt_bboxes.get(fid) is None)
    false_det_rate = float(false_det_count / n_absent) if n_absent > 0 else 0.0

    # Success AUC: area under P(IoU > t) curve for t in [0, 1]
    thresholds = np.linspace(0, 1, 21)
    success_auc = float(np.mean([np.mean([iou >= t for iou in ious]) for t in thresholds]))

    return {
        'mean_iou': float(np.mean(ious)),
        'success_auc': success_auc,
        'precision': float(np.mean([iou >= iou_threshold for iou in ious])),
        'precision_25': float(np.mean([iou >= 0.25 for iou in ious])),
        'precision_75': float(np.mean([iou >= 0.75 for iou in ious])),
        'freeze_rate': float(freeze_count / freeze_denom),
        'null_rate': float(null_count / n_eval),
        'false_det_rate': false_det_rate,
        'visible_iou': float(np.mean(visible_ious)) if visible_ious else 0.0,
        'occluded_iou': float(np.mean(occluded_ious)) if occluded_ious else 0.0,
        'n_eval_frames': n_eval,
    }


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class VANTAGE_SOT(VideoBaseDataset):
    """
    Single-Object Tracking benchmark for VLMs.

    One sample = one short sequence of N frames as individual images.
    Frame 0 comes with the GT bbox as initialization.
    The VLM must output bboxes for frames 1..N-1 in a single JSON response.

    Data is fetched from the VANTAGE-bench S3 stage (HF-release-with-annotations
    subset) on first instantiation and cached under LMUDataRoot().
    """

    MD5 = ''
    TYPE = 'SOT'

    DEFAULT_FAMILY = 'cr'

    DATASET_CONFIGS = {
        'VANTAGE_SOT': {},
        'VANTAGE_SOT_tiny': {
            'clip_frames': 8,
            'frame_stride': 15,
        },
        'VANTAGE_SOT_16f': {
            'clip_frames': 16,
            'frame_stride': 15,
        },
        'VANTAGE_SOT_32f': {
            'clip_frames': 32,
            'frame_stride': 15,
        },
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_CONFIGS.keys())

    def __init__(
        self,
        dataset: str = 'VANTAGE_SOT',
        prepared_data_dir: Optional[str] = None,
        # Optional filters
        scene: Optional[str] = None,
        camera: Optional[str] = None,
        object_id: Optional[int] = None,
        # Sampling (overridden by DATASET_CONFIGS presets)
        clip_frames: int = DEFAULT_CLIP_FRAMES,
        frame_stride: int = DEFAULT_FRAME_STRIDE,
        verbose: bool = False,
        # Path to metadata.jsonl (needed for new-format benchmarks missing init_bbox)
        metadata_path: Optional[str] = None,
        # VideoBaseDataset compat args (accepted but unused internally)
        pack: bool = False,
        nframe: int = 0,
        fps: float = -1,
        total_pixels=None,
        max_pixels=None,
        max_frames=None,
        **kwargs,
    ):
        self.dataset_name = dataset
        self.verbose = verbose
        self.nframe = nframe
        self.fps = fps
        self.model_family = kwargs.pop('model_family', self.DEFAULT_FAMILY)

        preset = self.DATASET_CONFIGS.get(dataset, {})
        self.scene_filter = scene
        self.camera_filter = camera
        self.object_id_filter = int(object_id) if object_id is not None else None
        self.clip_frames = preset.get('clip_frames', clip_frames)
        self.frame_stride = preset.get('frame_stride', frame_stride)

        # Build seq_id → init_bbox index for new-format benchmarks
        self._metadata_index: Dict[str, dict] = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._metadata_index[entry['seq_id']] = entry

        if prepared_data_dir:
            self.prepared_data_dir = prepared_data_dir
        else:
            self.prepared_data_dir = self._download_from_s3()

        self._prepare_data_from_dir()

    def _download_from_s3(self) -> str:
        """Fetch the prepared 8-frame SOT sequences from the VANTAGE-bench
        S3 stage into LMUDataRoot()/datasets/VANTAGE_SOT/ (one-shot)."""
        local_dir = Path(LMUDataRoot()) / 'datasets' / 'VANTAGE_SOT'
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        has_sequences = (
            local_dir.exists()
            and any(local_dir.iterdir())
            and any((d / 'gt.json').exists() for d in local_dir.iterdir() if d.is_dir())
        )
        if has_sequences:
            return str(local_dir)

        try:
            from s3fs import S3FileSystem
        except ImportError as e:
            raise ImportError(
                "s3fs is required for S3 access. Install with: pip install s3fs"
            ) from e

        s3 = S3FileSystem(
            anon=False,
            profile='team-cosmos',
            client_kwargs={'endpoint_url': 'https://pdx.s8k.io'},
        )
        s3_path = f'{S3_BUCKET}/{S3_PREFIX}'
        print(f"Downloading VANTAGE_SOT from s3://{s3_path} to {local_dir} ...")
        s3.get(s3_path, str(local_dir), recursive=True)
        print(f"VANTAGE_SOT download complete: {local_dir}")
        return str(local_dir)

    def _prepare_data_from_dir(self):
        """Load sequences from a prepared directory (pre-extracted frames + gt.json)."""
        prep_root = Path(self.prepared_data_dir).resolve()
        seq_dirs = sorted(d for d in prep_root.iterdir()
                          if d.is_dir() and (d / 'gt.json').exists())

        rows, gt_cache = [], {}
        for idx, seq_dir in enumerate(seq_dirs):
            with open(seq_dir / 'gt.json') as f:
                meta = json.load(f)

            # Support two gt.json formats:
            # - Full format (8f benchmark): has frame_ids, gt_bboxes, init_bbox, label
            # - Compact format (prepare_benchmark.py): has bboxes, nframes, stride
            if 'frame_ids' in meta:
                frame_ids = meta['frame_ids']
                raw_gt = meta.get('gt_bboxes') or {}
                gt_bboxes = {int(k): v for k, v in raw_gt.items()}
                init_bbox = meta['init_bbox']
                label = meta['label']
                scene = meta.get('scene', '')
                camera = meta.get('camera', '')
                object_id = str(meta.get('object_id', ''))
                object_type = meta.get('object_type', 'Person')
            else:
                # Compact format: bboxes keyed "frame_1".."frame_{N-1}"
                nframes = meta['nframes']
                frame_ids = list(range(nframes))
                bboxes_raw = meta.get('bboxes', {})
                gt_bboxes = {}
                for k, v in bboxes_raw.items():
                    m = re.search(r'(\d+)', str(k))
                    if m:
                        gt_bboxes[int(m.group(1))] = v
                # init_bbox: try metadata index, then fall back to None
                seq_id = seq_dir.name
                meta_entry = self._metadata_index.get(seq_id, {})
                init_bbox = meta.get('init_bbox') or meta_entry.get('init_bbox')
                if init_bbox is None:
                    print(f"[WARN] {seq_id}: no init_bbox found, skipping. "
                          f"Pass metadata_path to fix.")
                    continue
                gt_bboxes[0] = init_bbox
                label = meta_entry.get('label') or seq_id.replace('__', '/')
                scene = meta_entry.get('scene', '')
                camera = meta_entry.get('camera', '')
                object_id = str(meta_entry.get('object_id', ''))
                object_type = meta_entry.get('object_type', 'Person')

            # Apply filters
            if self.scene_filter and scene != self.scene_filter:
                continue
            if self.camera_filter and camera != self.camera_filter:
                continue
            if self.object_id_filter is not None:
                if str(self.object_id_filter) != object_id:
                    continue

            frames_dir = seq_dir / 'frames'
            frame_paths = [str(frames_dir / f'f{i:02d}.png') for i in range(len(frame_ids))]
            crop_path = str(seq_dir / 'frames' / 'crop.png')

            gt_cache[idx] = {
                'frame_ids':   frame_ids,
                'gt_bboxes':   gt_bboxes,
                'occluded':    {},
                'init_bbox':   init_bbox,
                'object_type': object_type,
                'label':       label,
                'video_path':  meta.get('video_path', ''),
                'frame_paths': frame_paths,
                'crop_path':   crop_path,
                'seq_dir':     str(seq_dir),
            }
            rows.append({
                'index':       idx,
                'label':       label,
                'scene':       scene,
                'camera':      camera,
                'object_id':   object_id,
                'object_type': object_type,
                'n_frames':    len(frame_ids),
                'video_path':  meta.get('video_path', ''),
                'frame_ids':   json.dumps(frame_ids),
                'init_bbox':   json.dumps(init_bbox),
            })

        self._gt_cache = gt_cache
        self.data = pd.DataFrame(rows) if rows else pd.DataFrame()

        if not rows:
            print(
                f"WARNING: VANTAGE_SOT found 0 sequences under {self.prepared_data_dir}. "
                "Check filters and S3 download."
            )
        elif self.verbose:
            print(f"VANTAGE_SOT: {len(rows)} sequences from {self.prepared_data_dir}")

    def __len__(self):
        return len(self.data)

    def build_prompt(self, line, video_llm=False):
        """
        VLM-native: returns individual frame images + text prompt.
        video_llm=False by default — we pass frames as images, not mp4.

        Message layout:
          [text: "Target object:"]
          [image: crop of target from frame 0]
          [text: SOT instructions with bbox coords + frame count]
          [text: "Frame 0:"] [image: frame0 with annotated bbox]
          [text: "Frame 1:"] [image: frame1]
          ...
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        idx = int(line['index'])
        cache = self._gt_cache[idx]

        frame_ids = cache['frame_ids']
        init_bbox = cache['init_bbox']
        label = cache['label']
        frame_paths = cache['frame_paths']

        # Ensure frame 0 has the init bbox drawn
        f0_path = frame_paths[0] if frame_paths else None
        if f0_path and os.path.exists(f0_path):
            ann_path = f0_path.replace('f00.png', 'f00_ann.png')
            if not os.path.exists(ann_path):
                import cv2 as _cv2
                img = _cv2.imread(f0_path)
                if img is not None:
                    img = _draw_init_bbox(img, init_bbox)
                    _cv2.imwrite(ann_path, img)
            if os.path.exists(ann_path):
                frame_paths = [ann_path] + frame_paths[1:]

        crop_path = cache.get('crop_path', '')
        if not os.path.exists(crop_path) and f0_path and os.path.exists(f0_path):
            import cv2 as _cv2
            img = _cv2.imread(f0_path)
            if img is not None:
                extract_target_crop(img[:, :, ::-1], init_bbox, crop_path)
        crop_ok = os.path.exists(crop_path)

        n_frames = len(frame_ids)
        object_type = cache.get('object_type', 'object')
        prompt_text = build_sot_prompt(
            n_frames=n_frames,
            init_bbox=init_bbox,
            object_type=object_type,
            model_family=self.model_family,
        )

        if not frame_paths:
            print(f"WARNING: No frames found for {label}")
            return [dict(type='text', value=prompt_text)]

        msgs = []
        if crop_ok:
            msgs.append(dict(type='text', value='Target object:'))
            msgs.append(dict(type='image', value=crop_path))

        msgs.append(dict(type='text', value=prompt_text))

        for i, img_path in enumerate(frame_paths):
            msgs.append(dict(type='text', value=f'Frame {i}:'))
            msgs.append(dict(type='image', value=img_path))

        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs) -> Dict:
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv']
        data = load(eval_file)
        verbose = judge_kwargs.get('verbose', False) or self.verbose

        if verbose:
            print(f"\nVANTAGE_SOT Evaluation | {eval_file} | {len(data)} predictions")

        results = {}
        all_metrics = defaultdict(list)
        parse_failures = 0

        # Build crowded count: number of other tracks sharing same camera window
        window_track_counts: Dict[str, int] = defaultdict(int)
        for cache in self._gt_cache.values():
            lbl = cache['label']
            parts = lbl.split('/')
            window_key = '/'.join(parts[:-1])
            window_track_counts[window_key] += 1

        for _, row in data.iterrows():
            idx = row.get('index')
            cache = self._gt_cache.get(idx)
            if cache is None:
                continue

            label = cache['label']
            frame_ids = cache['frame_ids']
            n_frames = len(frame_ids)

            if not cache['gt_bboxes']:
                if verbose:
                    print(f"  {label}: no GT (public dataset) — skipping evaluation")
                continue

            raw_pred = row.get('prediction', '')
            pred_bboxes = parse_sot_response(raw_pred, n_frames, model_family=self.model_family)

            if not pred_bboxes and raw_pred and str(raw_pred).strip() not in ('', '{}'):
                parse_failures += 1
                if verbose:
                    print(f"  {label}: parse failure — {str(raw_pred)[:120]}")

            metrics = compute_sot_metrics(
                gt_bboxes=cache['gt_bboxes'],
                pred_bboxes=pred_bboxes,
                frame_ids=frame_ids,
                occluded=cache['occluded'],
            )

            parts = label.split('/')
            window_key = '/'.join(parts[:-1])
            n_other = window_track_counts[window_key] - 1
            tags = compute_seq_tags(
                init_bbox=cache['init_bbox'],
                gt_bboxes=cache['gt_bboxes'],
                object_type=cache['object_type'],
                n_other_tracks=n_other,
            )

            results[label] = {
                **metrics,
                'object_type': cache['object_type'],
                'tags': tags,
                'n_pred_frames': len([v for v in pred_bboxes.values() if v is not None]),
            }

            for k, v in metrics.items():
                if isinstance(v, float):
                    all_metrics[k].append(v)

            if verbose:
                print(
                    f"  {label}: IoU={metrics['mean_iou']:.4f} "
                    f"AUC={metrics['success_auc']:.4f} "
                    f"prec={metrics['precision']:.4f} "
                    f"freeze={metrics['freeze_rate']:.4f} "
                    f"null={metrics['null_rate']:.4f} "
                    f"[vis={metrics['visible_iou']:.4f} "
                    f"occ={metrics['occluded_iou']:.4f}] "
                    f"[{','.join(tags)}]"
                )

        if parse_failures:
            print(f"WARNING: Parse failed for {parse_failures} sequences")

        if all_metrics:
            results['Overall'] = {k: float(np.mean(v)) for k, v in all_metrics.items()}

        # Print table
        print(f"\n{'Sequence':<55}{'IoU':>7}{'AUC':>7}{'P@.5':>7}{'P@.25':>7}{'P@.75':>7}{'Freeze':>8}{'Null':>7}")
        print("=" * 115)
        for label in sorted(k for k in results if k != 'Overall'):
            v = results[label]
            print(
                f"{label:<55}{v['mean_iou']:>7.4f}{v['success_auc']:>7.4f}"
                f"{v['precision']:>7.4f}{v['precision_25']:>7.4f}{v['precision_75']:>7.4f}"
                f"{v['freeze_rate']:>8.4f}{v['null_rate']:>7.4f}"
            )
        if 'Overall' in results:
            ov = results['Overall']
            print(
                f"{'Overall':<55}{ov['mean_iou']:>7.4f}{ov['success_auc']:>7.4f}"
                f"{ov['precision']:>7.4f}{ov['precision_25']:>7.4f}{ov['precision_75']:>7.4f}"
                f"{ov['freeze_rate']:>8.4f}{ov['null_rate']:>7.4f}"
            )

        # Tag-based breakdown
        tag_metrics: Dict[str, list] = defaultdict(list)
        for label, v in results.items():
            if label == 'Overall':
                continue
            for tag in v.get('tags', []):
                tag_metrics[tag].append(v)

        print(f"\n{'=== BREAKDOWN BY TAG ==='}")
        print(f"{'Tag':<20}{'n':>5}{'IoU':>8}{'AUC':>8}{'P@.5':>8}{'Null':>8}{'Freeze':>8}")
        print("-" * 65)
        for tag in sorted(tag_metrics):
            seqs = tag_metrics[tag]
            n = len(seqs)
            avg = lambda k: sum(s[k] for s in seqs) / n
            print(
                f"{tag:<20}{n:>5}{avg('mean_iou'):>8.3f}{avg('success_auc'):>8.3f}"
                f"{avg('precision'):>8.3f}{avg('null_rate'):>8.3f}{avg('freeze_rate'):>8.3f}"
            )

        # Save outputs
        json_path = get_intermediate_file_path(eval_file, '_sot_results', 'json')
        dump(results, json_path)

        csv_path = get_intermediate_file_path(eval_file, '_sot_metrics', 'csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Sequence', 'ObjectType', 'Tags', 'MeanIoU', 'SuccessAUC',
                'Precision50', 'Precision25', 'Precision75',
                'FreezeRate', 'NullRate', 'VisibleIoU', 'OccludedIoU', 'NEvalFrames',
            ])
            for label, v in results.items():
                writer.writerow([
                    label,
                    v.get('object_type', ''),
                    '|'.join(v.get('tags', [])),
                    f"{v['mean_iou']:.4f}",
                    f"{v.get('success_auc', 0.0):.4f}",
                    f"{v['precision']:.4f}",
                    f"{v.get('precision_25', 0.0):.4f}",
                    f"{v.get('precision_75', 0.0):.4f}",
                    f"{v['freeze_rate']:.4f}",
                    f"{v['null_rate']:.4f}",
                    f"{v['visible_iou']:.4f}",
                    f"{v['occluded_iou']:.4f}",
                    v.get('n_eval_frames', ''),
                ])

        print(f"\nResults: {json_path}\nCSV:     {csv_path}")
        # Primary rollup: success_auc (VOT-standard AUC; matches VANTAGE paper Table 2).
        return {k: v['success_auc'] for k, v in results.items()}
