"""
2D Detection Dataset for images in KITTI format.

The validation data should be in KITTI format with:
- images/ directory containing image files
- labels/ directory containing label files (KITTI format)

For evaluation, predicted labels and groundtruth labels are mapped to "person"
and evaluated using F1 score.
"""

import os
import random

import numpy as np
import pandas as pd
import PIL.Image
from PIL import ImageOps

from ...smp import LMUDataRoot, get_logger, load
from ..image_base import ImageBaseDataset
from .utils import (compute_2d_iou, load_dataset_config, parse_bbox_2d_from_text,
                    parse_kitti_label, scale_bbox)

# Categories that map to "person" for evaluation
PERSON_CATEGORIES = {'person', "Person", "people", "People", "pedestrian", "Pedestrian"}

# Default minimum bbox area in pixels (used when not set per-dataset)
DEFAULT_MIN_BBOX_AREA = 0

# Final path segment that uses MIN_BBOX_AREA=400 by default
SEQ_WITH_NOISE = 'IVA-0009-KPI-05_190916_10ft-60-deg.mp4'


def compute_bbox_area(bbox):
    """
    Compute the area of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2] format

    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox[:4]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


# Per-family detection prompt — `_PARSERS` chain (Qwen + Gemini + bare-array)
# handles both output shapes, so only the prompt needs family dispatch here.
_DETECTION_PROMPT_QWEN = (
    "Locate every instance that belongs to the following categories: 'person'. "
    'Report bbox coordinates as JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
    "Coordinates normalized to 0-1000."
)
_DETECTION_PROMPT_GEMINI = (
    "Locate every instance that belongs to the following categories: 'person'. "
    'Report bbox coordinates as JSON: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}]. '
    "Coordinates normalized to 0-1000."
)
# MiMo (Qwen2.5-VL backbone) inherits absolute-pixel grounding prior; same
# JSON key/order as Qwen, only the scale clause differs.
_DETECTION_PROMPT_MIMO = (
    "Locate every instance that belongs to the following categories: 'person'. "
    'Report bbox coordinates as JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
    "Coordinates are absolute pixel values (not normalized to 0-1000)."
)
_DETECTION_PROMPTS = {
    'cr':     _DETECTION_PROMPT_QWEN,
    'qwen3':  _DETECTION_PROMPT_QWEN,
    'gemini': _DETECTION_PROMPT_GEMINI,
    'gemma4': _DETECTION_PROMPT_GEMINI,
    'mimo':   _DETECTION_PROMPT_MIMO,
}


def map_label_to_person(label):
    """
    Map various person labels to 'person'.

    Args:
        label: Original label string

    Returns:
        'person' if label is a person type, otherwise original label
    """
    if label.lower() in PERSON_CATEGORIES:
        return 'person'
    return label.lower()


class Astro2DDetectionDataset(ImageBaseDataset):
    """Dataset class for 2D object detection evaluation in KITTI format."""

    TYPE = 'VQA'  # Use VQA type so predictions are treated as text
    MODALITY = 'IMAGE'

    DEFAULT_FAMILY = 'cr'

    # Per-dataset-name layout dispatch for the VANTAGE-bench stage. Cosmos rows
    # continue to read from datasets.yaml (legacy compat); VANTAGE rows live
    # here so the JSON config row only needs `dataset=...` to switch sources.
    # Each entry: data_root (s3:// URL), images_subdir, labels_subdir.
    _VANTAGE_S3_BASE = (
        's3://cosmos_understanding/benchmark/'
        'vantage_benchmark_hf_release_annotations/2dbbox'
    )
    _LAYOUT = {
        'VANTAGE_Astro2D_C0065': dict(
            data_root=f'{_VANTAGE_S3_BASE}/C0065-QA.m4v',
            images_subdir='images',
            labels_subdir='labels',
        ),
        'VANTAGE_Astro2D_IVAKPI05': dict(
            data_root=f'{_VANTAGE_S3_BASE}/IVA-0009-KPI-05_190916_10ft-60-deg.mp4',
            images_subdir='images',
            labels_subdir='labels',
        ),
        'VANTAGE_Astro2D_IVAKPI11': dict(
            data_root=(
                f'{_VANTAGE_S3_BASE}/IVA-0009-KPI-11_220320_NVR_ch2_main_'
                '20220201180404_20220201190000_4_cut_4.mp4'
            ),
            images_subdir='images',
            labels_subdir='labels',
        ),
    }

    @classmethod
    def supported_datasets(cls):
        return [
            'Astro2D', 'Astro2DBench_C0065', 'Astro2DBench_IVAKPI05', 'Astro2DBench_IVAKPI11',
            *cls._LAYOUT.keys(),
        ]

    def __init__(self, dataset='Astro2D', data_root=None, **kwargs):
        """
        Args:
            dataset: Dataset name. VANTAGE_Astro2D_* names dispatch via the
                class-level _LAYOUT table (native-resolution 1080p stage);
                cosmos names dispatch via datasets.yaml (960×544 downsampled stage).
            data_root: Optional override for the dispatched data_root.
            min_bbox_area: Optional. Min bbox area in pixels; bboxes smaller than this are
                filtered out. If not set in datasets.yaml or kwargs, defaults to 400 when
                the path ends with "IVA-0009-KPI-05_190916_10ft-60-deg.mp4", else 0.
        """
        self.model_family = kwargs.pop('model_family', self.DEFAULT_FAMILY)
        self.dataset_name = dataset
        self.data_root = data_root

        layout = self._LAYOUT.get(dataset)
        dataset_cfg = {}
        if layout is not None:
            # VANTAGE branch: subdirs and S3 path come from the in-class table.
            if self.data_root is None:
                self.data_root = layout['data_root']
            images_subdir = layout['images_subdir']
            labels_subdir = layout['labels_subdir']
            cache_subdir_root = 'vantage_astro2d'
        else:
            # Cosmos branch: subdirs + S3 path come from datasets.yaml (the
            # 960×544 downsampled stage). Untouched by the dispatch refactor.
            dataset_cfg = load_dataset_config(dataset) or {}
            if data_root is None and 'data_root' in dataset_cfg:
                self.data_root = dataset_cfg['data_root']
            images_subdir = 'images_hres'
            labels_subdir = 'labels_hres'
            cache_subdir_root = 'metropolis2d_astro'

        if self.data_root is None:
            raise ValueError(
                f"data_root must be specified or configured in datasets.yaml for dataset '{dataset}'"
            )

        if self.data_root.startswith('s3://'):
            from pathlib import Path

            from s3fs import S3FileSystem

            cache_dir = LMUDataRoot()
            seq_name = self.data_root.split('/')[-1]
            dataset_dir_path = Path(cache_dir) / cache_subdir_root / seq_name
            dataset_dir_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"preparing dataset to {dataset_dir_path}")

            # Copy from S3 if not already exists
            if not dataset_dir_path.exists():
                print(f"copying dataset from {self.data_root} to {dataset_dir_path}")

                # Use S3FileSystem with team-cosmos profile and custom endpoint
                s3 = S3FileSystem(
                    anon=False,
                    profile='team-cosmos',
                    client_kwargs={'endpoint_url': 'https://pdx.s8k.io'}
                )

                # Download all files from S3 recursively
                s3.get(self.data_root, str(dataset_dir_path), recursive=True)
                print(f"Successfully downloaded dataset from S3")
            self.img_root = str(dataset_dir_path)
            self.images_dir = os.path.join(dataset_dir_path, images_subdir)
            self.labels_dir = os.path.join(dataset_dir_path, labels_subdir)
        else:
            self.img_root = self.data_root
            self.images_dir = os.path.join(self.data_root, images_subdir)
            self.labels_dir = os.path.join(self.data_root, labels_subdir)

        # min_bbox_area: from config, or 400 if path ends with IVA-0009-KPI-05_190916_10ft-60-deg.mp4, else 0
        if 'min_bbox_area' in dataset_cfg:
            self.min_bbox_area = int(dataset_cfg['min_bbox_area'])
        elif kwargs.get('min_bbox_area') is not None:
            self.min_bbox_area = int(kwargs['min_bbox_area'])
        else:
            final_name = self.data_root.rstrip('/').split('/')[-1]
            self.min_bbox_area = 400 if final_name == SEQ_WITH_NOISE else DEFAULT_MIN_BBOX_AREA

        # Cache for ground truth
        self._gt_cache = {}

        # Build data structure
        self.data = self._build_data_structure()

        # Call post build hook for compatibility
        try:
            self.post_build(self.dataset_name)
        except Exception:
            pass

    def _get_image_files(self):
        """Get list of image files from the images directory."""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []

        for filename in sorted(os.listdir(self.images_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                image_files.append(filename)

        return image_files

    def _load_ground_truth(self, image_filename):
        """
        Load ground truth labels for an image.

        Args:
            image_filename: Name of the image file

        Returns:
            List of ground truth objects with 'label' and 'bbox'
        """
        if image_filename in self._gt_cache:
            return self._gt_cache[image_filename]

        # Construct label path (same name as image, but .txt extension)
        base_name = os.path.splitext(image_filename)[0]
        label_path = os.path.join(self.labels_dir, base_name + '.txt')

        gt_objects = parse_kitti_label(label_path)

        # Map labels to 'person'
        for obj in gt_objects:
            obj['original_label'] = obj['label']
            obj['label'] = map_label_to_person(obj['label'])

        self._gt_cache[image_filename] = gt_objects
        return gt_objects

    def _select_prompt(self) -> str:
        return _DETECTION_PROMPTS.get(self.model_family, _DETECTION_PROMPTS[self.DEFAULT_FAMILY])

    def _build_data_structure(self):
        """Build the data structure for VLMEvalKit format."""
        logger = get_logger('Astro2D')

        image_files = self._get_image_files()
        logger.info(f"Found {len(image_files)} images in {self.images_dir}")

        prompt = self._select_prompt()
        data_list = []
        for idx, image_filename in enumerate(image_files):
            image_path = os.path.join(self.images_dir, image_filename)

            # Load ground truth to check if there are objects
            gt_objects = self._load_ground_truth(image_filename)

            row = {
                'index': str(idx),
                'image_path': image_path,
                'image_filename': image_filename,
                'question': prompt,
                'num_gt_objects': len(gt_objects),
            }
            data_list.append(row)

        logger.info(f"Built dataset with {len(data_list)} samples")
        return pd.DataFrame(data_list)

    def build_prompt(self, line):
        """Build prompt for 2D detection."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']

        question = line['question']

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=question)
        ]

        return msgs

    def _compute_f1_at_iou(self, all_predictions, all_gt_boxes, iou_threshold):
        """
        Compute F1 score at a specific IoU threshold.

        Args:
            all_predictions: List of (pred_boxes_person, gt_boxes_person) tuples per image
            all_gt_boxes: Not used, kept for API compatibility
            iou_threshold: IoU threshold for matching

        Returns:
            Tuple of (precision, recall, f1, tp, fp, fn)
        """
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for pred_boxes_person, gt_boxes_person in all_predictions:
            total_gt += len(gt_boxes_person)

            # Sort predictions by confidence (descending) and shuffle for tie-breaking
            pred_boxes_sorted = sorted(pred_boxes_person, key=lambda x: x.get('score', 1.0), reverse=True)
            # random.shuffle(pred_boxes_sorted)

            # Match predictions to ground truth at IoU threshold
            gt_matched = [False] * len(gt_boxes_person)

            for pred in pred_boxes_sorted:
                pred_bbox = pred['bbox']
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_boxes_person):
                    if gt_matched[gt_idx]:
                        continue
                    iou = compute_2d_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    total_tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    total_fp += 1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1, total_tp, total_fp, total_gt - total_tp

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate predictions using Precision, Recall, and F1 score at multiple IoU thresholds.

        All predictions and ground truth labels are mapped to 'person' category
        before evaluation. Bboxes with area smaller than min_bbox_area are filtered out.

        Reports:
        - F1@0.5: F1 score at IoU threshold 0.5
        - F1@0.95: F1 score at IoU threshold 0.95
        - F1@mIOU: Mean F1 score across IoU thresholds from 0.5 to 0.95 (step 0.05)
        """
        logger = get_logger('Astro2D')

        # Try different file extensions if the specified one doesn't exist
        if not os.path.exists(eval_file):
            base = os.path.splitext(eval_file)[0]
            for ext in ['.xlsx', '.tsv', '.json', '.pkl']:
                candidate = base + ext
                if os.path.exists(candidate):
                    eval_file = candidate
                    logger.info(f'Using alternate file format: {eval_file}')
                    break
            else:
                logger.error(f'No prediction file found with base: {base}')
                return {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'f1_0.95': 0.0,
                    'f1_mIOU': 0.0,
                    'total_predictions': 0,
                    'valid_bbox_predictions': 0,
                    'error': 'Prediction file not found'
                }

        try:
            data = load(eval_file)
            logger.info(f'Loaded {len(data)} predictions from {eval_file}')
        except Exception as e:
            logger.error(f'Failed to load predictions: {e}')
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'f1_0.95': 0.0,
                'f1_mIOU': 0.0,
                'total_predictions': 0,
                'valid_bbox_predictions': 0,
                'error': str(e)
            }

        # Collect all predictions and ground truths for multi-threshold evaluation
        all_predictions = []  # List of (pred_boxes_person, gt_boxes_person) tuples
        total_pred = 0
        total_gt = 0
        valid_count = 0
        total_gt_filtered = 0  # GT boxes filtered due to small size
        total_pred_filtered = 0  # Pred boxes filtered due to small size

        coord_scale = 'pixel' if self.model_family == 'mimo' else 'normalized'
        # Process each image
        for idx, row in data.iterrows():
            # Parse prediction
            pred_text = str(row.get('prediction', ''))
            pred_boxes_raw = parse_bbox_2d_from_text(pred_text, coord_scale=coord_scale)

            image_path = row.get('image_path', '')
            # Load image
            try:
                pil_image = PIL.Image.open(image_path)
                pil_image = pil_image.convert('RGB')
                pil_image = ImageOps.exif_transpose(pil_image)
                width, height = pil_image.size
            except Exception as e:
                logger.error(f"Failed to load image {image_path}: {e}")
                width, height = 640, 480

            # Normalize prediction format
            pred_boxes = []
            for pred in pred_boxes_raw:
                if isinstance(pred, dict):
                    # Handle different bbox key names
                    bbox = None
                    for key in ['bbox_2d']:
                        if key in pred:
                            bbox = pred[key]
                            break

                    if bbox is not None:
                        bbox = scale_bbox(bbox, height, width, scale_factor=1000, coord_scale=coord_scale)

                    if bbox is not None and len(bbox) >= 4:
                        pred_boxes.append({
                            'bbox': bbox[:4],
                            'label': map_label_to_person(pred.get('label', 'person')),
                            'score': pred.get('score', pred.get('confidence', 1.0))
                        })

            if len(pred_boxes) > 0:
                valid_count += 1

            # Get ground truth
            image_filename = row.get('image_filename', '')
            if not image_filename:
                image_filename = os.path.basename(image_path)

            gt_boxes = self._load_ground_truth(image_filename)

            # Filter to only 'person' category
            gt_boxes_person_raw = [gt for gt in gt_boxes if gt['label'] == 'person']
            pred_boxes_person_raw = [p for p in pred_boxes if p['label'] == 'person']

            # Filter out small bboxes (area < min_bbox_area)
            gt_boxes_person = [gt for gt in gt_boxes_person_raw if compute_bbox_area(gt['bbox']) >= self.min_bbox_area]
            pred_boxes_person = [p for p in pred_boxes_person_raw if compute_bbox_area(p['bbox']) >= self.min_bbox_area]

            total_gt_filtered += len(gt_boxes_person_raw) - len(gt_boxes_person)
            total_pred_filtered += len(pred_boxes_person_raw) - len(pred_boxes_person)

            total_gt += len(gt_boxes_person)
            total_pred += len(pred_boxes_person)

            all_predictions.append((pred_boxes_person, gt_boxes_person))

        # Compute F1 at multiple IoU thresholds
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5, 0.55, ..., 0.95
        f1_scores = {}

        for iou_thresh in iou_thresholds:
            precision, recall, f1, tp, fp, fn = self._compute_f1_at_iou(all_predictions, None, iou_thresh)
            f1_scores[iou_thresh] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        # Get metrics at IoU=0.5
        metrics_05 = f1_scores[0.5]
        precision = metrics_05['precision']
        recall = metrics_05['recall']
        f1_05 = metrics_05['f1']
        total_tp = metrics_05['tp']
        total_fp = metrics_05['fp']

        # Get F1 at IoU=0.95
        f1_095 = f1_scores[0.95]['f1']

        # Compute mean F1 across all thresholds (F1@mIOU)
        f1_mIOU = np.mean([f1_scores[t]['f1'] for t in iou_thresholds])

        # Primary rollup: f1 (F1@IoU=0.5) — matches VANTAGE paper Table 2.
        # f1_mIOU is retained as a side metric for continuity with previously
        # reported numbers; downstream callers reading `result['f1']` continue
        # to see F1@0.5 (no key rename).
        result = {
            'f1': float(f1_05 * 100),
            'precision': float(precision * 100),
            'recall': float(recall * 100),
            'f1_0.95': float(f1_095 * 100),
            'f1_mIOU': float(f1_mIOU * 100),
            'total_predictions': len(data),
            'valid_bbox_predictions': valid_count,
            'valid_rate': valid_count / len(data) if len(data) > 0 else 0,
            'total_gt_objects': total_gt,
            'total_pred_objects': total_pred,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_gt - total_tp,
            'gt_filtered_small': total_gt_filtered,
            'pred_filtered_small': total_pred_filtered,
        }

        logger.info(f"Precision@IoU=0.5: {result['precision']:.2f}%")
        logger.info(f"Recall@IoU=0.5: {result['recall']:.2f}%")
        logger.info(f"F1@IoU=0.5: {result['f1']:.2f}%")
        logger.info(f"F1@IoU=0.95: {result['f1_0.95']:.2f}%")
        logger.info(f"F1@mIOU (0.5:0.05:0.95): {result['f1_mIOU']:.2f}%")
        logger.info(f"TP: {total_tp}, FP: {total_fp}, FN: {total_gt - total_tp}")
        logger.info(f"Valid predictions: {valid_count}/{len(data)} ({result['valid_rate']:.2%})")
        logger.info(f"Filtered small bboxes - GT: {total_gt_filtered}, Pred: {total_pred_filtered}")

        return result


# Deferred-class factory: ConcatDataset is defined in `vlmeval/dataset/__init__.py`,
# which imports this module at top — top-level `from .. import ConcatDataset` would
# be circular. The factory is invoked from `__init__.py` after ConcatDataset exists.
def _build_astro2dbench_dataset(concat_base):
    class Astro2DBenchDataset(concat_base):
        DATASET_SETS = {
            'Astro2DBench': ['Astro2DBench_C0065', 'Astro2DBench_IVAKPI05', 'Astro2DBench_IVAKPI11'],
            'VANTAGE_Astro2D': [
                'VANTAGE_Astro2D_C0065',
                'VANTAGE_Astro2D_IVAKPI05',
                'VANTAGE_Astro2D_IVAKPI11',
            ],
        }

        def evaluate(self, eval_file, **judge_kwargs):
            result = super().evaluate(eval_file, **judge_kwargs)
            # f1 (F1@0.5) first so dict iteration / Format-D parsers pick it
            # up as the primary; f1_mIOU is kept as a side metric for continuity.
            for metric in ('f1', 'precision', 'recall', 'f1_0.95', 'f1_mIOU'):
                vals = [
                    result[f'{sub}:{metric}']
                    for sub in self.datasets
                    if f'{sub}:{metric}' in result
                ]
                if vals:
                    result[metric] = float(np.mean(vals))
            return result

    return Astro2DBenchDataset
