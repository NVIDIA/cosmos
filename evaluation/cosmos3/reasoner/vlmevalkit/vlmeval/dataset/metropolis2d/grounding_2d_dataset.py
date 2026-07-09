"""2D Grounding Dataset for images in RefCOCO or JSONL format.

Following the structure of detection_2d_dataset.py, this dataset evaluates
2D visual grounding using Accuracy@IoU metrics.

Supported annotation formats:

1. RefCOCO format (JSON):
   - images/ directory containing image files
   - A JSON annotation file with referring expressions and bounding boxes

2. JSONL format (one JSON per line):
   {"image_path": "path/to/image.jpg", "gt": {"car": [[x1,y1,x2,y2], ...], ...}, 
    "categories": ["car", "truck", ...], "dataset_name": "...", "task_name": "..."}

The question for each sample:
f"Locate {object description} in the provided image and output its bbox coordinates using JSON format"

For evaluation, common grounding metrics are used:
- Acc@0.5: Accuracy at IoU threshold 0.5
- Acc@0.25: Accuracy at IoU threshold 0.25
- Mean IoU: Average IoU across all samples

Note: One expression can be associated with one or multiple bounding boxes.
When multiple GT boxes exist, a prediction is correct if it matches ANY of them.
"""

import os
import json
import numpy as np
import pandas as pd
import PIL.Image
import yaml
from PIL import ImageOps
from ..image_base import ImageBaseDataset
from ...smp import get_logger, load, LMUDataRoot
from .utils import _PARSERS, load_dataset_config, scale_bbox, compute_2d_iou


def convert_bbox_xywh_to_xyxy(bbox, image_width=None, image_height=None):
    """
    Convert bbox from xywh to xyxy format if needed.

    Args:
        bbox: Bounding box [x, y, w, h] or [x1, y1, x2, y2]
        image_width: Image width for heuristic detection
        image_height: Image height for heuristic detection

    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    if len(bbox) != 4:
        return bbox

    # Heuristic: if bbox[2] and bbox[3] are small relative to image size,
    # it's likely xywh format
    max_dim = max(image_width or 10000, image_height or 10000)
    if bbox[2] < max_dim / 2 and bbox[3] < max_dim / 2:
        # xywh format -> convert to xyxy
        x1, y1, w, h = bbox
        return [x1, y1, x1 + w, y1 + h]
    return bbox


def parse_refcoco_annotations(annotation_path):
    """
    Parse RefCOCO format annotation file.

    Expected JSON format:
    {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            ...
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [x, y, width, height],  # COCO format (xywh) - single box
                "bboxes": [[x1,y1,w1,h1], [x2,y2,w2,h2]],  # Multiple boxes (optional)
                "sentence": "the red car on the left",
                "category": "car"  # optional
            },
            ...
        ]
    }

    Alternative simplified format:
    [
        {
            "image": "image1.jpg",
            "bbox": [x1, y1, x2, y2],  # xyxy format - single box
            "bboxes": [[x1,y1,x2,y2], [x1,y1,x2,y2]],  # Multiple boxes (optional)
            "sentence": "the red car on the left"
        },
        ...
    ]

    Note: One expression can be associated with one or multiple bounding boxes.
    Use 'bbox' for single box or 'bboxes' for multiple boxes.

    Returns:
        List of dicts with 'image', 'bboxes' (list of [x1, y1, x2, y2]), 'sentence'
    """
    if not os.path.exists(annotation_path):
        return []

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    annotations = []

    # Handle COCO-style format
    if isinstance(data, dict) and 'images' in data and 'annotations' in data:
        # Build image_id to filename mapping
        id_to_image = {img['id']: img for img in data['images']}

        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in id_to_image:
                continue

            image_info = id_to_image[image_id]
            img_w = image_info.get('width')
            img_h = image_info.get('height')

            # Handle multiple bboxes per expression
            bboxes = []
            if 'bboxes' in ann and ann['bboxes']:
                # Multiple boxes provided
                for bbox in ann['bboxes']:
                    bbox = convert_bbox_xywh_to_xyxy(bbox, img_w, img_h)
                    bboxes.append(bbox)
            elif 'bbox' in ann:
                # Single box
                bbox = convert_bbox_xywh_to_xyxy(ann['bbox'], img_w, img_h)
                bboxes.append(bbox)

            if not bboxes:
                continue

            # Handle multiple sentences per annotation
            sentences = ann.get('sentences', [])
            if not sentences and 'sentence' in ann:
                sentences = [ann['sentence']]
            elif not sentences and 'raw' in ann:
                sentences = [ann['raw']]

            for sentence in sentences:
                if isinstance(sentence, dict):
                    sentence = sentence.get('raw', sentence.get('sent', str(sentence)))

                annotations.append({
                    'image': image_info['file_name'],
                    'image_id': image_id,
                    'bboxes': bboxes,  # List of bboxes
                    'sentence': sentence,
                    'category': ann.get('category', ann.get('category_name', '')),
                    'width': img_w,
                    'height': img_h,
                })

    # Handle simplified list format
    elif isinstance(data, list):
        for ann in data:
            img_w = ann.get('width')
            img_h = ann.get('height')

            # Handle multiple bboxes per expression
            bboxes = []
            if 'bboxes' in ann and ann['bboxes']:
                for bbox in ann['bboxes']:
                    bboxes.append(bbox)  # Assume already in xyxy format
            elif 'bbox' in ann:
                bboxes.append(ann['bbox'])  # Assume already in xyxy format

            if not bboxes:
                continue

            annotations.append({
                'image': ann.get('image', ann.get('file_name', '')),
                'image_id': ann.get('image_id', ''),
                'bboxes': bboxes,  # List of bboxes
                'sentence': ann.get('sentence', ann.get('expression', '')),
                'category': ann.get('category', ''),
                'width': img_w,
                'height': img_h,
            })

    return annotations


def parse_jsonl_annotations(annotation_path, data_root=None):
    """
    Parse JSONL format annotation file where each line is a JSON object.

    Expected JSONL format (one JSON object per line):
    {
        "image_path": "visdrone/image.jpg",  # relative path to image
        "gt": {
            "car": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],  # category -> list of bboxes
            "truck": [[x1, y1, x2, y2], ...],
            ...
        },
        "categories": ["car", "truck", "bus", "van"],
        "dataset_name": "VisDrone",
        "task_name": "common_object_detection"
    }

    This function creates one annotation entry per category per image.
    Each category's bboxes become the ground truth for that grounding query.

    Args:
        annotation_path: Path to JSONL annotation file
        data_root: Root directory for resolving relative image paths

    Returns:
        List of dicts with 'image', 'image_path', 'bboxes', 'sentence', 'category'
    """
    if not os.path.exists(annotation_path):
        return []

    annotations = []

    with open(annotation_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num + 1}: {e}")
                continue

            image_path = data.get('image_path', '')
            gt = data.get('gt', {})
            categories = data.get('categories', [])

            if not image_path or not gt:
                continue

            # Resolve full image path
            if data_root and not os.path.isabs(image_path):
                full_image_path = os.path.join(data_root, image_path)
            else:
                full_image_path = image_path

            # Create one annotation entry per category
            for category in categories:
                if category not in gt:
                    continue

                bboxes = gt[category]
                if not bboxes:
                    continue

                # Ensure bboxes is a list of lists
                if bboxes and isinstance(bboxes[0], (int, float)):
                    # Single bbox, wrap it
                    bboxes = [bboxes]

                annotations.append({
                    'image': image_path,  # Relative path
                    'image_path': full_image_path,  # Full path
                    'image_id': f"{line_num}",  # TODO(yuw): add category? _{category}
                    'bboxes': bboxes,  # List of bboxes in xyxy format
                    'sentence': category,  # Category name as the grounding query
                    'category': category,
                    'dataset_name': data.get('dataset_name', ''),
                    'task_name': data.get('task_name', ''),
                    'width': None,
                    'height': None,
                })
    return annotations


def parse_llava_grounding_annotations(annotation_path, data_root=None):
    """Parse VANTAGE's LLaVA-conversation-shape grounding annotation file.

    Expected format (JSON array, one item per (image, sentence) pair):
        [
          {
            "media": "images/<file>.jpg",
            "conversations": [
              {"from": "human", "value": "...Locate \"<sentence>\"..."},
              {"from": "gpt",   "value": "[[x1, y1, x2, y2], ...]"}
            ],
            "_meta": {"sentence": str, "image_width": int, "image_height": int,
                      "source": str, "object_category": str, ...}
          },
          ...
        ]

    GT bboxes live in conversations[1].value as a stringified JSON list. Sentence
    comes from _meta.sentence. Output shape matches parse_jsonl_annotations so the
    rest of the class is format-agnostic.
    """
    if not os.path.exists(annotation_path):
        return []

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    annotations = []
    for idx, item in enumerate(data):
        media = item.get('media', '')
        if not media:
            continue
        meta = item.get('_meta', {}) or {}
        sentence = meta.get('sentence', '')
        if not sentence:
            continue
        # GT bboxes are the assistant's "ideal response" — stringified JSON list.
        convs = item.get('conversations', [])
        gpt_turn = next((c for c in convs if c.get('from') == 'gpt'), None)
        if gpt_turn is None:
            continue
        try:
            bboxes = json.loads(gpt_turn.get('value', ''))
        except json.JSONDecodeError:
            continue
        if not isinstance(bboxes, list):
            continue

        full_image_path = (
            os.path.join(data_root, media) if data_root and not os.path.isabs(media) else media
        )
        annotations.append({
            'image': media,
            'image_path': full_image_path,
            'image_id': f'{idx}',
            'bboxes': bboxes,
            'sentence': sentence,
            'category': meta.get('object_category', item.get('category', '')),
            'dataset_name': meta.get('source', ''),
            'task_name': item.get('task', 'referring_expressions'),
            'width': meta.get('image_width'),
            'height': meta.get('image_height'),
        })
    return annotations


def detect_annotation_format(annotation_path):
    """
    Detect the format of the annotation file.

    Returns:
        'jsonl', 'llava' (VANTAGE LLaVA-conversation shape), or 'json' (refcoco).
    """
    if annotation_path.endswith('.jsonl'):
        return 'jsonl'

    # Try to detect by reading first line
    try:
        with open(annotation_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line.startswith('{') and not first_line.endswith('}'):
                # Could be multi-line JSON
                return 'json'
            if first_line.startswith('{'):
                # Try to parse as single JSON object (JSONL line)
                data = json.loads(first_line)
                # Check for JSONL-specific keys
                if 'gt' in data and 'categories' in data:
                    return 'jsonl'
            if first_line.startswith('['):
                # JSON array — could be refcoco or LLaVA-conversation. Sniff
                # the first item: LLaVA items have a 'conversations' list.
                try:
                    with open(annotation_path, 'r') as g:
                        items = json.load(g)
                    if isinstance(items, list) and items:
                        first = items[0]
                        if isinstance(first, dict) and 'conversations' in first and 'media' in first:
                            return 'llava'
                except Exception:
                    pass
                return 'json'
    except Exception:
        pass

    return 'json'  # Default to JSON


def parse_bbox_from_text(text: str, coord_scale: str = 'normalized') -> list:
    """Grounding-shape: `[[x1,y1,x2,y2]_norm, ...]`.

    Dispatches across Qwen / Gemini / bare-array shapes via the shared
    `_PARSERS` registry in `metropolis2d/utils.py`. `coord_scale` is forwarded
    to each parser's range gate (relaxes the 0-1000 upper bound for 'pixel').
    """
    for parser in _PARSERS:
        tuples = parser(text, coord_scale=coord_scale)
        if tuples:
            return [coords for _, coords, is_point in tuples if not is_point]
    return []


class Metropolis2DGroundingDataset(ImageBaseDataset):
    """Dataset class for 2D visual grounding evaluation in RefCOCO format."""

    TYPE = 'VQA'  # Use VQA type so predictions are treated as text
    MODALITY = 'IMAGE'

    DEFAULT_FAMILY = 'cr'

    # Per-family prompt templates. model_family is injected via dataset_conf (--profile <family>).
    _CR_PROMPT = (
        'As an AI visual assistant, your task is to identify and locate specific objects in the provided image.\n\n'
        'Supplied Description: {description}\n\n'
        'Task:\n'
        'Based on the description and the image content, identify the key groups of objects mentioned. '
        'For each group, provide a descriptive label and the precise bounding box coordinates for every '
        'individual instance in that group.\n\n'
        'Coordinates must be normalized to a 0-1000 scale in [x1, y1, x2, y2] format.\n\n'
        'Output Format:\n'
        'For each group of objects, output one line in exactly this format:\n'
        'The [object description]: [[x1, y1, x2, y2], [x3, y3, x4, y4]]\n\n'
        'Example:\n'
        'The blue cars parked on the right: [[579, 454, 690, 636], [342, 441, 435, 608]]'
    )
    _GEMINI_PROMPT = (
        'You are performing referring-expression grounding.\n\n'
        'Your task is to locate only the target object or objects described by the '
        'referring expression below.\n\n'
        'Referring expression:\n'
        '{description}\n\n'
        'Important rules:\n'
        '1. Do not detect every object of the same category.\n'
        '2. Use the spatial words in the expression, such as left, right, top, bottom, '
        'near, far, closest, or largest, to choose the correct instance.\n'
        '3. If the expression describes one object, return one bounding box.\n'
        '4. If the expression clearly describes multiple objects, return one bounding box '
        'for each target object.\n'
        '5. Coordinates must be normalized to a 0-1000 scale.\n'
        '6. Use [x1, y1, x2, y2] format.\n'
        '7. Return only valid JSON. Do not include explanations, markdown, or extra text.\n\n'
        'Output format:\n'
        '{{"bbox_2d": [[x1, y1, x2, y2]]}}'
    )
    # MiMo (Qwen2.5-VL backbone) inherits the absolute-pixel grounding prior;
    # prompt key/order matches Qwen, only the scale clause differs.
    _MIMO_PROMPT = (
        'As an AI visual assistant, your task is to identify and locate specific objects in the provided image.\n\n'
        'Supplied Description: {description}\n\n'
        'Task:\n'
        'Based on the description and the image content, identify the key groups of objects mentioned. '
        'For each group, provide a descriptive label and the precise bounding box coordinates for every '
        'individual instance in that group.\n\n'
        'Coordinates must be absolute pixel values (not normalized to 0-1000) in [x1, y1, x2, y2] format.\n\n'
        'Output Format:\n'
        'For each group of objects, output one line in exactly this format:\n'
        'The [object description]: [[x1, y1, x2, y2], [x3, y3, x4, y4]]\n\n'
        'Example:\n'
        'The blue cars parked on the right: [[579, 454, 690, 636], [342, 441, 435, 608]]'
    )
    PROMPTS = {
        'cr':     _CR_PROMPT,
        'qwen3':  _CR_PROMPT,  # placeholder — branch when qwen3-specific variant is decided
        'gemini': _GEMINI_PROMPT,
        'mimo':   _MIMO_PROMPT,
    }

    @classmethod
    def supported_datasets(cls):
        return ['Metropolis2DGrounding', 'Metropolis2DGrounding_val', 'Metropolis2DGround', 'VANTAGE_2DGrounding']

    def __init__(self, dataset='Metropolis2DGrounding', data_root=None, annotation_file=None, **kwargs):
        """
        Args:
            dataset: Dataset name (used to look up config in datasets.yaml)
            data_root: Root directory containing 'images' subdirectory.
                       If None, will be loaded from datasets.yaml based on dataset name.
            annotation_file: Path to RefCOCO format annotation JSON file
        """
        self.model_family = kwargs.pop('model_family', self.DEFAULT_FAMILY)
        self.dataset_name = dataset
        self.data_root = data_root
        self.annotation_file = annotation_file

        if data_root is None or annotation_file is None:
            # Try to load from datasets.yaml
            dataset_cfg = load_dataset_config(dataset, task='grounding')
            if dataset_cfg:
                if data_root is None and 'data_root' in dataset_cfg:
                    self.data_root = dataset_cfg['data_root']
                if annotation_file is None and 'annotation_file' in dataset_cfg:
                    self.annotation_file = dataset_cfg['annotation_file']

        if self.data_root is None:
            raise ValueError(
                f"data_root must be specified or configured in datasets.yaml for dataset '{dataset}'"
            )

        if self.data_root.startswith('s3://'):
            from pathlib import Path
            from s3fs import S3FileSystem

            cache_dir = LMUDataRoot()
            s3_anno_url = self.annotation_file
            task_name = s3_anno_url.split('/')[-1]
            # Per-dataset cache subdir: cosmos and VANTAGE rows have different
            # image-tree layouts ('visdrone/' vs 'images/') and different S3
            # sources; sharing one cache dir would corrupt whichever runs second.
            cache_subdir = 'vantage_2d_grounding' if dataset.startswith('VANTAGE_') else 'metropolis2d_grounding'
            dataset_dir_path = Path(cache_dir) / cache_subdir
            dataset_dir_path.parent.mkdir(parents=True, exist_ok=True)
            local_anno_path = dataset_dir_path / 'annotations' / task_name
            print(f"preparing dataset to {dataset_dir_path}")

            s3 = None

            # First-time fetch: pull the full data_root recursively (includes
            # annotations/ and image files).
            if not dataset_dir_path.exists():
                print(f"copying dataset from {self.data_root} to {dataset_dir_path}")
                s3 = S3FileSystem(
                    anon=False,
                    profile='team-cosmos',
                    client_kwargs={'endpoint_url': 'https://pdx.s8k.io'}
                )
                s3.get(self.data_root, str(dataset_dir_path), recursive=True)
                print(f"Successfully downloaded dataset from S3")

            # Cache-miss fallback: dir exists (e.g. populated by a prior run
            # against a different annotation file), but the requested annotation
            # isn't there. Fetch just that one file rather than re-pulling the
            # whole image tree — guards against stale NFS cache when the
            # annotation_file in datasets.yaml changes between runs.
            if not local_anno_path.exists():
                print(f"annotation cache miss; fetching {s3_anno_url} to {local_anno_path}")
                if s3 is None:
                    s3 = S3FileSystem(
                        anon=False,
                        profile='team-cosmos',
                        client_kwargs={'endpoint_url': 'https://pdx.s8k.io'}
                    )
                local_anno_path.parent.mkdir(parents=True, exist_ok=True)
                s3.get(s3_anno_url, str(local_anno_path))

            self.img_root = str(dataset_dir_path)
            self.annotation_file = str(local_anno_path)
        else:
            self.img_root = self.data_root

        # Find annotation file
        if self.annotation_file is None:
            # Try common annotation file names
            for ann_name in ['annotations.json', 'refs.json', 'grounding.json', 'val.json']:
                ann_path = os.path.join(self.data_root, ann_name)
                if os.path.exists(ann_path):
                    self.annotation_file = ann_path
                    break

        if self.annotation_file is None and os.path.exists(self.annotation_file):
            raise ValueError("annotation_file must be specified or found in data_root")

        # Detect annotation format and load annotations
        self.annotation_format = detect_annotation_format(self.annotation_file)
        if self.annotation_format == 'jsonl':
            self.annotations = parse_jsonl_annotations(self.annotation_file, self.img_root)
        elif self.annotation_format == 'llava':
            self.annotations = parse_llava_grounding_annotations(self.annotation_file, self.img_root)
        else:
            self.annotations = parse_refcoco_annotations(self.annotation_file)
        assert len(self.annotations) > 0, f"No annotations found in {self.annotation_file}"
        # Build data structure
        self.data = self._build_data_structure()

        # Call post build hook for compatibility
        try:
            self.post_build(self.dataset_name)
        except Exception:
            pass

    def _build_data_structure(self):
        """Build the data structure for VLMEvalKit format."""
        logger = get_logger('Metropolis2DGrounding')
        logger.info(f"Loaded {len(self.annotations)} annotations from {self.annotation_file} "
                    f"(format: {self.annotation_format})")

        data_list = []
        skipped_count = 0

        for idx, ann in enumerate(self.annotations):
            # Handle image path based on annotation format
            if 'image_path' in ann and ann['image_path']:
                # JSONL format: image_path is already resolved
                image_path = ann['image_path']
                image_filename = os.path.join(self.img_root, os.path.basename(image_path))
            else:
                # RefCOCO format: construct path from img_root
                image_filename = ann['image']
                image_path = os.path.join(self.img_root, image_filename)

            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                skipped_count += 1
                if skipped_count <= 5:
                    logger.warning(f"Image not found: {image_path}")
                elif skipped_count == 6:
                    logger.warning("Suppressing further 'Image not found' warnings...")
                continue

            sentence = ann['sentence']
            gt_bboxes = ann['bboxes']  # List of bboxes

            # Generate the question/prompt
            question = self._select_prompt().format(description=sentence)

            row = {
                'index': str(idx),
                'image_path': image_path,
                'image_filename': image_filename,
                'question': question,
                'sentence': sentence,
                'gt_bboxes': json.dumps(gt_bboxes),  # Store list of bboxes as JSON string
                'num_gt_bboxes': len(gt_bboxes),
                'category': ann.get('category', ''),
                'image_width': ann.get('width'),
                'image_height': ann.get('height'),
            }
            data_list.append(row)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} annotations due to missing images")

        logger.info(f"Built dataset with {len(data_list)} samples")
        return pd.DataFrame(data_list)

    def _select_prompt(self) -> str:
        return self.PROMPTS.get(self.model_family, self.PROMPTS[self.DEFAULT_FAMILY])

    def build_prompt(self, line):
        """Build prompt for visual grounding."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = line['image_path']
        question = line['question']

        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=question)
        ]

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate grounding predictions using standard metrics.

        Metrics:
        - Acc@0.5: Accuracy at IoU threshold 0.5 (standard RefCOCO metric)
        - Acc@0.25: Accuracy at IoU threshold 0.25
        - Acc@0.75: Accuracy at IoU threshold 0.75
        - Mean IoU: Average IoU across all samples

        Note: When multiple GT boxes exist for one expression, a prediction is
        considered correct if it matches ANY of the GT boxes (max IoU is used).
        """
        logger = get_logger('Metropolis2DGrounding')

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
                    'Acc@0.5': 0.0,
                    'Acc@0.25': 0.0,
                    'Acc@0.75': 0.0,
                    'Mean_IoU': 0.0,
                    'error': 'Prediction file not found'
                }

        try:
            data = load(eval_file)
            logger.info(f'Loaded {len(data)} predictions from {eval_file}')
        except Exception as e:
            logger.error(f'Failed to load predictions: {e}')
            return {
                'Acc@0.5': 0.0,
                'Acc@0.25': 0.0,
                'Acc@0.75': 0.0,
                'Mean_IoU': 0.0,
                'error': str(e)
            }

        # Evaluation metrics
        iou_thresholds = [0.25, 0.5, 0.75]
        correct_at_threshold = {t: 0 for t in iou_thresholds}
        all_ious = []
        valid_count = 0
        total_count = len(data)

        coord_scale = 'pixel' if self.model_family == 'mimo' else 'normalized'
        for idx, row in data.iterrows():
            # Parse prediction
            pred_text = str(row.get('prediction', ''))
            pred_bbox = parse_bbox_from_text(pred_text, coord_scale=coord_scale)

            # Get ground truth bboxes (can be multiple)
            gt_bboxes_str = row.get('gt_bboxes', '[]')
            try:
                gt_bboxes = json.loads(gt_bboxes_str)
            except json.JSONDecodeError:
                gt_bboxes = []

            # Ensure gt_bboxes is a list of bboxes
            if gt_bboxes and isinstance(gt_bboxes[0], (int, float)):
                # Single bbox stored as flat list, wrap it
                gt_bboxes = [gt_bboxes]

            if len(pred_bbox) == 0 or len(gt_bboxes) == 0:
                all_ious.append(0.0)
                continue

            valid_count += 1

            # Get image dimensions for scaling
            image_path = row.get('image_path', '')
            try:
                pil_image = PIL.Image.open(image_path)
                pil_image = ImageOps.exif_transpose(pil_image)
                img_width, img_height = pil_image.size
            except Exception:
                img_width = row.get('image_width', 1000)
                img_height = row.get('image_height', 1000)
                if img_width is None:
                    img_width = 1000
                if img_height is None:
                    img_height = 1000

            pred_bboxes_scaled = [
                scale_bbox(pb, img_height, img_width, scale_factor=1000, coord_scale=coord_scale)
                for pb in pred_bbox
            ]

            # Best 1-to-1 matching: for each pred bbox find its best GT IoU,
            # then take the overall maximum across all pred bboxes.
            max_iou = 0.0
            for pb_scaled in pred_bboxes_scaled:
                for gt_bbox in gt_bboxes:
                    if len(gt_bbox) >= 4:
                        iou = compute_2d_iou(pb_scaled, gt_bbox)
                        max_iou = max(max_iou, iou)

            all_ious.append(max_iou)

            # Check accuracy at each threshold (using max IoU)
            for thresh in iou_thresholds:
                if max_iou >= thresh:
                    correct_at_threshold[thresh] += 1

        # Compute metrics
        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0

        result = {
            'Acc@0.25': float(correct_at_threshold[0.25] / total_count * 100) if total_count > 0 else 0.0,
            'Acc@0.5': float(correct_at_threshold[0.5] / total_count * 100) if total_count > 0 else 0.0,
            'Acc@0.75': float(correct_at_threshold[0.75] / total_count * 100) if total_count > 0 else 0.0,
            'Mean_IoU': float(mean_iou * 100),
            'total_samples': total_count,
            'valid_predictions': valid_count,
            'valid_rate': valid_count / total_count if total_count > 0 else 0.0,
        }

        logger.info(f"Acc@0.25: {result['Acc@0.25']:.2f}%")
        logger.info(f"Acc@0.5: {result['Acc@0.5']:.2f}%")
        logger.info(f"Acc@0.75: {result['Acc@0.75']:.2f}%")
        logger.info(f"Mean IoU: {result['Mean_IoU']:.2f}%")
        logger.info(f"Valid predictions: {valid_count}/{total_count} ({result['valid_rate']:.2%})")

        return result
