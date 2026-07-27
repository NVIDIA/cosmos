# Astro2D Detection Dataset

A 2D object detection dataset for evaluating Vision-Language Models (VLMs) on person detection tasks using KITTI format annotations.

## Overview

This dataset evaluates VLMs on their ability to detect and localize people in images. The evaluation uses **F1 score** metrics at multiple IoU thresholds.

## Data Format

The dataset expects data in **KITTI format** with the following directory structure:

```
data_root/
├── images_hres/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── labels_hres/
    ├── 000001.txt
    ├── 000002.txt
    └── ...
```

### Label Format (KITTI)

Each label file is a space-separated text file with one object per line:

```
<type> <truncated> <occluded> <alpha> <x1> <y1> <x2> <y2> <h> <w> <l> <x> <y> <z> <rotation_y>
```

Where:
- `type`: Object class (e.g., 'person', 'Person', 'pedestrian', 'Pedestrian', 'people', 'People')
- `x1, y1, x2, y2`: 2D bounding box coordinates (top left and bottom right corners in pixels bounded by the image size)
- Other fields: 3D information (not used for 2D evaluation)

## Usage

### Running Evaluation

```bash
# if using s3:// path, remember to set LMUData
python run.py \
    --data Astro2D \
    --model <your_model> \
    --work-dir ./output \
    --verbose
```

### Supported Dataset Names

- `Astro2D`

### Configuration

The dataset can be configured via `datasets.yaml` with a `data_root` path. The path can be:
- A local filesystem path
- An S3 path (e.g., `s3://bucket/path`) - data will be automatically downloaded to cache

## Prompt

The VLM is prompted with:

> "Locate every instance that belongs to the following categories: 'person'. Report bbox coordinates in JSON format."

## Expected Output Format

The model should output a JSON array of detected objects:

```json
[
    {"bbox_2d": [x1, y1, x2, y2], "label": "person"},
    {"bbox_2d": [x1, y1, x2, y2], "label": "person"},
    ...
]
```

**Note**: Bounding box coordinates should be in normalized format (0-1000 scale), which will be scaled back to pixel coordinates during evaluation.

## Evaluation Metrics

### Label Mapping

For evaluation, all person-related labels are mapped to a unified `"person"` category:
- `person` → `person`
- `Person` → `person`
- `people` → `person`
- `People` → `person`
- `pedestrian` → `person`
- `Pedestrian` → `person`

### Metrics

| Metric | Description |
|--------|-------------|
| **F1@0.5** | F1 score at IoU threshold 0.5 |
| **F1@0.95** | F1 score at IoU threshold 0.95 |
| **F1@mIOU** | Mean F1 score across IoU thresholds [0.5, 0.55, ..., 0.95] |
| **Precision** | Precision at IoU threshold 0.5 |
| **Recall** | Recall at IoU threshold 0.5 |

### Output

The evaluation produces:
- `f1`: F1 score at IoU=0.5 (0-100%)
- `f1_0.95`: F1 score at IoU=0.95 (0-100%)
- `f1_mIOU`: Mean F1 score across IoU thresholds (0-100%)
- `precision`: Precision at IoU=0.5 (0-100%)
- `recall`: Recall at IoU=0.5 (0-100%)
- `valid_bbox_predictions`: Number of predictions with valid bounding boxes
- `total_gt_objects`: Total ground truth objects
- `total_pred_objects`: Total predicted objects
- `true_positives`: Number of true positive detections
- `false_positives`: Number of false positive detections
- `false_negatives`: Number of missed ground truth objects

## Example

### Ground Truth (labels_hres/000001.txt)
```
person 0.0 0 0.0 100 150 300 350 1.5 1.8 4.2 0.0 0.0 0.0 0.0
person 0.0 0 0.0 400 200 600 400 2.5 2.5 6.0 0.0 0.0 0.0 0.0
```

### Expected Model Output
```json
[
    {"bbox_2d": [78, 117, 234, 273], "label": "person"},
    {"bbox_2d": [312, 156, 468, 312], "label": "person"}
]
```

### Expected Evaluation Output
```
[2025-12-22 15:14:53] INFO - Astro2D - astro_2d_dataset.py: evaluate - Precision@IoU=0.5: 83.33%
[2025-12-22 15:14:53] INFO - Astro2D - astro_2d_dataset.py: evaluate - Recall@IoU=0.5: 62.50%
[2025-12-22 15:14:53] INFO - Astro2D - astro_2d_dataset.py: evaluate - F1@IoU=0.5: 71.43%
[2025-12-22 15:14:53] INFO - Astro2D - astro_2d_dataset.py: evaluate - F1@IoU=0.95: 45.00%
[2025-12-22 15:14:53] INFO - Astro2D - astro_2d_dataset.py: evaluate - F1@mIOU (0.5:0.05:0.95): 57.25%
[2025-12-22 15:14:53] INFO - Astro2D - astro_2d_dataset.py: evaluate - TP: 15, FP: 3, FN: 9
[2025-12-22 15:14:53] INFO - Astro2D - astro_2d_dataset.py: evaluate - Valid predictions: 2/2 (100.00%)
[2025-12-22 15:14:53] INFO - RUN - run.py: main - The evaluation of model qwen3_8b_local x dataset Astro2D has finished!
[2025-12-22 15:14:53] INFO - RUN - run.py: main - Evaluation Results:
{
    "precision": 83.33,
    "recall": 62.5,
    "f1": 71.43,
    "f1_0.95": 45.0,
    "f1_mIOU": 57.25,
    "total_predictions": 2,
    "valid_bbox_predictions": 2,
    "valid_rate": 1.0,
    "total_gt_objects": 24,
    "total_pred_objects": 18,
    "true_positives": 15,
    "false_positives": 3,
    "false_negatives": 9
}
```

## File Structure

```
metropolis2d/
├── __init__.py
├── astro_2d_dataset.py       # Main dataset implementation
├── utils.py                  # Utility functions
├── datasets.yaml             # Dataset configuration
└── README_detection.md       # This file
```

## Dependencies

- numpy
- pandas
- Pillow (PIL)
- s3fs (for S3 data access)

## Notes

1. Images should be in standard formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
2. Label files should have the same base name as corresponding images
3. EXIF orientation is automatically handled for images
4. S3 paths are supported - data will be cached locally in `LMUDataRoot()/metropolis2d_astro/`
5. Small bounding boxes can be filtered using `MIN_BBOX_AREA` threshold (default: 0)
