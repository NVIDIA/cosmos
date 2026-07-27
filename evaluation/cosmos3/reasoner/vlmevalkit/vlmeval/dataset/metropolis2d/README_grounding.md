# Metropolis2D Grounding Dataset

A 2D visual grounding dataset for evaluating Vision-Language Models (VLMs) on referring expression comprehension tasks.

## Overview

This dataset evaluates VLMs on their ability to localize objects based on natural language referring expressions. Given an image and a text description (e.g., "the red car on the left"), the model must output the bounding box of the referred object.

**Note**: One expression can be associated with one or multiple bounding boxes. When multiple GT boxes exist for an expression (e.g., "all the red cars"), a prediction is considered correct if it matches ANY of the GT boxes.

## Supported Annotation Formats

Two annotation formats are supported:
1. **JSONL format** - One JSON object per line, for object detection and grounding
2. **RefCOCO format** (JSON) - Standard referring expression format

## Data Format

### Directory Structure

```
data_root/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations.json  (or annotations.jsonl)
```

### Format 1: JSONL Format (Recommended)

```jsonl
{"image_path": "visdrone/0000189_00297_d_0000198.jpg", "gt": {"The white vans parked on the left side of the road.": [[214.0, 216.0, 250.0, 260.0]]}, "categories": ["The white vans parked on the left side of the road."], "dataset_name": "RefDrone_test", "task_name": "referring_object_detection"}
{"image_path": "visdrone/9999952_00000_d_0000047.jpg", "gt": {"car": [[1243.0, 408.0, 1307.0, 441.0], [1165.0, 403.0, 1228.0, 428.0], [944.0, 403.0, 1001.0, 430.0], [1333.0, 54.0, 1358.0, 69.0]], "truck": [[509.0, 652.0, 555.0, 732.0], [397.0, 632.0, 499.0, 782.0], [107.0, 409.0, 171.0, 476.0], [383.0, 8.0, 416.0, 40.0], [262.0, 346.0, 350.0, 387.0]], "bus": [[263.0, 421.0, 306.0, 464.0], [413.0, 346.0, 470.0, 369.0], [599.0, 354.0, 654.0, 382.0], [868.0, 399.0, 929.0, 428.0]]}, "categories": ["car", "truck", "bus", "van"], "dataset_name": "VisDrone", "task_name": "common_object_detection"}
```

**Fields:**
- `image_path`: Relative path to image (resolved from data_root)
- `gt`: Dictionary mapping category names or referring expressions to lists of bboxes in xyxy format (top left and bottom right coordinates in pixels bounded by the image size)
- `categories`: List of categories to use as grounding queries
- `dataset_name`: (optional) Source dataset name
- `task_name`: (optional) Task identifier

**Note:** Each category in `categories` creates a separate grounding sample with that category name as the referring expression.

### Format 2: RefCOCO Format (JSON)

#### COCO-style Format

```json
{
    "images": [
        {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "bbox": [100, 150, 200, 100],
            "sentence": "the red car on the left",
            "category": "car"
        },
        {
            "id": 2,
            "image_id": 1,
            "bboxes": [[100, 150, 200, 100], [400, 200, 150, 80]],
            "sentence": "all the red cars",
            "category": "car"
        },
        {
            "id": 3,
            "image_id": 1,
            "bbox": [400, 200, 150, 80],
            "sentences": [
                {"raw": "person walking"},
                {"raw": "the man in blue"}
            ]
        }
    ]
}
```

**Notes:**
- `bbox`: Single bounding box in `[x, y, width, height]` format (top left corner of the bounding box and its width and height in pixels)
- `bboxes`: Multiple bounding boxes as list of `[x, y, w, h]` (top left corner of the bounding box and its width and height in pixels)
- `sentence` or `sentences`: Referring expression(s)
- Bboxes are auto-converted from xywh to xyxy format

## Usage

### Running Evaluation

```bash
# if using s3:// path, remember to set LMUData
python run.py \
    --data Metropolis2DGrounding \
    --model <your_model> \
    --work-dir ./output \
    --verbose
```

### Supported Dataset Names

- `Metropolis2DGrounding`
- `Metropolis2DGrounding_val`
- `Metropolis2DGround`

### Configuration

Configure via `datasets.yaml`:

```yaml
Metropolis2DGrounding:
  data_root: /path/to/grounding/data
  annotation_file: annotations.jsonl
  task: grounding
```

Paths can be:
- Local filesystem paths
- S3 paths (e.g., `s3://bucket/path`) - data will be automatically downloaded to cache

## Prompt

The VLM is prompted with:

> Locate "{description}" in the provided image and output its bbox coordinates using JSON format. Output format: {"bbox_2d": [x1, y1, x2, y2]} where coordinates are normalized to 0-1000 scale.

For example:
> Locate "the red car on the left" in the provided image...

For JSONL format, the category name becomes the description:
> Locate "car" in the provided image...

## Expected Output Format

The model should output a JSON object with the bounding box:

```json
{"bbox_2d": [156, 234, 468, 390]}
```

**Note**: Coordinates should be normalized to 0-1000 scale.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Acc@0.5** | Accuracy at IoU threshold 0.5 (standard RefCOCO metric) |
| **Acc@0.25** | Accuracy at IoU threshold 0.25 |
| **Acc@0.75** | Accuracy at IoU threshold 0.75 |
| **Mean_IoU** | Average IoU across all samples |

### Handling Multiple GT Boxes

When an expression is associated with multiple ground truth bounding boxes:
- IoU is computed between the predicted box and **each** GT box
- The **maximum IoU** is used for evaluation
- A prediction is **correct** if max_IoU ≥ threshold

This means the model only needs to correctly localize ONE of the valid objects to be counted as correct.

### Output

The evaluation produces:
- `Acc@0.25`: Percentage of predictions with max_IoU ≥ 0.25
- `Acc@0.5`: Percentage of predictions with max_IoU ≥ 0.5
- `Acc@0.75`: Percentage of predictions with max_IoU ≥ 0.75
- `Mean_IoU`: Average of max_IoU across all samples (0-100%)
- `valid_predictions`: Number of predictions with valid bounding boxes
- `total_samples`: Total number of samples
- `valid_rate`: Ratio of valid predictions

## File Structure

```
metropolis2d/
├── __init__.py
├── astro_2d_dataset.py       # Astro2D detection dataset
├── detection_2d_dataset.py   # Legacy detection dataset
├── grounding_2d_dataset.py   # Grounding dataset (this)
├── utils.py                  # Shared utility functions
├── datasets.yaml             # Dataset configuration
├── README_detection.md       # Detection README
└── README_grounding.md       # This file
```

## Dependencies

- numpy
- pandas
- Pillow (PIL)
- pyyaml
- s3fs (for S3 data access)

## Comparison with Detection Dataset

| Aspect | Detection | Grounding |
|--------|-----------|-----------|
| Task | Find all objects of given categories | Find one specific object by description |
| Input | Fixed category list | Natural language description |
| Output | Multiple bounding boxes | Single bounding box |
| Format | KITTI | RefCOCO / JSONL |
| Metrics | F1@IoU | Acc@IoU, Mean IoU |

## Notes

1. Each annotation creates one sample (one referring expression → one or more bounding boxes)
2. For JSONL format, each category creates a separate grounding sample
3. Multiple referring expressions for the same object(s) create multiple samples
4. The model should output exactly one bounding box per query
5. When multiple GT boxes exist, prediction is correct if it matches ANY GT box (max IoU used)
6. EXIF orientation is automatically handled for images
7. Invalid predictions (parsing failures) count as IoU = 0
8. S3 paths are supported - data will be cached locally in `LMUDataRoot()/metropolis2d_grounding/`
9. Annotation format (JSON vs JSONL) is auto-detected
