import math
from collections.abc import Iterable
from numbers import Number


def calculate_iou(predicted: Iterable[Number], target: Iterable[Number]):
    """Calculate the IoU between predicted and target bounding boxes."""

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def box_iou(box1, box2):
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = box_area(box1)
        box2_area = box_area(box2)
        union = box1_area + box2_area - intersection

        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou

    # Calculate IoU for each pair of predicted and target boxes
    iou_scores = []
    for pred_box in predicted:
        best_iou = 0
        for target_box in target:
            iou = box_iou(pred_box, target_box)
            best_iou = max(best_iou, iou)
        iou_scores.append(best_iou)

    return iou_scores


def set_relevance_score(denominator_fn, predicted: Iterable, target: Iterable) -> float:
    """Calculate the relevance score."""
    pred = set(predicted)
    tget = set(target)
    denominator = denominator_fn(pred, tget)
    if not denominator:
        return 1
    return len(pred & tget) / denominator


def _union_denominator(pred: set, tget: set) -> int:
    return len(pred | tget)


def _pred_denominator(pred: set, _: set) -> int:
    return len(pred)


def _tget_denominator(_: set, tget: set) -> int:
    return len(tget)


def jaccard_index(predicted: Iterable, target: Iterable) -> float:
    """Calculate the Jaccard Index."""
    return set_relevance_score(_union_denominator, predicted, target)


def set_precision(predicted: Iterable, target: Iterable) -> float:
    """Calculate the precision, using sets."""
    return set_relevance_score(_pred_denominator, predicted, target)


def set_recall(predicted: Iterable, target: Iterable) -> float:
    """Calculate the recall, using sets."""
    return set_relevance_score(_tget_denominator, predicted, target)


def longest_common_prefix(list1: list, list2: list) -> list:
    """Return the longest common prefix."""
    index_first_difference = next(
        (i for i, (a, b) in enumerate(zip(list1, list2)) if a != b),
        min(len(list1), len(list2)),
    )
    return list1[:index_first_difference]


def mse(predicted: Number, target: Number) -> Number:
    """Return the mean squared error."""
    return (predicted - target) ** 2


def point_distance(predicted: tuple[float, ...], target: tuple[float, ...]):
    """Return the distance between two points."""
    if len(predicted) != len(target):
        raise ValueError(
            "point_distance: Predicted and target points have different dimensions."
        )
    return math.sqrt(
        sum((comp_res - comp_tar) ** 2 for comp_res, comp_tar in zip(predicted, target))
    )
