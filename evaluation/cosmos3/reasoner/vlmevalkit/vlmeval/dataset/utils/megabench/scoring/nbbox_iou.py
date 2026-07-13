import ast
import logging

import numpy as np

from .common.conversions import str_to_bboxes
from .common.metrics import calculate_iou


class NbboxIouTuple:
    """Calculates the IoU, for all bounding boxes, for all predicted bounding boxes.
    For each predicted bounding box, it uses the IoU with the target bounding box with
    the highest IoU.

    Assumes that co-ordinates are normalized between 0 and 1 and that the bounding boxes
    are of the form (x1, y1, x2, y2), where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right.
    """

    @classmethod
    def match(cls, responses, targets) -> float:
        """Exact match between targets and responses."""
        logging.debug(f"{responses=}, {targets=}")
        if not isinstance(responses, (tuple | list)):
            responses = str_to_bboxes(responses)
        if not isinstance(targets, (tuple | list)):
            targets = str_to_bboxes(targets)

        try:
            iou_scores = calculate_iou(responses, targets)
        except:
            return 0

        if not iou_scores:
            return 0

        # Take the mean IoU score for now.
        return sum(iou_scores) / len(iou_scores)


class NbboxIouSingle:
    """
    Single bbox IoU metric
    """

    @classmethod
    def match(cls, responses, targets) -> float:
        """Exact match between targets and responses."""
        logging.debug(f"{responses=}, {targets=}")
        targets = ast.literal_eval(targets)
        try:
            responses = ast.literal_eval(responses)
        except SyntaxError:
            return 0

        try:
            iou_scores = calculate_iou(
                [
                    responses,
                ],
                [
                    targets,
                ],
            )
            if not iou_scores:
                return 0
        except:
            return 0

        # Take the mean IoU score for now.
        return sum(iou_scores) / len(iou_scores)


class NbboxIouSequence:
    """
    Metric for a sequence of bboxes (used for single object tracking).
    The number of predicted boxes must match the ground truth.
    """

    @classmethod
    def match(cls, responses, targets) -> float:
        """Exact match between targets and responses."""
        if not isinstance(responses, (tuple | list)):
            responses = str(responses) if not isinstance(responses, str) else responses
            responses = str_to_bboxes(responses)
        if not isinstance(targets, (tuple | list)):
            targets = str_to_bboxes(targets)

        if len(targets) != len(responses):
            return 0

        scores = []
        for res, tgt in zip(responses, targets):
            scores.append(
                calculate_iou(
                    [
                        res,
                    ],
                    [
                        tgt,
                    ],
                )
            )
        avg_iou = np.mean(scores)

        return avg_iou
