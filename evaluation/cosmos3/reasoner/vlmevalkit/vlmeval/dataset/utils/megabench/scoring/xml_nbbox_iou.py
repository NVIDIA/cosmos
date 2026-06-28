import logging
from numbers import Number

from .common.conversions import parse_bboxes_from_xml
from .common.metrics import calculate_iou


class XmlNbboxIouSingle:
    """Calculates the IoU of bounding box.

    Assumes that co-ordinates are normalized between 0 and 1 and that the bounding boxes
    are of the form <box>top_left_x, top_left_y, bottom_right_x, bottom_right_y</box>
    """

    @classmethod
    def match(cls, responses, targets) -> float:

        logging.debug(f"{responses=}, {targets=}")
        if not isinstance(responses, (tuple | list)):
            responses = parse_bboxes_from_xml(responses)
        if not isinstance(targets, (tuple | list)):
            targets = parse_bboxes_from_xml(targets)

        if len(responses) == 0:
            return 0
        elif isinstance(responses[0], Number) and len(responses) == 4:
            responses = [responses]

        iou_scores = calculate_iou(responses, targets)
        if not iou_scores:
            return 0

        # Take the mean IoU score for now.
        return sum(iou_scores) / len(iou_scores)
