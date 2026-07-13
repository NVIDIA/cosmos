"""Return the normalized point distance."""

from .common.conversions import parse_point_2d_from_xml
from .common.metrics import point_distance


class XmlNormPointDistance:
    """Determines the distance between two points in XML notation.

    Assumes that co-ordinates are normalized between 0 and 1 and that the 2D point is
    of the form <point>x, y</point>.
    """

    @classmethod
    def parse_2d_point(cls, point) -> tuple[float, float]:
        """Parse a 2D point encoded in XML as <point>x, y</point>."""
        if not isinstance(point, (tuple | list)):
            point = parse_point_2d_from_xml(point)
            if not point:
                raise ValueError("Point could not be parsed from XML string.")
        elif len(point) != 2:
            raise ValueError("Point is not 2D.")
        if not all(0 <= comp <= 1 for comp in point):
            raise ValueError("Point is not normalized.")
        return tuple(point)

    @classmethod
    def match(cls, responses, targets) -> float:
        """Determine the normalized distance between two points."""
        try:
            responses = cls.parse_2d_point(responses)
            targets = cls.parse_2d_point(targets)
        except ValueError:
            return 0

        # Instead of normalizing by 1/sqrt(2), we just set it to 0 if the distance is above 1.
        return max(0, 1 - point_distance(responses, targets))
