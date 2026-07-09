import ast
import json
import re
from numbers import Number
from typing import Tuple, Union

from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from vlmeval.dataset.utils.megabench.parsing.common.parsers import parse_json


def freeze_structure(obj):
    """Freeze a structure and make it hashable."""
    if isinstance(obj, dict):
        return frozenset((k, freeze_structure(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return tuple(freeze_structure(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(obj)
    else:
        return obj


def cast_to_set(object) -> set:
    """Try to cast an object as a set."""
    object = freeze_structure(object)
    if isinstance(object, (frozenset, set, tuple)):
        return set(object)
    return str_to_set(object)


def cast_to_dict(object) -> dict:
    """Try to cast an object as a dict."""
    if isinstance(object, dict):
        return {key: cast_to_dict(val) for key, val in object.items()}
    elif isinstance(object, str):
        extract_json_attempt = parse_json(object)
        if extract_json_attempt:
            return extract_json_attempt
        return object
    else:
        return object


def str_to_iterable(func, iterable_str):
    """Converts a string representation of an iterable to an iterable."""
    if not isinstance(iterable_str, str):
        return func()

    iterable_str = iterable_str.strip(" ")
    if not iterable_str:
        return func()

    is_in_iterable = True
    if iterable_str[0] == "(":
        if not iterable_str.endswith(")"):
            return func()
    elif iterable_str[0] == "{":
        if not iterable_str.endswith("}"):
            return func()
    elif iterable_str[0] == "[":
        if not iterable_str.endswith("]"):
            return func()
    else:
        is_in_iterable = False

    # We may have a nested object, so try to use eval first
    try:
        eval_ = ast.literal_eval(iterable_str)
        if eval_ is None:
            return ""
        if isinstance(eval_, (int, float)):
            eval_ = [
                eval_,
            ]
        return func(eval_)
    except (SyntaxError, ValueError):
        if is_in_iterable:
            iterable_str = iterable_str[1:-1]
        items = [item.strip() for item in iterable_str.split(",")]
        return func(items)


def str_to_set(iterable_str) -> set:
    """Converts a string representation of an iterable to a set."""
    return str_to_iterable(set, iterable_str)


def str_to_list(iterable_str) -> set:
    """Converts a string representation of an iterable to a set."""
    return str_to_iterable(list, iterable_str)


def str_to_bboxes(bbox_list) -> list:
    if not isinstance(bbox_list, str):
        return []
    try:
        bboxes = ast.literal_eval(bbox_list)
    except (SyntaxError, ValueError):
        try:
            bboxes = json.loads(bbox_list)
        except json.JSONDecodeError:
            return []

    if len(bboxes) == 4 and isinstance(bboxes[0], Number):
        bboxes = [bboxes]

    if not isinstance(bboxes, (tuple | list)):
        return []

    new_bboxes = []
    for bbox in bboxes:
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            continue
        if any(not isinstance(coord, (float, int)) for coord in bbox):
            continue
        new_bboxes.append(bbox)
    return new_bboxes


def str_to_coords(coord_list, dim=2) -> list:
    if not isinstance(coord_list, str):
        return []
    try:
        coords = ast.literal_eval(coord_list)
    except SyntaxError:
        try:
            coords = json.loads(coord_list)
        except json.JSONDecodeError:
            return []

    new_coords = []
    for coord in coords:
        if not isinstance(coord, (tuple, list)) or len(coord) != dim:
            continue
        if any(not isinstance(coord, (float, int)) for coord in coord):
            continue
        new_coords.append(coord)
    return new_coords


def parse_point_2d_from_xml(xml_string) -> Union[Tuple[float, float], None]:
    """Parse an (x, y) point from XML formatted like this: <point>x, y</point>"""
    if not isinstance(xml_string, str):
        return None

    point_pattern = re.compile(r"<point>(.*?)<\/point>")
    matches = point_pattern.findall(xml_string)
    if len(matches) >= 2:
        return None

    if matches:
        coords = matches[0].split(",")
        if len(coords) != 2:
            return None
        try:
            return tuple(float(coord.strip()) for coord in coords)
        except ValueError:
            return None


def parse_bboxes_from_xml(xml_string: str) -> list:

    if not isinstance(xml_string, str):
        return []

    bbox_pattern = re.compile(r"<box>(.*?)<\/box>")
    matches = bbox_pattern.findall(xml_string)

    new_bboxes = []
    for match in matches:

        coords = match.split(",")
        if len(coords) != 4:
            continue
        try:
            bbox = tuple(float(coord.strip()) for coord in coords)
        except ValueError:
            continue

        if len(bbox) == 4 and all(isinstance(coord, float) for coord in bbox):
            new_bboxes.append(bbox)

    return new_bboxes


MONOSPACE_FONTS = ("Courier New", "DejaVu Sans Mono", "Consolas", "SF Mono")

MONOSPACE_FONT_FILES = []
for font_name in MONOSPACE_FONTS:
    try:
        MONOSPACE_FONT_FILES.append(
            font_manager.findfont(font_name, fallback_to_default=False)
        )
    except ValueError:
        continue


def ascii_text_to_image(
    text,
    width,
    height,
    font_size=20,
    padding=10,
    line_spacing=1,
    bg_color="white",
    text_color="black",
):
    """Convert ASCII text into an image."""
    # Split the text into lines
    lines = text.splitlines()

    # Calculate initial image size based on text
    char_width = font_size * 0.6  # Approximate width of a character
    init_width = int(max(len(line) for line in lines) * char_width + 2 * padding)
    init_height = int(
        (len(lines) * font_size * line_spacing) + 2 * padding
    )  # 1.2 for line spacing

    # Create a new image with the calculated size
    image = Image.new("RGB", (init_width, init_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    # Load a monospace font
    font = None
    for font_name in MONOSPACE_FONT_FILES:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except IOError:
            continue
    if font is None:
        raise ValueError("Cannot properly render ASCII art: missing monospace font.")

    # Draw each line of text
    y_text = padding
    for line in lines:
        draw.text((padding, y_text), line, font=font, fill=text_color)
        y_text += font_size * line_spacing  # Move to the next line

    # Resize the image to the specified dimensions
    image = image.resize((width, height), Image.Resampling.LANCZOS)

    # Convert the image to a NumPy array
    return image
