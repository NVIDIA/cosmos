import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cairosvg


@dataclass
class InputData:
    svg_path: str
    output_dir: str
    width: int
    height: int


def is_valid_svg(path: str) -> bool:
    try:
        ET.parse(path)
        return True
    except ET.ParseError:
        return False


def raster_svg(input_data: InputData):
    try:
        output_path = os.path.join(input_data.output_dir,
                                   os.path.basename(input_data.svg_path).replace('.svg', '.png'))
        if not is_valid_svg(input_data.svg_path):
            print(f"Invalid SVG file: {input_data.svg_path}")
            return
        cairosvg.svg2png(url=input_data.svg_path, write_to=output_path,
                         background_color='white', output_width=input_data.width, output_height=input_data.height)
    except Exception as e:
        print(f"Error rastering {input_data.svg_path}: {e}")
