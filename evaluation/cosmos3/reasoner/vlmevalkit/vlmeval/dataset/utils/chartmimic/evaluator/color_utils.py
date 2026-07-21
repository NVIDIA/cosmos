import numpy as np

# This is a patch for color map, which is not updated for newer version of
# numpy


def patch_asscalar(a):
    return a.item()


setattr(np, "asscalar", patch_asscalar)


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_lab(rgb):
    """
    Convert an RGB color to Lab color space.
    RGB values should be in the range [0, 255].
    """
    # Create an sRGBColor object from RGB values
    from colormath.color_conversions import convert_color
    from colormath.color_objects import LabColor, sRGBColor
    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)

    # Convert to Lab color space
    lab_color = convert_color(rgb_color, LabColor)

    return lab_color


def calculate_similarity_single(c1, c2):
    if c1.startswith("#") and c2.startswith("#"):
        # c1 = rgb2lab(np.array([hex_to_rgb(c1)]))
        # c2 = rgb2lab(np.array([hex_to_rgb(c2)]))
        c1 = hex_to_rgb(c1)
        c2 = hex_to_rgb(c2)
        lab1 = rgb_to_lab(c1)
        lab2 = rgb_to_lab(c2)
        # return max(0, 1 - deltaE_cie76(c1, c2)[0] / 100)
        from colormath.color_diff import delta_e_cie2000
        return max(0, 1 - (delta_e_cie2000(lab1, lab2) / 100))
    elif not c1.startswith("#") and not c2.startswith("#"):

        return 1 if c1 == c2 else 0
    else:
        return 0


def filter_color(color_list):
    filtered_color_list = []
    len_color_list = len(color_list)
    for i in range(len_color_list):
        if i != 0:
            put_in = True
            for item in filtered_color_list:
                similarity = calculate_similarity_single(
                    color_list[i].split("--")[1], item.split("--")[1])
                if similarity > 0.7:
                    put_in = False
                    break
            if put_in:
                filtered_color_list.append(color_list[i])
        else:
            filtered_color_list.append(color_list[i])
    # print("Filtered color list: ", filtered_color_list)
    return filtered_color_list


def group_color(color_list):
    color_dict = {}

    for color in color_list:
        chart_type = color.split("--")[0]
        color = color.split("--")[1]

        if chart_type not in color_dict:
            color_dict[chart_type] = [color]
        else:
            color_dict[chart_type].append(color)

    return color_dict
