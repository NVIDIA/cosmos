# # flake8: noqa
import os
import warnings

import squarify

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys

if os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"] not in sys.path:
    sys.path.insert(0, os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"])

import inspect

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx
import networkx as nx
import networkx.drawing.nx_pylab as nx_pylab
import numpy as np
from evaluator.color_utils import filter_color
from matplotlib.axes._axes import Axes
from matplotlib.axes._base import _process_plot_var_args
from matplotlib.image import NonUniformImage
from matplotlib.patches import Circle, Ellipse
from matplotlib.projections.polar import PolarAxes
from matplotlib_venn._common import VennDiagram
from mpl_toolkits.mplot3d import Axes3D

# from chart2code.utils.evaluator.color_utils import filter_color


drawed_colors = []
drawed_objects = {}
in_decorator = False


def convert_color_to_hex(color):
    'Convert color from name, RGBA, or hex to a hex format.'
    try:
        # First, try to convert from color name to RGBA to hex
        if isinstance(color, str):
            # Check if it's already a hex color (start with '#' and length
            # either 7 or 9)
            if color.startswith('#') and (len(color) == 7 or len(color) == 9):
                return color.upper()
            else:
                return mcolors.to_hex(mcolors.to_rgba(color)).upper()
        # Then, check if it's in RGBA format
        elif isinstance(color, (list, tuple, np.ndarray)) and (len(color) == 4 or len(color) == 3):
            return mcolors.to_hex(color).upper()
        else:
            raise ValueError("Unsupported color format")
    except ValueError as e:
        print(color)
        print("Error converting color:", e)
        return None


def log_function_specific_for_draw_networkx_labels(func):
    def wrapper(
        G,
        pos,
        labels=None,
        font_size=12,
        font_color="k",
        font_family="sans-serif",
        font_weight="normal",
        alpha=None,
        bbox=None,
        horizontalalignment="center",
        verticalalignment="center",
        ax=None,
        clip_on=True,
    ):
        global drawed_colors
        global in_decorator

        if not in_decorator:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(
                G,
                pos,
                labels=labels,
                font_size=font_size,
                font_color=font_color,
                font_family=font_family,
                font_weight=font_weight,
                alpha=alpha,
                bbox=bbox,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                ax=ax,
                clip_on=clip_on
            )

            for item in result.values():
                color = convert_color_to_hex(item.get_color())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = item

            in_decorator = False
        else:
            return func(
                G,
                pos,
                labels=labels,
                font_size=font_size,
                font_color=font_color,
                font_family=font_family,
                font_weight=font_weight,
                alpha=alpha,
                bbox=bbox,
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                ax=ax,
                clip_on=clip_on
            )
        return result
    wrapper.__name__ = func.__name__
    return wrapper


def log_function_specific_for_draw_networkx_edges(func):
    def wrapper(
        G,
        pos,
        edgelist=None,
        width=1.0,
        edge_color="k",
        style="solid",
        alpha=None,
        arrowstyle=None,
        arrowsize=10,
        edge_cmap=None,
        edge_vmin=None,
        edge_vmax=None,
        ax=None,
        arrows=None,
        label=None,
        node_size=300,
        nodelist=None,
        node_shape="o",
        connectionstyle="arc3",
        min_source_margin=0,
        min_target_margin=0,
    ):
        global drawed_colors
        global in_decorator

        if not in_decorator:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(
                G,
                pos,
                edgelist=edgelist,
                width=width,
                edge_color=edge_color,
                style=style,
                alpha=alpha,
                arrowstyle=arrowstyle,
                arrowsize=arrowsize,
                edge_cmap=edge_cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax,
                ax=ax,
                arrows=arrows,
                label=label,
                node_size=node_size,
                nodelist=nodelist,
                node_shape=node_shape,
                connectionstyle=connectionstyle,
                min_source_margin=min_source_margin,
                min_target_margin=min_target_margin
            )

            if isinstance(result, list):
                for line in result:
                    color = convert_color_to_hex(line.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
                if len(result) > 0:
                    drawed_objects[func_name + "--" + color] = result
            else:
                for item in result.get_edgecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "--" + color)
                if len(result.get_edgecolors().tolist()) > 0:
                    drawed_objects[func_name + "--" +
                                   color] = result  # ! Attention

            in_decorator = False
        else:
            return func(
                G,
                pos,
                edgelist=edgelist,
                width=width,
                edge_color=edge_color,
                style=style,
                alpha=alpha,
                arrowstyle=arrowstyle,
                arrowsize=arrowsize,
                edge_cmap=edge_cmap,
                edge_vmin=edge_vmin,
                edge_vmax=edge_vmax,
                ax=ax,
                arrows=arrows,
                label=label,
                node_size=node_size,
                nodelist=nodelist,
                node_shape=node_shape,
                connectionstyle=connectionstyle,
                min_source_margin=min_source_margin,
                min_target_margin=min_target_margin
            )
        return result
    wrapper.__name__ = func.__name__
    return wrapper


def log_function_specific_for_draw_networkx_nodes(func):
    def wrapper(
        G,
        pos,
        nodelist=None,
        node_size=300,
        node_color="#1f78b4",
        node_shape="o",
        alpha=None,
        cmap=None,
        vmin=None,
        vmax=None,
        ax=None,
        linewidths=None,
        edgecolors=None,
        label=None,
        margins=None,
    ):
        global drawed_colors
        global in_decorator

        if not in_decorator:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(
                G,
                pos,
                nodelist=nodelist,
                node_size=node_size,
                node_color=node_color,
                node_shape=node_shape,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                linewidths=linewidths,
                edgecolors=edgecolors,
                label=label,
                margins=margins
            )

            for item in result.get_facecolors().tolist():
                color = convert_color_to_hex(item)
                drawed_colors.append(func_name + "--" + color)
            drawed_objects[func_name + "--" + color] = result

            in_decorator = False
        else:
            return func(
                G,
                pos,
                nodelist=nodelist,
                node_size=node_size,
                node_color=node_color,
                node_shape=node_shape,
                alpha=alpha,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                linewidths=linewidths,
                edgecolors=edgecolors,
                label=label,
                margins=margins
            )
        return result
    wrapper.__name__ = func.__name__
    return wrapper


def log_function_for_3d(func):
    def wrapper(*args, **kwargs):
        global drawed_colors
        global in_decorator

        if not in_decorator:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(*args, **kwargs)

            if func.__name__ == "scatter":
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if isinstance(kwargs["cmap"], str):
                        drawed_colors.append(
                            func_name + "_3d--" + kwargs["cmap"])
                        drawed_objects[func_name + "_3d--" +
                                       kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(
                            func_name + "_3d--" + kwargs["cmap"].name)
                        drawed_objects[func_name + "_3d--" +
                                       kwargs["cmap"].name] = result
                else:
                    for item in result.get_facecolors().tolist():
                        color = convert_color_to_hex(item)
                        drawed_colors.append(func_name + "_3d--" + color)
                    drawed_objects[func_name + "_3d--" +
                                   color] = result  # ! Attention
            elif func.__name__ == "plot":
                for line in result:
                    color = convert_color_to_hex(line.get_color())
                    drawed_colors.append(func_name + "_3d--" + color)
                    drawed_objects[func_name + "_3d--" + color] = line
            elif func.__name__ == "plot_surface":
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if isinstance(kwargs["cmap"], str):
                        drawed_colors.append(
                            func_name + "_3d--" + kwargs["cmap"])
                        drawed_objects[func_name + "_3d--" +
                                       kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(
                            func_name + "_3d--" + kwargs["cmap"].name)  # ! Attention
                        drawed_objects[func_name + "_3d--" +
                                       kwargs["cmap"].name] = result
                else:
                    colors = result.get_facecolors().tolist()
                    drawed_colors.append(
                        func_name +
                        "_3d--" +
                        convert_color_to_hex(
                            colors[0]))
                    # ! Attention
                    drawed_objects[func_name + "_3d--" +
                                   convert_color_to_hex(colors[0])] = result
            elif func.__name__ == "bar3d":
                colors = result.get_facecolors().tolist()
                drawed_colors.append(
                    func_name +
                    "_3d--" +
                    convert_color_to_hex(
                        colors[0]))
                # ! Attention
                drawed_objects[func_name + "_3d--" +
                               convert_color_to_hex(colors[0])] = result
            elif func.__name__ == "bar":
                for item in result:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "_3d--" + color)
                    drawed_objects[func_name + "_3d--" + color] = item
            elif func.__name__ == "add_collection3d":
                colors = result.get_facecolors().tolist()
                for color in colors:
                    drawed_colors.append(
                        func_name + "_3d--" + convert_color_to_hex(color))
                drawed_objects[func_name + "_3d--" +
                               convert_color_to_hex(color)] = result

            in_decorator = False
        else:
            return func(*args, **kwargs)
        return result
    wrapper.__name__ = func.__name__
    return wrapper


def log_function(func):
    def wrapper(*args, **kwargs):
        global drawed_colors
        global in_decorator

        if not in_decorator:
            in_decorator = True

            func_name = inspect.getfile(func) + "/" + func.__name__

            result = func(*args, **kwargs)

            if func.__name__ == "_makeline":
                color = convert_color_to_hex(result[1]["color"])
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result[0]
            elif func.__name__ == "axhline":
                color = convert_color_to_hex(result.get_color())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "axvline":
                color = convert_color_to_hex(result.get_color())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "_fill_between_x_or_y":
                color = convert_color_to_hex(list(result.get_facecolors()[0]))
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "bar":
                for item in result:
                    color = convert_color_to_hex(
                        list(item._original_facecolor))
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" + color] = item
            elif func.__name__ == "scatter" and not isinstance(args[0], PolarAxes):
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if isinstance(kwargs["cmap"], str):
                        drawed_colors.append(func_name + "--" + kwargs["cmap"])
                        drawed_objects[func_name + "--" +
                                       kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(
                            func_name + "--" + kwargs["cmap"].name)  # ! Attention
                        drawed_objects[func_name + "--" +
                                       kwargs["cmap"].name] = result
                else:
                    if len(result.get_facecolor()) != 0:
                        color = convert_color_to_hex(
                            list(result.get_facecolor()[0]))
                        drawed_colors.append(func_name + "--" + color)
                        drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "pie":
                for item in result[0]:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" + color] = item
            elif func.__name__ == "axvspan":
                color = convert_color_to_hex(result.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "axhspan":
                color = convert_color_to_hex(result.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = result
            elif func.__name__ == "hlines":
                for item in result.get_edgecolors():
                    color = convert_color_to_hex(list(item))
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" +
                               color] = result   # ! Attention
            elif func.__name__ == "vlines":
                for item in result.get_edgecolors():
                    color = convert_color_to_hex(list(item))
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" +
                               color] = result   # ! Attention
            elif func.__name__ == "boxplot":
                for item in result["boxes"]:
                    if isinstance(item, matplotlib.patches.PathPatch):
                        color = convert_color_to_hex(
                            list(item.get_facecolor()))
                        drawed_colors.append(func_name + "--" + color)
                        drawed_objects[func_name + "--" +
                                       color] = item  # ! Attention
            elif func.__name__ == "violinplot":
                for item in result["bodies"]:
                    color = convert_color_to_hex(list(item.get_facecolor()[0]))
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" +
                                   color] = item  # ! Attention
            elif func.__name__ == "hist":
                tops, bins, patches = result
                if not isinstance(patches, matplotlib.cbook.silent_list):
                    for item in patches:
                        color = convert_color_to_hex(
                            list(item.get_facecolor()))
                        drawed_colors.append(func_name + "--" + color)
                        drawed_objects[func_name + "--" + color] = item
                else:
                    for container in patches:
                        for item in container:
                            color = convert_color_to_hex(
                                list(item.get_facecolor()))
                            drawed_colors.append(func_name + "--" + color)
                            drawed_objects[func_name + "--" + color] = item
            elif func.__name__ == "quiver":
                for item in result.get_facecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" +
                               color] = result  # ! Attention
            elif func.__name__ == "plot" and len(args) > 0 and isinstance(args[0], PolarAxes):
                lines = result
                for line in lines:
                    color = convert_color_to_hex(line.get_color())
                    # print("color", color)
                    drawed_colors.append(func_name + "_polar" + "--" + color)
                    drawed_objects[func_name + "_polar" + "--" + color] = line
            elif func.__name__ == "scatter" and isinstance(args[0], PolarAxes):
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    if isinstance(kwargs["cmap"], str):
                        drawed_colors.append(
                            func_name + "_polar" + "--" + kwargs["cmap"])
                        drawed_objects[func_name +
                                       "_polar--" + kwargs["cmap"]] = result
                    else:
                        drawed_colors.append(
                            func_name + "_polar" + "--" + kwargs["cmap"].name)
                        drawed_objects[func_name + "_polar" +
                                       "--" + kwargs["cmap"].name] = result
                else:
                    if len(result.get_facecolor()) != 0:
                        color = convert_color_to_hex(
                            list(result.get_facecolor()[0]))
                        drawed_colors.append(
                            func_name + "_polar" + "--" + color)
                        drawed_objects[func_name + "_polar" +
                                       "--" + color] = result  # ! Attention
            elif func.__name__ == "plot" and "squarify" in func_name:
                # get ax
                ax = result
                # get container
                containers = ax.containers
                for container in containers:
                    for item in container:
                        color = convert_color_to_hex(
                            list(item.get_facecolor()))
                        drawed_colors.append(
                            func_name + "_squarify" + "--" + color)
                        drawed_objects[func_name +
                                       "_squarify" + "--" + color] = item
            elif func.__name__ == "imshow":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" +
                               colormap] = result  # ! Attention
            elif func.__name__ == "pcolor":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" +
                               colormap] = result  # ! Attention
            elif func.__name__ == "contour":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" +
                               colormap] = result  # ! Attention
            elif func.__name__ == "contourf":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" +
                               colormap] = result  # ! Attention
            elif func.__name__ == "fill":
                patches = result
                for patch in patches:
                    color = convert_color_to_hex(list(patch.get_facecolor()))
                    drawed_colors.append(func_name + "--" + color)
                    drawed_objects[func_name + "--" + color] = patch
            elif func.__name__ == "__init__" and isinstance(args[0], NonUniformImage):
                colormap = args[0].get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" + colormap] = args[0]
            elif func.__name__ == "broken_barh":
                colors = result.get_facecolors().tolist()
                for color in colors:
                    drawed_colors.append(
                        func_name + "--" + convert_color_to_hex(color))
                drawed_objects[func_name + "--" +
                               convert_color_to_hex(color)] = result
            elif func.__name__ == "__init__" and isinstance(args[0], Ellipse):
                color = convert_color_to_hex(args[0].get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = args[0]
            elif func.__name__ == "tripcolor":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
                drawed_objects[func_name + "--" +
                               colormap] = result  # ! Attention
            elif func.__name__ == "__init__" and isinstance(args[0], VennDiagram):
                for item in args[0].patches:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = args[0]
            elif func.__name__ == "__init__" and isinstance(args[0], Circle):
                color = convert_color_to_hex(args[0].get_facecolor())
                drawed_colors.append(func_name + "--" + color)
                drawed_objects[func_name + "--" + color] = args[0]
            in_decorator = False
        else:
            return func(*args, **kwargs)
        return result

    wrapper.__name__ = func.__name__
    return wrapper


def update_drawed_colors(drawed_obejcts):
    drawed_colors = []
    for name, obj in drawed_objects.items():
        func_name = name.split("--")[0]
        color = name.split("--")[1]

        if "/_makeline" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/axhline" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/axvline" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/_fill_between_x_or_y" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolors()[0]))
            drawed_colors.append(func_name + "--" + color)
        elif "/bar" in func_name and "_3d" not in func_name:
            color = convert_color_to_hex(list(obj._original_facecolor))
            if color is not None:
                drawed_colors.append(func_name + "--" + color)
        elif "/scatter" in func_name and "polar" not in func_name and "3d" not in func_name:
            # check whether cmap is used by checking whether color is hex
            if color.startswith("#") is False:
                drawed_colors.append(func_name + "--" + color)
            else:
                if len(obj.get_facecolor()) != 0:
                    color = convert_color_to_hex(list(obj.get_facecolor()[0]))
                    drawed_colors.append(func_name + "--" + color)
        elif "/pie" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/axvspan" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/axhspan" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/hlines" in func_name:
            for item in obj.get_edgecolors():
                color = convert_color_to_hex(list(item))
                drawed_colors.append(func_name + "--" + color)
        elif "/vlines" in func_name:
            for item in obj.get_edgecolors():
                color = convert_color_to_hex(list(item))
                drawed_colors.append(func_name + "--" + color)
        elif "/boxplot" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/violinplot" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()[0]))
            drawed_colors.append(func_name + "--" + color)
        elif "/hist" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/quiver" in func_name:
            for item in obj.get_facecolors().tolist():
                color = convert_color_to_hex(item)
                drawed_colors.append(func_name + "--" + color)
        elif "/plot" in func_name and "polar" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "_polar--" + color)
        elif "/scatter" in func_name and "polar" in func_name:
            # check whether cmap is used by checking whether color is hex
            if color.startswith("#") is False:
                drawed_colors.append(func_name + "_polar--" + color)
            else:
                if len(obj.get_facecolor()) != 0:
                    color = convert_color_to_hex(list(obj.get_facecolor()[0]))
                    drawed_colors.append(func_name + "_polar--" + color)
        elif "/plot" in func_name and "_squarify" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/imshow" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/pcolor" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/contour" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/contourf" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/fill" in func_name:
            color = convert_color_to_hex(list(obj.get_facecolor()))
            drawed_colors.append(func_name + "--" + color)
        elif "/__init__" in func_name and isinstance(obj, NonUniformImage):
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/broken_barh" in func_name:
            colors = obj.get_facecolors().tolist()
            for color in colors:
                drawed_colors.append(
                    func_name + "--" + convert_color_to_hex(color))
        elif "/__init__" in func_name and isinstance(obj, Ellipse):
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/tripcolor" in func_name:
            colormap = obj.get_cmap().name
            drawed_colors.append(func_name + "--" + colormap)
        elif "/__init__" in func_name and isinstance(obj, VennDiagram):
            for item in obj.patches:
                color = convert_color_to_hex(item.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
        elif "/__init__" in func_name and isinstance(obj, Circle):
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "--" + color)
        elif "/scatter" in func_name and "3d" in func_name:
            # check whether cmap is used by checking whether color is hex
            if color.startswith("#") is False:
                drawed_colors.append(func_name + "_3d--" + color)
            else:
                for item in obj.get_facecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "_3d--" + color)
        elif "/plot" in func_name and "3d" in func_name and "plot_surface" not in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "_3d--" + color)
        elif "/plot_surface" in func_name:
            if color.startswith("#") is False:
                drawed_colors.append(func_name + "_3d--" + color)
            else:
                colors = obj.get_facecolors().tolist()
                drawed_colors.append(
                    func_name +
                    "_3d--" +
                    convert_color_to_hex(
                        colors[0]))
        elif "/bar3d" in func_name:
            colors = obj.get_facecolors().tolist()
            drawed_colors.append(
                func_name +
                "_3d--" +
                convert_color_to_hex(
                    colors[0]))
        elif "/bar" in func_name and "3d" in func_name:
            color = convert_color_to_hex(obj.get_facecolor())
            drawed_colors.append(func_name + "_3d--" + color)
        elif "/add_collection3d" in func_name:
            colors = obj.get_facecolors().tolist()
            for color in colors:
                drawed_colors.append(
                    func_name + "_3d--" + convert_color_to_hex(color))
        elif "/draw_networkx_labels" in func_name:
            color = convert_color_to_hex(obj.get_color())
            drawed_colors.append(func_name + "--" + color)
        elif "/draw_networkx_edges" in func_name:
            if isinstance(obj, list):
                for line in obj:
                    color = convert_color_to_hex(line.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
            else:
                for item in obj.get_edgecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "--" + color)
        elif "/draw_networkx_nodes" in func_name:
            for item in obj.get_facecolors().tolist():
                color = convert_color_to_hex(item)
                drawed_colors.append(func_name + "--" + color)

    drawed_colors = list(set(drawed_colors))

    return drawed_colors

try:
    _process_plot_var_args._makeline = log_function(
        _process_plot_var_args._makeline)
except:
    _process_plot_var_args._make_line = log_function(
        _process_plot_var_args._make_line)
Axes.bar = log_function(Axes.bar)
Axes.scatter = log_function(Axes.scatter)
Axes.axhline = log_function(Axes.axhline)
Axes.axvline = log_function(Axes.axvline)
Axes._fill_between_x_or_y = log_function(Axes._fill_between_x_or_y)
Axes.pie = log_function(Axes.pie)
Axes.axvspan = log_function(Axes.axvspan)
Axes.axhspan = log_function(Axes.axhspan)
Axes.hlines = log_function(Axes.hlines)
Axes.vlines = log_function(Axes.vlines)
Axes.boxplot = log_function(Axes.boxplot)
Axes.violinplot = log_function(Axes.violinplot)
Axes.hist = log_function(Axes.hist)
# Axes.plot = log_function(Axes.plot)
PolarAxes.plot = log_function(PolarAxes.plot)
Axes.quiver = log_function(Axes.quiver)
Axes.imshow = log_function(Axes.imshow)
Axes.pcolor = log_function(Axes.pcolor)
Axes.contour = log_function(Axes.contour)
Axes.contourf = log_function(Axes.contourf)
Axes.fill = log_function(Axes.fill)
NonUniformImage.__init__ = log_function(NonUniformImage.__init__)
Ellipse.__init__ = log_function(Ellipse.__init__)
Axes.broken_barh = log_function(Axes.broken_barh)

nx_pylab.draw_networkx_nodes = log_function_specific_for_draw_networkx_nodes(
    nx_pylab.draw_networkx_nodes)
nx_pylab.draw_networkx_edges = log_function_specific_for_draw_networkx_edges(
    nx_pylab.draw_networkx_edges)
nx_pylab.draw_networkx_labels = log_function_specific_for_draw_networkx_labels(
    nx_pylab.draw_networkx_labels)

nx.draw_networkx_nodes = log_function_specific_for_draw_networkx_nodes(
    nx.draw_networkx_nodes)
nx.draw_networkx_edges = log_function_specific_for_draw_networkx_edges(
    nx.draw_networkx_edges)
nx.draw_networkx_labels = log_function_specific_for_draw_networkx_labels(
    nx.draw_networkx_labels)


squarify.plot = log_function(squarify.plot)

Axes3D.scatter = log_function_for_3d(Axes3D.scatter)
Axes3D.plot = log_function_for_3d(Axes3D.plot)
Axes3D.plot_surface = log_function_for_3d(Axes3D.plot_surface)
Axes3D.bar3d = log_function_for_3d(Axes3D.bar3d)
Axes3D.bar = log_function_for_3d(Axes3D.bar)
Axes3D.add_collection3d = log_function_for_3d(Axes3D.add_collection3d)

Axes.tripcolor = log_function(Axes.tripcolor)

VennDiagram.__init__ = log_function(VennDiagram.__init__)

Circle.__init__ = log_function(Circle.__init__)
