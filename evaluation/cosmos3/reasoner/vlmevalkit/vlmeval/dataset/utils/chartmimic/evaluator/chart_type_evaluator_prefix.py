import os
import sys

# sys.path.insert(0, os.environ['PROJECT_PATH'])
if os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"] not in sys.path:
    sys.path.insert(0, os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"])

import inspect
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import networkx.drawing.nx_pylab as nx_pylab
import squarify
from matplotlib.axes import Axes
from matplotlib.axes._base import _process_plot_var_args
from matplotlib.image import NonUniformImage
from matplotlib.patches import Circle, Ellipse
from matplotlib.projections.polar import PolarAxes
from matplotlib.tri._tripcolor import tripcolor
from matplotlib_venn._common import VennDiagram
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# sys.path.insert(0, os.environ['PROJECT_PATH'])

called_functions = {}
in_decorator = False


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

            file_name = inspect.getfile(func) + "/" + func.__name__
            name = file_name + "-" + func.__name__
            called_functions[name] = called_functions.get(name, 0) + 1

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

            file_name = inspect.getfile(func) + "/" + func.__name__
            name = file_name + "-" + func.__name__
            called_functions[name] = called_functions.get(name, 0) + 1

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

            file_name = inspect.getfile(func) + "/" + func.__name__
            name = file_name + "-" + func.__name__
            called_functions[name] = called_functions.get(name, 0) + 1

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


def log_function(func):
    def wrapper(*args, **kwargs):
        global in_decorator
        if not in_decorator:
            in_decorator = True
            if len(args) > 0 and isinstance(
                    args[0], PolarAxes) and func.__name__ == "plot":
                file_name = inspect.getfile(func)
                file_name += "_polar"
            else:
                file_name = inspect.getfile(func)
            name = file_name + "-" + func.__name__
            called_functions[name] = called_functions.get(name, 0) + 1
            result = func(*args, **kwargs)
            in_decorator = False
            return result
        else:
            return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


Axes.bar = log_function(Axes.bar)
Axes.barh = log_function(Axes.barh)   # The same as the bar

# _process_plot_var_args._makeline = log_function(_process_plot_var_args._makeline)
Axes.plot = log_function(Axes.plot)     # Special Case for polar plot
Axes.axhline = log_function(Axes.axhline)
Axes.axvline = log_function(Axes.axvline)
Axes.axvspan = log_function(Axes.axvspan)
Axes.axhspan = log_function(Axes.axhspan)
Axes.hlines = log_function(Axes.hlines)
Axes.vlines = log_function(Axes.vlines)

Axes.errorbar = log_function(Axes.errorbar)   # The same as the line

Axes.boxplot = log_function(Axes.boxplot)

Axes.violinplot = log_function(Axes.violinplot)
Axes.violin = log_function(Axes.violin)

Axes.hist = log_function(Axes.hist)

# Axes._fill_between_x_or_y = log_function(Axes._fill_between_x_or_y)
Axes.fill_between = log_function(Axes.fill_between)
Axes.fill_betweenx = log_function(Axes.fill_betweenx)

Axes.scatter = log_function(Axes.scatter)

nx_pylab.draw_networkx_nodes = log_function_specific_for_draw_networkx_nodes(
    nx_pylab.draw_networkx_nodes)
nx_pylab.draw_networkx_edges = log_function_specific_for_draw_networkx_edges(
    nx_pylab.draw_networkx_edges)
nx_pylab.draw_networkx_labels = log_function_specific_for_draw_networkx_labels(
    nx_pylab.draw_networkx_labels)

# nx_pylab.draw_networkx_nodes = log_function_specific_for_draw_networkx_nodes(nx_pylab.draw_networkx_nodes)
# nx_pylab.draw_networkx_edges = log_function_specific_for_draw_networkx_edges(nx_pylab.draw_networkx_edges)
# nx_pylab.draw_networkx_labels = log_function_specific_for_draw_networkx_labels(nx_pylab.draw_networkx_labels)

nx.draw_networkx_nodes = log_function_specific_for_draw_networkx_nodes(
    nx.draw_networkx_nodes)
nx.draw_networkx_edges = log_function_specific_for_draw_networkx_edges(
    nx.draw_networkx_edges)
nx.draw_networkx_labels = log_function_specific_for_draw_networkx_labels(
    nx.draw_networkx_labels)

Axes.quiver = log_function(Axes.quiver)

Axes3D.scatter = log_function(Axes3D.scatter)
Axes3D.plot = log_function(Axes3D.plot)
Axes3D.plot_surface = log_function(Axes3D.plot_surface)
Axes3D.bar3d = log_function(Axes3D.bar3d)
Axes3D.bar = log_function(Axes3D.bar)
Axes3D.add_collection3d = log_function(Axes3D.add_collection3d)

Axes.pie = log_function(Axes.pie)

Axes.fill = log_function(Axes.fill)

squarify.plot = log_function(squarify.plot)

Axes.imshow = log_function(Axes.imshow)
Axes.pcolor = log_function(Axes.pcolor)
NonUniformImage.__init__ = log_function(NonUniformImage.__init__)

Axes.contour = log_function(Axes.contour)
Axes.contourf = log_function(Axes.contourf)

Ellipse.__init__ = log_function(Ellipse.__init__)
Axes.broken_barh = log_function(Axes.broken_barh)

Axes.tripcolor = log_function(Axes.tripcolor)

VennDiagram.__init__ = log_function(VennDiagram.__init__)

Circle.__init__ = log_function(Circle.__init__)

# Axes.plot = log_function(Axes.plot)
# Axes.loglog = log_function(Axes.loglog)
# Axes.scatter = log_function(Axes.scatter)
# Axes.bar = log_function(Axes.bar)
# Axes.barh = log_function(Axes.barh)
# Axes.axhline = log_function(Axes.axhline)
# Axes.axvline = log_function(Axes.axvline)
# Axes.errorbar = log_function(Axes.errorbar)
# Axes.matshow = log_function(Axes.matshow)
# Axes.hist = log_function(Axes.hist)
# Axes.pie = log_function(Axes.pie)
# Axes.boxplot = log_function(Axes.boxplot)
# Axes.arrow = log_function(Axes.arrow)
# Axes.fill_between = log_function(Axes.fill_between)
# Axes.fill_betweenx = log_function(Axes.fill_betweenx)
# Axes.imshow = log_function(Axes.imshow)
# Axes.contour = log_function(Axes.contour)
# Axes.contourf = log_function(Axes.contourf)
# Axes.violinplot = log_function(Axes.violinplot)
# Axes.violin = log_function(Axes.violin)

# squarify.plot = log_function(squarify.plot)
