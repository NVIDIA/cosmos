# flake8: noqa
import inspect
import warnings

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx.drawing.nx_pylab as nx_pylab
import squarify
from matplotlib.axes._axes import Axes
from matplotlib.axes._base import _process_plot_var_args
from matplotlib.image import NonUniformImage
from matplotlib.patches import Ellipse
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore", category=UserWarning)

# sys.path.insert(0, f'{os.environ["PROJECT_PATH"]}')


drawed_colors = []
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
        elif isinstance(color, (list, tuple)) and len(color) == 4:
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

            for item in result.get_edgecolors().tolist():
                color = convert_color_to_hex(item)
                drawed_colors.append(func_name + "--" + color)

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
                    drawed_colors.append(func_name + "--" + kwargs["cmap"])
                else:
                    for item in result.get_facecolors().tolist():
                        color = convert_color_to_hex(item)
                        drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "plot":
                for line in result:
                    color = convert_color_to_hex(line.get_color())
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "plot_surface":
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    drawed_colors.append(func_name + "--" + kwargs["cmap"])
                else:
                    colors = result.get_facecolors().tolist()
                    drawed_colors.append(
                        func_name +
                        "--" +
                        convert_color_to_hex(
                            colors[0]))
            elif func.__name__ == "bar3d":
                colors = result.get_facecolors().tolist()
                drawed_colors.append(
                    func_name +
                    "--" +
                    convert_color_to_hex(
                        colors[0]))
            elif func.__name__ == "bar":
                for item in result:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "add_collection3d":
                colors = result.get_facecolors().tolist()
                for color in colors:
                    drawed_colors.append(
                        func_name + "--" + convert_color_to_hex(color))

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
            elif func.__name__ == "axhline":
                color = convert_color_to_hex(result.get_color())
                drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "axvline":
                color = convert_color_to_hex(result.get_color())
                drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "_fill_between_x_or_y":
                color = convert_color_to_hex(list(result.get_facecolors()[0]))
                drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "bar":
                for item in result:
                    color = convert_color_to_hex(
                        list(item._original_facecolor))
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "scatter" and not isinstance(args[0], PolarAxes):
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    drawed_colors.append(func_name + "--" + kwargs["cmap"])
                else:
                    color = convert_color_to_hex(
                        list(result.get_facecolor()[0]))
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "pie":
                for item in result[0]:
                    color = convert_color_to_hex(item.get_facecolor())
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "axvspan":
                color = convert_color_to_hex(result.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "axhspan":
                color = convert_color_to_hex(result.get_facecolor())
                drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "hlines":
                for item in result.get_edgecolors():
                    color = convert_color_to_hex(list(item))
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "vlines":
                for item in result.get_edgecolors():
                    color = convert_color_to_hex(list(item))
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "boxplot":
                for item in result["boxes"]:
                    if isinstance(item, matplotlib.patches.PathPatch):
                        color = convert_color_to_hex(
                            list(item.get_facecolor()))
                        drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "violinplot":
                for item in result["bodies"]:
                    color = convert_color_to_hex(list(item.get_facecolor()[0]))
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "hist":
                tops, bins, patches = result
                if not isinstance(patches, matplotlib.cbook.silent_list):
                    for item in patches:
                        color = convert_color_to_hex(
                            list(item.get_facecolor()))
                        drawed_colors.append(func_name + "--" + color)
                else:
                    for container in patches:
                        for item in container:
                            color = convert_color_to_hex(
                                list(item.get_facecolor()))
                            drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "quiver":
                for item in result.get_facecolors().tolist():
                    color = convert_color_to_hex(item)
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "plot" and len(args) > 0 and isinstance(args[0], PolarAxes):
                lines = result
                for line in lines:
                    color = convert_color_to_hex(line.get_color())
                    drawed_colors.append(func_name + "_polar" + "--" + color)
            elif func.__name__ == "scatter" and isinstance(args[0], PolarAxes):
                # check whether cmap is used
                if "cmap" in kwargs and kwargs["cmap"] is not None:
                    print("cmap is used", kwargs["cmap"])
                    drawed_colors.append(func_name + "--" + kwargs["cmap"])
                else:
                    color = convert_color_to_hex(
                        list(result.get_facecolor()[0]))
                    drawed_colors.append(func_name + "_polar" + "--" + color)
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
            elif func.__name__ == "imshow":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
            elif func.__name__ == "pcolor":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
            elif func.__name__ == "contour":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
            elif func.__name__ == "contourf":
                colormap = result.get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
            elif func.__name__ == "fill":
                patches = result
                for patch in patches:
                    color = convert_color_to_hex(list(patch.get_facecolor()))
                    drawed_colors.append(func_name + "--" + color)
            elif func.__name__ == "__init__" and isinstance(args[0], NonUniformImage):
                colormap = args[0].get_cmap().name
                drawed_colors.append(func_name + "--" + colormap)
            elif func.__name__ == "broken_barh":
                colors = result.get_facecolors().tolist()
                for color in colors:
                    drawed_colors.append(
                        func_name + "--" + convert_color_to_hex(color))
            elif func.__name__ == "__init__" and isinstance(args[0], Ellipse):
                color = convert_color_to_hex(args[0].get_facecolor())
                drawed_colors.append(func_name + "--" + color)

            in_decorator = False
        else:
            return func(*args, **kwargs)
        return result
    wrapper.__name__ = func.__name__
    return wrapper


_process_plot_var_args._makeline = log_function(
    _process_plot_var_args._makeline)
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
Axes.plot = log_function(Axes.plot)
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


squarify.plot = log_function(squarify.plot)

Axes3D.scatter = log_function_for_3d(Axes3D.scatter)
Axes3D.plot = log_function_for_3d(Axes3D.plot)
Axes3D.plot_surface = log_function_for_3d(Axes3D.plot_surface)
Axes3D.bar3d = log_function_for_3d(Axes3D.bar3d)
Axes3D.bar = log_function_for_3d(Axes3D.bar)
Axes3D.add_collection3d = log_function_for_3d(Axes3D.add_collection3d)

# barh test
# draw a simple barh plot
# fig, ax = plt.subplots()
# ax.barh(np.arange(5), np.random.rand(5))
# ax.barh(np.arange(5), np.random.rand(5))
# plt.show()

# axhline test
# fig, ax = plt.subplots()
# ax.axhline(0.5)
# ax.axhline(0.8)
# plt.show()

# axvline test
# fig, ax = plt.subplots()
# ax.axvline(0.5)
# ax.axvline(0.8)
# plt.show()

# errorbar test
# fig, ax = plt.subplots()
# x = np.arange(10)
# y = np.sin(x)
#
# ax.errorbar(x, y, yerr=0.1)
# ax.errorbar(x, y, yerr=0.2)
# plt.show()

# squarify test
# fig, ax = plt.subplots()
# sizes = [50, 25, 25]
# squarify.plot(sizes=sizes, ax=ax)
# plt.savefig("tmp.png")
# plt.show()

# loglog test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y = x**2
# ax.loglog(x, y)
# plt.show()

# fill_between test
# fig, ax = plt.subplots()
# x = np.arange(10)
# y1 = np.sin(x)
# y2 = np.cos(x)
# ax.fill_between(x, y1, y2, cmap='viridis')
# plt.show()

# fill_betweenx test
# fig, ax = plt.subplots()
# x = np.arange(10)
# y1 = np.sin(x)
# y2 = np.cos(x)
# ax.fill_betweenx(x, y1, y2, cmap='viridis')
# plt.show()

# pie test
# fig, ax = plt.subplots()
# sizes = [50, 25, 25]
# ax.pie(sizes)
# plt.savefig("tmp.png")
# plt.show()

# axvspan test
# fig, ax = plt.subplots()
# ax.axvspan(0.2, 0.3, color='red', alpha=0.5)
# ax.axvspan(0.5, 0.7, color='blue', alpha=0.5)
# plt.show()

# axhspan test
# fig, ax = plt.subplots()
# ax.axhspan(0.2, 0.3, color='red', alpha=0.5)
# ax.axhspan(0.5, 0.7, color='blue', alpha=0.5)
# plt.show()


# hlines test
# fig, ax = plt.subplots()
# y_values = [1, 2, 3, 4, 5]
# xmin = 0
# xmax = 10
# ax.hlines(y=y_values, xmin=xmin, xmax=xmax, linestyles='dashed')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# plt.savefig("tmp.png")
# plt.show()

# vlines test
# fig, ax = plt.subplots()
# x_values = [1, 2, 3, 4, 5]
# ymin = 0
# ymax = 10
# ax.vlines(x=x_values, ymin=ymin, ymax=ymax, linestyles='dashed')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# plt.savefig("tmp.png")
# plt.show()

# boxplot test
# fig, ax = plt.subplots()
# data = np.random.rand(10, 3)
# ax.boxplot(data, patch_artist=True)
# plt.savefig("tmp.png")
# plt.show()

# violin test
# fig, ax = plt.subplots()
# data = np.random.rand(10, 3)
# ax.violinplot(data)
# plt.savefig("tmp.png")
# plt.show()

# hist test
# fig, ax = plt.subplots()
# data = np.random.rand(100, 1)
# ax.hist(data, bins=10)
# plt.savefig("tmp.png")
# plt.show()


# networkx test
# fig, ax = plt.subplots()
# G = networkx.complete_graph(5)
# draw the graph, give each node a different color, and a label. make the edges red and blue, with labels
# networkx.draw(G, ax=ax, node_color='r', edge_color='b', labels={0: '0', 1: '1', 2: '2', 3: '3', 4: '4'})
# plt.savefig("tmp.png")
# plt.show()

# quiver test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 10)
# y = np.linspace(0, 10, 10)
# u = np.zeros(10)
# v = np.ones(10)
# # draw the quiver plot, with color red
# ax.quiver(x, y, u, v, color='r')
# plt.savefig("tmp.png")
# plt.show()

# 3d scatter test
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# x = np.random.rand(10)
# y = np.random.rand(10)
# z = np.random.rand(10)
# draw the scatter plot, with color red
# ax.scatter3D(x, y, z, c='#ff2395')
# plt.savefig("tmp.png")
# plt.show()

# 3d plot test
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# draw two lines in 3d, with color red and blue
# ax.plot([0, 1], [0, 1], [0, 1], color='r')
# ax.plot([0, 1], [0, 1], [1, 0], color='b')

# 3d plot_surface test
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# draw a surface plot, with a beautiful colormap
# X = np.linspace(-5, 5, 100)
# Y = np.linspace(-5, 5, 100)
# X, Y = np.meshgrid(X, Y)
# Z = np.sin(np.sqrt(X**2 + Y**2))
# ax.plot_surface(X, Y, Z, cmap='viridis')
# plt.savefig("tmp.png")
# plt.show()

# 3d bar test
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# x = np.arange(10)
# y = np.random.rand(10)
# z = np.zeros(10)
# dx = np.ones(10)
# dy = np.ones(10)
# dz = np.random.rand(10)
# # draw the 3d bar plot, with color red
# ax.bar3d(x, y, z, dx, dy, dz)
# plt.savefig("tmp.png")
# plt.show()

# # bar2d in axes3d test
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# x = np.arange(10)
# y = np.random.rand(10)
# z = np.zeros(10)
# dx = np.ones(10)
# dy = np.ones(10)
# dz = np.random.rand(10)
# # draw the 2d bar plot, with color red
# ax.bar(x, y, z, zdir='y', color=['r', 'b', 'g', 'y', 'm', 'c', 'k', 'w', 'r', 'b'])
# plt.savefig("tmp.png")
# plt.show()


# plot in test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# draw the plot, with color red
# ax.plot(x, y, color='r')
# plt.savefig("tmp.png")
# plt.show()

# matshow in test
# fig, ax = plt.subplots()
# data = np.random.rand(10, 10)
# draw the matshow plot, with a beautiful colormap
# ax.imshow(data, cmap='pink')
# plt.savefig("tmp.png")
# plt.show()

# pcolor in test
# fig, ax = plt.subplots()
# data = np.random.rand(10, 10)
# draw the pcolor plot, with a beautiful colormap
# ax.pcolor(data)
# plt.savefig("tmp.png")
# plt.show()

# # contour in test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y = np.linspace(0, 10, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(X) * np.cos(Y)
# # draw the contour plot, with a beautiful colormap
# ax.contour(X, Y, Z)
# plt.savefig("tmp.png")
# plt.show()

# # contourf in test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y = np.linspace(0, 10, 100)
# X, Y = np.meshgrid(x, y)
# Z = np.sin(X) * np.cos(Y)
# # draw the contourf plot, with a beautiful colormap
# ax.contourf(X, Y, Z, cmap='viridis')
# plt.savefig("tmp.png")
# plt.show()

# stackplot in test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.tan(x)
# draw the stackplot, with beautiful colors
# ax.stackplot(x, y1, y2, y3, colors=['r', 'g', 'b'])
# plt.savefig("tmp.png")
# plt.show()

# fill in test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
# draw the fill plot, with color red
# ax.fill(x, y1, color='r')
# plt.savefig("tmp.png")
# plt.show()


# # NonUniformImage in test
# fig, ax = plt.subplots()
# data = np.random.rand(10, 10)
# x = np.linspace(-4, 4, 9)
# y = np.linspace(-4, 4, 9)
# z = np.sqrt(x[np.newaxis, :] ** 2 + y[:, np.newaxis] ** 2)
# im = NonUniformImage(ax, interpolation='bilinear')
# im.set_data(x, y , z)
# ax.add_image(im)
# plt.savefig("tmp.png")
# plt.show()

# broken_barh in test
# fig, ax = plt.subplots()
# x = np.linspace(0, 10, 100)
# y = np.sin(x)
# draw the broken_barh plot, with color red
# ax.broken_barh([(1, 2), (3, 4)], (0, 1), facecolors='r')
# plt.savefig("tmp.png")
# plt.show()


# Ellipse in test
fig, ax = plt.subplots()
e = matplotlib.patches.Ellipse((0.5, 0.5), 0.4, 0.2, color='r')
ax.add_patch(e)
plt.savefig("tmp.png")
plt.show()


# # radar plot in test
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# theta = np.linspace(0, 2*np.pi, 100)
# r = np.sin(3*theta)**2
# # draw the radar plot, with color red
# ax.plot(theta, r, color='r')
# plt.savefig("tmp.png")
# plt.show()


# import numpy as np; np.random.seed(0)

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D

# # ===================
# # Part 2: Data Preparation
# # ===================
# # Data for PC1 and PC2
# values_pc1 = [0.8, 0.7, 0.6, 0.85, 0.9, 0.75, 0.7, 0.65, 0.8, 0.9]
# values_pc2 = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]
# num_vars = len(values_pc1)

# # Compute angle for each axis
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# # The plot is circular, so we need to "complete the loop" and append the start to the end.
# values_pc1 += values_pc1[:1]
# values_pc2 += values_pc2[:1]
# angles += angles[:1]

# # ===================
# # Part 3: Plot Configuration and Rendering
# # ===================
# # Draw the radar chart
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# ax.fill(angles, values_pc1, color="black", alpha=0.1)
# ax.plot(angles, values_pc1, color="black", linewidth=2, label="Loadings PC1")
# ax.scatter(angles[:-1], values_pc1[:-1], color="black", s=50)
# ax.fill(angles, values_pc2, color="red", alpha=0.1)
# ax.plot(angles, values_pc2, color="red", linewidth=2, label="Loadings PC2")
# ax.scatter(angles[:-1], values_pc2[:-1], color="red", s=50)

# # Add labels to the plot
# ax.set_yticklabels([])
# grid_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
# ax.set_xticks(grid_angles)
# angle_labels = [f"{i*45}°" for i in range(8)]
# ax.set_xticklabels(angle_labels)

# # Add grid lines and labels for the concentric circles
# ax.set_rgrids(
#     [0.2, 0.4, 0.6, 0.8, 1.0],
#     labels=["0.2", "0.4", "0.6", "0.8", "1.0"],
#     angle=30,
#     color="black",
#     size=10,
# )

# # Create legend handles manually
# legend_elements = [
#     Line2D(
#         [0],
#         [0],
#         color="black",
#         linewidth=2,
#         marker="o",
#         markersize=8,
#         label="Loadings PC1",
#     ),
#     Line2D(
#         [0],
#         [0],
#         color="red",
#         linewidth=2,
#         marker="o",
#         markersize=8,
#         label="Loadings PC2",
#     ),
# ]

# # Add legend and title
# ax.legend(
#     handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1.1), frameon=False
# )

# # ===================
# # Part 4: Saving Output
# # ===================
# # Adjust layout and save the plot
# plt.tight_layout()
# plt.savefig('tmp.png')


# poly3d in test
# import math
# import matplotlib.pyplot as plt
# import numpy as np; np.random.seed(0)

# from matplotlib.collections import PolyCollection

# # ===================
# # Part 2: Data Preparation
# # ===================
# # Fixing random state for reproducibility
# def polygon_under_graph(x, y):
#     """
#     Construct the vertex list which defines the polygon filling the space under
#     the (x, y) line graph. This assumes x is in ascending order.
#     """
#     return [(x[0], 0.0), *zip(x, y), (x[-1], 0.0)]


# x = np.linspace(0.0, 10.0, 31)
# vaccination_numbers = range(1, 4)

# # verts[i] is a list of (x, y) pairs defining polygon i.
# gamma = np.vectorize(math.gamma)
# verts = [
#     polygon_under_graph(x, v**x * np.exp(-v) / gamma(x + 1))
#     for v in vaccination_numbers
# ]

# # ===================
# # Part 3: Plot Configuration and Rendering
# # ===================
# ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
# facecolors = plt.colormaps["viridis_r"](np.linspace(0, 1, len(verts)))

# poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
# ax.add_collection3d(poly, zs=vaccination_numbers, zdir="y")

# ax.set(
#     xlim=(0, 10),
#     ylim=(1, 4),
#     zlim=(0, 0.35),
#     xlabel="Age",
#     ylabel="Vaccination Number",
#     zlabel="Incidence Rate",
# )

# ax.set_yticks([1, 2, 3])
# ax.set_box_aspect(aspect=None, zoom=0.8)

# # ===================
# # Part 4: Saving Output
# # ===================
# plt.tight_layout()
# plt.savefig('3d_14.pdf', bbox_inches='tight')


drawed_colors = set(drawed_colors)
print("drawed_colors", drawed_colors)
