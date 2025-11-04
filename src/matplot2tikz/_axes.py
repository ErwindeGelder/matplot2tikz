from __future__ import annotations

import re
from collections.abc import Iterable, Sized
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Subplot
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from . import _color
from ._util import _common_texification

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar

    from ._tikzdata import TikzData


class MyAxes:
    def __init__(self, data: TikzData, obj: Axes) -> None:
        """Returns the PGFPlots code for an axis environment."""
        self.data = data
        self.obj = obj
        self.content: list[str] = []

        # Are we dealing with an axis that hosts a colorbar? Skip then, those are
        # treated implicitily by the associated axis.
        self.is_colorbar = _is_colorbar_heuristic(obj)
        if self.is_colorbar:
            return

        # instantiation
        self.nsubplots = 1
        self.subplot_index = 0
        self.is_subplot = False

        if isinstance(obj, Subplot):
            self._subplot()

        data.current_axis_options = set()  # Reset axis options

        self._set_hide_axis()
        self._set_plot_title()
        self._set_axis_titles()
        xlim, ylim = self._set_axis_limits()
        self._set_axis_scaling()
        self._set_axis_on_top()
        self._set_axis_dimensions(self._get_aspect_ratio(), xlim, ylim)
        self._set_axis_positions()
        self._set_ticks()
        self._set_grid()
        self._set_axis_line_styles()
        self._set_background_color()
        self._set_colorbar()
        self._content_end()

    def _set_hide_axis(self) -> None:
        # check if axes need to be displayed at all
        if not self.obj.axison:
            self.data.current_axis_options.add("hide x axis")
            self.data.current_axis_options.add("hide y axis")

    def _set_plot_title(self) -> None:
        title = self.obj.get_title()
        self.data.current_axis_title = title
        if title:
            title = _common_texification(title)
            self.data.current_axis_options.add(f"title={{{title}}}")

    def _set_axis_titles(self) -> None:
        xlabel = self.obj.get_xlabel()
        if xlabel:
            xlabel = _common_texification(xlabel)

            labelcolor = self.obj.xaxis.label.get_color()

            if labelcolor != "black":
                col, _ = _color.mpl_color2xcolor(self.data, labelcolor)
                self.data.current_axis_options.add(f"xlabel=\\textcolor{{{col}}}{{{xlabel}}}")
            else:
                self.data.current_axis_options.add(f"xlabel={{{xlabel}}}")

            xrotation = self.obj.xaxis.get_label().get_rotation()
            if xrotation != 0:
                self.data.current_axis_options.add(f"xlabel style={{rotate={xrotation - 90}}}")

        ylabel = self.obj.get_ylabel()
        if ylabel:
            ylabel = _common_texification(ylabel)

            labelcolor = self.obj.yaxis.label.get_color()
            if labelcolor != "black":
                col, _ = _color.mpl_color2xcolor(self.data, labelcolor)
                self.data.current_axis_options.add(f"ylabel=\\textcolor{{{col}}}{{{ylabel}}}")
            else:
                self.data.current_axis_options.add(f"ylabel={{{ylabel}}}")

            yrotation = self.obj.yaxis.get_label().get_rotation()
            if yrotation != 90:  # noqa: PLR2004
                self.data.current_axis_options.add(f"ylabel style={{rotate={yrotation - 90}}}")

    def _set_axis_limits(self) -> tuple[list[float], list[float]]:
        ff = self.data.float_format
        xlim = list(self.obj.get_xlim())
        xlim0, xlim1 = sorted(xlim)
        ylim = list(self.obj.get_ylim())
        ylim0, ylim1 = sorted(ylim)
        # Sort the limits so make sure that the smaller of the two is actually *min.
        self.data.current_axis_options.add(f"xmin={xlim0:{ff}}, xmax={xlim1:{ff}}")
        self.data.current_axis_options.add(f"ymin={ylim0:{ff}}, ymax={ylim1:{ff}}")
        # When the axis is inverted add additional option
        if xlim != sorted(xlim):
            self.data.current_axis_options.add("x dir=reverse")
        if ylim != sorted(ylim):
            self.data.current_axis_options.add("y dir=reverse")
        return xlim, ylim

    def _set_axis_scaling(self) -> None:
        if self.obj.get_xscale() == "log":
            self.data.current_axis_options.add("xmode=log")
            self.data.current_axis_options.add(
                f"log basis x={{{_try_f2i(self.obj.xaxis._scale.base)}}}"  # type: ignore[attr-defined]  # noqa: SLF001
            )
        if self.obj.get_yscale() == "log":
            self.data.current_axis_options.add("ymode=log")
            self.data.current_axis_options.add(
                f"log basis y={{{_try_f2i(self.obj.yaxis._scale.base)}}}"  # type: ignore[attr-defined]  # noqa: SLF001
            )

    def _set_axis_on_top(self) -> None:
        # Possible values for get_axisbelow():
        #   True (zorder = 0.5):   Ticks and gridlines are below all Artists.
        #   'line' (zorder = 1.5): Ticks and gridlines are above patches (e.g.
        #                          rectangles) but still below lines / markers.
        #   False (zorder = 2.5):  Ticks and gridlines are above patches and lines /
        #                          markers.
        if not self.obj.get_axisbelow():
            self.data.current_axis_options.add("axis on top")

    def _get_aspect_ratio(self) -> float | None:
        aspect = self.obj.get_aspect()
        if aspect in ["auto", "normal"]:
            return None  # just take the given width/height values
        if aspect == "equal":
            return 1.0
        return float(aspect)

    def _set_axis_dimensions(
        self, aspect_num: float | None, xlim: list[float], ylim: list[float]
    ) -> None:
        if self.data.axis_width and self.data.axis_height:
            # width and height overwrite aspect ratio
            self.data.current_axis_options.add("width=" + self.data.axis_width)
            self.data.current_axis_options.add("height=" + self.data.axis_height)
        elif self.data.axis_width:
            # only self.data.axis_width given. calculate height by the aspect ratio
            self.data.current_axis_options.add("width=" + self.data.axis_width)
            if aspect_num:
                alpha = aspect_num * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
                if alpha == 1.0:
                    self.data.axis_height = self.data.axis_width
                else:
                    # Concatenate the literals, as self.data.axis_width could as well
                    # be a LaTeX length variable such as \figurewidth.
                    self.data.axis_height = str(alpha) + "*" + self.data.axis_width
                self.data.current_axis_options.add("height=" + self.data.axis_height)
        elif self.data.axis_height:
            # only self.data.axis_height given. calculate width by the aspect ratio
            self.data.current_axis_options.add("height=" + self.data.axis_height)
            if aspect_num:
                alpha = aspect_num * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
                if alpha == 1.0:
                    self.data.axis_width = self.data.axis_height
                else:
                    # Concatenate the literals, as self.data.axis_height could as
                    # well be a LaTeX length variable such as \figureheight.
                    self.data.axis_width = str(1.0 / alpha) + "*" + self.data.axis_height
                self.data.current_axis_options.add("width=" + self.data.axis_width)

    def _set_axis_positions(self) -> None:
        xaxis_pos = self.obj.get_xaxis().label_position
        if xaxis_pos == "top":
            # By default, x-axis position is "bottom"
            self.data.current_axis_options.add("axis x line=top")

        yaxis_pos = self.obj.get_yaxis().label_position
        if yaxis_pos == "right":
            # By default, y-axis position is "left"
            self.data.current_axis_options.add("axis y line=right")

    def _set_ticks(self) -> None:
        self._get_ticks()
        self._get_tick_colors()
        self._get_tick_direction()
        self._set_tick_rotation()
        self._set_tick_positions()

    def _set_grid(self) -> None:
        # Don't use get_{x,y}gridlines for gridlines; see discussion on
        # <http://sourceforge.net/p/matplotlib/mailman/message/25169234/> Coordinate of
        # the lines are entirely meaningless, but styles (colors,...) are respected.

        try:
            # mpl 3.3.3+
            # <https://github.com/matplotlib/matplotlib/pull/18769>
            has_major_xgrid = self.obj.xaxis._major_tick_kw["gridOn"]  # type: ignore[attr-defined]  # noqa: SLF001
            has_minor_xgrid = self.obj.xaxis._minor_tick_kw["gridOn"]  # type: ignore[attr-defined]  # noqa: SLF001
            has_major_ygrid = self.obj.yaxis._major_tick_kw["gridOn"]  # type: ignore[attr-defined]  # noqa: SLF001
            has_minor_ygrid = self.obj.yaxis._minor_tick_kw["gridOn"]  # type: ignore[attr-defined]  # noqa: SLF001
        except KeyError:
            has_major_xgrid = self.obj.xaxis._gridOnMajor  # type: ignore[attr-defined]  # noqa: SLF001
            has_minor_xgrid = self.obj.xaxis._gridOnMinor  # type: ignore[attr-defined]  # noqa: SLF001
            has_major_ygrid = self.obj.yaxis._gridOnMajor  # type: ignore[attr-defined]  # noqa: SLF001
            has_minor_ygrid = self.obj.yaxis._gridOnMinor  # type: ignore[attr-defined]  # noqa: SLF001

        if has_major_xgrid:
            self.data.current_axis_options.add("xmajorgrids")
        if has_minor_xgrid:
            self.data.current_axis_options.add("xminorgrids")

        xlines = self.obj.get_xgridlines()
        if xlines:
            xgridcolor = xlines[0].get_color()
            col, _ = _color.mpl_color2xcolor(self.data, xgridcolor)
            if col != "black":
                self.data.current_axis_options.add(f"x grid style={{{col}}}")

        if has_major_ygrid:
            self.data.current_axis_options.add("ymajorgrids")
        if has_minor_ygrid:
            self.data.current_axis_options.add("yminorgrids")

        ylines = self.obj.get_ygridlines()
        if ylines:
            ygridcolor = ylines[0].get_color()
            col, _ = _color.mpl_color2xcolor(self.data, ygridcolor)
            if col != "black":
                self.data.current_axis_options.add(f"y grid style={{{col}}}")

    def _set_axis_line_styles(self) -> None:
        # Assume that the bottom edge color is the color of the entire box.
        axcol = self.obj.spines["bottom"].get_edgecolor()
        col, _ = _color.mpl_color2xcolor(self.data, axcol)
        if col != "black":
            self.data.current_axis_options.add(f"axis line style={{{col}}}")

    def _set_background_color(self) -> None:
        bgcolor = self.obj.get_facecolor()
        col, _ = _color.mpl_color2xcolor(self.data, bgcolor)
        if col != "white":
            self.data.current_axis_options.add(f"axis background/.style={{fill={col}}}")

    def _set_colorbar(self) -> None:
        colorbar = _find_associated_colorbar(self.obj)
        if not colorbar:
            return

        colorbar_styles = []

        orientation = colorbar.orientation
        limits = colorbar.mappable.get_clim()
        if orientation == "horizontal":
            self.data.current_axis_options.add("colorbar horizontal")

            colorbar_ticks = colorbar.ax.get_xticks()
            colorbar_ticks_minor = colorbar.ax.get_xticks(minor=True)
            axis_limits = colorbar.ax.get_xlim()

            # In matplotlib, the colorbar color limits are determined by get_clim(), and
            # the tick positions are as usual with respect to {x,y}lim. In PGFPlots,
            # however, they are mixed together.  Hence, scale the tick positions just
            # like {x,y}lim are scaled to clim.
            colorbar_ticks = (colorbar_ticks - axis_limits[0]) / (
                axis_limits[1] - axis_limits[0]
            ) * (limits[1] - limits[0]) + limits[0]
            colorbar_ticks_minor = (colorbar_ticks_minor - axis_limits[0]) / (
                axis_limits[1] - axis_limits[0]
            ) * (limits[1] - limits[0]) + limits[0]
            # Getting the labels via get_* might not actually be suitable:
            # they might not reflect the current state.
            colorbar_ticklabels = colorbar.ax.get_xticklabels()
            colorbar_ticklabels_minor = colorbar.ax.get_xticklabels(minor=True)

            colorbar_styles.extend(_get_ticks(self.data, "x", colorbar_ticks, colorbar_ticklabels))
            colorbar_styles.extend(
                _get_ticks(self.data, "minor x", colorbar_ticks_minor, colorbar_ticklabels_minor)
            )

        elif orientation == "vertical":
            self.data.current_axis_options.add("colorbar")
            colorbar_ticks = colorbar.ax.get_yticks()
            colorbar_ticks_minor = colorbar.ax.get_yticks(minor=True)
            axis_limits = colorbar.ax.get_ylim()

            # In matplotlib, the colorbar color limits are determined by get_clim(), and
            # the tick positions are as usual with respect to {x,y}lim. In PGFPlots,
            # however, they are mixed together.  Hence, scale the tick positions just
            # like {x,y}lim are scaled to clim.
            colorbar_ticks = (colorbar_ticks - axis_limits[0]) / (
                axis_limits[1] - axis_limits[0]
            ) * (limits[1] - limits[0]) + limits[0]
            colorbar_ticks_minor = (colorbar_ticks_minor - axis_limits[0]) / (
                axis_limits[1] - axis_limits[0]
            ) * (limits[1] - limits[0]) + limits[0]

            # Getting the labels via get_* might not actually be suitable:
            # they might not reflect the current state.
            colorbar_ticklabels = colorbar.ax.get_yticklabels()
            colorbar_ylabel = colorbar.ax.get_ylabel()
            colorbar_ticklabels_minor = colorbar.ax.get_yticklabels(minor=True)
            colorbar_styles.extend(_get_ticks(self.data, "y", colorbar_ticks, colorbar_ticklabels))
            colorbar_styles.extend(
                _get_ticks(self.data, "minor y", colorbar_ticks_minor, colorbar_ticklabels_minor)
            )
            colorbar_styles.append("ylabel={" + colorbar_ylabel + "}")
        else:
            msg = f"Orientation must be either 'horizontal' or 'vertical', but is '{orientation}'."
            raise ValueError(msg)

        mycolormap, is_custom_cmap = _mpl_cmap2pgf_cmap(colorbar.mappable.get_cmap(), self.data)
        if is_custom_cmap:
            self.data.current_axis_options.add("colormap=" + mycolormap)
        else:
            self.data.current_axis_options.add("colormap/" + mycolormap)

        ff = self.data.float_format
        self.data.current_axis_options.add(f"point meta min={limits[0]:{ff}}")
        self.data.current_axis_options.add(f"point meta max={limits[1]:{ff}}")

        if colorbar_styles:
            self.data.current_axis_options.add(
                "colorbar style={{{}}}".format(",".join(colorbar_styles))
            )

    def _content_end(self) -> None:
        if self.is_subplot:
            self.content.append("\n\\nextgroupplot")
        else:
            self.content.append(self.data.flavor.start("axis"))

    def get_begin_code(self) -> list[str]:
        if self.data.current_axis_options:
            # Put axis_options in a deterministic order to avoid diff churn.
            self.content.append(
                "[\n" + ",\n".join(sorted(self.data.current_axis_options)) + "\n]\n"
            )
        return self.content

    def get_end_code(self) -> str:
        if not self.is_subplot:
            return self.data.flavor.end("axis") + "\n\n"
        if self.is_subplot and self.nsubplots == self.subplot_index:
            self.data.is_in_groupplot_env = False
            return self.data.flavor.end("groupplot") + "\n\n"

        return ""

    def _get_ticks(self) -> None:
        self.data.current_axis_options.update(
            _get_ticks(self.data, "x", self.obj.get_xticks(), self.obj.get_xticklabels())
        )
        self.data.current_axis_options.update(
            _get_ticks(self.data, "y", self.obj.get_yticks(), self.obj.get_yticklabels())
        )
        self.data.current_axis_options.update(
            _get_ticks(
                self.data,
                "minor x",
                self.obj.get_xticks(minor=True),
                self.obj.get_xticklabels(minor=True),
            )
        )
        self.data.current_axis_options.update(
            _get_ticks(
                self.data,
                "minor y",
                self.obj.get_yticks(minor=True),
                self.obj.get_yticklabels(minor=True),
            )
        )

    def _get_tick_colors(self) -> None:
        try:
            l0 = self.obj.get_xticklines()[0]
        except IndexError:
            pass
        else:
            c0 = l0.get_color()
            xtickcolor, _ = _color.mpl_color2xcolor(self.data, c0)
            self.data.current_axis_options.add(f"xtick style={{color={xtickcolor}}}")

        try:
            l0 = self.obj.get_yticklines()[0]
        except IndexError:
            pass
        else:
            c0 = l0.get_color()
            ytickcolor, _ = _color.mpl_color2xcolor(self.data, c0)
            self.data.current_axis_options.add(f"ytick style={{color={ytickcolor}}}")

    def _get_tick_direction(self) -> None:
        # For new matplotlib versions, we could replace the direction getter by
        # `get_ticks_direction()`, see
        # <https://github.com/matplotlib/matplotlib/pull/5290>.  Unfortunately, _tickdir
        # doesn't seem to be quite accurate. See
        # <https://github.com/matplotlib/matplotlib/issues/5311>.  For now, just take
        # the first tick direction of each of the axes.
        x_tick_dirs = [tick._tickdir for tick in self.obj.xaxis.get_major_ticks()]  # type: ignore[attr-defined]  # noqa: SLF001
        y_tick_dirs = [tick._tickdir for tick in self.obj.yaxis.get_major_ticks()]  # type: ignore[attr-defined]  # noqa: SLF001
        if x_tick_dirs or y_tick_dirs:
            if x_tick_dirs and y_tick_dirs:
                direction = x_tick_dirs[0] if x_tick_dirs[0] == y_tick_dirs[0] else None
            elif x_tick_dirs:
                direction = x_tick_dirs[0]
            else:
                # y_tick_dirs must be present
                direction = y_tick_dirs[0]

            if direction:
                if direction == "in":
                    # 'tick align=inside' is the PGFPlots default
                    pass
                elif direction == "out":
                    self.data.current_axis_options.add("tick align=outside")
                elif direction == "inout":
                    self.data.current_axis_options.add("tick align=center")
                else:
                    msg = f"Direction can be 'in', 'out', or 'inout', but is '{direction}'."
                    raise ValueError(msg)

    def _set_tick_rotation(self) -> None:
        x_tick_rotation_and_horizontal_alignment = (
            self._get_label_rotation_and_horizontal_alignment("x")
        )
        if x_tick_rotation_and_horizontal_alignment:
            self.data.current_axis_options.add(x_tick_rotation_and_horizontal_alignment)

        y_tick_rotation_and_horizontal_alignment = (
            self._get_label_rotation_and_horizontal_alignment("y")
        )
        if y_tick_rotation_and_horizontal_alignment:
            self.data.current_axis_options.add(y_tick_rotation_and_horizontal_alignment)

    def _set_tick_positions(self) -> None:
        x_tick_position_string, x_tick_position = _get_tick_position(self.obj, "x")
        y_tick_position_string, y_tick_position = _get_tick_position(self.obj, "y")

        if x_tick_position == y_tick_position and x_tick_position is not None:
            self.data.current_axis_options.add(f"tick pos={x_tick_position}")
        else:
            if x_tick_position_string is not None:
                self.data.current_axis_options.add(x_tick_position_string)
            if y_tick_position_string is not None:
                self.data.current_axis_options.add(y_tick_position_string)

    def _subplot(self) -> None:
        # https://github.com/matplotlib/matplotlib/issues/7225#issuecomment-252173667
        subplotspec = self.obj.get_subplotspec()
        if subplotspec is None:
            return
        geom = subplotspec.get_topmost_subplotspec().get_geometry()

        self.nsubplots = geom[0] * geom[1]
        if self.nsubplots > 1:
            # Is this an axis-colorbar pair? No need for groupplot then.
            is_groupplot = self.nsubplots != 2 or not _find_associated_colorbar(self.obj)  # noqa: PLR2004

            if is_groupplot:
                self.is_subplot = True
                # subplotspec geometry positioning is 0-based
                self.subplot_index = geom[2] + 1
                if not self.data.is_in_groupplot_env:
                    group_style = [f"group size={geom[1]} by {geom[0]}"]
                    group_style.extend(self.data.extra_groupstyle_options)
                    options = ["group style={{{}}}".format(", ".join(group_style))]
                    self.content.append(
                        self.data.flavor.start("groupplot") + f"[{', '.join(options)}]"
                    )
                    self.data.is_in_groupplot_env = True
                    self.data.pgfplots_libs.add("groupplots")

    def _get_label_rotation_and_horizontal_alignment(self, x_or_y: str) -> str:
        label_style = ""

        major_tick_labels = (
            self.obj.xaxis.get_majorticklabels()
            if x_or_y == "x"
            else self.obj.yaxis.get_majorticklabels()
        )

        if not major_tick_labels:
            return ""

        tick_label_text_width_identifier = f"{x_or_y} tick label text width"
        if tick_label_text_width_identifier in self.data.current_axis_options:
            self.data.current_axis_options.remove(tick_label_text_width_identifier)

        values = []

        tick_labels_rotation = [label.get_rotation() for label in major_tick_labels]
        if len(set(tick_labels_rotation)) == 1:
            if tick_labels_rotation[0] != 0:
                values.append(f"rotate={tick_labels_rotation[0]}")
        else:
            values.append(
                "rotate={{{},0}}[\\ticknum]".format(",".join(str(x) for x in tick_labels_rotation))
            )

        tick_labels_horizontal_alignment = [
            label.get_horizontalalignment() for label in major_tick_labels
        ]
        if len(set(tick_labels_horizontal_alignment)) == 1:
            anchor = {"right": "east", "left": "west", "center": "center"}[
                tick_labels_horizontal_alignment[0]
            ]
            if not (x_or_y == "x" and anchor == "center") and not (
                x_or_y == "y" and anchor == "east"
            ):
                values.append(f"anchor={anchor}")

        if values:
            label_style = "{}ticklabel style={{{}}}".format(x_or_y, ",".join(values))

        # Ignore horizontal alignment if no '{x,y} tick label text width' has been
        # passed in the 'extra' parameter

        return label_style


def _get_tick_position(obj: Axes, x_or_y: str) -> tuple[str | None, str | None]:
    major_ticks = obj.xaxis.majorTicks if x_or_y == "x" else obj.yaxis.majorTicks

    major_ticks_bottom = [tick.tick1line.get_visible() for tick in major_ticks]
    major_ticks_top = [tick.tick2line.get_visible() for tick in major_ticks]

    major_ticks_bottom_show_all = False
    if len(set(major_ticks_bottom)) == 1 and major_ticks_bottom[0] is True:
        major_ticks_bottom_show_all = True

    major_ticks_top_show_all = False
    if len(set(major_ticks_top)) == 1 and major_ticks_top[0] is True:
        major_ticks_top_show_all = True

    position_string = None
    major_ticks_position = None
    if not major_ticks_bottom_show_all and not major_ticks_top_show_all:
        position_string = f"{x_or_y}majorticks=false"
    elif major_ticks_bottom_show_all and major_ticks_top_show_all:
        major_ticks_position = "both"
    elif major_ticks_bottom_show_all:
        major_ticks_position = "left"
    elif major_ticks_top_show_all:
        major_ticks_position = "right"

    if major_ticks_position:
        position_string = f"{x_or_y}tick pos={major_ticks_position}"

    return position_string, major_ticks_position


def _get_ticks(data: TikzData, xy: str, ticks: list | np.ndarray, ticklabels: list) -> list[str]:
    """Gets a {'x','y'}, a number of ticks and ticks labels.

    Returns the necessary axis options for the given configuration.
    """
    axis_options = []
    is_label_required = _is_label_required(ticks, ticklabels)
    pgfplots_ticklabels = _get_pgfplots_ticklabels(ticklabels)

    # if the labels are all missing, then we need to output an empty set of labels
    if len(ticklabels) == 0 and len(ticks) != 0:
        axis_options.append(f"{xy}ticklabels={{}}")
        # remove the multiplier too
        axis_options.append(f"scaled {xy} ticks=" + r"manual:{}{\pgfmathparse{#1}}")

    # Leave the ticks to PGFPlots if not in STRICT mode and if there are no explicit
    # labels.
    if data.strict or is_label_required:
        if len(ticks):
            ff = data.float_format
            axis_options.append(
                "{}tick={{{}}}".format(xy, ",".join([f"{el:{ff}}" for el in ticks]))
            )
        else:
            val = "{}" if "minor" in xy else "\\empty"
            axis_options.append(f"{xy}tick={val}")

        if is_label_required:
            length = sum(len(label) for label in pgfplots_ticklabels)
            max_line_length = 75
            sep = ("", ",", "") if length < max_line_length else ("\n  ", ",\n  ", "\n")
            string = sep[1].join(pgfplots_ticklabels)
            axis_options.append(f"{xy}ticklabels={{{sep[0]}{string}{sep[2]}}}")
    return axis_options


def _is_label_required(ticks: list | np.ndarray, ticklabels: list) -> bool:
    """Check if the label is necessary.

    If one of the labels is, then all of them must appear in the TikZ plot.
    """
    for tick, ticklabel in zip(ticks, ticklabels, strict=False):
        # store the label anyway
        label = ticklabel.get_text()

        if not ticklabel.get_visible():
            return True

        if not label:
            continue

        try:
            label_float = float(label.replace("\N{MINUS SIGN}", "-"))
        except ValueError:
            # Check if label is in format "$\matchdefault{<base>^{<exponent>}}$" (for log plots)
            match = re.search(r"\$\\mathdefault\{(\d+)\^\{(-?\d+(?:\.\d+)?)\}\}\$", label)
            if match is None:
                return True
            label_float = float(match.group(1)) ** float(match.group(2))
        if abs(label_float - tick) > 1.0e-10 + 1.0e-10 * abs(tick):
            return True
    return False


def _get_pgfplots_ticklabels(ticklabels: list) -> list[str]:
    pgfplots_ticklabels = []
    for ticklabel in ticklabels:
        label = ticklabel.get_text()
        if "," in label:
            label = "{" + label + "}"
        pgfplots_ticklabels.append(_common_texification(label))
    return pgfplots_ticklabels


def _is_colorbar_heuristic(obj: Axes) -> bool:
    """Find out if the object is in fact a color bar.

    To come up with something more accurate, see
    # <https://discourse.matplotlib.org/t/find-out-if-an-axes-object-is-a-colorbar/22563>
    # Might help: Are the colorbars exactly the l.collections.PolyCollection's?
    """
    # Not sure if these properties are always present
    if hasattr(obj, "_colorbar") or hasattr(obj, "_colorbar_info"):
        return True

    try:
        aspect = float(obj.get_aspect())
    except ValueError:
        # e.g., aspect in ['equal', 'auto']
        return False

    # Assume that something is a colorbar if and only if the ratio is above 5.0
    # and there are no ticks on the corresponding axis. This isn't always true,
    # though: The ratio of a color can be freely adjusted by the aspect
    # keyword, e.g.,
    #
    #    plt.colorbar(im, aspect=5)  # noqa: ERA001
    #
    threshold_ratio = 5.0

    return (aspect >= threshold_ratio and len(obj.get_xticks()) == 0) or (
        aspect <= 1.0 / threshold_ratio and len(obj.get_yticks()) == 0
    )


def _mpl_cmap2pgf_cmap(cmap: Colormap, data: TikzData) -> tuple[str, bool]:
    """Converts a color map as given in matplotlib to a color map as represented in PGFPlots."""
    if isinstance(cmap, LinearSegmentedColormap):
        return _handle_linear_segmented_color_map(cmap, data)
    if isinstance(cmap, ListedColormap):
        return _handle_listed_color_map(cmap, data)
    msg = "Only LinearSegmentedColormap and ListedColormap are supported."
    raise NotImplementedError(msg)


def _handle_linear_segmented_color_map(
    cmap: LinearSegmentedColormap, data: TikzData
) -> tuple[str, bool]:
    if cmap.is_gray():
        is_custom_colormap = False
        return "blackwhite", is_custom_colormap

    # For an explanation of what _segmentdata contains, see
    # http://matplotlib.org/mpl_examples/pylab_examples/custom_cmap.py
    # A key sentence:
    # If there are discontinuities, then it is a little more complicated. Label the 3
    # elements in each row in the cdict entry for a given color as (x, y0, y1). Then
    # for values of x between x[i] and x[i+1] the color value is interpolated between
    # y1[i] and y0[i+1].
    segdata = cmap._segmentdata  # type: ignore[attr-defined]  # noqa: SLF001
    red = segdata["red"]
    green = segdata["green"]
    blue = segdata["blue"]

    # Loop over the data, stop at each spot where the linear interpolations is
    # interrupted, and set a color mark there.
    #
    # Set initial color.
    k_red = 0
    k_green = 0
    k_blue = 0
    colors = []
    xx = []
    while True:
        # find next x
        x = min(red[k_red][0], green[k_green][0], blue[k_blue][0])

        if red[k_red][0] == x:
            red_comp = red[k_red][1]
            k_red += 1
        else:
            red_comp = _linear_interpolation(
                x,
                (red[k_red - 1][0], red[k_red][0]),
                (red[k_red - 1][2], red[k_red][1]),
            )

        if green[k_green][0] == x:
            green_comp = green[k_green][1]
            k_green += 1
        else:
            green_comp = _linear_interpolation(
                x,
                (green[k_green - 1][0], green[k_green][0]),
                (green[k_green - 1][2], green[k_green][1]),
            )

        if blue[k_blue][0] == x:
            blue_comp = blue[k_blue][1]
            k_blue += 1
        else:
            blue_comp = _linear_interpolation(
                x,
                (blue[k_blue - 1][0], blue[k_blue][0]),
                (blue[k_blue - 1][2], blue[k_blue][1]),
            )

        xx.append(x)
        colors.append((red_comp, green_comp, blue_comp))

        if x >= 1.0:
            break

    # The PGFPlots color map has an actual physical scale, like (0cm,10cm), and the
    # points where the colors change is also given in those units. As of now
    # (2010-05-06) it is crucial for PGFPlots that the difference between two successive
    # points is an integer multiple of a given unity (parameter to the colormap; e.g.,
    # 1cm).  At the same time, TeX suffers from significant round-off errors, so make
    # sure that this unit is not too small such that the round-off errors don't play
    # much of a role. A unit of 1pt, e.g., does most often not work.
    unit = "pt"

    # Scale to integer (too high integers will firstly be slow and secondly may produce
    # dimension errors or memory errors in latex)
    # 0-1000 is the internal granularity of PGFplots.
    # 16300 was the maximum value for pgfplots<=1.13
    xx = _scale_to_int(np.array(xx), 1000)

    color_changes = []
    ff = data.float_format
    for k, x in enumerate(xx):
        color_changes.append(
            f"rgb({x}{unit})=({colors[k][0]:{ff}},{colors[k][1]:{ff}},{colors[k][2]:{ff}})"
        )

    colormap_string = "{{mymap}}{{[1{}]\n  {}\n}}".format(unit, ";\n  ".join(color_changes))
    is_custom_colormap = True
    return colormap_string, is_custom_colormap


def _handle_listed_color_map(cmap: ListedColormap, data: TikzData) -> tuple[str, bool]:
    # check for predefined colormaps in both matplotlib and pgfplots
    cm_translate = {
        # All the rest are LinearSegmentedColorMaps. :/
        # 'autumn': 'autumn',  # noqa: ERA001
        # 'cool': 'cool',  # noqa: ERA001
        # 'copper': 'copper',  # noqa: ERA001
        # 'gray': 'blackwhite',  # noqa: ERA001
        # 'hot': 'hot2',  # noqa: ERA001
        # 'hsv': 'hsv',  # noqa: ERA001
        # 'jet': 'jet',  # noqa: ERA001
        # 'spring': 'spring',  # noqa: ERA001
        # 'summer': 'summer',  # noqa: ERA001
        "viridis": "viridis",
        # 'winter': 'winter',  # noqa: ERA001
    }
    for mpl_cm_name, pgf_cm in cm_translate.items():
        mpl_cm = plt.get_cmap(mpl_cm_name)
        if isinstance(mpl_cm, ListedColormap) and cmap.colors == mpl_cm.colors:
            is_custom_colormap = False
            return pgf_cm, is_custom_colormap

    unit = "pt"
    ff = data.float_format
    if cmap.N is None or (
        isinstance(cmap.colors, Sized)
        and len(cmap.colors) == cmap.N
        and isinstance(cmap.colors, Iterable)
    ):
        colors = [
            f"rgb({k}{unit})=({rgb[0]:{ff}},{rgb[1]:{ff}},{rgb[2]:{ff}})"
            for k, rgb in enumerate(cmap.colors)
            if isinstance(rgb, list)
        ]
    elif isinstance(cmap.colors, list):
        reps = int(float(cmap.N) / len(cmap.colors) - 0.5) + 1
        repeated_cols = reps * cmap.colors
        colors = [
            f"rgb({k}{unit})=({rgb[0]:{ff}},{rgb[1]:{ff}},{rgb[2]:{ff}})"
            for k, rgb in enumerate(repeated_cols[: cmap.N])
        ]
    colormap_string = "{{mymap}}{{[1{}]\n {}\n}}".format(unit, ";\n  ".join(colors))
    is_custom_colormap = True
    return colormap_string, is_custom_colormap


def _scale_to_int(array: np.ndarray, max_val: float) -> list[int]:
    """Scales the array X such that it contains only integers."""
    array = array / max(1 / max_val, _gcd_array(array))
    return [int(entry) for entry in array]


def _gcd_array(array: np.ndarray) -> float:
    """Return the largest real value h such that all elements in x are integer multiples of h."""
    greatest_common_divisor = 0.0
    for x in array:
        greatest_common_divisor = _gcd(greatest_common_divisor, x)

    return greatest_common_divisor


def _gcd(a: float, b: float) -> float:
    """Euclidean algorithm for calculating the GCD of two numbers a, b.

    This algorithm also works for real numbers:
    Find the greatest number h such that a and b are integer multiples of h.
    """
    # Keep the tolerance somewhat significantly above machine precision as otherwise
    # round-off errors will be accounted for, returning 1.0e-10 instead of 1.0 for the
    # values
    #   [1.0, 2.0000000001, 3.0, 4.0].
    tolerance = 1e-5
    while a > tolerance:
        a, b = b % a, a
    return b


def _linear_interpolation(x: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Given two data points [a,b], linearly interpolate those at x."""
    return (b[1] * (x - a[0]) + b[0] * (a[1] - x)) / (a[1] - a[0])


def _find_associated_colorbar(obj: Axes) -> Colorbar | None:
    """Find associated colorbar, if any.

    A rather poor way of telling whether an axis has a colorbar associated: Check the
    next axis environment, and see if it is de facto a color bar; if yes, return the
    color bar object.
    """
    for child in obj.get_children():
        try:
            cbar = child.colorbar  # type: ignore[attr-defined]
        except AttributeError:
            continue
        if cbar is not None:  # really necessary?
            # if fetch was successful, cbar contains
            # (reference to colorbar, reference to axis containing colorbar)
            return cbar
    return None


def _try_f2i(x: float) -> float:
    """If possible, convert float to int without rounding.

    Used for log base: if not used, base for log scale can be "10.0" (and then
    printed as such  by pgfplots).
    """
    return int(x) if int(x) == x else x
