from __future__ import annotations

import contextlib
import datetime
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.dates import num2date

from . import _color as mycol
from . import _files
from . import _path as mypath
from ._markers import _mpl_marker2pgfp_marker
from ._util import get_legend_text, has_legend, transform_to_data_coordinates

if TYPE_CHECKING:
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    from ._tikzdata import TikzData


@dataclass
class MarkerData:
    marker: str | None
    mark_options: list
    fc: (
        str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | tuple[str | tuple[float, float, float], float]
        | tuple[tuple[float, float, float, float], float]
        | None
    ) = None  # facecolor
    ec: (
        str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | tuple[str | tuple[float, float, float], float]
        | tuple[tuple[float, float, float, float], float]
        | None
    ) = None  # edgecolor
    lc: (
        str
        | tuple[float, float, float]
        | tuple[float, float, float, float]
        | tuple[str | tuple[float, float, float], float]
        | tuple[tuple[float, float, float, float], float]
        | None
    ) = None  # linecolor


def draw_line2d(data: TikzData, obj: Line2D) -> list[str]:
    r"""Returns the PGFPlots code for an Line2D environment.

    If line is of length 0, do nothing.  Otherwise, an empty \addplot table will be
    created, which will be interpreted as an external data source in either the file
    '' or '.tex'.  Instead, render nothing.
    """
    obj_xdata = obj.get_xdata()
    xdata = (
        obj_xdata
        if isinstance(obj_xdata, Iterable) and isinstance(obj_xdata, Sized)
        else [obj_xdata]
    )

    if len(xdata) == 0:
        return []

    # Get several plot options
    addplot_options = _get_line2d_options(data, obj)
    # Check if a line is in a legend and forget it if not.
    # Fixes <https://github.com/nschloe/tikzplotlib/issues/167>.
    legend_text = get_legend_text(obj)
    if legend_text is None and obj.axes is not None and has_legend(obj.axes):
        addplot_options.append("forget plot")

    # process options
    content = ["\\addplot "]
    if addplot_options:
        opts = ", ".join(addplot_options)
        content.append(f"[{opts}]\n")

    content += _table(data, obj)

    if legend_text is not None:
        content.append(f"\\addlegendentry{{{legend_text}}}\n")

    return content


def _get_line2d_options(data: TikzData, obj: Line2D) -> list[str]:
    addplot_options = []

    line_width = mypath.mpl_linewidth2pgfp_linewidth(data, obj.get_linewidth())
    if line_width:
        addplot_options.append(line_width)
    line_xcolor = _get_linecolor_line2d(data, obj)
    addplot_options.append(line_xcolor)
    drawstyle = _get_drawstyle_line2d(obj)
    if drawstyle is not None:
        addplot_options.append(drawstyle)
    alpha = obj.get_alpha()
    if alpha is not None:
        addplot_options.append(f"opacity={alpha}")
    linestyle = mypath.mpl_linestyle2pgfplots_linestyle(data, obj.get_linestyle(), line=obj)
    if linestyle is not None and linestyle != "solid":
        addplot_options.append(linestyle)

    marker_face_color = obj.get_markerfacecolor()
    marker_edge_color = obj.get_markeredgecolor()

    is_filled = marker_face_color is not None and not (
        isinstance(marker_face_color, str) and marker_face_color.lower() == "none"
    )
    mpl_marker = obj.get_marker()
    if not isinstance(mpl_marker, str):
        raise NotImplementedError
    marker, extra_mark_options = _mpl_marker2pgfp_marker(data, mpl_marker, is_filled=is_filled)
    if marker:
        _marker(
            data,
            obj,
            MarkerData(
                marker=marker,
                mark_options=extra_mark_options,
                fc=marker_face_color,
                ec=marker_edge_color,
                lc=line_xcolor,
            ),
            addplot_options,
        )

    if marker and linestyle is None:
        addplot_options.append("only marks")

    return addplot_options


def _get_linecolor_line2d(data: TikzData, obj: Line2D) -> str:
    color = obj.get_color()
    return mycol.mpl_color2xcolor(data, color)[0]


def _get_drawstyle_line2d(obj: Line2D) -> str | None:
    drawstyle = obj.get_drawstyle()
    if drawstyle in [None, "default"]:
        return None
    if drawstyle == "steps-mid":
        return "const plot mark mid"
    if drawstyle in ["steps-pre", "steps"]:
        return "const plot mark right"
    if drawstyle == "steps-post":
        return "const plot mark left"
    msg = f"Unknown drawstyle '{drawstyle}'."
    raise NotImplementedError(msg)


def draw_linecollection(data: TikzData, obj: LineCollection) -> list[str]:
    """Returns Pgfplots code for a number of patch objects."""
    content = []

    edgecolors = obj.get_edgecolors()  # type: ignore[attr-defined]
    linestyles = obj.get_linestyles()  # type: ignore[attr-defined]
    linewidths = obj.get_linewidths()  # type: ignore[attr-defined]
    paths = obj.get_paths()

    for i, path in enumerate(paths):
        color = edgecolors[i] if i < len(edgecolors) else edgecolors[0]
        style = linestyles[i] if i < len(linestyles) else linestyles[0]
        # Ensure that if style is a tuple, that first element is a float
        if isinstance(style, tuple):
            style = (float(style[0]), style[1])
        width = float(linewidths[i] if i < len(linewidths) else linewidths[0])

        options = mypath.get_draw_options(
            data, mypath.LineData(obj=obj, ec=color, ls=style, lw=width)
        )

        cont, _ = mypath.draw_path(data, path, draw_options=options, simplify=False)
        content.append(cont + "\n")

    return content


def _marker(
    data: TikzData, obj: Line2D, marker_data: MarkerData, addplot_options: list[str]
) -> None:
    if marker_data.marker is None:
        msg = "Marker must be set."
        raise ValueError(msg)
    addplot_options.append("mark=" + marker_data.marker)

    _marker_size(data, obj, addplot_options)
    _marker_every(obj, addplot_options)
    _marker_options(data, marker_data, addplot_options)


def _marker_size(data: TikzData, obj: Line2D, addplot_options: list[str]) -> None:
    mark_size = obj.get_markersize()
    if mark_size:
        ff = data.float_format
        # setting half size because pgfplots counts the radius/half-width
        pgf_size = 0.5 * mark_size
        addplot_options.append(f"mark size={pgf_size:{ff}}")


def _marker_every(obj: Line2D, addplot_options: list[str]) -> None:
    mark_every = obj.get_markevery()
    if mark_every:
        if isinstance(mark_every, (int, float)):
            addplot_options.append(f"mark repeat={mark_every:d}")
        elif isinstance(mark_every, slice):
            raise NotImplementedError
        else:
            # python starts at index 0, pgfplots at index 1
            pgf_marker = [1 + m for m in mark_every]
            addplot_options.append("mark indices = {" + ", ".join(map(str, pgf_marker)) + "}")


def _marker_options(data: TikzData, marker_data: MarkerData, addplot_options: list[str]) -> None:
    mark_options = ["solid"]
    mark_options += marker_data.mark_options
    if marker_data.fc is None or (isinstance(marker_data.fc, str) and marker_data.fc == "none"):
        mark_options.append("fill opacity=0")
    else:
        face_xcolor, _ = mycol.mpl_color2xcolor(data, marker_data.fc)
        if face_xcolor != marker_data.lc:
            mark_options.append("fill=" + face_xcolor)

    face_and_edge_have_equal_color = marker_data.ec == marker_data.fc
    # Sometimes, the colors are given as arrays. Collapse them into a
    # single boolean.
    if isinstance(face_and_edge_have_equal_color, Iterable):
        face_and_edge_have_equal_color = all(face_and_edge_have_equal_color)

    if not face_and_edge_have_equal_color and marker_data.ec is not None:
        draw_xcolor, _ = mycol.mpl_color2xcolor(data, marker_data.ec)
        if draw_xcolor != marker_data.lc:
            mark_options.append("draw=" + draw_xcolor)
    opts = ",".join(mark_options)
    addplot_options.append(f"mark options={{{opts}}}")


def _table(data: TikzData, obj: Line2D) -> list[str]:
    xdata, ydata = _get_xy_data(data, obj)
    ydata_mask = _get_ydata_mask(obj)

    if isinstance(xdata[0], datetime.datetime):
        xdata = np.array([date.strftime("%Y-%m-%d %H:%M") for date in xdata])
        xformat = ""
        col_sep = ","
        opts = ["header=false", "col sep=comma"]
        data.current_axis_options.add("date coordinates in=x")
        # Replace float xmin/xmax by datetime
        # <https://github.com/matplotlib/matplotlib/issues/13727>.
        data.current_axis_options = {
            option for option in data.current_axis_options if not option.startswith("xmin")
        }
        if data.current_mpl_axes is None:
            msg = "Matplotlib axes should be set to get the x-axis limits."
            raise ValueError(msg)
        xmin, xmax = data.current_mpl_axes.get_xlim()
        mindate = num2date(xmin).strftime("%Y-%m-%d %H:%M")
        maxdate = num2date(xmax).strftime("%Y-%m-%d %H:%M")
        data.current_axis_options.add(f"xmin={mindate}, xmax={maxdate}")

        # Also remove xtick stuff, as it will result in compilation error in LaTeX
        data.current_axis_options = {
            option
            for option in data.current_axis_options
            if not option.startswith(("xtick=", "xticklabels="))
        }
    else:
        opts = []
        xformat = data.float_format
        col_sep = " "

    if data.table_row_sep != "\n":
        # don't want the \n in the table definition, just in the data (below)
        opts.append("row sep=" + data.table_row_sep.strip())

    table_row_sep = data.table_row_sep
    ydata[ydata_mask] = np.nan
    # matplotlib jumps at masked or nan values, while PGFPlots by default
    # interpolates. Hence, if we have a masked plot, make sure that PGFPlots jumps
    # as well.
    if (
        np.any(ydata_mask) or ~np.all(np.isfinite(ydata))
    ) and "unbounded coords=jump" not in data.current_axis_options:
        data.current_axis_options.add("unbounded coords=jump")

    ff = data.float_format
    plot_table = [
        f"{x:{xformat}}{col_sep}{y:{ff}}{table_row_sep}" for x, y in zip(xdata, ydata, strict=True)
    ]

    min_extern_length = 3

    content = []
    if data.externalize_tables and len(xdata) >= min_extern_length:
        filepath, rel_filepath = _files.new_filepath(data, "table", ".dat")
        with filepath.open("w") as f:
            # No encoding handling required: plot_table is only ASCII
            f.write("".join(plot_table))

        if data.externals_search_path is not None:
            esp = data.externals_search_path
            opts.append(f"search path={{{esp}}}")

        opts_str = ("[" + ",".join(opts) + "] ") if len(opts) > 0 else ""
        posix_filepath = rel_filepath.as_posix()
        content.append(f"table {{{opts_str}}}{{{posix_filepath}}};\n")
    else:
        if len(opts) > 0:
            opts_str = ",".join(opts)
            content.append(f"table [{opts_str}] {{%\n")
        else:
            content.append("table {%\n")
        content.extend(plot_table)
        content.append("};\n")

    return content


def _get_xy_data(data: TikzData, obj: Line2D) -> tuple[np.ndarray, np.ndarray]:
    # get_xydata() always gives float data, no matter what
    xy = obj.get_xydata()
    if isinstance(xy, np.ndarray):
        xdata, ydata = xy.T
    else:
        msg = "xy data must be a numpy array."
        raise TypeError(msg)

    # get_{x,y}data gives datetime or string objects if so specified in the plotter
    xdata_alt = obj.get_xdata()
    xdata_iterable = list(xdata_alt) if isinstance(xdata_alt, Iterable) else [xdata_alt]

    ff = data.float_format

    if isinstance(xdata_iterable[0], datetime.datetime):
        xdata = np.array(xdata_iterable)
    else:
        if isinstance(xdata_iterable[0], str):
            # Remove old xtick,xticklabels (if any).
            data.current_axis_options = {
                option
                for option in data.current_axis_options
                if not option.startswith(("xtick=", "xticklabels="))
            }
            data.current_axis_options.update(
                [
                    "xtick={{{}}}".format(",".join([f"{x:{ff}}" for x in xdata])),
                    "xticklabels={{{}}}".format(",".join([str(x) for x in xdata_iterable])),
                ]
            )
        xdata, ydata = transform_to_data_coordinates(obj, xdata, ydata)

    # matplotlib allows plotting of data containing `astropy.units`, but they will break
    # the formatted string here. Try to strip the units from the data.
    with contextlib.suppress(AttributeError):
        xdata = xdata.value
    with contextlib.suppress(AttributeError):
        ydata = ydata.value

    return xdata, ydata


def _get_ydata_mask(obj: Line2D) -> np.ndarray:
    ydata = obj.get_ydata()
    if not hasattr(ydata, "mask"):
        return np.array([], dtype=bool)
    ydata_mask = ydata.mask
    if isinstance(ydata_mask, np.bool_) and not ydata_mask:
        return np.array([], dtype=bool)
    if callable(ydata_mask):
        # pandas.Series have the method mask
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.mask.html
        return np.array([], dtype=bool)
    return ydata_mask
