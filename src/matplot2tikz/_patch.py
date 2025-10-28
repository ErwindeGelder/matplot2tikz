from __future__ import annotations

from collections.abc import Generator, Iterable
from itertools import cycle, islice, tee
from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Patch, Rectangle
from matplotlib.transforms import Affine2D

from . import _path as mypath
from ._text import _get_arrow_style

if TYPE_CHECKING:
    from matplotlib.collections import Collection

    from ._tikzdata import TikzData


def draw_patch(data: TikzData, obj: Patch) -> list[str]:
    """Return the PGFPlots code for patches."""
    if isinstance(obj, FancyArrowPatch):
        draw_options = mypath.get_draw_options(
            data,
            mypath.LineData(
                obj=obj,
                ec=obj.get_edgecolor(),
                fc=None,  # get_fillcolor for the arrow refers to the head, not the path
                ls=obj.get_linestyle(),
                lw=obj.get_linewidth(),
                hatch=obj.get_hatch(),
            ),
        )
        return _draw_fancy_arrow(data, obj, draw_options)

    # Gather the draw options.
    draw_options = mypath.get_draw_options(
        data,
        mypath.LineData(
            obj=obj,
            ec=obj.get_edgecolor(),
            fc=obj.get_facecolor(),
            ls=obj.get_linestyle(),
            lw=obj.get_linewidth(),
            hatch=obj.get_hatch(),
        ),
    )

    if isinstance(obj, Rectangle):
        # rectangle specialization
        return _draw_rectangle(data, obj, draw_options)
    if isinstance(obj, Ellipse):
        # ellipse specialization
        return _draw_ellipse(data, obj, draw_options)
    # regular patch
    return _draw_polygon(data, obj, draw_options)


def _is_in_legend(obj: Collection | Patch) -> bool:
    label = obj.get_label()
    if obj.axes is None:
        return False
    leg = obj.axes.get_legend()
    if leg is None:
        return False
    return label in [txt.get_text() for txt in leg.get_texts()]


def _patch_legend(obj: Collection | Patch, draw_options: list, legend_type: str) -> str:
    """Decorator for handling legend Collection and Patch."""
    legend = ""
    if _is_in_legend(obj):
        # Unfortunately, patch legend entries need \addlegendimage in Pgfplots.
        do = ", ".join([legend_type, *draw_options]) if draw_options else ""
        label = obj.get_label()
        legend += f"\\addlegendimage{{{do}}}\n\\addlegendentry{{{label}}}\n\n"

    return legend


def zip_modulo(*iterables: Iterable) -> Generator:
    # Make copies to avoid exhausing any iterables
    tees = [tee(iterable) for iterable in iterables]

    # Get the length of the longest iterable
    max_length = max(sum(1 for _ in t[0]) for t in tees)

    # Build infinite cycles for each iterable, then limit it to max_length
    max_length_iterables = [islice(cycle(t[1]), max_length) for t in tees]

    for _ in range(max_length):
        yield tuple(next(iterable) for iterable in max_length_iterables)


def draw_patchcollection(data: TikzData, obj: Collection) -> list[str]:
    """Returns PGFPlots code for a number of patch objects."""
    content = []

    # recompute the face colors
    obj.update_scalarmappable()

    def ensure_list(x: Iterable | float) -> Iterable:
        if isinstance(x, float):
            return [x]
        return x if sum(1 for _ in x) else [None]

    ecs = ensure_list(obj.get_edgecolor())
    fcs = ensure_list(obj.get_facecolor())
    lss = ensure_list(obj.get_linestyle())
    lws = ensure_list(obj.get_linewidth())
    ts = ensure_list(obj.get_transforms())
    offs_tmp = obj.get_offsets()
    offs = offs_tmp if isinstance(offs_tmp, Iterable) else [offs_tmp]
    hatches = ensure_list(obj.get_hatch()) if obj.get_hatch() is not None else [None]

    paths = obj.get_paths()
    for path, ec, fc, ls, lw, t, off, hatch in zip_modulo(
        paths, ecs, fcs, lss, lws, ts, offs, hatches
    ):
        draw_options = mypath.get_draw_options(
            data, mypath.LineData(obj=obj, ec=ec, fc=fc, ls=ls, lw=lw, hatch=hatch)
        )
        cont, is_area = mypath.draw_path(
            data,
            path.transformed(Affine2D(t).translate(*off)) if t is not None else path,
            draw_options=draw_options,
        )
        content.append(cont)

    legend_type = "area legend" if is_area else "line legend"
    content.append(_patch_legend(obj, draw_options, legend_type) or "\n")

    return content


def _draw_polygon(data: TikzData, obj: Patch, draw_options: list) -> list[str]:
    str_path, is_area = mypath.draw_path(data, obj.get_path(), draw_options=draw_options)
    legend_type = "area legend" if is_area else "line legend"
    return [str_path, _patch_legend(obj, draw_options, legend_type)]


def _draw_rectangle(data: TikzData, obj: Rectangle, draw_options: list) -> list[str]:
    """Return the PGFPlots code for rectangles."""
    # Objects with labels are plot objects (from bar charts, etc).  Even those without
    # labels explicitly set have a label of "_nolegend_".  Everything else should be
    # skipped because they likely correspong to axis/legend objects which are handled by
    # PGFPlots
    label = obj.get_label()
    if label == "":
        return []

    # Get actual label, bar charts by default only give rectangles labels of
    # "_nolegend_". See <https://stackoverflow.com/q/35881290/353337>.
    if isinstance(obj.axes, Axes):
        handles, labels = obj.axes.get_legend_handles_labels()
        labels_found = [label for h, label in zip(handles, labels) if obj in h.get_children()]
        if len(labels_found) == 1:
            label = labels_found[0]

    left_lower_x = obj.get_x()
    left_lower_y = obj.get_y()

    # If we are dealing with a bar plot, left_lower_y will be 0. This is a problem if the y-scale is
    # logarithmic (see https://github.com/ErwindeGelder/matplot2tikz/issues/25)
    # To resolve this, the lower y limit will be used as lower_left_y
    if data.current_mpl_axes is not None and data.current_mpl_axes.get_yscale() == "log":
        left_lower_y = data.current_mpl_axes.get_ylim()[0]

    ff = data.float_format
    do = ",".join(draw_options)
    right_upper_x = left_lower_x + obj.get_width()
    right_upper_y = left_lower_y + obj.get_height()
    content = [
        f"\\draw[{do}] (axis cs:{left_lower_x:{ff}},{left_lower_y:{ff}}) "
        f"rectangle (axis cs:{right_upper_x:{ff}},{right_upper_y:{ff}});\n"
    ]

    if label != "_nolegend_" and str(label) not in data.rectangle_legends:
        data.rectangle_legends.add(str(label))
        draw_opts = ",".join(draw_options)
        content.append(f"\\addlegendimage{{ybar,ybar legend,{draw_opts}}}\n")
        content.append(f"\\addlegendentry{{{label}}}\n\n")

    return content


def _draw_ellipse(data: TikzData, obj: Ellipse, draw_options: list) -> list[str]:
    """Return the PGFPlots code for ellipses."""
    if isinstance(obj, Circle):
        # circle specialization
        return _draw_circle(data, obj, draw_options)
    x, y = obj.center
    ff = data.float_format

    if obj.angle != 0:
        draw_options.append(f"rotate around={{{obj.angle:{ff}}:(axis cs:{x:{ff}},{y:{ff}})}}")

    do = ",".join(draw_options)
    content = [
        f"\\draw[{do}] (axis cs:{x:{ff}},{y:{ff}}) ellipse "
        f"({0.5 * obj.width:{ff}} and {0.5 * obj.height:{ff}});\n"
    ]
    content.append(_patch_legend(obj, draw_options, "area legend"))

    return content


def _draw_circle(data: TikzData, obj: Circle, draw_options: list) -> list[str]:
    """Return the PGFPlots code for circles."""
    x, y = obj.center
    ff = data.float_format
    do = ",".join(draw_options)
    return [
        f"\\draw[{do}] (axis cs:{x:{ff}},{y:{ff}}) circle ({obj.get_radius():{ff}});\n",
        _patch_legend(obj, draw_options, "area legend"),
    ]


def _draw_fancy_arrow(data: TikzData, obj: FancyArrowPatch, draw_options: list) -> list[str]:
    style = _get_arrow_style(data, obj)
    ff = data.float_format
    if obj._posA_posB is not None:  # type: ignore[attr-defined]  # noqa: SLF001  (no known method to obtain posA and posB)
        pos_a, pos_b = obj._posA_posB  # type: ignore[attr-defined]  # noqa: SLF001
        do = ",".join(style)
        str_path = (
            f"\\draw[{do}] (axis cs:{pos_a[0]:{ff}},{pos_a[1]:{ff}}) -- "
            f"(axis cs:{pos_b[0]:{ff}},{pos_b[1]:{ff}});\n"
        )
    else:
        str_path, _ = mypath.draw_path(
            data,
            obj._path_original,  # type: ignore[attr-defined]  # noqa: SLF001
            draw_options=draw_options + style,
        )
    return [str_path, _patch_legend(obj, draw_options, "line legend")]
