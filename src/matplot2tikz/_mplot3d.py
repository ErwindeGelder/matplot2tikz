from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import numpy as np
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import (
    Line3D,
    Line3DCollection,
    Path3DCollection,
    Poly3DCollection,
    Text3D,
)

from . import _files, _line2d
from . import _path as mypath
from ._axes import _mpl_cmap2pgf_cmap
from ._util import get_legend_text, has_legend

if TYPE_CHECKING:
    from matplotlib.collections import Collection, LineCollection, PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from ._tikzdata import TikzData

MIN_POLYGON_VERTICES = 3
RGBA_LENGTH = 4
ColorArray = np.ndarray
HashableColor = tuple[float, ...]
LineStyle = str | tuple[float, Sequence[float] | None] | None
T = TypeVar("T")


@dataclass
class Poly3DStyle:
    facecolor: ColorArray | None
    edgecolor: ColorArray | None
    linewidth: float | None
    linestyle: LineStyle


@dataclass
class Poly3DStyleArrays:
    facecolors: ColorArray | None
    edgecolors: ColorArray
    linewidths: Sequence[float | None] | np.ndarray
    linestyles: Sequence[LineStyle]


@dataclass
class Poly3DColorGroup:
    segments: list[np.ndarray]
    colors: list[ColorArray | None]
    style_options: list[str]
    vertex_count: int


@dataclass
class Quiver3DData:
    coordinates: np.ndarray
    options: list[str]


def is_quiver3d_collection(obj: Collection) -> bool:
    # Since a quiver3d collection is represented as a Line3DCollection with a
    # specific structure of segments and uniform style, it can only be identifed
    # by checking both
    if not isinstance(obj, Line3DCollection):
        return False
    style_arrays = _poly3d_style_arrays(obj)
    if not _quiver3d_has_uniform_style(style_arrays):
        return False
    return _quiver3d_coordinates(get_segments3d(obj)) is not None


def is_contour3d_collection(obj: Collection) -> bool:
    return hasattr(obj, "_3dverts_codes")


def get_text3d_position(obj: Text) -> tuple[float, float, float]:
    if not isinstance(obj, Text3D):
        msg = f"Expected Text3D, got {type(obj)}."
        raise TypeError(msg)
    return obj.get_position_3d()


def get_line3d_data(obj: Line2D) -> np.ndarray:
    if not isinstance(obj, Line3D):
        msg = f"Expected Line3D, got {type(obj)}."
        raise TypeError(msg)
    xdata, ydata, zdata = obj.get_data_3d()
    return np.column_stack([xdata, ydata, zdata])


def get_offsets3d(obj: PathCollection) -> np.ndarray:
    if not isinstance(obj, Path3DCollection):
        msg = f"Expected Path3DCollection, got {type(obj)}."
        raise TypeError(msg)
    offsets = obj._offsets3d  # noqa: SLF001
    xdata, ydata, zdata = (np.ma.getdata(offset) for offset in offsets)
    return np.column_stack([xdata, ydata, zdata])


def get_segments3d(obj: LineCollection) -> list[np.ndarray]:
    if not isinstance(obj, Line3DCollection):
        msg = f"Expected Line3DCollection, got {type(obj)}."
        raise TypeError(msg)
    segments = obj._segments3d  # noqa: SLF001
    return [np.asarray(segment, dtype=float) for segment in segments]


def get_poly3d_segments(obj: Collection) -> list[np.ndarray]:
    if not isinstance(obj, Poly3DCollection):
        msg = f"Expected Poly3DCollection, got {type(obj)}."
        raise TypeError(msg)

    vec = obj._vec  # noqa: SLF001
    segslices = obj._segslices  # noqa: SLF001
    return [np.asarray(vec[:3, segslice].T, dtype=float) for segslice in segslices]


def get_poly3d_facecolors(obj: Collection) -> np.ndarray:
    facecolors = obj._facecolor3d  # type: ignore[attr-defined]  # noqa: SLF001
    return np.asarray(facecolors)


def get_poly3d_edgecolors(obj: Collection) -> np.ndarray:
    edgecolors = obj._edgecolor3d  # type: ignore[attr-defined]  # noqa: SLF001
    return np.asarray(edgecolors)


def get_contour3d_verts_and_codes(obj: Collection) -> list[tuple[np.ndarray, np.ndarray | None]]:
    verts_codes = obj._3dverts_codes  # type: ignore[attr-defined]  # noqa: SLF001
    return [
        (
            np.asarray(vertices, dtype=float),
            None if codes is None else np.asarray(codes),
        )
        for vertices, codes in verts_codes
    ]


def draw_line3d(data: TikzData, obj: Line2D) -> list[str]:
    """Return PGFPlots code for a 3D line."""
    coordinates = get_line3d_data(obj)
    if len(coordinates) == 0:
        return []

    addplot_options = _line2d._get_line2d_options(data, obj)  # noqa: SLF001
    legend_text = get_legend_text(obj)
    if legend_text is None and obj.axes is not None and has_legend(obj.axes):
        addplot_options.append("forget plot")

    content = ["\\addplot3 "]
    if addplot_options:
        content.append("[{}]\n".format(", ".join(addplot_options)))
    content.extend(table(data, coordinates))

    if legend_text is not None:
        content.append(f"\\addlegendentry{{{legend_text}}}\n")

    return content


def draw_quiver3d(data: TikzData, obj: Line3DCollection) -> list[str]:
    """Return PGFPlots code for a 3D quiver plot."""
    style_arrays = _poly3d_style_arrays(obj)
    segments = get_segments3d(obj)
    quiver_data = _quiver3d_data(data, obj, segments, style_arrays)
    if quiver_data is not None:
        return addplot_table(
            data,
            quiver_data.coordinates,
            command="addplot3",
            options=quiver_data.options,
            table_options=["x=x", "y=y", "z=z"],
            column_names=["x", "y", "z", "u", "v", "w"],
        )
    return []


def draw_line3dcollection(data: TikzData, obj: Line3DCollection) -> list[str]:
    """Return PGFPlots code for 3D line collections such as wireframes and quivers."""
    style_arrays = _poly3d_style_arrays(obj)
    segments = get_segments3d(obj)

    content = []
    for i, segment in enumerate(segments):
        color = _cycle_array(style_arrays.edgecolors, i)
        style = _cycle_array(style_arrays.linestyles, i)
        if isinstance(style, tuple):
            style = (float(style[0]), style[1])
        linewidth = _cycle_array(style_arrays.linewidths, i)
        width = None if linewidth is None else float(linewidth)

        options = mypath.get_draw_options(
            data, mypath.LineData(obj=obj, ec=color, ls=style, lw=width)
        )
        content.extend(
            addplot_table(
                data,
                segment,
                command="addplot3",
                options=options,
                externalize_min_rows=3,
            )
        )

    return content


def _quiver3d_data(
    data: TikzData,
    obj: LineCollection,
    segments: list[np.ndarray],
    style_arrays: Poly3DStyleArrays,
) -> Quiver3DData | None:
    quiver_coordinates = _quiver3d_coordinates(segments)
    if quiver_coordinates is None:
        return None

    style = _cycle_array(style_arrays.linestyles, 0)
    if isinstance(style, tuple):
        style = (float(style[0]), style[1])
    linewidth = _cycle_array(style_arrays.linewidths, 0)
    width = None if linewidth is None else float(linewidth)
    options = mypath.get_draw_options(
        data,
        mypath.LineData(obj=obj, ec=_cycle_array(style_arrays.edgecolors, 0), ls=style, lw=width),
    )
    options.extend(
        [
            "-latex",
            "mark=none",
            r"quiver={u=\thisrow{u}, v=\thisrow{v}, w=\thisrow{w}}",
        ]
    )
    return Quiver3DData(quiver_coordinates, options)


def _quiver3d_coordinates(segments: list[np.ndarray]) -> np.ndarray | None:
    if not segments or len(segments) % 3 != 0:
        return None

    arrow_count = len(segments) // 3
    shafts = segments[:arrow_count]
    first_heads = segments[arrow_count : 2 * arrow_count]
    second_heads = segments[2 * arrow_count :]
    rows = []
    for shaft, first_head, second_head in zip(shafts, first_heads, second_heads, strict=True):
        if shaft.shape != (2, 3) or first_head.shape != (2, 3) or second_head.shape != (2, 3):
            return None
        tip = shaft[0]
        tail = shaft[1]
        if not (np.allclose(first_head[0], tip) and np.allclose(second_head[0], tip)):
            return None
        vector = tip - tail
        if np.allclose(vector, 0.0):
            return None
        rows.append([*tail, *vector])

    return np.asarray(rows, dtype=float)


def _quiver3d_has_uniform_style(style_arrays: Poly3DStyleArrays) -> bool:
    return (
        _array_len(style_arrays.edgecolors) <= 1
        and _array_len(style_arrays.linewidths) <= 1
        and len(style_arrays.linestyles) <= 1
    )


def _array_len(values: Sequence[object] | np.ndarray | None) -> int:
    return 0 if values is None else len(values)


def draw_path3dcollection(data: TikzData, obj: PathCollection) -> list[str]:
    """Return PGFPlots code for 3D scatter/path collections."""
    pcd = mypath.make_pathcollection_data(
        data,
        obj,
        mypath.PathCollectionCoordinates(
            offsets=get_offsets3d(obj),
            labels=["x", "y", "z"],
            table_options=["x=x", "y=y", "z=z"],
            is_contour=False,
            command="\\addplot3",
        ),
    )
    return mypath.draw_pathcollection_data(data, pcd)


def draw_poly3dcollection(data: TikzData, obj: Collection) -> list[str]:
    """Returns PGFPlots patch-plot code for 3D polygon collections."""
    indexed_segments = [
        (index, vertices)
        for index, vertices in enumerate(get_poly3d_segments(obj))
        if len(vertices) >= MIN_POLYGON_VERTICES
    ]
    if not indexed_segments:
        return []

    data.pgfplots_libs.add("patchplots")
    array = obj.get_array()
    if array is not None:
        color_data = np.ma.getdata(array)
        if len(color_data) > max(index for index, _ in indexed_segments):
            segment_colors = [
                (vertices, float(color_data[index])) for index, vertices in indexed_segments
            ]
            return _draw_poly3dcollection_colormapped(data, obj, segment_colors)

    return _draw_poly3dcollection_explicit_colors(data, obj, indexed_segments)


def draw_contour3d(data: TikzData, obj: Collection) -> list[str]:
    """Returns PGFPlots code for 3D contour collections."""
    facecolors = np.asarray(obj.get_facecolor())
    if len(facecolors):
        return _draw_contour3d_filled(data, obj, facecolors)
    return _draw_contour3d_lines(data, obj)


def _draw_contour3d_lines(data: TikzData, obj: Collection) -> list[str]:
    content: list[str] = []
    style_arrays = _poly3d_style_arrays(obj)
    for i, (vertices, codes) in enumerate(get_contour3d_verts_and_codes(obj)):
        options = _poly3d_style_options(
            data, obj, _poly3d_style_at(style_arrays, i, include_facecolor=False)
        )
        for segment in _split_contour3d_vertices(vertices, codes):
            if len(segment) < 2:  # noqa: PLR2004
                continue
            content.extend(addplot_table(data, segment, command="addplot3", options=options))
    return content


def _draw_contour3d_filled(data: TikzData, obj: Collection, facecolors: np.ndarray) -> list[str]:
    data.pgfplots_libs.add("patchplots")
    content: list[str] = []
    style_arrays = _poly3d_style_arrays(obj, facecolors=facecolors)
    for i, (vertices, codes) in enumerate(get_contour3d_verts_and_codes(obj)):
        segments = [
            segment
            for segment in _split_contour3d_vertices(vertices, codes)
            if len(segment) >= MIN_POLYGON_VERTICES
        ]
        if not segments:
            continue
        style = _poly3d_style_at(style_arrays, i)
        for vertex_count, grouped_segments in _group_segments_only_by_vertex_count(segments):
            options = _poly3d_base_options(vertex_count)
            options.extend(_poly3d_style_options(data, obj, style))
            options.append(_patch_table(grouped_segments))
            content.extend(_poly3d_addplot(data, grouped_segments, options))
    return content


def _split_contour3d_vertices(vertices: np.ndarray, codes: np.ndarray | None) -> list[np.ndarray]:
    if len(vertices) == 0:
        return []
    if codes is None:
        return [vertices]

    segments: list[np.ndarray] = []
    current_segment: list[np.ndarray] = []
    for vertex, code in zip(vertices, codes, strict=False):
        if code == Path.MOVETO:
            if current_segment:
                segments.append(np.asarray(current_segment, dtype=float))
            current_segment = [vertex]
        elif code == Path.CLOSEPOLY:
            if current_segment:
                segments.append(np.asarray(current_segment, dtype=float))
                current_segment = []
        else:
            current_segment.append(vertex)
    if current_segment:
        segments.append(np.asarray(current_segment, dtype=float))
    return segments


def _draw_poly3dcollection_colormapped(
    data: TikzData, obj: Collection, segment_colors: list[tuple[np.ndarray, float]]
) -> list[str]:
    content: list[str] = []
    cmap = obj.get_cmap()
    if cmap is not None:
        mycolormap, is_custom_cmap = _mpl_cmap2pgf_cmap(cmap, data)
        colormap_option = "colormap" + ("=" if is_custom_cmap else "/") + mycolormap
        data.current_axis_options.add(colormap_option)

    for vertex_count, group in _group_segments_by_vertex_count(segment_colors):
        grouped_segments = [segment for segment, _ in group]
        grouped_color_data = [color_value for _, color_value in group]
        options = _poly3d_base_options(vertex_count)
        options.extend(_poly3d_collection_options(data, obj))
        options.append(_patch_table(grouped_segments, grouped_color_data))
        content.extend(_poly3d_addplot(data, grouped_segments, options))

    return content


def _draw_poly3dcollection_explicit_colors(
    data: TikzData, obj: Collection, segments: list[tuple[int, np.ndarray]]
) -> list[str]:
    facecolors = get_poly3d_facecolors(obj)
    edgecolors = get_poly3d_edgecolors(obj)
    style_arrays = _poly3d_style_arrays(obj, facecolors=facecolors, edgecolors=edgecolors)

    group_key: tuple[int, HashableColor | None, float | None, str, float | None]
    grouped: dict[
        tuple[int, HashableColor | None, float | None, str, float | None], Poly3DColorGroup
    ] = {}
    for index, vertices in segments:
        style = _poly3d_style_at(style_arrays, index)
        fc = style.facecolor
        ec = style.edgecolor
        lw = style.linewidth
        ls = style.linestyle
        face_alpha = _color_alpha(fc)
        group_key = (
            len(vertices),
            _hashable_color(ec),
            float(lw) if lw is not None else None,
            repr(ls),
            face_alpha,
        )
        if group_key not in grouped:
            grouped[group_key] = Poly3DColorGroup(
                segments=[],
                colors=[],
                style_options=_poly3d_edge_options(data, obj, ec, lw, ls)
                + _poly3d_fill_opacity_options(data, face_alpha),
                vertex_count=len(vertices),
            )
        grouped[group_key].segments.append(vertices)
        grouped[group_key].colors.append(_rgb_color(fc))

    content: list[str] = []
    for group in grouped.values():
        options = _poly3d_base_options(group.vertex_count)
        options.extend(group.style_options)
        color_indices = _poly3d_explicit_color_indices(group.colors)
        if not color_indices.unique_colors:
            options.append(_patch_table(group.segments))
        elif len(color_indices.unique_colors) > 1:
            options.extend(_poly3d_colormap_options(data, color_indices.unique_colors))
            options.append(_patch_table(group.segments, color_indices.indices))
        else:
            color = color_indices.unique_colors[0]
            options.extend(
                _poly3d_style_options(
                    data,
                    obj,
                    Poly3DStyle(color, None, None, None),
                )
            )
            options.append(_patch_table(group.segments))
        content.extend(_poly3d_addplot(data, group.segments, options))
    return content


@dataclass
class Poly3DColorIndices:
    unique_colors: list[ColorArray]
    indices: list[float]


def _poly3d_base_options(vertex_count: int) -> list[str]:
    return [
        "patch",
        "patch type=polygon",
        f"vertex count={vertex_count}",
        "table/row sep=\\\\",
        "z buffer=sort",
    ]


def _poly3d_collection_options(data: TikzData, obj: Collection) -> list[str]:
    facecolors = get_poly3d_facecolors(obj)
    edgecolors = get_poly3d_edgecolors(obj)
    style = _poly3d_style_at(
        _poly3d_style_arrays(obj, facecolors=facecolors, edgecolors=edgecolors), 0
    )
    face_alpha = _color_alpha(style.facecolor)
    return _poly3d_edge_options(
        data,
        obj,
        style.edgecolor,
        style.linewidth,
        style.linestyle,
    ) + _poly3d_fill_opacity_options(data, face_alpha)


def _poly3d_edge_options(
    data: TikzData,
    obj: Collection,
    edgecolor: ColorArray | None,
    linewidth: float | None,
    linestyle: LineStyle,
) -> list[str]:
    return _poly3d_style_options(
        data,
        obj,
        Poly3DStyle(
            facecolor=None,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        ),
    )


def _poly3d_style_arrays(
    obj: Collection,
    *,
    facecolors: ColorArray | None = None,
    edgecolors: ColorArray | None = None,
) -> Poly3DStyleArrays:
    return Poly3DStyleArrays(
        facecolors=facecolors,
        edgecolors=np.asarray(obj.get_edgecolor()) if edgecolors is None else edgecolors,
        linewidths=np.atleast_1d(obj.get_linewidth()),
        linestyles=cast("Sequence[LineStyle]", obj.get_linestyle()),
    )


def _poly3d_style_at(
    style_arrays: Poly3DStyleArrays,
    index: int,
    *,
    include_facecolor: bool = True,
) -> Poly3DStyle:
    return Poly3DStyle(
        facecolor=_cycle_array(style_arrays.facecolors, index) if include_facecolor else None,
        edgecolor=_cycle_array(style_arrays.edgecolors, index),
        linewidth=_cycle_array(style_arrays.linewidths, index),
        linestyle=_cycle_array(style_arrays.linestyles, index),
    )


def _poly3d_style_options(
    data: TikzData,
    obj: Collection,
    style: Poly3DStyle,
) -> list[str]:
    line_data = mypath.LineData(
        obj=obj,
        ec=style.edgecolor,
        fc=style.facecolor,
        ls=style.linestyle if isinstance(style.linestyle, (str, tuple)) else None,
        lw=style.linewidth,
    )
    draw_options = mypath.get_draw_options(data, line_data)
    if (
        (style.edgecolor is None or style.linewidth == 0)
        and "draw=none" not in draw_options
        and not any(option.startswith("draw=") for option in draw_options)
    ):
        draw_options.append("draw=none")
    if (
        style.facecolor is not None
        and len(style.facecolor) == RGBA_LENGTH
        and style.facecolor[3] == 0
    ):
        draw_options.append("fill opacity=0")
    return draw_options


def _poly3d_explicit_color_indices(colors: list[ColorArray | None]) -> Poly3DColorIndices:
    non_null_colors = [color for color in colors if color is not None]
    if len(non_null_colors) != len(colors):
        return Poly3DColorIndices([], [])

    unique_colors: list[ColorArray] = []
    indices: list[float] = []
    for color in non_null_colors:
        color_index = _matching_color_index(unique_colors, color)
        if color_index is None:
            color_index = len(unique_colors)
            unique_colors.append(color)
        indices.append(float(color_index))
    return Poly3DColorIndices(unique_colors, indices)


def _matching_color_index(colors: list[ColorArray], color: ColorArray) -> int | None:
    for i, unique_color in enumerate(colors):
        if np.allclose(color, unique_color):
            return i
    return None


def _poly3d_colormap_options(data: TikzData, colors: list[ColorArray]) -> list[str]:
    name = f"matplot2tikzpoly{data.custom_colormap_id}"
    data.custom_colormap_id += 1

    color_changes: list[str] = []
    for i, color in enumerate(colors):
        red, green, blue = color[:3]
        ff = data.float_format
        color_changes.append(f"rgb({i}pt)=({red:{ff}},{green:{ff}},{blue:{ff}})")

    return [
        "colormap={" + name + "}{[1pt]\n " + ";\n ".join(color_changes) + "\n}",
        "point meta min=0",
        f"point meta max={len(colors) - 1}",
    ]


def _poly3d_addplot(data: TikzData, segments: list[np.ndarray], options: list[str]) -> list[str]:
    content = [
        "\\addplot3 [\n",
        ",\n".join(options),
        "\n]\n",
        "table [row sep=\\\\] {%\n",
        "x y z\\\\\n",
    ]

    ff = data.float_format
    for segment in segments:
        for x, y, z in segment:
            content.append(f"{x:{ff}} {y:{ff}} {z:{ff}}\\\\\n")
    content.append("};\n")
    return content


def _patch_table(segments: list[np.ndarray], color_data: list[float] | None = None) -> str:
    rows: list[str] = []
    first_index = 0
    for i, segment in enumerate(segments):
        indices: list[int | float] = list(range(first_index, first_index + len(segment)))
        first_index += len(segment)
        if color_data is not None:
            indices.append(color_data[i])
        rows.append(" ".join(str(index) for index in indices) + r"\\")

    key = "patch table with point meta" if color_data is not None else "patch table"
    return key + "={%\n" + "\n".join(rows) + "\n}"


def _group_segments_by_vertex_count(
    segment_colors: list[tuple[np.ndarray, float]],
) -> Iterable[tuple[int, list[tuple[np.ndarray, float]]]]:
    grouped: dict[int, list[tuple[np.ndarray, float]]] = {}
    for segment, color_value in segment_colors:
        grouped.setdefault(len(segment), []).append((segment, float(color_value)))
    return grouped.items()


def _group_segments_only_by_vertex_count(
    segments: list[np.ndarray],
) -> Iterable[tuple[int, list[np.ndarray]]]:
    grouped: dict[int, list[np.ndarray]] = {}
    for segment in segments:
        grouped.setdefault(len(segment), []).append(segment)
    return grouped.items()


def _cycle_array(values: Sequence[T] | np.ndarray | None, index: int) -> T | None:
    if values is None:
        return None
    length = len(values)
    if length == 0:
        return None
    return values[index % length]


def _hashable_color(color: ColorArray | None) -> HashableColor | None:
    if color is None:
        return None
    return tuple(float(value) for value in np.asarray(color).reshape(-1))


def _rgb_color(color: ColorArray | None) -> ColorArray | None:
    if color is None:
        return None
    color_array = np.asarray(color, dtype=float).reshape(-1)
    if len(color_array) < 3:  # noqa: PLR2004
        return None
    return color_array[:3]


def _color_alpha(color: ColorArray | None) -> float | None:
    if color is None:
        return None
    color_array = np.asarray(color, dtype=float).reshape(-1)
    if len(color_array) < RGBA_LENGTH:
        return None
    return float(color_array[3])


def _poly3d_fill_opacity_options(data: TikzData, alpha: float | None) -> list[str]:
    if alpha is None or alpha == 1.0:
        return []
    return [f"fill opacity={alpha:{data.float_format}}"]


def addplot_table(  # noqa: PLR0913
    data: TikzData,
    coordinates: np.ndarray,
    *,
    command: str = "addplot",
    options: Sequence[str] | None = None,
    table_options: Sequence[str] | None = None,
    column_names: Sequence[str] | None = None,
    externalize_min_rows: int = 3,
) -> list[str]:
    """Return a PGFPlots addplot command with inline or external table data."""
    coordinates = _as_2d_array(coordinates)
    if len(coordinates) == 0:
        return []

    content = [f"\\{command}"]
    if options:
        content.append(" [" + ", ".join(options) + "]")
    content.append("\n")
    content.extend(
        table(
            data,
            coordinates,
            table_options=table_options,
            column_names=column_names,
            externalize_min_rows=externalize_min_rows,
        )
    )
    return content


def table(
    data: TikzData,
    coordinates: np.ndarray,
    *,
    table_options: Sequence[str] | None = None,
    column_names: Sequence[str] | None = None,
    externalize_min_rows: int = 3,
) -> list[str]:
    """Return PGFPlots table code for numeric coordinate data."""
    coordinates = _as_2d_array(coordinates)
    if not np.all(np.isfinite(coordinates)) and "unbounded coords=jump" not in (
        data.current_axis_options
    ):
        data.current_axis_options.add("unbounded coords=jump")

    opts = list(table_options or [])
    if data.table_row_sep != "\n":
        opts.append("row sep=" + data.table_row_sep.strip())

    plot_table = table_rows(data, coordinates, column_names=column_names)
    content = []
    if data.externalize_tables and len(coordinates) >= externalize_min_rows:
        filepath, rel_filepath = _files.new_filepath(data, "table", ".dat")
        with filepath.open("w") as f:
            f.write("".join(plot_table))

        if data.externals_search_path is not None:
            opts.append(f"search path={{{data.externals_search_path}}}")

        opts_str = ("[" + ",".join(opts) + "] ") if opts else ""
        content.append(f"table {{{opts_str}}}{{{rel_filepath.as_posix()}}};\n")
        return content

    if opts:
        content.append("table [" + ",".join(opts) + "] {%\n")
    else:
        content.append("table {%\n")
    content.extend(plot_table)
    content.append("};\n")
    return content


def table_rows(
    data: TikzData,
    coordinates: np.ndarray,
    *,
    column_names: Sequence[str] | None = None,
) -> list[str]:
    coordinates = _as_2d_array(coordinates)
    ff = data.float_format
    table_row_sep = data.table_row_sep
    rows = []
    if column_names:
        rows.append(" ".join(column_names) + table_row_sep)
    rows.extend(
        " ".join(_format_value(value, ff) for value in row) + table_row_sep for row in coordinates
    )
    return rows


def _as_2d_array(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 1:
        return coordinates.reshape((-1, 1))
    if coordinates.ndim != 2:  # noqa: PLR2004
        msg = f"Expected 2D coordinate array, got shape {coordinates.shape}."
        raise ValueError(msg)
    return coordinates


def _format_value(value: str | float | np.number, float_format: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable):
        msg = f"Unexpected nested table value {value!r}."
        raise TypeError(msg)
    numeric_value = value
    if not isinstance(numeric_value, (int, float, np.number)):
        msg = f"Unexpected table value {value!r}."
        raise TypeError(msg)
    return f"{numeric_value:{float_format}}"
