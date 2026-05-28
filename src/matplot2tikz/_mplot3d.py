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
from ._clip3d import (
    ClipBox3D,
    clip_box_from_axes,
    clip_line_to_box,
    clip_polygon_to_box,
    points_inside,
)
from ._util import get_legend_text, has_legend

if TYPE_CHECKING:
    from matplotlib.collections import Collection, LineCollection, PathCollection
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from ._tikzdata import TikzData

MIN_POLYGON_VERTICES = 3
RGBA_LENGTH = 4
HashableColor = tuple[float, ...]
LineStyle = str | tuple[float, Sequence[float] | None] | None
T = TypeVar("T")


@dataclass
class Poly3DStyle:
    facecolor: np.ndarray | None
    edgecolor: np.ndarray | None
    linewidth: float | None
    linestyle: LineStyle


@dataclass
class Poly3DStyleArrays:
    facecolors: np.ndarray | None
    edgecolors: np.ndarray
    linewidths: Sequence[float | None] | np.ndarray
    linestyles: Sequence[LineStyle]


@dataclass
class Poly3DColorGroup:
    segments: list[np.ndarray]
    colors: list[np.ndarray | None]
    style_options: list[str]
    vertex_count: int


@dataclass
class Quiver3DData:
    coordinates: np.ndarray
    options: list[str]


def is_quiver3d_collection(obj: Collection) -> bool:
    # Since a quiver3d collection is represented as a Line3DCollection with a
    # specific structure of segments and uniform style, both must be checked.
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

    # Matplotlib <= 3.10 stores Poly3DCollection vertices in _vec/_segslices.
    if hasattr(obj, "_vec") and hasattr(obj, "_segslices"):
        vec = obj._vec  # noqa: SLF001
        segslices = obj._segslices  # noqa: SLF001
        return [np.asarray(vec[:3, segslice].T, dtype=float) for segslice in segslices]

    # Matplotlib 3.11.0rc1 switched to padded _faces plus _invalid_vertices.
    faces = getattr(obj, "_faces", None)
    if faces is None:
        msg = "Poly3DCollection has neither _vec/_segslices nor _faces."
        raise AttributeError(msg)

    faces = np.asarray(faces, dtype=float)
    if faces.ndim != 3 or faces.shape[-1] != 3:  # noqa: PLR2004
        msg = f"Expected Poly3DCollection._faces with shape (n, m, 3), got {faces.shape}."
        raise ValueError(msg)

    invalid_vertices = np.asarray(getattr(obj, "_invalid_vertices", False), dtype=bool)
    if invalid_vertices.ndim == 0:
        if bool(invalid_vertices):
            return []
        return [np.asarray(face, dtype=float) for face in faces]

    return [
        np.asarray(face[~invalid_mask], dtype=float)
        for face, invalid_mask in zip(faces, invalid_vertices, strict=False)
    ]


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
    segments = _line_segments_for_export(data, coordinates)
    if not segments:
        return []

    addplot_options = _line2d._get_line2d_options(data, obj)  # noqa: SLF001
    legend_text = get_legend_text(obj)
    if legend_text is None and obj.axes is not None and has_legend(obj.axes):
        addplot_options.append("forget plot")

    content = []
    for segment in segments:
        content.append("\\addplot3 ")
        if addplot_options:
            content.append("[{}]\n".format(", ".join(addplot_options)))
        content.extend(table(data, segment))

    if legend_text is not None:
        content.append(f"\\addlegendentry{{{legend_text}}}\n")

    return content


def draw_quiver3d(data: TikzData, obj: Line3DCollection) -> list[str]:
    """Return PGFPlots code for a 3D quiver plot."""
    if data.clip_3d == "clip":
        return draw_line3dcollection(data, obj)

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
        for export_segment in _line_segments_for_export(data, segment):
            content.extend(
                addplot_table(
                    data,
                    export_segment,
                    command="addplot3",
                    options=options,
                    externalize_min_rows=3,
                )
            )

    return content


def _line_segments_for_export(data: TikzData, points: np.ndarray) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    clip_box = _clip_box(data)
    if clip_box is None:
        return [points]
    return clip_line_to_box(points, clip_box, data.clip_3d)


def _clip_box(data: TikzData) -> ClipBox3D | None:
    if data.clip_3d == "none" or data.current_mpl_axes is None:
        return None
    return clip_box_from_axes(data.current_mpl_axes)


def _quiver3d_data(
    data: TikzData,
    obj: LineCollection,
    segments: list[np.ndarray],
    style_arrays: Poly3DStyleArrays,
) -> Quiver3DData | None:
    quiver_coordinates = _quiver3d_coordinates(segments)
    if quiver_coordinates is None:
        return None
    quiver_coordinates = _clip_quiver_coordinates(data, quiver_coordinates)
    if len(quiver_coordinates) == 0:
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


def _clip_quiver_coordinates(data: TikzData, coordinates: np.ndarray) -> np.ndarray:
    clip_box = _clip_box(data)
    if clip_box is None:
        return coordinates
    tails = coordinates[:, :3]
    tips = coordinates[:, :3] + coordinates[:, 3:6]
    mask = points_inside(tails, clip_box) & points_inside(tips, clip_box)
    return coordinates[mask]


def draw_path3dcollection(data: TikzData, obj: PathCollection) -> list[str]:
    """Return PGFPlots code for 3D scatter/path collections."""
    offsets = get_offsets3d(obj)
    mask = _scatter_mask_for_export(data, offsets)
    if mask is None:
        return _draw_path3dcollection(data, obj, offsets)
    if not np.any(mask):
        return []
    return _draw_clipped_path3dcollection(data, obj, offsets, mask)


def _draw_path3dcollection(data: TikzData, obj: PathCollection, offsets: np.ndarray) -> list[str]:
    pcd = mypath.make_pathcollection_data(
        data,
        obj,
        mypath.PathCollectionCoordinates(
            offsets=offsets,
            labels=["x", "y", "z"],
            table_options=["x=x", "y=y", "z=z"],
            is_contour=False,
            command="\\addplot3",
        ),
    )
    return mypath.draw_pathcollection_data(data, pcd)


def _scatter_mask_for_export(data: TikzData, offsets: np.ndarray) -> np.ndarray | None:
    clip_box = _clip_box(data)
    if clip_box is None:
        return None
    return points_inside(offsets, clip_box)


def _draw_clipped_path3dcollection(
    data: TikzData, obj: PathCollection, offsets: np.ndarray, mask: np.ndarray
) -> list[str]:
    obj3d = cast("Path3DCollection", obj)
    old_offsets = obj3d._offsets3d  # noqa: SLF001
    old_array = obj.get_array()
    old_sizes = obj.get_sizes()
    old_edgecolors = obj.get_edgecolors()  # type: ignore[attr-defined]
    old_facecolors = obj.get_facecolors()  # type: ignore[attr-defined]
    try:
        obj3d._offsets3d = tuple(offsets[mask].T)  # noqa: SLF001
        if old_array is not None and len(old_array) == len(mask):
            obj.set_array(np.asarray(old_array)[mask])
        if len(old_sizes) == len(mask):
            obj.set_sizes(old_sizes[mask])
        if len(old_edgecolors) == len(mask):
            obj.set_edgecolors(old_edgecolors[mask])  # type: ignore[attr-defined]
        if len(old_facecolors) == len(mask):
            obj.set_facecolors(old_facecolors[mask])  # type: ignore[attr-defined]
        return _draw_path3dcollection(data, obj, offsets[mask])
    finally:
        obj3d._offsets3d = old_offsets  # noqa: SLF001
        obj.set_array(old_array)
        obj.set_sizes(old_sizes)
        obj.set_edgecolors(old_edgecolors)  # type: ignore[attr-defined]
        obj.set_facecolors(old_facecolors)  # type: ignore[attr-defined]


def draw_poly3dcollection(data: TikzData, obj: Collection) -> list[str]:
    """Returns PGFPlots patch-plot code for 3D polygon collections."""
    indexed_segments = _poly_segments_for_export(data, get_poly3d_segments(obj))
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


def _poly_segments_for_export(
    data: TikzData, segments: list[np.ndarray]
) -> list[tuple[int, np.ndarray]]:
    clip_box = _clip_box(data)
    if clip_box is None:
        return [
            (index, vertices)
            for index, vertices in enumerate(segments)
            if len(vertices) >= MIN_POLYGON_VERTICES
        ]

    indexed_segments: list[tuple[int, np.ndarray]] = []
    for index, vertices in enumerate(segments):
        if len(vertices) < MIN_POLYGON_VERTICES:
            continue
        if data.clip_3d == "clip" and np.all(points_inside(vertices, clip_box)):
            indexed_segments.append((index, vertices))
            continue

        clipped = clip_polygon_to_box(vertices, clip_box, data.clip_3d)
        if len(clipped) >= MIN_POLYGON_VERTICES:
            indexed_segments.append((index, clipped))
    return indexed_segments


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
        for segment in _split_contour3d_vertices_for_export(data, vertices, codes):
            content.extend(addplot_table(data, segment, command="addplot3", options=options))
    return content


def _draw_contour3d_filled(data: TikzData, obj: Collection, facecolors: np.ndarray) -> list[str]:
    data.pgfplots_libs.add("patchplots")
    content: list[str] = []
    style_arrays = _poly3d_style_arrays(obj, facecolors=facecolors)
    for i, (vertices, codes) in enumerate(get_contour3d_verts_and_codes(obj)):
        segments = _contourf_segments_for_export(data, vertices, codes)
        if not segments:
            continue
        style = _poly3d_style_at(style_arrays, i)
        for vertex_count, grouped_segments in _group_segments_only_by_vertex_count(segments):
            options = _poly3d_base_options(data, vertex_count)
            options.extend(_poly3d_style_options(data, obj, style))
            options.append(_patch_table(data, grouped_segments))
            content.extend(_poly3d_addplot(data, grouped_segments, options))
    return content


def _split_contour3d_vertices_for_export(
    data: TikzData, vertices: np.ndarray, codes: np.ndarray | None
) -> list[np.ndarray]:
    segments = _split_contour3d_vertices(vertices, codes)
    clip_box = _clip_box(data)
    if clip_box is None:
        return [segment for segment in segments if len(segment) >= 2]  # noqa: PLR2004

    export_segments = []
    for segment in segments:
        export_segments.extend(clip_line_to_box(segment, clip_box, data.clip_3d))
    return export_segments


def _contourf_segments_for_export(
    data: TikzData, vertices: np.ndarray, codes: np.ndarray | None
) -> list[np.ndarray]:
    segments = _split_contour3d_vertices(vertices, codes)
    clip_box = _clip_box(data)
    if clip_box is None:
        return [segment for segment in segments if len(segment) >= MIN_POLYGON_VERTICES]

    export_segments = []
    for segment in segments:
        clipped = clip_polygon_to_box(segment, clip_box, data.clip_3d)
        if len(clipped) >= MIN_POLYGON_VERTICES:
            export_segments.append(clipped)
    return export_segments


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

    # Triangulate if shading is enabled and any polygon has more than 4 vertices, since PGFPlots
    # only supports shading for triangles and bilinear shading for quads.
    if data.shader != "none" and any(len(vertices) > 4 for vertices, _ in segment_colors):  # noqa: PLR2004
        segment_colors = [
            (segment, color_value)
            for vertices, color_value in segment_colors
            for segment in _triangulate_polygon(vertices)
        ]

    for vertex_count, group in _group_segments_by_vertex_count(segment_colors):
        grouped_segments = [segment for segment, _ in group]
        grouped_color_data = [color_value for _, color_value in group]
        options = _poly3d_base_options(data, vertex_count, use_shader=data.shader != "none")
        options.extend(_poly3d_collection_options(data, obj))

        if data.shader != "none":
            options.append(r"point meta=\thisrow{meta}")
            options.append(_patch_table(data, grouped_segments))
            content.extend(
                _poly3d_addplot(
                    data,
                    grouped_segments,
                    options,
                    point_meta_segments=_poly3d_z_point_meta(grouped_segments),
                )
            )
        else:
            options.append(_patch_table(data, grouped_segments, grouped_color_data))
            content.extend(_poly3d_addplot(data, grouped_segments, options))

    return content


def _triangulate_polygon(vertices: np.ndarray) -> list[np.ndarray]:
    if len(vertices) < MIN_POLYGON_VERTICES:
        return []
    if len(vertices) == MIN_POLYGON_VERTICES:
        return [vertices]
    return [
        np.asarray([vertices[0], vertices[i], vertices[i + 1]], dtype=float)
        for i in range(1, len(vertices) - 1)
    ]


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
        options = _poly3d_base_options(data, group.vertex_count)
        options.extend(group.style_options)
        color_indices = _poly3d_explicit_color_indices(group.colors)
        if not color_indices.unique_colors:
            options.append(_patch_table(data, group.segments))
        elif len(color_indices.unique_colors) > 1:
            options.extend(_poly3d_colormap_options(data, color_indices.unique_colors))
            options.append(_patch_table(data, group.segments, color_indices.indices))
        else:
            color = color_indices.unique_colors[0]
            options.extend(
                _poly3d_style_options(
                    data,
                    obj,
                    Poly3DStyle(color, None, None, None),
                )
            )
            options.append(_patch_table(data, group.segments))
        content.extend(_poly3d_addplot(data, group.segments, options))
    return content


@dataclass
class Poly3DColorIndices:
    unique_colors: list[np.ndarray]
    indices: list[float]


def _poly3d_base_options(
    data: TikzData, vertex_count: int, *, use_shader: bool = False
) -> list[str]:
    options = [
        "patch",
        *_poly3d_patch_type_options(use_shader, vertex_count),
        "table/row sep=\\\\",
        "z buffer=sort",
    ]
    if use_shader:
        options.append(_shader_option(data.shader))
    return options


def _poly3d_patch_type_options(use_shader: bool, vertex_count: int) -> list[str]:  # noqa: FBT001
    if use_shader and vertex_count == 3:  # noqa: PLR2004
        return ["patch type=triangle"]
    if use_shader and vertex_count == 4:  # noqa: PLR2004
        return ["patch type=bilinear"]
    return ["patch type=polygon", f"vertex count={vertex_count}"]


def _shader_option(shader: str) -> str:
    shader = shader.removeprefix(",").strip()
    return shader if shader.startswith("shader=") else f"shader={shader}"


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
    edgecolor: np.ndarray | None,
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
    facecolors: np.ndarray | None = None,
    edgecolors: np.ndarray | None = None,
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


def _poly3d_explicit_color_indices(colors: list[np.ndarray | None]) -> Poly3DColorIndices:
    non_null_colors = [color for color in colors if color is not None]
    if len(non_null_colors) != len(colors):
        return Poly3DColorIndices([], [])

    unique_colors: list[np.ndarray] = []
    indices: list[float] = []
    for color in non_null_colors:
        color_index = _matching_color_index(unique_colors, color)
        if color_index is None:
            color_index = len(unique_colors)
            unique_colors.append(color)
        indices.append(float(color_index))
    return Poly3DColorIndices(unique_colors, indices)


def _matching_color_index(colors: list[np.ndarray], color: np.ndarray) -> int | None:
    for i, unique_color in enumerate(colors):
        if np.allclose(color, unique_color):
            return i
    return None


def _poly3d_colormap_options(data: TikzData, colors: list[np.ndarray]) -> list[str]:
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


def _poly3d_z_point_meta(segments: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(segment[:, 2], dtype=float) for segment in segments]


def _poly3d_addplot(
    data: TikzData,
    segments: list[np.ndarray],
    options: list[str],
    *,
    point_meta_segments: list[np.ndarray] | None = None,
) -> list[str]:
    column_names = "x y z meta" if point_meta_segments is not None else "x y z"
    content = [
        "\\addplot3 [\n",
        ",\n".join(options),
        "\n]\n",
        "table [row sep=\\\\] {%\n",
        column_names + "\\\\\n",
    ]

    ff = data.float_format
    if point_meta_segments is None:
        for segment in segments:
            for x, y, z in segment:
                content.append(f"{x:{ff}} {y:{ff}} {z:{ff}}\\\\\n")
    else:
        for segment, point_meta in zip(segments, point_meta_segments, strict=True):
            for (x, y, z), meta in zip(segment, point_meta, strict=True):
                content.append(f"{x:{ff}} {y:{ff}} {z:{ff}} {meta:{ff}}\\\\\n")
    content.append("};\n")
    return content


def _patch_table(
    data: TikzData, segments: list[np.ndarray], color_data: list[float] | None = None
) -> str:
    rows: list[str] = []
    first_index = 0
    for i, segment in enumerate(segments):
        indices = [str(index) for index in range(first_index, first_index + len(segment))]
        first_index += len(segment)
        if color_data is not None:
            indices.append(_format_value(color_data[i], data.float_format))
        rows.append(" ".join(indices) + r"\\")

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


def _hashable_color(color: np.ndarray | None) -> HashableColor | None:
    if color is None:
        return None
    return tuple(float(value) for value in np.asarray(color).reshape(-1))


def _rgb_color(color: np.ndarray | None) -> np.ndarray | None:
    if color is None:
        return None
    color_array = np.asarray(color, dtype=float).reshape(-1)
    if len(color_array) < 3:  # noqa: PLR2004
        return None
    return color_array[:3]


def _color_alpha(color: np.ndarray | None) -> float | None:
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
        content.append(f"table {opts_str}{{{rel_filepath.as_posix()}}};\n")
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
