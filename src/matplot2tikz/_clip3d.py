"""Clip exported 3D artists to Matplotlib axis limits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.transforms import Transform

Clip3DMode = Literal["none", "hide", "clip"]
POLYGON_RANK = 2


@dataclass
class ClipBox3D:
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    zlim: tuple[float, float]
    xtransform: Transform
    ytransform: Transform
    ztransform: Transform


def clip_box_from_axes(axes: Axes) -> ClipBox3D:
    """Return the 3D clipping box in transformed axis coordinates."""
    xlim = _transformed_limits(axes.xaxis.get_transform(), axes.get_xlim())
    ylim = _transformed_limits(axes.yaxis.get_transform(), axes.get_ylim())
    zlim = _transformed_limits(axes.zaxis.get_transform(), axes.get_zlim())  # type: ignore[attr-defined]
    return ClipBox3D(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        xtransform=axes.xaxis.get_transform(),
        ytransform=axes.yaxis.get_transform(),
        ztransform=axes.zaxis.get_transform(),  # type: ignore[attr-defined]
    )


def points_inside(points: np.ndarray, box: ClipBox3D) -> np.ndarray:
    """Return a mask for points inside the 3D clipping box."""
    transformed = transform_points(points, box)
    return _points_inside_transformed(transformed, box)


def clip_line_to_box(points: np.ndarray, box: ClipBox3D, mode: Clip3DMode) -> list[np.ndarray]:
    """Return line segments hidden or clipped to the 3D clipping box.

    Line clipping uses a Liang-Barsky style parametric interval against the six
    axis-aligned box planes after applying the Matplotlib axis transforms.
    """
    transformed = transform_points(points, box)
    segments = _line_segments(transformed, box, mode)
    return [inverse_transform_points(segment, box) for segment in segments]


def clip_polygon_to_box(poly: np.ndarray, box: ClipBox3D, mode: Clip3DMode) -> np.ndarray:
    """Return a polygon hidden or clipped to the 3D clipping box.

    Polygon clipping uses Sutherland-Hodgman clipping against each box plane
    after applying the Matplotlib axis transforms.
    """
    transformed = transform_points(poly, box)
    if mode == "hide":
        if np.all(_points_inside_transformed(transformed, box)):
            return poly
        return np.empty((0, 3), dtype=float)
    if mode == "clip":
        clipped = _clip_polygon_transformed(transformed, box)
        return inverse_transform_points(clipped, box) if len(clipped) else clipped
    return poly


def transform_points(points: np.ndarray, box: ClipBox3D) -> np.ndarray:
    """Transform data coordinates into axis-scale coordinates for clipping."""
    points = np.asarray(points, dtype=float)
    return np.column_stack(
        [
            box.xtransform.transform(points[:, 0]),
            box.ytransform.transform(points[:, 1]),
            box.ztransform.transform(points[:, 2]),
        ]
    )


def inverse_transform_points(points: np.ndarray, box: ClipBox3D) -> np.ndarray:
    """Transform clipped axis-scale coordinates back into data coordinates."""
    points = np.asarray(points, dtype=float)
    return np.column_stack(
        [
            box.xtransform.inverted().transform(points[:, 0]),
            box.ytransform.inverted().transform(points[:, 1]),
            box.ztransform.inverted().transform(points[:, 2]),
        ]
    )


def _transformed_limits(transform: Transform, limits: tuple[float, float]) -> tuple[float, float]:
    transformed = np.asarray(transform.transform(limits), dtype=float)
    finite = transformed[np.isfinite(transformed)]
    if len(finite) != 2:  # noqa: PLR2004
        return (np.nan, np.nan)
    return (float(np.min(finite)), float(np.max(finite)))


def _points_inside_transformed(points: np.ndarray, box: ClipBox3D) -> np.ndarray:
    return np.logical_and.reduce(
        (
            np.isfinite(points).all(axis=1),
            points[:, 0] >= box.xlim[0],
            points[:, 0] <= box.xlim[1],
            points[:, 1] >= box.ylim[0],
            points[:, 1] <= box.ylim[1],
            points[:, 2] >= box.zlim[0],
            points[:, 2] <= box.zlim[1],
        )
    )


def _line_segments(points: np.ndarray, box: ClipBox3D, mode: Clip3DMode) -> list[np.ndarray]:
    if len(points) < 2:  # noqa: PLR2004
        return []

    inside = _points_inside_transformed(points, box)
    segments: list[np.ndarray] = []
    previous_edge_visible = False
    for i, (p0, p1) in enumerate(pairwise(points)):
        clipped = np.empty((0, 3), dtype=float)
        if mode == "hide":
            if inside[i] and inside[i + 1]:
                clipped = np.asarray([p0, p1], dtype=float)
        elif mode == "clip":
            clipped = _clip_line_segment_transformed(p0, p1, box)
        if len(clipped):
            _append_line_segment(segments, clipped, merge=previous_edge_visible)
            previous_edge_visible = True
        else:
            previous_edge_visible = False
    return segments


def _append_line_segment(segments: list[np.ndarray], segment: np.ndarray, *, merge: bool) -> None:
    if len(segment) < 2:  # noqa: PLR2004
        return
    if merge and segments and np.allclose(segments[-1][-1], segment[0]):
        segments[-1] = np.vstack([segments[-1], segment[1:]])
        return
    segments.append(segment)


def _clip_line_segment_transformed(p0: np.ndarray, p1: np.ndarray, box: ClipBox3D) -> np.ndarray:
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return np.empty((0, 3), dtype=float)

    direction = p1 - p0
    t0 = 0.0
    t1 = 1.0
    for axis, (lower, upper) in enumerate((box.xlim, box.ylim, box.zlim)):
        if not np.isfinite([lower, upper]).all():
            return np.empty((0, 3), dtype=float)
        if direction[axis] == 0:
            if p0[axis] < lower or p0[axis] > upper:
                return np.empty((0, 3), dtype=float)
            continue

        t_lower = (lower - p0[axis]) / direction[axis]
        t_upper = (upper - p0[axis]) / direction[axis]
        t_enter = min(t_lower, t_upper)
        t_exit = max(t_lower, t_upper)
        t0 = max(t0, t_enter)
        t1 = min(t1, t_exit)
        if t0 > t1:
            return np.empty((0, 3), dtype=float)

    start = p0 + t0 * direction
    end = p0 + t1 * direction
    if np.allclose(start, end):
        return np.empty((0, 3), dtype=float)
    return np.asarray([start, end], dtype=float)


def _clip_polygon_transformed(poly: np.ndarray, box: ClipBox3D) -> np.ndarray:
    if _is_degenerate_polygon(poly):
        return np.empty((0, 3), dtype=float)

    for axis, value, keep_greater in (
        (0, box.xlim[0], True),
        (0, box.xlim[1], False),
        (1, box.ylim[0], True),
        (1, box.ylim[1], False),
        (2, box.zlim[0], True),
        (2, box.zlim[1], False),
    ):
        if not np.isfinite(value):
            return np.empty((0, 3), dtype=float)
        poly = _clip_polygon_against_plane(poly, axis, value, keep_greater=keep_greater)
        if len(poly) < 3:  # noqa: PLR2004
            return np.empty((0, 3), dtype=float)

    if _is_degenerate_polygon(poly):
        return np.empty((0, 3), dtype=float)
    return poly


def _clip_polygon_against_plane(
    poly: np.ndarray, axis: int, value: float, *, keep_greater: bool
) -> np.ndarray:
    clipped = []
    previous = poly[-1]
    previous_inside = _inside_plane(previous, axis, value, keep_greater=keep_greater)

    for current in poly:
        current_inside = _inside_plane(current, axis, value, keep_greater=keep_greater)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            if not np.isclose(denominator, 0.0):
                t = np.clip((value - previous[axis]) / denominator, 0.0, 1.0)
                clipped.append(previous + t * (current - previous))
        if current_inside:
            clipped.append(current)

        previous = current
        previous_inside = current_inside

    return np.asarray(clipped, dtype=float) if clipped else np.empty((0, 3), dtype=float)


def _inside_plane(point: np.ndarray, axis: int, value: float, *, keep_greater: bool) -> bool:
    return bool(point[axis] >= value if keep_greater else point[axis] <= value)


def _is_degenerate_polygon(poly: np.ndarray, tol: float = 1e-12) -> bool:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 3 or len(poly) < 3:  # noqa: PLR2004
        return True
    if not np.isfinite(poly).all():
        return True

    centered = poly - poly.mean(axis=0)
    scale = np.linalg.norm(centered, axis=1).max(initial=0)
    if scale == 0:
        return True
    return bool(np.linalg.matrix_rank(centered, tol=tol * scale) < POLYGON_RANK)
