from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D, art3d

if TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.figure import FigureBase


STEP_DRAW_STYLES = ["steps-pre", "steps-post", "steps-mid"]


def initial_data() -> np.ndarray:
    return np.array([])


def initial_axis_limits() -> np.ndarray:
    return np.array([-np.inf, np.inf])


@dataclass
class CleanFigureData:
    fig: FigureBase
    axes: Axes | Axes3D
    target_resolution: int | list[int] | np.ndarray
    scale_precision: float
    data: np.ndarray = field(default_factory=initial_data)
    visual_data: np.ndarray | None = None
    x_lim: np.ndarray = field(default_factory=initial_axis_limits)
    y_lim: np.ndarray = field(default_factory=initial_axis_limits)
    has_lines: bool | None = None
    has_markers: bool | None = None


def clean_figure(
    fig: FigureBase | None = None,
    target_resolution: int | list[int] | np.ndarray = 600,
    scale_precision: float = 1.0,
) -> None:
    r"""Cleans figure as a preparation for tikz export.

    This will minimize the number of points required for the tikz figure.
    If the figure has subplots, it will recursively clean then up.

    Note that this function modifies the figure directly (impure function).

    :param fig: Matplotlib figure handle (Default value = None)
    :param target_resolution: target resolution of final figure in PPI.
                              If a scalar integer is provided, it is assumed to be
                              square in both axis.  If a list or an np.array is
                              provided, it is interpreted as [H, W].
                              By default 600
    :param scale_precision: scalar value indicating precision when scaling down.
                           By default 1

    Examples:
    --------
        1. 2D lineplot
        ```python
            from matplot2tikz import get_tikz_code, cleanfigure

            x = np.linspace(1, 100, 20)
            y = np.linspace(1, 100, 20)

            with plt.rc_context(rc=RC_PARAMS):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.plot(x, y)
                ax.set_ylim([20, 80])
                ax.set_xlim([20, 80])
                raw = get_tikz_code()

                clean_figure(fig)
                clean = get_tikz_code()

                # Use number of lines to test if it worked.
                # the baseline (raw) should have 20 points
                # the clean version (clean) should have 2 points
                # the difference in line numbers should therefore be 2
                num_lines_raw = raw.count("\n")
                num_lines_clean = clean.count("\n")
                print("number of tikz lines saved", num_lines_raw - num_lines_clean)
        ```

        2. 3D lineplot
        ```python
            from matplot2tikz import get_tikz_code, cleanfigure

            theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
            z = np.linspace(-2, 2, 100)
            r = z ** 2 + 1
            x = r * np.sin(theta)
            y = r * np.cos(theta)

            with plt.rc_context(rc=RC_PARAMS):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(x, y, z)
                ax.set_xlim([-2, 2])
                ax.set_ylim([-2, 2])
                ax.set_zlim([-2, 2])
                ax.view_init(30, 30)
                raw = get_tikz_code(fig)

                clean_figure(fig)
                clean = get_tikz_code()

                # Use number of lines to test if it worked.
                num_lines_raw = raw.count("\n")
                num_lines_clean = clean.count("\n")
                assert num_lines_raw - num_lines_clean == 14
        ```
    """
    if fig is None or fig == "gcf":
        fig = plt.gcf()
    _recursive_cleanfigure(
        fig, target_resolution=target_resolution, scale_precision=scale_precision
    )


def _recursive_cleanfigure(
    obj: Artist, target_resolution: int | list[int] | np.ndarray, scale_precision: float
) -> None:
    """Recursively visit child objects and clean them.

    :param obj: object
    :param target_resolution: target resolution of final figure in PPI.
        If a scalar integer is provided, it is assumed to be square in both axis.
        If a list or an np.array is provided, it is interpreted as [H, W].
    :param scale_precision: scalar value indicating precision when scaling down.
    """
    for child in obj.get_children():
        if isinstance(child, Axes):
            # Note: containers contain Patches but are not child objects.
            # This is a problem because a bar plot creates a Barcontainer.
            _clean_containers(child)
            _recursive_cleanfigure(child, target_resolution, scale_precision)
        elif isinstance(child, Axes3D):
            _clean_containers(child)
            _recursive_cleanfigure(child, target_resolution, scale_precision)
        elif isinstance(child, (Line2D, art3d.Line3D)):
            _cleanline(child, target_resolution, scale_precision)
        elif isinstance(child, (PathCollection, art3d.Path3DCollection)):
            _clean_collections(child, target_resolution, scale_precision)
        elif isinstance(child, mpl.collections.LineCollection):
            warnings.warn(
                "Cleaning Line Collections (scatter plot) is not supported yet.", stacklevel=2
            )
        elif isinstance(child, art3d.Line3DCollection):
            warnings.warn("Cleaning Line3DCollection is not supported yet.", stacklevel=2)
        elif isinstance(child, art3d.Poly3DCollection):
            warnings.warn("Cleaning Poly3DCollections is not supported yet.", stacklevel=2)
        elif isinstance(child, QuadContourSet):
            warnings.warn("Cleaning QuadContourSet is not supported yet.", stacklevel=2)
        # The following objects are passed:
        # Spine, AxesImage, Patch


def _clean_containers(axes: Axes) -> None:
    """Containers are not children of axes. They need to be visited separately."""
    for container in axes.containers:
        if isinstance(container, BarContainer):
            warnings.warn("Cleaning Bar Container (bar plot) is not supported yet.", stacklevel=2)


def _cleanline(
    linehandle: Line2D | art3d.Line3D,
    target_resolution: int | list[int] | np.ndarray,
    scale_precision: float,
) -> None:
    """Clean a 2D or 3D Line plot figure."""
    axes = linehandle.axes
    if axes is None:
        return
    figure = axes.figure
    if figure is None:
        return
    cfd = CleanFigureData(
        fig=figure,
        axes=axes,
        target_resolution=target_resolution,
        scale_precision=scale_precision,
    )

    if _is_step(linehandle):
        warnings.warn("step plot simplification not yet implemented.", stacklevel=2)
    else:
        cfd.data = _get_line_data(linehandle)
        cfd.x_lim, cfd.y_lim = _get_visual_limits(cfd.axes)
        cfd.visual_data = _get_visual_data(cfd.axes, cfd.data)
        cfd.has_lines = _line_has_lines(linehandle)

        cfd.data = _prune_outside_box(cfd)
        cfd.visual_data = _get_visual_data(cfd.axes, cfd.data)

        if not isinstance(linehandle, art3d.Line3D):
            cfd.visual_data = _move_points_closer(cfd.x_lim, cfd.y_lim, cfd.visual_data)

        cfd.has_markers = linehandle.get_marker() != "None"
        cfd.has_lines = linehandle.get_linestyle() != "None"
        data = _simplify_line(cfd)
        data = _limit_precision(cfd.axes, data, cfd.scale_precision)
        _update_line_data(linehandle, data)


def _clean_collections(
    collection: PathCollection | art3d.Path3DCollection,
    target_resolution: int | list[int] | np.ndarray,
    scale_precision: float,
) -> None:
    """Clean a 2D or 3D collection, i.e., scatter plot."""
    axes = collection.axes
    if axes is None:
        return
    figure = axes.figure
    if figure is None:
        return
    cfd = CleanFigureData(
        fig=figure,
        axes=axes,
        target_resolution=target_resolution,
        scale_precision=scale_precision,
    )

    cfd.data = _get_collection_data(collection)
    cfd.x_lim, cfd.y_lim = _get_visual_limits(cfd.axes)
    cfd.visual_data = _get_visual_data(cfd.axes, cfd.data)

    cfd.has_lines = True

    cfd.data = _prune_outside_box(cfd)
    cfd.visual_data = _get_visual_data(cfd.axes, cfd.data)

    if not isinstance(collection, art3d.Path3DCollection):
        cfd.visual_data = _move_points_closer(cfd.x_lim, cfd.y_lim, cfd.visual_data)
        cfd.visual_data = _get_visual_data(cfd.axes, cfd.visual_data)

    cfd.has_markers = True
    cfd.has_lines = False
    data = _simplify_line(cfd)
    data = _limit_precision(cfd.axes, data, cfd.scale_precision)
    collection.set_offsets(data)


def _is_step(linehandle: Line2D | art3d.Line3D) -> bool:
    """Check if plot is a step plot."""
    return linehandle._drawstyle in STEP_DRAW_STYLES  # type: ignore[union-attr]  # noqa: SLF001


def _get_visual_limits(axhandle: Axes) -> tuple[np.ndarray, np.ndarray]:
    """Returns the visual representation of the axis limits (x & y).

    Respecting possible log_scaling and projection into the image plane.
    """
    x_lim = np.array(axhandle.get_xlim())
    if _ax_is_xlog(axhandle):
        x_lim = np.log10(x_lim)

    y_lim = np.array(axhandle.get_ylim())
    if _ax_is_ylog(axhandle):
        y_lim = np.log10(y_lim)

    if isinstance(axhandle, Axes3D):
        z_lim = np.array(axhandle.get_zlim())
        if _ax_is_zlog(axhandle):
            z_lim = np.log10(z_lim)

        p = _get_projection_matrix(axhandle)

        corners = _corners3d(x_lim, y_lim, z_lim)

        # Add the canonical 4th dimension
        corners = np.concatenate([corners, np.ones((8, 1))], axis=1)
        corners_projected = p @ corners.T

        x_corners = corners_projected[0, :] / corners_projected[3, :]
        y_corners = corners_projected[1, :] / corners_projected[3, :]

        x_lim = np.array([np.min(x_corners), np.max(x_corners)])
        y_lim = np.array([np.min(y_corners), np.max(y_corners)])

    return x_lim, y_lim


def _replace_data_with_nan(data: np.ndarray, id_replace: np.ndarray) -> np.ndarray:
    """Replaces data at id_replace with NaNs."""
    if _isempty(id_replace):
        return data

    if data.shape[1] == 3:  # noqa: PLR2004
        x_data, y_data, z_data = _split_data_3d(data)
    else:
        x_data, y_data = _split_data_2d(data)

    x_data[id_replace] = np.nan
    y_data[id_replace] = np.nan
    if data.shape[1] == 3:  # noqa: PLR2004
        z_data = z_data.copy()
        z_data[id_replace] = np.nan
        return _stack_data_3d(x_data, y_data, z_data)
    return _stack_data_2d(x_data, y_data)


def _remove_data(data: np.ndarray, id_remove: np.ndarray) -> np.ndarray:
    """Remove data at id_remove."""
    if _isempty(id_remove):
        return data

    if data.shape[1] == 3:  # noqa: PLR2004
        x_data, y_data, z_data = _split_data_3d(data)
    else:
        x_data, y_data = _split_data_2d(data)

    x_data = np.delete(x_data, id_remove, axis=0)
    y_data = np.delete(y_data, id_remove, axis=0)
    if data.shape[1] == 3:  # noqa: PLR2004
        z_data = np.delete(z_data, id_remove, axis=0)
        return _stack_data_3d(x_data, y_data, z_data)
    return _stack_data_2d(x_data, y_data)


def _update_line_data(linehandle: Line2D | art3d.Line3D, data: np.ndarray) -> None:
    if isinstance(linehandle, art3d.Line3D):
        x_data, y_data, z_data = _split_data_3d(data)
        # I don't understand why I need to set both to get tikz code reduction to work
        linehandle.set_data_3d(x_data, y_data, z_data)
        linehandle.set_data(x_data, y_data)
    else:
        x_data, y_data = _split_data_2d(data)
        linehandle.set_xdata(x_data)
        linehandle.set_ydata(y_data)


def _split_data_2d(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert data to 2 different arrays."""
    x_data, y_data = np.split(data, 2, axis=1)
    return x_data.reshape((-1,)), y_data.reshape((-1,))


def _stack_data_2d(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """Stack 2 arrays to one big one."""
    return np.stack([x_data, y_data], axis=1)


def _split_data_3d(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert data to 3 different arrays."""
    x_data, y_data, z_data = np.split(data, 3, axis=1)
    return x_data.reshape((-1,)), y_data.reshape((-1,)), z_data.reshape((-1,))


def _stack_data_3d(x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray) -> np.ndarray:
    """Stack 3 arrays to one big one."""
    return np.stack([x_data, y_data, z_data], axis=1)


def _remove_nans(data: np.ndarray) -> np.ndarray:
    """Removes superfluous NaNs in the data.

    I.e., those at the end/beginning of the data and consecutive ones.
    """
    id_nan = np.any(np.isnan(data), axis=1)
    id_remove = np.argwhere(id_nan).reshape((-1,))
    if not _isempty(id_remove):
        id_remove = id_remove[
            np.concatenate([np.diff(id_remove, axis=0) == 1, np.array([False]).reshape((-1,))])
        ]

    id_first = np.argwhere(np.logical_not(id_nan))[0]
    id_last = np.argwhere(np.logical_not(id_nan))[-1]

    if _isempty(id_first):
        # remove entire data
        id_remove = np.arange(len(data))
    else:
        id_remove = np.concatenate(
            [np.arange(0, id_first[0]), id_remove, np.arange(id_last[0] + 1, len(data))]
        )
    return np.delete(data, id_remove, axis=0)


def _is_in_box(data: np.ndarray, x_lim: np.ndarray, y_lim: np.ndarray) -> np.ndarray:
    """Returns a mask that indicates, whether a data point is within the limits."""
    mask_x = np.logical_and(data[:, 0] > x_lim[0], data[:, 0] < x_lim[1])
    mask_y = np.logical_and(data[:, 1] > y_lim[0], data[:, 1] < y_lim[1])
    return np.logical_and(mask_x, mask_y)


def _ax_is_xlog(axhandle: Axes) -> bool:
    return axhandle.get_xscale() == "log"


def _ax_is_ylog(axhandle: Axes) -> bool:
    return axhandle.get_yscale() == "log"


def _ax_is_zlog(axhandle: Axes3D) -> bool:
    return axhandle.get_zscale() == "log"


def _get_line_data(linehandle: Line2D | art3d.Line3D) -> np.ndarray:
    """Retrieve 2D or 3D data from line object.

    :param linehandle: matplotlib linehandle object

    :returns : (data, is3D)
    """
    if isinstance(linehandle, art3d.Line3D):
        x_data, y_data, z_data = linehandle.get_data_3d()
        data = _stack_data_3d(x_data, y_data, z_data)
    else:
        x_data = np.asarray(linehandle.get_xdata()).astype(np.float32)
        y_data = np.asarray(linehandle.get_ydata()).astype(np.float32)
        data = _stack_data_2d(x_data, y_data)
    return data


def _get_collection_data(collection: PathCollection | art3d.Path3DCollection) -> np.ndarray:
    if isinstance(collection, art3d.Path3DCollection):
        # https://stackoverflow.com/questions/51716696/extracting-data-from-a-3d-scatter-plot-in-matplotlib
        offsets = collection._offsets3d  # noqa: SLF001
        x_data, y_data, z_data = (o.data for o in offsets)
        z_data = np.array(z_data)  # Needed, because it can be a memoryview.
        data = _stack_data_3d(x_data, y_data, z_data)
    else:
        offsets = collection.get_offsets()
        data = offsets.data  # type: ignore[union-attr]
    return data


def _get_visual_data(axhandle: Axes | Axes3D, data: np.ndarray) -> np.ndarray:
    """Returns the visual representation of the data.

    Respecting possible log_scaling and projection into the image plane.

    :returns : visualData
    """
    if isinstance(axhandle, Axes3D):
        x_data, y_data, z_data = _split_data_3d(data)
    else:
        x_data, y_data = _split_data_2d(data)

    if axhandle.get_xscale() == "log":
        x_data = np.log10(x_data)
    if axhandle.get_yscale() == "log":
        y_data = np.log10(y_data)
    if isinstance(axhandle, Axes3D):
        if axhandle.get_zscale() == "log":
            z_data = np.log10(z_data)

        p = _get_projection_matrix(axhandle)

        points = np.stack([x_data, y_data, z_data, np.ones_like(z_data)], axis=1)
        data_projected = p @ points.T
        x_data = data_projected[0, :] / data_projected[-1, :]
        y_data = data_projected[1, :] / data_projected[-1, :]

    x_data = np.reshape(x_data, (-1,))
    y_data = np.reshape(y_data, (-1,))
    return _stack_data_2d(x_data, y_data)


def _isempty(array: np.ndarray) -> bool:
    """Proxy for matlab / octave isempty function.

    :param array: array to check if it is empty
    :type array: np.ndarray
    """
    return array.size == 0


def _line_has_lines(linehandle: Line2D | art3d.Line3D) -> bool:
    """Check if linestyle is not None and linewidth is larger than 0."""
    return (linehandle.get_linestyle() is not None) and (linehandle.get_linewidth() > 0.0)


def _prune_outside_box(cfd: CleanFigureData) -> np.ndarray:
    """Some sections of the line may sit outside of the visible box. Cut those off.

    This method is not pure because it updates the linehandle object's data.
    """
    if cfd.visual_data is None or cfd.visual_data.size == 0:
        return cfd.data

    tol = 1.0e-10
    relaxed_x_lim = cfd.x_lim + np.array([-tol, tol])
    relaxed_y_lim = cfd.y_lim + np.array([-tol, tol])

    data_is_in_box = _is_in_box(cfd.visual_data, relaxed_x_lim, relaxed_y_lim)

    should_plot = data_is_in_box
    if cfd.has_lines:
        segvis = _segment_visible(cfd.visual_data, data_is_in_box, cfd.x_lim, cfd.y_lim)
        should_plot = np.logical_or(
            should_plot, np.concatenate([np.array([False]).reshape((-1,)), segvis])
        )
        should_plot = np.logical_or(
            should_plot, np.concatenate([segvis, np.array([False]).reshape((-1,))])
        )

    id_replace = np.array([[]])
    id_remove = np.array([[]])

    if not np.all(should_plot):
        id_remove = np.argwhere(np.logical_not(should_plot))

        # If there are consecutive data points to be removed, only replace
        # the first one by a NaN. Consecutive data points have
        # diff(id_remove)==1, so replace diff(id_remove)>1 by NaN and remove
        # the rest
        idx = np.diff(id_remove, axis=0) > 1
        idx = np.concatenate([np.array([True]).reshape((-1, 1)), idx], axis=0)

        id_replace = id_remove[idx]
        id_remove = id_remove[np.logical_not(idx)]

    data = _replace_data_with_nan(cfd.data, id_replace)
    data = _remove_data(data, id_remove)
    return _remove_nans(data)


def _move_points_closer(x_lim: np.ndarray, y_lim: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Move points closer if needed.

    Move all points outside a box much larger than the visible one
    to the boundary of that box and make sure that lines in the visible
    box are preserved. This typically involves replacing one point by
    two new ones and a NaN.

    Not implemented: 3D simplification of frontal 2D projection. This requires the
    full transformation rather than the projection, as we have to calculate
    the inverse transformation to project back into 3D.
    """
    # Calculate the extension of the extended box
    x_width = x_lim[1] - x_lim[0]
    y_width = y_lim[1] - y_lim[0]

    # Don't choose the larger box too large to make sure that the values inside
    # it can still be treated by TeX.
    extended_factor = 0.1
    large_xlim = x_lim + extended_factor * np.array([-x_width, x_width])
    large_ylim = y_lim + extended_factor * np.array([-y_width, y_width])

    data_is_in_large_box = _is_in_box(data, large_xlim, large_ylim)
    data_is_in_large_box = np.logical_or(data_is_in_large_box, np.any(np.isnan(data), axis=1))
    id_replace = np.argwhere(np.logical_not(data_is_in_large_box))

    data_insert = np.array([[]])
    if not _isempty(id_replace):
        msg = (
            "There is data outside of the box. Don't know how to handle during cleaning. "
            "Please check if x/ylim is to tight"
        )
        raise NotImplementedError(msg)
    data = _insert_data(data, id_replace, data_insert)
    if _isempty(id_replace):
        return data
    raise NotImplementedError


def _insert_data(
    data: np.ndarray,
    id_insert: np.ndarray,
    data_insert: np.ndarray,  # noqa: ARG001
) -> np.ndarray:
    """Inserts the elements of the cell array dataInsert at position id_insert."""
    if _isempty(id_insert):
        return data
    raise NotImplementedError


def _simplify_line(cfd: CleanFigureData) -> np.ndarray:
    """Reduce the number of data points in the line 'handle'.

    Applies a path-simplification algorithm if there are no markers or
    pixelization otherwise. Changes are visually negligible at the target
    resolution.

    The target resolution is either specified as the number of PPI or as
    the [Width, Height] of the figure in pixels.
    A scalar value of INF or 0 disables path simplification.
    (default = 600)
    """
    if (
        (
            not isinstance(cfd.target_resolution, (list, np.ndarray))
            and (np.isinf(cfd.target_resolution) or cfd.target_resolution == 0)
        )
        or (
            isinstance(cfd.target_resolution, (list, np.ndarray))
            and any(np.logical_or(np.isinf(cfd.target_resolution), cfd.target_resolution == 0))
        )
        or cfd.visual_data is None
    ):
        return cfd.data
    width, height = _get_width_height_in_pixels(cfd.fig, cfd.target_resolution)
    x_data_vis, y_data_vis = _split_data_2d(cfd.visual_data)
    # Only simplify if there are more than 2 points
    if np.size(x_data_vis) <= 2 or np.size(y_data_vis) <= 2:  # noqa: PLR2004
        return cfd.data

    # Automatically guess a tol based on the area of the figure and
    # the area and resolution of the output
    x_range = cfd.x_lim[1] - cfd.x_lim[0]
    y_range = cfd.y_lim[1] - cfd.y_lim[0]

    # Conversion factors of data units into pixels
    x_to_pix = width / x_range
    y_to_pix = height / y_range

    id_remove = np.array([])
    # If the path has markers, perform pixelation instead of simplification
    if cfd.has_markers and not cfd.has_lines:
        # Pixelate data at the zoom multiplier
        mask = _pixelate(x_data_vis, y_data_vis, x_to_pix, y_to_pix)
        id_remove = np.argwhere(mask * 1 == 0)
    elif cfd.has_lines and not cfd.has_markers:
        # Get the width of a pixel
        x_pixel_width = 1 / x_to_pix
        y_pixel_width = 1 / y_to_pix
        tol = min(x_pixel_width, y_pixel_width)

        # Split up lines which are separated by NaNs
        id_nan = np.logical_or(np.isnan(x_data_vis), np.isnan(y_data_vis))

        # If lines were separated by a NaN, diff(~id_nan) would give 1 for
        # the start of a line and -1 for the index after the end of
        # a line.

        id_diff = np.diff(
            1 * np.concatenate([np.array([False]), np.logical_not(id_nan), np.array([False])]),
            axis=0,
        ).reshape((-1,))
        line_start = np.argwhere(id_diff == 1)
        line_start = line_start.reshape((-1,))
        line_end = np.argwhere(id_diff == -1) - 1
        line_end = line_end.reshape((-1,))
        num_lines = np.size(line_start)

        id_removes = [np.array([], dtype=np.int32).reshape((-1,))] * num_lines

        # Simplify the line segments
        for ii in np.arange(num_lines):
            # Actual data that inherits the simplifications
            x = x_data_vis[line_start[ii] : line_end[ii] + 1]
            y = y_data_vis[line_start[ii] : line_end[ii] + 1]

            # Line simplification
            if np.size(x) > 2:  # noqa: PLR2004
                mask = _opheim_simplify(x, y, tol)
                id_removes[ii] = np.argwhere(mask == 0).reshape((-1,)) + line_start[ii]
        # Merge the indices of the line segments
        id_remove = np.concatenate(id_removes)

    # remove the data points
    return _remove_data(cfd.data, id_remove)


def _pixelate(x: np.ndarray, y: np.ndarray, x_to_pix: float, y_to_pix: float) -> np.ndarray:
    """Rough reduction of data points at a multiple of the target resolution.

    The resolution is lost only beyond the multiplier magnification.

    :param x: x coordinates of data points. Shape [N, ]
    :param y: y coordinates of data points. Shape [N, ]
    :param x_to_pix: scalar converting x measure to pixel measure in x direction
    :param y_to_pix: scalar converting y measure to pixel measure in y direction

    :returns: mask
    """
    mult = 2
    data_pixel = np.round(np.stack([x * x_to_pix * mult, y * y_to_pix * mult], axis=1))
    id_orig = np.argsort(data_pixel[:, 0])
    data_pixel_sorted = data_pixel[id_orig, :]

    m = np.logical_or(np.diff(data_pixel_sorted[:, 0]) != 0, np.diff(data_pixel_sorted[:, 1]) != 0)
    mask_sorted = np.concatenate([np.array([True]).reshape((-1,)), m], axis=0)

    mask = np.ones(x.shape) == 0
    mask[id_orig] = mask_sorted
    mask[0] = True
    mask[-1] = True

    isnan = np.logical_or(np.isnan(x), np.isnan(y))
    mask[isnan] = True
    return mask


def _get_width_height_in_pixels(
    fighandle: FigureBase, target_resolution: float | list | np.ndarray
) -> tuple[float, float]:
    """Target resolution as ppi / dpi. Return width and height in pixels.

    :param fighandle: matplotlib figure object handle
    :param target_resolution: Target resolution in PPI/ DPI. If target_resolution is a scalar,
        calculate final pixels based on figure width and height.
    """
    if isinstance(target_resolution, (float, int)):
        # in matplotlib, the figsize units are always in inches
        if isinstance(fighandle, Figure):
            fig_width_inches = fighandle.get_figwidth()
            fig_height_inches = fighandle.get_figheight()
            if not isinstance(fig_width_inches, float) or not isinstance(fig_height_inches, float):
                raise TypeError
            width = fig_width_inches * target_resolution
            height = fig_height_inches * target_resolution
        else:
            raise TypeError
    else:
        width = target_resolution[0]
        height = target_resolution[1]
    return width, height


def _opheim_simplify(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    """Opheim path simplification algorithm.

     Given a path of vertices V and a tolerance TOL, the algorithm:
       1. selects the first vertex as the KEY;
       2. finds the first vertex farther than TOL from the KEY and links
          the two vertices with a LINE;
       3. finds the last vertex from KEY which stays within TOL from the
          LINE and sets it to be the LAST vertex. Removes all points in
          between the KEY and the LAST vertex;
       4. sets the KEY to the LAST vertex and restarts from step 2.

     The Opheim algorithm can produce unexpected results if the path
     returns back on itself while remaining within TOL from the LINE.
     This behaviour can be seen in the following example:

       x   = [1,2,2,2,3];
       y   = [1,1,2,1,1];
       tol < 1

     The algorithm undesirably removes the second last point. See
     https://github.com/matlab2tikz/matlab2tikz/pull/585#issuecomment-89397577
     for additional details.

     To rectify this issues, step 3 is modified to find the LAST vertex as
     follows:
       3*. finds the last vertex from KEY which stays within TOL from the
           LINE, or the vertex that connected to its previous point forms
           a segment which spans an angle with LINE larger than 90
           degrees.

    :param x: x coordinates of path to simplify. Shape [N, ]
    :type x: np.ndarray
    :param y: y coordinates of path to simplify. Shape [N, ]
    :type y: np.ndarray
    :param tol: scalar float specifying the tolerance for path simplification
    :type tol: float
    :returns: boolean array of shape [N, ] that masks out elements that need not be drawn
    :rtype: np.ndarray

    References:
    ----------
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.95.5882&rep=rep1&type=pdf
    """
    mask = np.zeros_like(x) == 1
    mask[0] = True
    mask[-1] = True
    n = np.size(x)
    i = 0
    while i <= n - 2 - 1:
        j = i + 1
        v = np.array([x[j] - x[i], y[j] - y[i]])
        while j < n - 1 and np.linalg.norm(v) <= tol:
            j = j + 1
            v = np.array([x[j] - x[i], y[j] - y[i]])
        v = v / np.linalg.norm(v)

        # Unit normal to the line between point i and point j
        normal = np.array([v[1], -v[0]])

        # Find the last point which stays within TOL from the line
        # connecting i to j, or the last point within a direction change
        # of pi/2.
        # Starts from the j+1 points, since all previous points are within
        # TOL by construction.

        while j < n - 1:
            # Calculate the perpendicular distance from the i->j line
            v1 = np.array([x[j + 1] - x[i], y[j + 1] - y[i]])
            d = np.abs(np.dot(normal, v1))
            if d > tol:
                break

            # Calculate the angle between the line from the i->j and the
            # line from j -> j+1. If
            v2 = np.array([x[j + 1] - x[j], y[j + 1] - y[i]])
            anglecosine = np.dot(v, v2)
            if anglecosine <= 0:
                break
            j = j + 1
        i = j
        mask[i] = True
    return mask


def _limit_precision(axhandle: Axes | Axes3D, data: np.ndarray, alpha: float) -> np.ndarray:
    """Limit the precision of the given data. If alpha is 0 or negative do nothing."""
    if alpha <= 0:
        return data

    is_xlog = axhandle.get_xscale() == "log"
    is_ylog = axhandle.get_yscale() == "log"
    if not isinstance(axhandle, Axes3D):
        x_data, y_data = _split_data_2d(data)
        data = np.stack([x_data, y_data], axis=1)
        is_log = np.array([is_xlog, is_ylog])
    else:
        x_data, y_data, z_data = _split_data_3d(data)
        is_zlog = axhandle.get_zscale() == "log"
        data = np.stack([x_data, y_data, z_data], axis=1)
        is_log = np.array([is_xlog, is_ylog, is_zlog])

    # Only do something if the data is not empty
    if _isempty(data) or np.isinf(data).all():
        return data

    # Scale to visual coordinates
    data[:, is_log] = np.log10(data[:, is_log])

    # Get the maximal value of the data, only considering finite values
    max_value = max(np.abs(data[np.isfinite(data)]))

    # The least significant bit is proportional to the numerical precision
    # of the largest number. Scale it with a user defined value alpha
    least_significant_bit = np.finfo(max_value).eps * alpha

    data = np.round(data / least_significant_bit) * least_significant_bit
    data[:, is_log] = 10.0 ** data[:, is_log]
    return data


def _segment_visible(
    data: np.ndarray, data_is_in_box: np.ndarray, x_lim: np.ndarray, y_lim: np.ndarray
) -> np.ndarray:
    """Given a bounding box, determine if a line is visible.

    Given a bounding box {x,y}Lim, determine whether the line between all
    pairs of subsequent data points [data(idx,:)<-->data(idx+1,:)] is visible.
    There are two possible cases:
    1: One of the data points is within the limits
    2: The line segments between the datapoints crosses the bounding box

    :param data: array of data points. Shape [N, 2]
    :param data_is_in_box: boolean mask that specifies if data point lies within visual box
    :param x_lim: x axes limits
    :param y_lim: y axes limits

    :returns: mask
    """
    n = np.shape(data)[0]
    mask = np.zeros((n - 1, 1)) == 1

    # Only check if there is more than 1 point
    if n > 1:
        # Define the vectors of data points for the segments X1--X2
        idx = np.arange(n - 1)
        x1 = data[idx, :]
        x2 = data[idx + 1, :]

        # One of the neighbors is inside the box and the other is finite
        this_visible = np.logical_and(data_is_in_box[idx], np.all(np.isfinite(x2), 1))
        next_visible = np.logical_and(data_is_in_box[idx + 1], np.all(np.isfinite(x1), 1))

        bottom_left, top_left, bottom_right, top_right = _corners2d(x_lim, y_lim)

        left = _segments_intersect(x1, x2, bottom_left, top_left)
        right = _segments_intersect(x1, x2, bottom_right, top_right)
        bottom = _segments_intersect(x1, x2, bottom_left, bottom_right)
        top = _segments_intersect(x1, x2, top_left, top_right)

        # Check the result
        mask1 = np.logical_or(this_visible, next_visible)
        mask2 = np.logical_or(left, right)
        mask3 = np.logical_or(top, bottom)

        mask = np.logical_or(mask1, mask2)
        mask = np.logical_or(mask3, mask)

    return mask


def _corners2d(
    x_lim: np.ndarray, y_lim: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Determine the corners of the axes as defined by xLim and yLim."""
    bottom_left = np.array([x_lim[0], y_lim[0]])
    top_left = np.array([x_lim[0], y_lim[1]])
    bottom_right = np.array([x_lim[1], y_lim[0]])
    top_right = np.array([x_lim[1], y_lim[1]])
    return bottom_left, top_left, bottom_right, top_right


def _corners3d(
    x_lim: list | np.ndarray, y_lim: list | np.ndarray, z_lim: list | np.ndarray
) -> np.ndarray:
    """Determine the corners of the 3D axes as defined by xLim, yLim and zLim."""
    # Lower square of the cube
    lower_bottom_left = np.array([x_lim[0], y_lim[0], z_lim[0]])
    lower_top_left = np.array([x_lim[0], y_lim[1], z_lim[0]])
    lower_bottom_right = np.array([x_lim[1], y_lim[0], z_lim[0]])
    lower_top_right = np.array([x_lim[1], y_lim[1], z_lim[0]])

    # Upper square of the cube
    upper_bottom_left = np.array([x_lim[0], y_lim[0], z_lim[1]])
    upper_top_left = np.array([x_lim[0], y_lim[1], z_lim[1]])
    upper_bottom_right = np.array([x_lim[1], y_lim[0], z_lim[1]])
    upper_top_right = np.array([x_lim[1], y_lim[1], z_lim[1]])

    return np.array(
        [
            lower_bottom_left,
            lower_top_left,
            lower_bottom_right,
            lower_top_right,
            upper_bottom_left,
            upper_top_left,
            upper_bottom_right,
            upper_top_right,
        ]
    )


def _get_projection_matrix(axhandle: Axes3D) -> np.ndarray:
    """Get Projection matrix that projects 3D points into 2D image plane.

    :returns: Projection matrix P
    """
    az = np.deg2rad(axhandle.azim)
    el = np.deg2rad(axhandle.elev)
    rotation_z = np.array(
        [
            [np.cos(-az), -np.sin(-az), 0, 0],
            [np.sin(-az), np.cos(-az), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    rotation_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.sin(el), np.cos(el), 0],
            [0, -np.cos(el), np.sin(el), 0],
            [0, 0, 0, 1],
        ]
    )
    x_lim = axhandle.get_xlim3d()
    y_lim = axhandle.get_ylim3d()
    z_lim = axhandle.get_zlim3d()

    aspect_ratio = np.array([x_lim[1] - x_lim[0], y_lim[1] - x_lim[0], z_lim[1] - z_lim[0]])
    aspect_ratio /= aspect_ratio[-1]
    scale_matrix = np.diag(np.concatenate([aspect_ratio, np.array([1.0])]))

    return rotation_x @ rotation_z @ scale_matrix


def _segments_intersect(
    x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray
) -> np.ndarray:
    """Checks whether the segments x1--x2 and x3--x4 intersect.

    A mask is returns with True if the lines intersect.
    """
    lam = _cross_lines(x1, x2, x3, x4)

    # Check whether lambda is in bound
    mask1 = np.logical_and(lam[:, 0] > 0.0, lam[:, 0] < 1.0)
    mask2 = np.logical_and(lam[:, 1] > 0.0, lam[:, 1] < 1.0)
    return np.logical_and(mask1, mask2)


def _cross_lines(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, x4: np.ndarray) -> np.ndarray:
    """Checks whether the segments x1--x2 and x3--x4 intersect.

    See https://en.wikipedia.org/wiki/Line-line_intersection for reference.
    Given four points x_k=(x_k,y_k), k in {1,2,3,4}, and the two lines defined by those,

        L1(lambda) = x1 + lam (x2 - x1)
        L2(lambda) = x3 + lam (x4 - x3)

    returns the lambda for which they intersect (and Inf if they are parallel).
    Technically, one needs to solve the 2x2 equation system

        x1 + lambda1 (x2-x1)  =  x3 + lambda2 (x4-x3)
        y1 + lambda1 (y2-y1)  =  y3 + lambda2 (y4-y3)

    for lambda1 and lambda2.

    Now x1 is a vector of all data points x1 and x2 is a vector of all
    consecutive data points x2
    n is the number of segments (not points in the plot!)
    """
    det_a = -(x2[:, 0] - x1[:, 0]) * (x4[1] - x3[1]) + (x2[:, 1] - x1[:, 1]) * (x4[0] - x3[0])

    id_det_a = det_a != 0

    n = x2.shape[0]
    lam = np.zeros((n, 2))
    if id_det_a.any():
        # NOTE: watch out for broadcasting
        rhs = -x1.reshape((-1, 2)) + x3.reshape((-1, 2))
        rotate = np.array([[0, -1], [1, 0]])
        lam[id_det_a, 0] = (rhs[id_det_a, :] @ rotate @ (x4 - x3).T) / det_a[id_det_a]
        lam[id_det_a, 1] = (
            np.sum(-(x2[id_det_a, :] - x1[id_det_a, :]) @ rotate * rhs[id_det_a, :], axis=1)
            / det_a[id_det_a]
        )
    return lam
