"""Test 3D plot export."""

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing_extensions import NotRequired, TypedDict, Unpack

import matplot2tikz

from .helpers import assert_equality

mpl.use("Agg")

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

Clip3DMode = Literal["none", "hide", "clip"]
ShaderMode = Literal["none", "interp"]


class _TikzCodeOptions(TypedDict):
    axis_width: NotRequired[str | None]
    axis_height: NotRequired[str | None]
    extra_axis_parameters: NotRequired[list[str] | None]
    strict: NotRequired[bool]
    add_axis_environment: NotRequired[bool]
    standalone: NotRequired[bool]
    externalize_tables: NotRequired[bool]
    clip_3d: NotRequired[Clip3DMode]
    shader: NotRequired[ShaderMode]


def plot_line_and_scatter() -> Figure:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))

    theta = np.linspace(0.0, 2.5 * np.pi, 18)
    radius = 0.25 + 0.07 * theta
    ax.plot(
        radius * np.cos(theta),
        radius * np.sin(theta),
        0.18 * theta,
        color="tab:red",
        marker="o",
        label="spiral",
    )
    ax.plot(
        np.linspace(-0.8, 0.8, 9),
        0.25 * np.sin(np.linspace(-2.0, 2.0, 9)),
        0.35 + 0.15 * np.cos(np.linspace(-2.0, 2.0, 9)),
        color="tab:blue",
        linestyle="--",
        marker="^",
        label="ridge",
    )

    xs = np.linspace(-0.7, 0.7, 4)
    ys = np.linspace(-0.6, 0.6, 3)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.3 + 0.45 * xx**2 + 0.25 * np.cos(np.pi * yy)
    ax.scatter(
        xx.ravel(),
        yy.ravel(),
        zz.ravel(),
        c=zz.ravel(),
        s=np.linspace(12.0, 52.0, zz.size),
        cmap="plasma",
        marker="D",
        edgecolors="black",
        linewidths=0.35,
        label="samples",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.9, 0.9)
    ax.set_zlim(0.0, 1.6)
    ax.view_init(elev=28.0, azim=38.0)
    ax.text(0.28, -0.72, 1.18, "3D text")
    ax.legend(loc="upper left")
    return fig


def plot_surface_and_wireframe() -> Figure:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    xx, yy = np.meshgrid(x, y)
    zz = np.sin(np.sqrt(xx**2 + yy**2))
    ax.plot_surface(
        xx,
        yy,
        zz,
        cmap=plt.get_cmap("viridis"),
        edgecolor="black",
        linewidth=0.25,
        alpha=0.88,
    )
    ax.plot_wireframe(
        xx,
        yy,
        zz + 0.75,
        color="black",
        linewidth=0.45,
        rstride=2,
        cstride=2,
    )
    ax.contour(
        xx,
        yy,
        zz,
        levels=[-0.2, 0.0, 0.2],
        colors=["navy", "darkorange", "crimson"],
        linewidths=0.9,
    )
    ax.contour(
        xx,
        yy,
        zz,
        levels=[-0.2, 0.0, 0.2],
        zdir="z",
        offset=-0.85,
        cmap="plasma",
        linewidths=0.6,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=24.0, azim=-55.0)
    return fig


def plot_bar3d() -> Figure:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    xpos, ypos = np.meshgrid([0.0, 1.1, 2.2], [0.0, 1.0])
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    dx = np.array([0.55, 0.75, 0.65, 0.55, 0.75, 0.65])
    dy = np.array([0.55, 0.45, 0.65, 0.75, 0.5, 0.6])
    dz = np.array([0.5, 1.15, 0.8, 1.35, 0.65, 1.55])
    colors = plt.get_cmap("cividis")((dz - dz.min()) / (dz.max() - dz.min()))
    ax.bar3d(
        xpos,
        ypos,
        zpos,
        dx,
        dy,
        dz,
        color=colors,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.82,
        shade=False,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.2, 3.05)
    ax.set_ylim(-0.2, 1.9)
    ax.set_zlim(0.0, 1.7)
    ax.view_init(elev=26.0, azim=35.0)
    return fig


def plot_quiver3d() -> Figure:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.array([0.0, 0.9, 0.0, 0.9])
    y = np.array([0.0, 0.0, 0.8, 0.8])
    z = np.array([0.0, 0.2, 0.1, 0.3])
    u = np.array([0.45, -0.2, 0.25, -0.3])
    v = np.array([0.15, 0.35, -0.25, -0.2])
    w = np.array([0.35, 0.25, 0.45, 0.2])
    ax.quiver(
        x,
        y,
        z,
        u,
        v,
        w,
        length=0.8,
        arrow_length_ratio=0.25,
        color="tab:green",
        linewidth=1.1,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(0.0, 0.8)
    ax.view_init(elev=30.0, azim=-45.0)
    return fig


def test_line_and_scatter() -> None:
    assert_equality(plot_line_and_scatter, "test_3d_line_and_scatter_reference.tex")


def test_colormapped_3d_scatter_preserves_marker_and_edge() -> None:
    code = _tikz_code(plot_line_and_scatter)

    assert "mark=diamond*" in code
    assert "draw=black" in code
    assert "line width=0.14pt" in code


def test_surface_and_wireframe() -> None:
    assert_equality(plot_surface_and_wireframe, "test_3d_surface_and_wireframe_reference.tex")


def test_3d_surface_shader_option() -> None:
    code = _tikz_code(plot_surface_and_wireframe, shader="interp")

    assert "shader=interp" in code
    assert "patch type=bilinear" in code
    assert r"point meta=\thisrow{meta}" in code
    assert "x y z meta\\\\" in code
    assert "patch table with point meta" not in code
    assert "patch type=polygon,\nvertex count=4" not in code


def test_3d_axis_uses_native_label_and_tick_layout() -> None:
    code = _tikz_code(plot_surface_and_wireframe)

    assert "ylabel style={rotate=-90.0}" not in code
    assert "yticklabel style={anchor=center}" not in code
    assert "tick pos=left" not in code
    assert "tick align=outside" not in code


def test_3d_grid_matches_mplot3d_default() -> None:
    code = _tikz_code(plot_surface_and_wireframe)

    assert "xmajorgrids" in code
    assert "ymajorgrids" in code
    assert "zmajorgrids" in code

    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    ax.grid(visible=False)
    code = matplot2tikz.get_tikz_code(fig, include_disclaimer=False, float_format=".8g")
    plt.close("all")

    assert "xmajorgrids" not in code
    assert "ymajorgrids" not in code
    assert "zmajorgrids" not in code


def test_3d_view_uses_pgfplots_azimuth_convention() -> None:
    code = _tikz_code(plot_line_and_scatter)

    assert "view={142}{28}" in code


def test_strict_3d_ticks_use_matplotlib_locations() -> None:
    code = _tikz_code(plot_surface_and_wireframe, strict=True)

    assert "xtick={-6,-4,-2,0,2,4,6}" in code
    assert "ytick={-6,-4,-2,0,2,4,6}" in code
    assert "ztick={-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1}" in code


def test_3d_custom_tick_labels_are_exported_without_strict() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    ax.set_xticks([0.0, 0.5, 1.0], ["left", "middle", "right"])
    ax.set_yticks([0.0, 1.0], ["front", "back"])
    ax.set_zticks([0.0, 1.0], ["low", "high"])

    code = matplot2tikz.get_tikz_code(fig, include_disclaimer=False, float_format=".8g")
    plt.close("all")

    assert "xtick={0,0.5,1}" in code
    assert "xticklabels={left,middle,right}" in code
    assert "ytick={0,1}" in code
    assert "yticklabels={front,back}" in code
    assert "ztick={0,1}" in code
    assert "zticklabels={low,high}" in code


def test_strict_3d_custom_tick_locations_are_exported() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    ax.set_xticks([0.0, 0.3, 0.9])
    ax.set_yticks([0.1, 0.4, 1.0])
    ax.set_zticks([0.2, 0.6, 1.0])

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", strict=True
    )
    plt.close("all")

    assert "xtick={0,0.3,0.9}" in code
    assert "ytick={0.1,0.4,1}" in code
    assert "ztick={0.2,0.6,1}" in code


def test_bar3d() -> None:
    assert_equality(plot_bar3d, "test_3d_bar3d_reference.tex", strict=True)


def test_bar3d_sorts_faces_in_one_patch_plot() -> None:
    code = _tikz_code(plot_bar3d, strict=True)

    assert code.count("\\addplot3 [") == 1
    assert "patch table with point meta" in code
    assert "colormap={matplot2tikzpoly0}" in code
    assert "ztick={0,0.25,0.5,0.75,1,1.25,1.5,1.75}" in code


def test_poly3d_colormap_keeps_colors_aligned_after_filtering() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    collection = Poly3DCollection(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
        ],
        edgecolor="black",
    )
    collection.set_array(np.array([0.0, 1.0, 2.0]))
    ax.add_collection3d(collection)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    code = matplot2tikz.get_tikz_code(fig, include_disclaimer=False, float_format=".8g")
    plt.close("all")

    assert "0 1 2 1\\\\" in code
    assert "3 4 5 2\\\\" in code
    assert "0 1 2 0\\\\" not in code


def test_clip3d_line_clips_to_axis_limits() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot([-1.0, 2.0], [0.5, 0.5], [0.0, 0.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="clip"
    )
    plt.close("all")

    assert "0 0.5 0" in code
    assert "1 0.5 0" in code
    assert "-1 0.5 0" not in code


def test_clip3d_line_preserves_contiguous_polyline() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot(
        [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0],
        [0.5] * 7,
        [0.5] * 7,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="clip"
    )
    plt.close("all")

    assert code.count("\\addplot3 [") == 1
    assert "0 0.5 0.5" in code
    assert "0.25 0.5 0.5" in code
    assert "1 0.5 0.5" in code
    assert "-1 0.5 0.5" not in code
    assert "2 0.5 0.5" not in code


def test_clip3d_line_preserves_disconnected_polyline_runs() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot(
        [0.0, 1.0, np.nan, 1.0, 0.0],
        [0.5, 0.5, np.nan, 0.5, 0.5],
        [0.5, 0.5, np.nan, 0.5, 0.5],
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="clip"
    )
    plt.close("all")

    assert code.count("\\addplot3 [") == 2  # noqa: PLR2004


def test_clip3d_line_uses_log_axis_coordinates() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot([0.1, 10.0], [0.0, 2.0], [1.0, 1.0])
    ax.set_xlim(1.0, 10.0)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(0.1, 10.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="clip"
    )
    plt.close("all")

    assert "1 1 1" in code
    assert "10 2 1" in code
    assert "0.1 0 1" not in code


def test_clip3d_scatter_hides_points_outside_limits() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.scatter(
        [0.2, 1.5, 0.8],
        [0.2, 0.5, 1.8],
        [0.2, 0.5, 0.8],
        c=[1.0, 2.0, 3.0],
        s=[10.0, 20.0, 30.0],
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="hide"
    )
    plt.close("all")

    assert "0.2 0.2 0.2 1" in code
    assert "1.5 0.5 0.5 2" not in code
    assert "0.8 1.8 0.8 3" not in code


def test_clip3d_poly_collection_clips_polygon_vertices() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    collection = Poly3DCollection(
        [np.array([[-0.5, 0.2, 0.2], [0.5, 0.2, 0.2], [0.5, 0.8, 0.2]])],
        facecolor="tab:red",
    )
    ax.add_collection3d(collection)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="clip"
    )
    plt.close("all")

    assert "0 0.2 0.2" in code
    assert "-0.5 0.2 0.2" not in code


def test_clip3d_surface_keeps_fully_inside_quad_patches() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.array([[0.0, 1.0], [0.0, 1.0]])
    y = np.array([[0.0, 0.0], [1.0, 1.0]])
    z = np.array([[0.0, 0.0], [0.0, 0.0]])
    ax.plot_surface(x, y, z, linewidth=0.0, edgecolor="none")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="clip"
    )
    plt.close("all")

    assert "vertex count=4" in code
    assert "vertex count=3" not in code


def test_clip3d_surface_keeps_clipped_quad_patches() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    collection = Poly3DCollection(
        [
            np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ]
            )
        ],
        facecolor="tab:blue",
        edgecolor="none",
    )
    ax.add_collection3d(collection)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="clip"
    )
    plt.close("all")

    assert "vertex count=4" in code
    assert "vertex count=3" not in code
    assert "0 0 0" in code
    assert "-1 0 0" not in code


def test_clip3d_shader_triangulates_collection_when_a_clipped_polygon_breaks() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    collection = Poly3DCollection(
        [
            np.array(
                [
                    [0.1, 0.1, 0.0],
                    [0.4, 0.1, 1.0],
                    [0.4, 0.4, 2.0],
                    [0.1, 0.4, 3.0],
                ]
            ),
            np.array(
                [
                    [-0.5, 0.2, 0.0],
                    [0.8, 0.2, 2.0],
                    [0.8, 0.8, 4.0],
                    [0.2, 0.8, 6.0],
                ]
            ),
        ],
        edgecolor="none",
    )
    collection.set_array(np.array([0.0, 1.0]))
    collection.set_cmap(plt.get_cmap("viridis"))
    ax.add_collection3d(collection)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 6.0)

    code = matplot2tikz.get_tikz_code(
        fig,
        include_disclaimer=False,
        float_format=".8g",
        clip_3d="clip",
        shader="interp",
    )
    plt.close("all")

    assert "patch type=triangle" in code
    assert "patch type=bilinear" not in code
    assert "patch type=polygon" not in code
    assert "shader=interp" in code
    assert r"point meta=\thisrow{meta}" in code


def test_clip3d_quiver_hide_keeps_semantic_quiver_for_inside_arrows() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.quiver(
        [0.2, 1.2],
        [0.2, 0.2],
        [0.2, 0.2],
        [0.2, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_zlim(0.0, 1.0)

    code = matplot2tikz.get_tikz_code(
        fig, include_disclaimer=False, float_format=".8g", clip_3d="hide"
    )
    plt.close("all")

    assert r"quiver={u=\thisrow{u}, v=\thisrow{v}, w=\thisrow{w}}" in code
    assert "0.2 0.2 0.2 0.2 0 0" in code
    assert "1.2 0.2 0.2 0.2 0 0" not in code


def test_clip3d_rejects_unknown_mode() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])

    with pytest.raises(ValueError, match="clip_3d"):
        matplot2tikz.get_tikz_code(
            fig,
            include_disclaimer=False,
            clip_3d=cast("Clip3DMode", "invalid"),
        )
    plt.close("all")


def test_quiver3d() -> None:
    assert_equality(plot_quiver3d, "test_3d_quiver_reference.tex")


def test_quiver3d_uses_semantic_pgfplots_quiver() -> None:
    code = _tikz_code(plot_quiver3d)

    assert code.count("\\addplot3 [") == 1
    assert r"quiver={u=\thisrow{u}, v=\thisrow{v}, w=\thisrow{w}}" in code
    assert "x y z u v w" in code
    assert "-latex" in code
    assert "mark=none" in code


def test_3d_axis_size_and_extra_parameters() -> None:
    code = _tikz_code(
        plot_line_and_scatter,
        axis_width="7cm",
        axis_height="5cm",
        extra_axis_parameters=["name=threeDaxis"],
    )

    assert "width=7cm" in code
    assert "height=5cm" in code
    assert "name=threeDaxis" in code


def test_3d_without_axis_environment() -> None:
    code = _tikz_code(plot_line_and_scatter, add_axis_environment=False)

    assert "\\begin{axis}" not in code
    assert "\\end{axis}" not in code
    assert "\\addplot3" in code


def test_3d_standalone_includes_dynamic_libraries() -> None:
    code = _tikz_code(plot_surface_and_wireframe, standalone=True)

    assert "\\documentclass{standalone}" in code
    assert "\\usepgfplotslibrary{patchplots}" in code
    assert "\\begin{document}" in code


def test_3d_externalize_tables() -> None:
    fig = plot_line_and_scatter()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "plot.tex"
        matplot2tikz.save(
            filepath,
            figure=fig,
            include_disclaimer=False,
            externalize_tables=True,
            float_format=".8g",
        )

        code = filepath.read_text(encoding="utf-8")
        table_files = list(Path(tmpdir).glob("plot-*.dat"))

    plt.close("all")

    assert "\\addplot3" in code
    assert table_files
    assert "plot-000.dat" in code


def test_contour3d_uses_native_coordinates() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    xx, yy = np.meshgrid(x, y)
    zz = xx + yy
    ax.contour(xx, yy, zz, levels=[1.0, 2.0, 3.0])

    code = matplot2tikz.get_tikz_code(fig, include_disclaimer=False, float_format=".8g")
    plt.close("all")

    assert "\\addplot3" in code
    assert "zmin=" in code
    assert "0 1 1" in code
    assert "0 2 2" in code


def test_contourf3d_uses_patchplots() -> None:
    fig = plt.figure()
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    xx, yy = np.meshgrid(x, y)
    zz = xx + yy
    ax.contourf(xx, yy, zz, levels=[1.0, 2.0, 3.0])

    code = matplot2tikz.get_tikz_code(fig, include_disclaimer=False, float_format=".8g")
    plt.close("all")

    assert "\\addplot3" in code
    assert "patch table" in code


def _tikz_code(plot: Callable[[], Figure], **kwargs: Unpack[_TikzCodeOptions]) -> str:
    fig = plot()
    code = matplot2tikz.get_tikz_code(fig, include_disclaimer=False, float_format=".8g", **kwargs)
    plt.close("all")
    return code
