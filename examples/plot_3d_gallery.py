"""Generate the 3D example gallery used by the README."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import matplot2tikz

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d import Axes3D

mpl.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples"
TEX_DIR = EXAMPLE_DIR / "output" / "tex"
PNG_DIR = EXAMPLE_DIR / "output" / "png"
BUILD_DIR = EXAMPLE_DIR / "output" / "build"
Clip3DMode = Literal["none", "hide", "clip"]


def line_scatter_text() -> Figure:
    """Return a 3D line, scatter, legend, and text example."""
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    theta = np.linspace(0.0, 2.6 * np.pi, 22)
    radius = 0.18 + 0.035 * theta
    ax.plot(
        radius * np.cos(theta),
        radius * np.sin(theta),
        0.12 * theta,
        color="tab:red",
        marker="o",
        label="spiral",
    )
    x = np.linspace(-0.8, 0.8, 8)
    ax.plot(x, 0.2 * np.sin(4 * x), 0.25 + 0.15 * np.cos(3 * x), "--", label="ridge")
    xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, 4), np.linspace(-0.4, 0.4, 3))
    zs = 0.2 + xs**2 + 0.35 * ys**2
    ax.scatter(xs.ravel(), ys.ravel(), zs.ravel(), c=zs.ravel(), s=32, cmap="viridis")
    ax.text(0.0, -0.55, 0.78, "3D text")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left")
    ax.view_init(elev=28.0, azim=38.0)
    return fig


def surface_wireframe() -> Figure:
    """Return a surface with an overlaid wireframe."""
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.linspace(-2.5, 2.5, 11)
    y = np.linspace(-2.5, 2.5, 11)
    xx, yy = np.meshgrid(x, y)
    zz = np.sin(np.hypot(xx, yy))
    ax.plot_surface(xx, yy, zz, cmap="viridis", edgecolor="black", linewidth=0.2, alpha=0.9)
    ax.plot_wireframe(xx, yy, zz + 0.75, color="black", linewidth=0.45, rstride=2, cstride=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=24.0, azim=-55.0)
    return fig


def contour_projection() -> Figure:
    """Return 3D contour lines and filled contour patches."""
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.linspace(-2.0, 2.0, 10)
    y = np.linspace(-2.0, 2.0, 10)
    xx, yy = np.meshgrid(x, y)
    zz = np.cos(xx) * np.sin(yy)
    ax.contour(
        xx,
        yy,
        zz,
        levels=[-0.6, -0.2, 0.2, 0.6],
        colors=["navy", "teal", "orange", "crimson"],
    )
    ax.contourf(xx, yy, zz, zdir="z", offset=-1.1, levels=5, cmap="cividis", alpha=0.65)
    ax.set_zlim(-1.1, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25.0, azim=105.0)
    return fig


def bars() -> Figure:
    """Return a 3D bar chart with custom tick labels."""
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    xpos, ypos = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0])
    heights = np.array([0.5, 1.2, 0.8, 1.4, 0.7, 1.6])
    colors = plt.get_cmap("cividis")((heights - heights.min()) / np.ptp(heights))
    ax.bar3d(
        xpos.ravel(),
        ypos.ravel(),
        np.zeros(6),
        0.55,
        0.55,
        heights,
        color=colors,
        edgecolor="black",
        shade=False,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xticks([0.25, 1.25, 2.25], ["low", "mid", "high"])
    ax.set_yticks([0.25, 1.25], ["front", "back"])
    ax.view_init(elev=26.0, azim=35.0)
    return fig


def quiver() -> Figure:
    """Return a semantic 3D quiver example."""
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    x = np.array([0.0, 0.9, 0.0, 0.9])
    y = np.array([0.0, 0.0, 0.8, 0.8])
    z = np.array([0.0, 0.2, 0.1, 0.3])
    u = np.array([0.45, -0.2, 0.25, -0.3])
    v = np.array([0.15, 0.35, -0.25, -0.2])
    w = np.array([0.35, 0.25, 0.45, 0.2])
    ax.quiver(x, y, z, u, v, w, length=0.8, arrow_length_ratio=0.25, color="tab:green")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30.0, azim=-45.0)
    return fig


def clipping_surface() -> Figure:
    """Return a surface and wireframe scene that crosses the 3D axis box."""
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
        zdir="z",
        offset=-0.85,
        cmap="viridis",
        linewidths=0.6,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_zlim(-1, 0.5)
    ax.view_init(elev=24.0, azim=-55.0)

    return fig


def clipping_surface_none() -> Figure:
    """Return the clipping comparison scene without export clipping."""
    return clipping_surface()


def clipping_surface_hide() -> Figure:
    """Return the clipping comparison scene for hide clipping."""
    return clipping_surface()


def clipping_surface_clip() -> Figure:
    """Return the clipping comparison scene for geometric clipping."""
    return clipping_surface()


def log_clipping() -> Figure:
    """Return a log-scaled 3D line clipped in axis-scale coordinates."""
    fig = plt.figure(figsize=(4.2, 3.2))
    ax = cast("Axes3D", fig.add_subplot(111, projection="3d"))
    ax.plot([0.1, 10.0], [0.0, 2.0], [1.0, 1.0], marker="o", color="tab:blue")
    ax.set_xlim(1.0, 10.0)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(0.0, 2.0)
    ax.set_xlabel(r"$\log_{10} \mathrm{X}$")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=24.0, azim=-35.0)
    return fig


EXAMPLES: tuple[tuple[str, str, Callable[[], Figure], Clip3DMode], ...] = (
    ("line_scatter_text", "Line, scatter, text", line_scatter_text, "none"),
    ("surface_wireframe", "Surface and wireframe", surface_wireframe, "none"),
    ("contour_projection", "Contours and filled contours", contour_projection, "none"),
    ("bar3d", "3D bars", bars, "none"),
    ("quiver3d", "Semantic quiver", quiver, "none"),
    ("log_clipping", "Log-scale clipping", log_clipping, "clip"),
    ("clipping_none", "No clipping", clipping_surface_none, "none"),
    ("clipping_hide", "Hide outside", clipping_surface_hide, "hide"),
    ("clipping_clip", "Clip to limits", clipping_surface_clip, "clip"),
)


def main() -> None:
    """Generate TeX examples and PNG previews when the local TeX tools exist."""
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    expected_names = {name for name, _title, _plot, _clip_3d in EXAMPLES}
    for output_dir, suffix in ((TEX_DIR, ".tex"), (PNG_DIR, ".png")):
        for path in output_dir.glob(f"*{suffix}"):
            if path.stem not in expected_names:
                path.unlink()

    pdflatex = shutil.which("pdflatex")
    pdftoppm = shutil.which("pdftoppm")
    for name, _title, plot, clip_3d in EXAMPLES:
        fig = plot()
        tex_path = TEX_DIR / f"{name}.tex"
        fig.savefig(BUILD_DIR / f"{name}_reference.png", dpi=fig.dpi)
        matplot2tikz.save(
            tex_path,
            figure=fig,
            standalone=True,
            include_disclaimer=False,
            float_format=".8g",
            clip_3d=clip_3d,
        )
        plt.close(fig)

        if pdflatex is None or pdftoppm is None:
            continue

        subprocess.run(  # noqa: S603
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(BUILD_DIR),
                str(tex_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        pdf_path = BUILD_DIR / f"{name}.pdf"
        png_stem = PNG_DIR / name
        subprocess.run(  # noqa: S603
            [pdftoppm, "-png", "-singlefile", "-r", "170", str(pdf_path), str(png_stem)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


if __name__ == "__main__":
    main()
