"""Test all kind of patches."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Patch, Polygon, Rectangle, Wedge
from matplotlib.path import Path

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    rng = np.random.default_rng(123)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n_points = 3
    x = rng.random(size=n_points)
    y = rng.random(size=n_points)
    radii = 0.1 * rng.random(size=n_points)
    patches: list[Patch] = []
    for x1, y1, r in zip(x, y, radii, strict=True):
        circle = Circle((x1, y1), r)
        patches.append(circle)

    rect = Rectangle(xy=(0.0, 0.25), width=1.0, height=0.5, angle=-45.0)
    patches.append(rect)

    x = rng.random(size=n_points)
    y = rng.random(size=n_points)
    radii = 0.1 * rng.random(size=n_points)
    theta1 = 360.0 * rng.random(size=n_points)
    theta2 = 360.0 * rng.random(size=n_points)
    for x1, y1, r, t1, t2 in zip(x, y, radii, theta1, theta2, strict=True):
        wedge = Wedge((x1, y1), r, t1, t2)
        patches.append(wedge)

    # Some limiting conditions on Wedge
    patches += [
        Wedge((0.3, 0.7), 0.1, 0, 360),  # Full circle
        Wedge((0.7, 0.8), 0.2, 0, 360, width=0.05),  # Full ring
        Wedge((0.8, 0.3), 0.2, 0, 45),  # Full sector
        Wedge((0.8, 0.3), 0.2, 45, 90, width=0.10),  # Ring sector
    ]

    for _ in range(n_points):
        polygon = Polygon(rng.random(size=(n_points, 2)), closed=True)
        patches.append(polygon)

    colors = 100 * rng.random(size=len(patches))
    p = PatchCollection(patches, cmap=plt.get_cmap("viridis"), alpha=0.4)
    p.set_array(np.array(colors))
    ax.add_collection(p)

    ellipse = Ellipse(xy=(1.0, 0.5), width=1.0, height=0.5, angle=45.0, alpha=0.4)
    ax.add_patch(ellipse)

    circle = Circle(xy=(0.0, 1.0), radius=0.5, color="r", alpha=0.4)
    ax.add_patch(circle)

    arrow = FancyArrowPatch(posA=(0.25, 0.25), posB=(0.5, 0.25), arrowstyle="->")
    ax.add_patch(arrow)

    curved_arrow = FancyArrowPatch(
        path=Path(
            [(0.3, 0.3), (0.5, 1.0), (1.0, 0.8), (0.8, 0.3)],
            [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
        ),
        arrowstyle="-|>",
    )
    ax.add_patch(curved_arrow)

    plt.colorbar(p)

    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
