"""Test plot with annotations."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1, 5), ylim=(-4, 3))
    t = np.arange(0.0, 5.0, 0.2)
    s = np.cos(2 * np.pi * t)
    ax.plot(t, s, color="blue")
    ax.annotate(
        "text",
        xy=(4.0, 1.0),
        xycoords="data",
        xytext=(4.5, 1.5),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "ec": "r"},
    )
    ax.annotate(
        "arrowstyle",
        xy=(0, 1),
        xycoords="data",
        xytext=(-50, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        "no arrow",
        xy=(0, 1),
        xycoords="data",
        xytext=(50, -30),
        textcoords="offset pixels",
    )
    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
