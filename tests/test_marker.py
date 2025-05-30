"""Test different marker options."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    fig, ax = plt.subplots()
    with plt.style.context("ggplot"):
        t = np.linspace(0, 2 * np.pi, 11)
        s = np.sin(t)
        c = np.cos(t)
        s2 = np.sin(2 * t)
        ax.plot(t, s, "ko-", mec="r", markevery=2, markersize=3)
        ax.plot(t, c, "ks--", mec="r", markevery=3)
        ax.scatter(t, s2, s=100, c="black", marker="+")
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.set_title("Simple plot")
        ax.grid(visible=True)
    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
