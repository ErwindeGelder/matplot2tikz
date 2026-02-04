"""Test Legend entries for secondary axis plots."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality


# https://github.com/ErwindeGelder/matplot2tikz/issues/32#issue-3413025718
def plot() -> Figure:
    # Create data
    t = np.linspace(0, 10, 10)
    y1 = t  # Linear function
    y2 = t**2  # Quadratic function

    # Create plot with secondary axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot on both axes
    ax1.plot(t, y1, color="b", label="Linear")
    ax2.plot(t, y2, color="r", label="Quadratic")  # ← On ax2!

    # Combine legends on ax1 (common matplotlib pattern)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
