"""Test legend with twinx plot.

See: https://github.com/ErwindeGelder/matplot2tikz/issues/32
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> None:
    # Create data
    t = np.linspace(0, 10, 6)
    y1 = t  # Linear function
    y2 = t**2  # Quadratic function

    # Create plot with secondary axis
    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Plot on both axes
    ax1.plot(t, y1, color='b', label='Linear')
    ax2.plot(t, y2, color='r', label='Quadratic')  # ← On ax2!

    # Combine legends on ax1 (common matplotlib pattern)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
