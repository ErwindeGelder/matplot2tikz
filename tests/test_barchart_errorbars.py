"""Bar Chart With Errorbar test.

This tests plots a bar chart with error bars.  The errorbars need to be drawn
at the correct z-order to be successful.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(3)
    y1 = [1, 2, 3]
    y1err = [0.1, 0.2, 0.5]
    y2 = [3, 2, 4]
    y2err = [0.4, 0.2, 0.5]
    y3 = [5, 3, 1]
    y3err = [0.1, 0.2, 0.1]
    w = 0.25

    err_bar_style = {"ecolor": "black", "lw": 5, "capsize": 8, "capthick": 5}

    ax.bar(x - w, y1, w, color="b", yerr=y1err, align="center", error_kw=err_bar_style)
    ax.bar(x, y2, w, color="g", yerr=y2err, align="center", error_kw=err_bar_style)
    ax.bar(x + w, y3, w, color="r", yerr=y3err, align="center", error_kw=err_bar_style)

    return fig


def test() -> None:
    try:
        assert_equality(plot, __file__[:-3] + "_reference.tex")
    except AssertionError:
        # Try other output, which is the old output with Python 3.9 and below.
        assert_equality(plot, __file__[:-3] + "_reference2.tex")
