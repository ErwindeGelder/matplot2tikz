"""Test tick positioning."""

import itertools

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> Figure:
    x = [1, 2, 3, 4]
    y = [1, 4, 9, 6]
    
    fig = plt.figure(figsize=(10, 10))
    
    # repeat=4 gives 16 combinations of (bottom, top, left, right)
    for i, (b, t, l, r) in enumerate(itertools.product([False, True], repeat=4)):
        ax = plt.subplot(4, 4, i + 1)
        plt.plot(x, y, "ro")
        
        # Set the visibility of the tick lines
        # We also turn off the labels so the plots don't overlap
        ax.tick_params(axis="x", which="both", bottom=b, top=t, labelbottom=False)
        ax.tick_params(axis="y", which="both", left=l, right=r, labelleft=False)

    plt.tight_layout()
    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
