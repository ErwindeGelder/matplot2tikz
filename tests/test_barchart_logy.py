"""Bar Chart test with y-axis on log scale.

See https://github.com/ErwindeGelder/matplot2tikz/issues/25
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from .helpers import assert_equality

mpl.use("Agg")


def plot() -> None:
    _, axes = plt.subplots()
    axes.set_yscale("log")
    axes.set_ylim(0.002, 200)
    axes.bar([1, 2, 3, 4, 5], [0.01, 0.1, 1, 10, 100])


def test() -> None:
    assert_equality(plot, "test_barchart_logy_reference.tex")
