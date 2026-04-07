"""Test for AnchoredText conversion (issue #53)."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredText

from .helpers import assert_equality


def plot() -> Figure:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    at = AnchoredText("Test label", loc="upper left")
    ax.add_artist(at)
    return fig


def test() -> None:
    assert_equality(plot, "test_anchored_text_reference.tex")
