"""Test multi-line title correctness."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .helpers import assert_equality


def plot() -> Figure:
    # Generate data
    t = np.linspace(0, 10, 100)
    y = np.sin(t)

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plot data
    ax.plot(t, y, color="blue", label="Sine Wave")

    # Add title with a newline character \n
    # Using a raw string (r"") is a best practice for LaTeX-related projects
    ax.set_title(r"Simple Sine Wave Plot" + "\n" + r"With a Multi-line Title")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(visible=True)

    return fig


def test() -> None:
    assert_equality(plot, __file__[:-3] + "_reference.tex")
