"""Test escaping of some characters.

https://github.com/nschloe/tikzplotlib/issues/332
"""

from collections.abc import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import matplot2tikz

mpl.use("Agg")


def plot() -> Figure:
    fig = plt.figure()
    plt.plot(0, 0, "kx")
    plt.title("Foo & Bar Dogs_N_Cats %")
    plt.xlabel("Foo & Bar Dogs_N_Cats %")
    plt.ylabel("Foo & Bar Dogs_N_Cats %")
    return fig


def _get_tikz_code(plot_func: Callable[[], Figure]) -> str:
    plot_func()
    code = matplot2tikz.get_tikz_code(
        include_disclaimer=False,
        float_format=".8g",
    )
    plt.close("all")
    return code


def test_text_mode_escaping() -> None:
    code = _get_tikz_code(plot)
    escaped = "Foo \\& Bar Dogs\\_N\\_Cats \\%"
    assert f"title={{{escaped}}}" in code
    assert f"xlabel={{{escaped}}}" in code
    assert f"ylabel={{{escaped}}}" in code


def plot_math_label() -> Figure:
    fig = plt.figure()
    plt.plot([0.1, 10.0], [0.0, 1.0])
    plt.xscale("log")
    plt.xlabel(r"$\\log_{10} \\mathrm{X}$")
    return fig


def test_math_mode_underscore_not_escaped() -> None:
    code = _get_tikz_code(plot_math_label)
    xlabel_line = next(line for line in code.splitlines() if line.strip().startswith("xlabel="))
    assert r"\\log_{10} \\mathrm{X}" in xlabel_line
    assert r"\\log\\_{10}" not in xlabel_line


def plot_escaped_dollar() -> Figure:
    fig = plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0])
    plt.xlabel(r"$\text{Price } \${}100 \approx 86\,\text{\euro}_{\text{in mai 2026}}$")
    return fig


def test_escaped_dollar_is_literal() -> None:
    code = _get_tikz_code(plot_escaped_dollar)
    xlabel_line = next(line for line in code.splitlines() if line.strip().startswith("xlabel="))
    assert (
        r"\(\displaystyle \text{Price } \${}100 \approx 86\,"
        r"\text{\euro}_{\text{in mai 2026}}\)" in xlabel_line
    )


def plot_multiple_math_segments() -> Figure:
    fig = plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0])
    plt.xlabel(r"gain $x_1$ vs $x_2$")
    return fig


def test_multiple_math_segments() -> None:
    code = _get_tikz_code(plot_multiple_math_segments)
    xlabel_line = next(line for line in code.splitlines() if line.strip().startswith("xlabel="))
    assert r"\(\displaystyle x_1\)" in xlabel_line
    assert r"\(\displaystyle x_2\)" in xlabel_line
    assert r"x\_1" not in xlabel_line
    assert r"x\_2" not in xlabel_line
