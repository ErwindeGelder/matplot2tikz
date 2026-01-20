"""Helper functions for running the tests."""

import difflib
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.text import Text

import matplot2tikz


def print_tree(obj: Artist, indent: str = "") -> None:
    """Recursively prints the tree structure of the matplotlib object."""
    if isinstance(obj, Text):
        print(indent, type(obj).__name__, f'("{obj.get_text()}")')  # noqa: T201
    else:
        print(indent, type(obj).__name__)  # noqa: T201

    for child in obj.get_children():
        print_tree(child, indent + "  ")


# https://stackoverflow.com/a/845432/353337
def _unidiff_output(expected: str, actual: str) -> str:
    expected_lines = expected.splitlines(keepends=True)
    actual_lines = actual.splitlines(keepends=True)
    diff = difflib.unified_diff(expected_lines, actual_lines)
    return "".join(diff)


def assert_equality(  # type: ignore[no-untyped-def]
    plot: Callable,
    filename: str,
    flavor: str = "latex",
    **extra_get_tikz_code_args,  # noqa: ANN003
) -> None:
    plot()
    code = matplot2tikz.get_tikz_code(
        include_disclaimer=False,
        float_format=".8g",
        flavor=flavor,
        **extra_get_tikz_code_args,
    )
    plt.close("all")

    this_dir = Path(__file__).resolve().parent
    with (this_dir / filename).open(encoding="utf-8") as f:
        reference = f.read()
    try:
        assert reference == code, filename + "\n" + _unidiff_output(reference, code)
    except AssertionError:
        with (this_dir / f"{filename[:-4]}_output.tex").open("w") as f:
            f.write(code)
    assert reference == code, filename + "\n" + _unidiff_output(reference, code)
