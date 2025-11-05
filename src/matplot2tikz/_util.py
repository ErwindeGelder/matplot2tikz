"""Several utility functions used at various parts of matplot2tikz library."""

from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

import matplotlib.transforms
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D


def has_legend(axes: Axes | Axes3D) -> bool:
    return axes.get_legend() is not None


def get_legend_text(obj: Line2D | PathCollection) -> str | None:
    """Check if line is in legend."""
    if obj.axes is None:
        return None
    leg = obj.axes.get_legend()
    if leg is None:
        return None

    try:
        leg_handles = leg.legend_handles  # matplotlib version >= 3.7.0
    except AttributeError:
        leg_handles = leg.legendHandles  # type: ignore[attr-defined]  # matplotlib version < 3.7.0
    keys = [h.get_label() for h in leg_handles if h is not None]
    values = [t.get_text() for t in leg.texts]

    label = obj.get_label()
    d = dict(zip(keys, values, strict=True))
    if label in d:
        return d[label]

    return None


def transform_to_data_coordinates(
    obj: Line2D, xdata: np.ndarray, ydata: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The coordinates might not be in data coordinates, but could be sometimes in axes coordinates.

    For example, the matplotlib command
      axes.axvline(2)
    will have the y coordinates set to 0 and 1, not to the limits. Therefore, a
    two-stage transform has to be applied:
      1. first transforming to display coordinates, then
      2. from display to data.
    """
    if obj.axes is not None and obj.get_transform() != obj.axes.transData:
        points = np.array([xdata, ydata]).T
        transform = matplotlib.transforms.composite_transform_factory(
            obj.get_transform(), obj.axes.transData.inverted()
        )
        xdata, ydata = transform.transform(points).T
        return xdata, ydata
    return xdata, ydata


_NO_ESCAPE = r"(?<!\\)(?:\\\\)*"
_split_math = re.compile(_NO_ESCAPE + r"\$").split
_replace_mathdefault = functools.partial(
    # Replace \mathdefault (when not preceded by an escape) by empty string.
    re.compile(_NO_ESCAPE + r"(\\mathdefault)").sub,
    "",
)


def _common_texification(text: str) -> str:
    return _tex_escape(text)


# https://github.com/nschloe/tikzplotlib/pull/603
def _tex_escape(text: str) -> str:
    r"""Do some necessary and/or useful substitutions for texts to be included in LaTeX documents.

    This distinguishes text-mode and math-mode by replacing the math separator
    ``$`` with ``\(\displaystyle %s\)``. Escaped math separators (``\$``)
    are ignored.
    """
    # Sometimes, matplotlib adds the unknown command \mathdefault.
    # Not using \mathnormal instead since this looks odd for the latex cm font.
    text = _replace_mathdefault(text)
    text = text.replace("\N{MINUS SIGN}", r"\ensuremath{-}")
    # Work around <https://github.com/matplotlib/matplotlib/issues/15493>
    text = text.replace("&", r"\&")
    text = text.replace("_", r"\_")
    text = text.replace("%", r"\%")
    # split text into normaltext and inline math parts
    parts = _split_math(text)
    for i, s in enumerate(parts):
        if i % 2:  # mathmode replacements
            parts[i] = rf"\(\displaystyle {s}\)"
    return "".join(parts)
