from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import webcolors
from matplotlib.colors import ColorConverter

if TYPE_CHECKING:
    from ._tikzdata import TikzData

# RGB values (as taken from xcolor.dtx):
builtin_colors = {
    "white": [1, 1, 1],
    "lightgray": [0.75, 0.75, 0.75],
    "gray": [0.5, 0.5, 0.5],
    "darkgray": [0.25, 0.25, 0.25],
    "black": [0, 0, 0],
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "brown": [0.75, 0.5, 0.25],
    "lime": [0.75, 1, 0],
    "orange": [1, 0.5, 0],
    "pink": [1, 0.75, 0.75],
    "purple": [0.75, 0, 0.25],
    "teal": [0, 0.5, 0.5],
    "violet": [0.5, 0, 0.5],
    # The colors cyan, magenta, yellow, and olive are also
    # predefined by xcolor, but their RGB approximation of the
    # native CMYK values is not very good. Don't use them here.
}


def _get_closest_colour_name(rgb: np.ndarray) -> tuple[str, int]:
    try:
        wnames: list[str] = webcolors.names("css3")
    except AttributeError:  # For older versions of webcolors
        wnames = sorted(webcolors.CSS3_NAMES_TO_HEX.keys())
    match = wnames[0]
    mindiff = 195076  # = 255**2 * 3 + 1 (maximum difference possible + 1)
    for name in wnames:
        wc_rgb = webcolors.name_to_rgb(name)

        diff = (
            int(rgb[0] - wc_rgb.red) ** 2
            + int(rgb[1] - wc_rgb.green) ** 2
            + int(rgb[2] - wc_rgb.blue) ** 2
        )
        if diff < mindiff:
            match = name
            mindiff = diff

        if mindiff == 0:
            break

    return match, mindiff


def mpl_color2xcolor(
    data: TikzData,
    matplotlib_color: str
    | tuple[float, float, float]
    | tuple[float, float, float, float]
    | tuple[str | tuple[float, float, float], float]
    | tuple[tuple[float, float, float, float], float]
    | np.ndarray,
) -> tuple[str, np.ndarray]:
    """Translates a matplotlib color specification into a proper LaTeX xcolor."""
    # Ensure type is right.
    if isinstance(matplotlib_color, np.ndarray):
        matplotlib_color = tuple(matplotlib_color)

    # Convert it to RGBA.
    rgba = ColorConverter().to_rgba(matplotlib_color)
    my_col = np.array(rgba)

    # If the alpha channel is exactly 0, then the color is really 'none'
    # regardless of the RGB channels.
    if my_col[-1] == 0.0:
        return "none", my_col

    # Check if it exactly matches any of the colors already available.
    # This case is actually treated below (alpha==1), but that loop
    # may pick up combinations with black before finding the exact
    # match. Hence, first check all colors.
    for name, rgb in builtin_colors.items():
        if all(my_col[:3] == rgb):
            return name, my_col

    # Don't handle gray colors separately. They can be specified in xcolor as
    #
    #  {gray}{0.6901960784313725}
    #
    # but this float representation hides the fact that this is actually an
    # RGB255 integer value, 176.

    # convert to RGB255
    rgb255 = np.array(my_col[:3] * 255, dtype=int)

    name, diff = _get_closest_colour_name(rgb255)
    if diff > 0:
        if np.all(my_col[0] == my_col[:3]):
            name = f"{name}{rgb255[0]}"
        else:
            name = f"{name}{rgb255[0]}{rgb255[1]}{rgb255[2]}"
    data.custom_colors[name] = ("RGB", ",".join([str(val) for val in rgb255]))

    return name, my_col
