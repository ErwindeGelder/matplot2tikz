"""Process text objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
from matplotlib.font_manager import font_scalings
from matplotlib.patches import ArrowStyle, BoxStyle, FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Annotation, Text

from . import _color

if TYPE_CHECKING:
    from ._tikzdata import TikzData


def draw_text(data: TikzData, obj: Text) -> list[str]:
    """Paints text on the graph.

    :return: Content for tikz plot.
    """
    content: list[str] = []
    properties: list[str] = []
    style: list[str] = []
    ff = data.float_format
    tikz_pos = _get_tikz_pos(data, obj, content)

    text = obj.get_text()

    if text in ["", data.current_axis_title]:
        # Text nodes which are direct children of Axes are typically titles.  They are
        # already captured by the `title` property of pgfplots axes, so skip them here.
        return content

    size = obj.get_fontsize()
    if isinstance(size, str):
        size = font_scalings[size]
    bbox = obj.get_bbox_patch()
    converter = mpl.colors.ColorConverter()
    # without the factor 0.5, the fonts are too big most of the time.
    scaling = 0.5 * size / data.font_size
    if scaling != 1.0:
        properties.append(f"scale={scaling:{ff}}")

    if bbox is not None:
        _bbox(data, bbox, properties, scaling)

    ha = obj.get_horizontalalignment()
    va = obj.get_verticalalignment()
    anchor = _transform_positioning(ha, va)
    if anchor:
        properties.append(anchor)
    col, _ = _color.mpl_color2xcolor(data, converter.to_rgb(obj.get_color()))
    properties.append(f"text={col}")
    properties.append(f"rotate={obj.get_rotation():.1f}")

    if obj.get_fontstyle() == "italic":
        style.append("\\itshape")
    elif obj.get_fontstyle() != "normal":
        msg = f"Object style '{obj.get_fontstyle()}' not implemented."
        raise NotImplementedError(msg)

    # get_weights returns a numeric value in the range 0-1000 or one of (value in parenthesis)
    # `ultralight` (100) `light` (200), `normal` (400), `regular` (400), `book` (400),
    # `medium` (500), `roman` (500), `semibold` (600), `demibold` (600), `demi` (600), `bold` (700),
    # `heavy` (800), `extra bold` (800), `black` (900)
    # (from matplotlib/font_manager.py)
    weight = obj.get_fontweight()
    min_weight_bold = 550
    if weight in ["semibold", "demibold", "demi", "bold", "heavy", "extra bold", "black"] or (
        isinstance(weight, int) and weight > min_weight_bold
    ):
        style.append("\\bfseries")

    if "\n" in text:
        # http://tex.stackexchange.com/a/124114/13262
        properties.append(f"align={ha}")
        # Manipulating the text here is actually against mpl2tikz's policy not
        # to do that. On the other hand, newlines should translate into
        # newlines.
        # We might want to remove this here in the future.
        text = text.replace("\n ", "\\\\")

    props = ",\n  ".join(properties)
    text = " ".join([*style, text])
    content.append(f"\\draw {tikz_pos} node[\n  {props}\n]{{{text}}};\n")
    return content


def _get_tikz_pos(data: TikzData, obj: Text, content: list[str]) -> str:
    """Gets the position in tikz format."""
    pos = _annotation(data, obj, content) if isinstance(obj, Annotation) else obj.get_position()

    if isinstance(pos, str):
        return pos
    if obj.axes:
        # If the coordinates are relative to an axis, use `axis cs`.
        return f"(axis cs:{pos[0]:{data.float_format}},{pos[1]:{data.float_format}})"
    # relative to the entire figure, it's a getting a littler harder. See
    # <http://tex.stackexchange.com/a/274902/13262> for a solution to the
    # problem:
    return (
        f"({{$(current bounding box.south west)!{pos[0]:{data.float_format}}!"
        "(current bounding box.south east)$}"
        "|-"
        f"{{$(current bounding box.south west)!{pos[1]:{data.float_format}}!"
        "(current bounding box.north west)$})"
    )


def _transform_positioning(horizontal_alignment: str, vertical_aligment: str) -> str:
    """Converts matplotlib positioning to pgf node positioning.

    Not quite accurate but the results are equivalent more or less.
    """
    if horizontal_alignment == "center" and vertical_aligment == "center":
        return ""

    ha_mpl_to_tikz = {"right": "east", "left": "west", "center": ""}
    va_mpl_to_tikz = {"top": "north", "bottom": "south", "center": "", "baseline": "base"}
    anchor = " ".join(
        [va_mpl_to_tikz[vertical_aligment], ha_mpl_to_tikz[horizontal_alignment]]
    ).strip()
    return f"anchor={anchor}"


def _parse_annotation_coords(float_format: str, coords: str, xy: tuple[float, float]) -> str:
    """Convert a coordinate name and xy into a tikz coordinate string."""
    if coords == "data":
        x, y = xy
        return f"(axis cs:{x:{float_format}},{y:{float_format}})"
    if coords in [
        "figure points",
        "figure pixels",
        "figure fraction",
        "axes points",
        "axes pixels",
        "axes fraction",
        "data",
        "polar",
    ]:
        raise NotImplementedError
    # unknown
    raise NotImplementedError


def _get_arrow_style(data: TikzData, obj: FancyArrowPatch) -> list:
    # get a style string from a FancyArrowPatch
    arrow_translate = {
        "-": ["-"],
        "->": ["->"],
        "<-": ["<-"],
        "<->": ["<->"],
        "<|-": ["latex-"],
        "-|>": ["-latex"],
        "<|-|>": ["latex-latex"],
        "]-": ["|-"],
        "-[": ["-|"],
        "]-[": ["|-|"],
        "|-|": ["|-|"],
        "]->": ["]->"],
        "<-[": ["<-["],
        "simple": ["-latex", "very thick"],
        "fancy": ["-latex", "very thick"],
        "wedge": ["-latex", "very thick"],
    }
    style_cls = type(obj.get_arrowstyle())

    # Sometimes, mpl adds new arrow styles to the ArrowStyle._style_list dictionary.
    # To support multiple mpl versions, check in a loop instead of a dictionary lookup.
    latex_style = None
    for key, value in arrow_translate.items():
        if key not in ArrowStyle._style_list:  # type: ignore[attr-defined]  # noqa: SLF001  (there is no other way; not all ArrowStyle contain the .arrow attribute)
            continue

        if ArrowStyle._style_list[key] == style_cls:  # type: ignore[attr-defined]  # noqa: SLF001
            latex_style = value
            break

    if latex_style is None:
        msg = f"Unknown arrow style {style_cls}"
        raise NotImplementedError(msg)

    col, _ = _color.mpl_color2xcolor(data, obj.get_edgecolor())
    return [*latex_style, "draw=" + col]


def _annotation(data: TikzData, obj: Annotation, content: list[str]) -> str | tuple[float, float]:
    ann_xy = obj.xy
    ann_xycoords = obj.xycoords
    if not isinstance(ann_xycoords, str):
        # Anything else except for explicit positioning is not supported yet
        return obj.get_position()
    ann_xytext = obj.xyann
    ann_textcoords = obj.anncoords

    ff = data.float_format

    try:
        xy_pos = _parse_annotation_coords(ff, ann_xycoords, ann_xy)
    except NotImplementedError:
        # Anything else except for explicit positioning is not supported yet
        return obj.get_position()

    # special cases only for text_coords
    if ann_textcoords == "offset points":
        x, y = ann_xytext
        unit = "pt"
        text_pos = f"{xy_pos} ++({x:{ff}}{unit},{y:{ff}}{unit})"
    else:
        try:
            text_pos = _parse_annotation_coords(ff, ann_xycoords, ann_xytext)
        except NotImplementedError:
            # Anything else except for explicit positioning is not supported yet
            return obj.get_position()

    if obj.arrow_patch:
        style = ",".join(_get_arrow_style(data, obj.arrow_patch))
        the_arrow = f"\\draw[{style}] {text_pos} -- {xy_pos};\n"
        content.append(the_arrow)
    return text_pos


def _bbox(data: TikzData, bbox: FancyBboxPatch, properties: list[str], scaling: float) -> None:
    bbox_style = bbox.get_boxstyle()
    if bbox.get_fill():
        facecolor, _ = _color.mpl_color2xcolor(data, bbox.get_facecolor())
        if facecolor:
            properties.append(f"fill={facecolor}")
    edgecolor, _ = _color.mpl_color2xcolor(data, bbox.get_edgecolor())
    if edgecolor:
        properties.append(f"draw={edgecolor}")
    ff = data.float_format
    line_width = bbox.get_linewidth() * 0.4
    properties.append(f"line width={line_width:{ff}}pt")
    inner_sep = bbox_style.pad * data.font_size  # type: ignore[attr-defined]
    properties.append(f"inner sep={inner_sep:{ff}}pt")
    if bbox.get_alpha():
        properties.append(f"fill opacity={bbox.get_alpha()}")

    # Process the style and linestyle of the bounding box.
    _bbox_style(data, bbox_style, properties)
    _bbox_linestyle(bbox, properties, scaling)


def _bbox_style(
    data: TikzData,
    bbox_style: BoxStyle
    | BoxStyle.Round
    | BoxStyle.RArrow
    | BoxStyle.LArrow
    | BoxStyle.DArrow
    | BoxStyle.Circle
    | BoxStyle.Roundtooth
    | BoxStyle.Sawtooth
    | BoxStyle.Square,
    properties: list[str],
) -> None:
    if isinstance(bbox_style, BoxStyle.Round):
        properties.append("rounded corners")
    elif isinstance(bbox_style, BoxStyle.RArrow):
        data.tikz_libs.add("shapes.arrows")
        properties.append("single arrow")
    elif isinstance(bbox_style, BoxStyle.LArrow):
        data.tikz_libs.add("shapes.arrows")
        properties.append("single arrow")
        properties.append("shape border rotate=180")
    elif isinstance(bbox_style, BoxStyle.DArrow):
        data.tikz_libs.add("shapes.arrows")
        properties.append("double arrow")
    elif isinstance(bbox_style, BoxStyle.Circle):
        properties.append("circle")
    elif isinstance(bbox_style, BoxStyle.Roundtooth):
        properties.append("decorate")
        properties.append("decoration={snake,amplitude=0.5,segment length=3}")
    elif isinstance(bbox_style, BoxStyle.Sawtooth):
        properties.append("decorate")
        properties.append("decoration={zigzag,amplitude=0.5,segment length=3}")
    elif not isinstance(bbox_style, BoxStyle.Square):
        msg = f"bbox_style '{type(bbox_style)}' not implemented."
        raise NotImplementedError(msg)


def _bbox_linestyle(bbox: FancyBboxPatch, properties: list[str], scaling: float) -> None:
    bbox_ls = bbox.get_linestyle()
    if bbox_ls == "dotted":
        properties.append("dotted")
    elif bbox_ls == "dashed":
        properties.append("dashed")
    elif bbox_ls == "dashdot":
        s1 = 1.0 / scaling
        s3 = 3.0 / scaling
        s6 = 6.0 / scaling
        properties.append(f"dash pattern=on {s1:.3g}pt off {s3:.3g}pt on {s6:.3g}pt off {s3:.3g}pt")
    elif bbox_ls != "solid":
        msg = f"bbox line style '{bbox_ls}' not implemented."
        raise NotImplementedError(msg)
