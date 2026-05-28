from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ._clip3d import Clip3DMode


@dataclass
class TikzData:
    flavor: Flavors

    externalize_tables: bool = False
    override_externals: bool = False
    include_disclaimer: bool = True
    wrap: bool = True
    add_axis_environment: bool = True
    show_info: bool = False
    strict: bool = False
    standalone: bool = False
    is_in_groupplot_env: bool = False
    clip_3d: Clip3DMode = "none"
    shader: str = "none"

    dpi: int = 100
    font_size: float = 10.0

    axis_width: str | None = None
    axis_height: str | None = None
    externals_search_path: str | None = None
    float_format: str = ".15g"
    table_row_sep: str = "\n"
    base_name: str = ""
    current_axis_title: str = ""

    rel_data_path: Path | None = None
    output_dir: Path = Path()

    tikz_libs: set[str] = field(default_factory=set)
    pgfplots_libs: set[str] = field(default_factory=set)
    rectangle_legends: set[str] = field(default_factory=set)
    extra_axis_parameters: set[str] = field(default_factory=set)
    extra_groupstyle_options: set[str] = field(default_factory=set)
    extra_tikzpicture_parameters: set[str] = field(default_factory=set)
    current_axis_options: set[str] = field(default_factory=set)

    legend_colors: list[str] = field(default_factory=list)
    extra_lines_start: list[str] = field(default_factory=list)

    custom_colors: dict = field(default_factory=dict)
    nb_keys: dict = field(default_factory=dict)

    current_mpl_axes: Axes | None = None
    custom_colormap_id: int = 0


class Flavors(enum.Enum):
    latex = (
        r"\begin{{{}}}",
        r"\end{{{}}}",
        "document",
        """\
\\documentclass{{standalone}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{pgfplots}}
\\DeclareUnicodeCharacter{{2212}}{{-}}
\\usepgfplotslibrary{{{pgfplotslibs}}}
\\usetikzlibrary{{{tikzlibs}}}
\\pgfplotsset{{compat=newest}}
""",
    )
    context = (
        r"\start{}",
        r"\stop{}",
        "text",
        """\
\\setupcolors[state=start]
\\usemodule[tikz]
\\usemodule[pgfplots]
\\usepgfplotslibrary[{pgfplotslibs}]
\\usetikzlibrary[{tikzlibs}]
\\pgfplotsset{{compat=newest}}
% groupplot doesn't define ConTeXt stuff
\\unexpanded\\def\\startgroupplot{{\\groupplot}}
\\unexpanded\\def\\stopgroupplot{{\\endgroupplot}}
""",
    )

    def start(self, what: str) -> str:
        return self.value[0].format(what)

    def end(self, what: str) -> str:
        return self.value[1].format(what)

    def preamble(self, data: TikzData | None = None) -> str:
        if data is None:
            pgfplotslibs = "groupplots,dateplot"
            tikzlibs = "patterns,shapes.arrows"
        else:
            pgfplotslibs = ",".join(data.pgfplots_libs)
            tikzlibs = ",".join(data.tikz_libs)
        return self.value[3].format(pgfplotslibs=pgfplotslibs, tikzlibs=tikzlibs)

    def standalone(self, code: str, data: TikzData | None = None) -> str:
        docenv = self.value[2]
        return f"{self.preamble(data)}{self.start(docenv)}\n{code}\n{self.end(docenv)}"
