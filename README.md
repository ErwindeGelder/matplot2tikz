# matplot2tikz
The artist formerly known as <em>tikzplotlib</em>.

![PyPI - Version](https://img.shields.io/pypi/v/matplot2tikz)
![GitHub License](https://img.shields.io/github/license/ErwindeGelder/matplot2tikz)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/matplot2tikz)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![codecov](https://codecov.io/gh/ErwindeGelder/matplot2tikz/graph/badge.svg?token=YE2VNWL9HI)](https://codecov.io/gh/ErwindeGelder/matplot2tikz)
[![PyPI Downloads](https://static.pepy.tech/badge/matplot2tikz/month)](https://pepy.tech/projects/matplot2tikz)

This is matplot2tikz, a Python tool for converting matplotlib figures into
[PGFPlots](https://www.ctan.org/pkg/pgfplots) ([PGF/TikZ](https://www.ctan.org/pkg/pgf))
figures like

![](https://raw.githubusercontent.com/ErwindeGelder/matplot2tikz/refs/heads/main/example.png)

for native inclusion into LaTeX or ConTeXt documents.

The output of matplot2tikz is in [PGFPlots](https://github.com/pgf-tikz/pgfplots/), a TeX
library that sits on top of [PGF/TikZ](https://en.wikipedia.org/wiki/PGF/TikZ) and
describes graphs in terms of axes, data etc. Consequently, the output of matplot2tikz

-   retains more information,
-   can be more easily understood, and
-   is more easily editable

than [raw TikZ output](https://matplotlib.org/users/whats_new.html#pgf-tikz-backend).
For example, the matplotlib figure

```python
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

t = np.arange(0.0, 2.0, 0.1)
s = np.sin(2 * np.pi * t)
s2 = np.cos(2 * np.pi * t)
plt.plot(t, s, "o-", lw=4.1)
plt.plot(t, s2, "o-", lw=4.1)
plt.xlabel("time (s)")
plt.ylabel("Voltage (mV)")
plt.title("Simple plot $\\frac{\\alpha}{2}$")
plt.grid(True)

import matplot2tikz

matplot2tikz.save("test.tex")
```

-->
(see above) gives

```latex
\begin{tikzpicture}

\definecolor{chocolate2267451}{RGB}{226,74,51}
\definecolor{dimgray85}{RGB}{85,85,85}
\definecolor{gainsboro229}{RGB}{229,229,229}
\definecolor{steelblue52138189}{RGB}{52,138,189}
\begin{axis}[
axis background/.style={fill=gainsboro229},
axis line style={white},
tick align=outside,
tick pos=left,
title={Simple plot \(\displaystyle \frac{\alpha}{2}\)},
x grid style={white},
xlabel=\textcolor{dimgray85}{time (s)},
xmajorgrids,
xmin=-0.095, xmax=1.995,
xtick style={color=dimgray85},
y grid style={white},
ylabel=\textcolor{dimgray85}{Voltage (mV)},
ymajorgrids,
ymin=-1.1, ymax=1.1,
ytick style={color=dimgray85}
]
\addplot [line width=1.64pt, chocolate2267451, mark=*, mark size=3, mark options={solid}]
table {%
0 0
% [...]
1.9 -0.587785252292473
};
\addplot [line width=1.64pt, steelblue52138189, mark=*, mark size=3, mark options={solid}]
table {%
0 1
% [...]
1.9 0.809016994374947
};
\end{axis}

\end{tikzpicture}
```

(Use `get_tikz_code()` instead of `save()` if you want the code as a string.)

Tweaking the plot is straightforward and can be done as part of your TeX work flow.
[The fantastic PGFPlots manual](http://pgfplots.sourceforge.net/pgfplots.pdf) contains
great examples of how to make your plot look even better.

Of course, not all figures produced by matplotlib can be converted without error.
Notably, [3D plots don't work](https://github.com/matplotlib/matplotlib/issues/7243).

## Installation

matplot2tikz is [available from the Python Package
Index](https://pypi.org/project/matplot2tikz/), so simply do

```
pip install matplot2tikz
```

to install.

## Usage

1. Generate your matplotlib plot as usual.

2. Instead of `pyplot.show()`, invoke matplot2tikz by

    ```python
    import matplot2tikz

    matplot2tikz.save("mytikz.tex")
    # or
    matplot2tikz.save("mytikz.tex", flavor="context")
    ```

    to store the TikZ file as `mytikz.tex`.

3. Add the contents of `mytikz.tex` into your TeX source code. A convenient way of doing
   so is via

    ```latex
    \input{/path/to/mytikz}
    ```

    Also make sure that the packages for PGFPlots and proper Unicode support and are
    included in the header of your document:

    ```latex
    \usepackage[utf8]{inputenc}
    \usepackage{pgfplots}
    \DeclareUnicodeCharacter{2212}{−}
    \usepgfplotslibrary{groupplots,dateplot}
    \usetikzlibrary{patterns,shapes.arrows}
    \pgfplotsset{compat=newest}
    ```

    or:

    ```latex
    \setupcolors[state=start]
    \usemodule[tikz]
    \usemodule[pgfplots]
    \usepgfplotslibrary[groupplots,dateplot]
    \usetikzlibrary[patterns,shapes.arrows]
    \pgfplotsset{compat=newest}
    \unexpanded\def\startgroupplot{\groupplot}
    \unexpanded\def\stopgroupplot{\endgroupplot}
    ```

    You can also get the code via:

    ```python
    import matplot2tikz

    matplot2tikz.Flavors.latex.preamble()
    # or
    matplot2tikz.Flavors.context.preamble()
    ```

4. [Optional] Clean up the figure before exporting to tikz using the `clean_figure`
   command.

    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    # ... do your plotting

    import matplot2tikz

    matplot2tikz.clean_figure()
    matplot2tikz.save("test.tex")
    ```

    The command will remove points that are outside the axes limits, simplify curves and
    reduce point density for the specified target resolution.

## matplot2tikz vs. tikzplotlib

This matplot2tikz library originated from the [tikzplotlib](https://github.com/nschloe/tikzplotlib)
project.
The reason a new library has been created is because tikzplotlib is no longer maintained and 
maintainance could only be done by the single owner of the tikzplotlib library.
If you need to use third-party code that already depends on tikzplotlib, it is suggested to change
the tikzplotlib dependency to matplot2tikz. 
If this is not possible, a workaround is to put the following code *before* importing the 
third-party code:

   ```
   import sys
   import matplot2tikz
   sys.modules["tikzplotlib"] = matplot2tikz
   # Do other imports, e.g., using `import my_third_party_library`
   # If tikzplotlib is used in this library, it will automatically use matplot2tikz instead.
   ```

## Contributing

If you experience bugs, would like to contribute, have nice examples of what matplot2tikz
can do, or if you are just looking for more information, then please visit
[matplot2tikz's GitHub page](https://github.com/ErwindeGelder/matplot2tikz).

For contributing, follow these steps:

1. Download the git repository, e.g., using
   `git clone git@github.com:ErwindeGelder/matplot2tikz.git`.
2. Create a virtual environment, e.g., using `python -m venv venv`.
3. Activate the virtual environment (e.g., on Windows, `venv\Scripts\activate.bat`).
4. Install the necessary libraries using `pip install -e .[dev]`.
5. The main branch is protected, meaning that you cannot directly push changes to this branch. 
   Therefore, if you want to make changes, do so in a seperate branch. For example, you can create 
   a new branch using `git checkout -b feature/my_awesome_new_feature`.
6. Before pushing changes, ensure that the code adheres to the linting rules and that the tests are 
   successful. To run the linting and testing, tox first needs to know where it can find the
   different Python versions that are supported. One way to do so is by making use of pyenv or 
   pyenv-win. Note that you only need to do this once for a single machine.
7. Run `tox`. This does a linting check and runs all test scripts. To manually perform these steps,
   use the following commands (note that to ensure the same output is generated, Python 3.9 is used 
   in step 2 above):
   1. Run `tox -e lint`. If issues arise, fix them. You can do the linting commands manually 
      using:
      1. `ruff format . --check` (remove the `--check` flag to let `ruff` do the formatting)
      2. `ruff check .`
      3. `mypy .`
   2. Run `tox -e py39`.
   3. Run `tox -e py310`.
   4. Run `tox -e py311`.
   5. Run `tox -e py312`.
   6. Run `tox -e py313`.
   7. Run `tox -e combine-test-reports`
8. Check if the tests covered everything using the coverage report in 
   `/reports/coverage_html/index.html`.

   NOTE: Currently, now all code is covered. Ideally, all code is covered, but for now, ensure that 
   all *new* code is covered by the testing.
9. Push changes to GitHub. If everything is OK and you want to merge your changes to the `main`
   branch, create a pull request.
   Ideally, there is at least one reviewer who reviews the pull request before the merge.

Note that currently only "Code owners" can merge pull requests onto the `main` branch. This is to
ensure that not everyone can break the main code (even unintentially). If you want to be a "Code
owner", let us know!

## License

matplot2tikz is published under the [MIT
license](https://en.wikipedia.org/wiki/MIT_License).
