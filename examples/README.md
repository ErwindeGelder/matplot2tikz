# 3D examples

Run the gallery generator from the repository root:

```powershell
$env:PYTHONPATH = "src"
python examples/plot_3d_gallery.py
```

or on POSIX shells:

```sh
PYTHONPATH=src python examples/plot_3d_gallery.py
```

The script writes standalone PGFPlots examples into `examples/output/tex/`. If
`pdflatex` and `pdftoppm` are available, it also compiles the examples and writes
PNG previews into `examples/output/png/`.
