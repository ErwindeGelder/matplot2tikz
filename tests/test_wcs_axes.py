"""Test plot with WCS axes."""

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS

import matplot2tikz

mpl.use("Agg")


def test_wcs_axes() -> None:
    """Test that WCS axes are handled without warnings."""
    # Create a simple WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [50, 50]
    wcs.wcs.cdelt = [0.1, 0.1]
    wcs.wcs.crval = [0, -90]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Create a figure with WCS projection
    fig, ax = plt.subplots(subplot_kw={"projection": wcs})

    # Add some data
    rng = np.random.default_rng(1337)
    data = rng.integers(5, size=(2, 4))
    vmin = data.min()
    vmax = data.max()
    mappable = ax.imshow(data, vmin=vmin, vmax=vmax, origin="lower", cmap="viridis")
    ax.grid(color="white", ls="solid", alpha=0.5)
    ax.set(xlabel="Galactic Longitude", ylabel="Galactic Latitude")
    _ = fig.colorbar(mappable, ax=ax, label="Test Unit")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tikz_code = matplot2tikz.get_tikz_code()

        # Filter warnings to only check for the specific warning about unknown objects
        unknown_object_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, UserWarning)
            and "Don't know how to handle object" in str(warning.message)
            and "_WCSAxesArtist" in str(warning.message)
        ]

        # Assert that we didn't get the _WCSAxesArtist warning
        assert len(unknown_object_warnings) == 0, (
            f"Got unexpected warning about _WCSAxesArtist: "
            f"{[str(warning.message) for warning in unknown_object_warnings]}"
        )

    # Basic checks that the output looks reasonable
    assert "\\begin{tikzpicture}" in tikz_code
    assert "\\end{tikzpicture}" in tikz_code
    # Check that image is included
    assert "\\addplot graphics" in tikz_code

    plt.close("all")
