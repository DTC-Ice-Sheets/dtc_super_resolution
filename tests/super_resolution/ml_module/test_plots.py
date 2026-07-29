"""Tests for the plots module."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import rioxarray  # noqa: F401  registers the .rio accessor
import xarray as xr
from typeguard import suppress_type_checks


def _make_predictions_da(n_months=5):
    """Create a minimal predictions DataArray."""
    x = np.arange(10, dtype=float)
    y = np.arange(8, dtype=float)
    times = np.arange(n_months, dtype=float)
    data = np.random.rand(n_months, len(y), len(x)).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": times, "y": y, "x": x},
        name="surface_mass_balance_glaciated",
    )
    return da.rio.write_crs("epsg:3031")


@patch("super_resolution.ml_module.plots.plt.close", new=MagicMock())
@patch("super_resolution.ml_module.plots.plt.savefig")
@patch("super_resolution.ml_module.plots.plt.subplots")
def test_predictions_plot_saves_file(mock_subplots, mock_savefig, tmp_path):
    from super_resolution.ml_module.plots import predictions_plot

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    da = _make_predictions_da(n_months=5)
    args = Namespace(output_dir=str(tmp_path), start_date="2021-01-01")

    with suppress_type_checks():
        predictions_plot(da, month=3, args=args)

    mock_savefig.assert_called_once()
    saved_path = mock_savefig.call_args[0][0]
    assert str(tmp_path) in saved_path


@patch("super_resolution.ml_module.plots.plt.close", new=MagicMock())
@patch("super_resolution.ml_module.plots.plt.savefig", new=MagicMock())
@patch("super_resolution.ml_module.plots.plt.subplots")
def test_predictions_plot_uses_correct_month_name(mock_subplots, tmp_path):
    from super_resolution.ml_module.plots import predictions_plot

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    da = _make_predictions_da(n_months=5)
    args = Namespace(output_dir=str(tmp_path), start_date="2021-01-01")

    with suppress_type_checks():
        predictions_plot(da, month=2, args=args)

    # month 2 from start_date 2021-01: start_month=1, plot_month=3 → March
    # xarray .plot() also calls set_title once; check any call contains "March"
    titles = [c[0][0] for c in mock_ax.set_title.call_args_list]
    assert any("March" in t for t in titles)


@patch("super_resolution.ml_module.plots.plt.close")
@patch("super_resolution.ml_module.plots.plt.savefig", new=MagicMock())
@patch("super_resolution.ml_module.plots.plt.subplots")
def test_predictions_plot_closes_figure(mock_subplots, mock_close, tmp_path):
    from super_resolution.ml_module.plots import predictions_plot

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    da = _make_predictions_da(n_months=3)
    args = Namespace(output_dir=str(tmp_path), start_date="2021-03-01")

    with suppress_type_checks():
        predictions_plot(da, month=1, args=args)

    mock_close.assert_called_once()


@patch("super_resolution.ml_module.plots.plt.close", new=MagicMock())
@patch("super_resolution.ml_module.plots.plt.savefig")
@patch("super_resolution.ml_module.plots.plt.subplots")
def test_predictions_plot_filename_contains_year_and_month(mock_subplots, mock_savefig, tmp_path):
    """Saved filename encodes the year and plot month from start_date."""
    from super_resolution.ml_module.plots import predictions_plot

    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_subplots.return_value = (mock_fig, mock_ax)

    da = _make_predictions_da(n_months=5)
    args = Namespace(output_dir=str(tmp_path), start_date="2021-02-01")

    with suppress_type_checks():
        predictions_plot(da, month=0, args=args)

    saved_path = str(mock_savefig.call_args[0][0])
    # plot_month = 2 + 0 = 2, year = 2021
    assert "2021" in saved_path
    assert "_2_" in saved_path or "2021_2" in saved_path
