"""`softae-thickness fit` — the entry point `fit_geometry_series` did not have.

The module was written, tested against synthetic spectra, exported from the package —
and callable from nothing in `src/`. That is the shape that reads as finished from
inside the module and is the reason this file exists: it drives the analysis the way an
operator would, from a store, not from a hand-built list of `SeriesMember`.
"""

from __future__ import annotations

import numpy as np
import pytest

from softae.analysis.eis_data import EISResult
from softae.core.data_store import DataStore
from softae.tools.thickness import build_parser

L_GAP = L_STRIPE = 0.2
GEOM = L_STRIPE / L_GAP
FREQ = np.geomspace(1e5, 1.0, 40)


def _spectrum(t_um, sigma=1e-5, G_fix=0.0):
    t_cm = t_um * 1e-4
    omega = 2.0 * np.pi * FREQ
    Y = sigma * GEOM * t_cm + G_fix + 1j * omega * (1e-9 / 0.015 * t_cm + 12e-12)
    Z = 1.0 / Y
    return EISResult.from_arrays(channel=0, f=FREQ, z_real=Z.real, z_imag_neg=-Z.imag)


@pytest.fixture()
def store_with_series(tmp_path):
    """A crossed 4-level series, cast and measured, with spectra on disk."""
    ds = DataStore(tmp_path / "proj")
    run_id = ds.start_run("geometry_series")
    order = [100, 200, 150, 250, 200, 100, 250, 150]      # crossed vs channel
    for i, t_um in enumerate(order):
        ch = i + 1
        eis = _spectrum(t_um)
        eis.channel = ch
        path = ds.eis_dir(run_id) / f"eis_ch{ch}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        eis.save(path)
        eis.raw_file_path = str(path)
        ds.record_measurement(run_id, eis, role="sample")
        ds.record_thickness(channel=ch, thickness_um=float(t_um), plan_id="geo-1",
                            run_id=run_id, level_um=float(t_um))
    yield ds, run_id
    ds.close()


def _run(argv, project):
    args = build_parser().parse_args(argv + ["--project", str(project)])
    return args.func(args)


class TestFitCommand:
    def test_it_fits_a_recorded_series_end_to_end(self, store_with_series, capsys):
        ds, run_id = store_with_series
        code = _run(["fit", "--plan", "geo-1"], ds.project_dir)
        out = capsys.readouterr().out
        assert code == 0
        assert "geometry series" in out
        assert "4 levels" in out
        assert "S/cm" in out

    def test_it_reports_the_confound_verdict_not_only_the_fit(self, store_with_series,
                                                              capsys):
        ds, _ = store_with_series
        _run(["fit", "--plan", "geo-1"], ds.project_dir)
        out = capsys.readouterr().out
        assert "confounding: ok" in out

    def test_dead_height_is_unavailable_without_a_measured_g_fixture(
            self, store_with_series, capsys):
        # The honest state, and it names the one artifact that changes it rather than
        # printing a number derived from an assumed fixture conductance.
        ds, _ = store_with_series
        _run(["fit", "--plan", "geo-1", "--fixture", "no-such-fixture"],
             ds.project_dir)
        out = capsys.readouterr().out
        assert "dead height: UNAVAILABLE" in out
        assert "open blank" in out

    def test_a_channel_without_a_spectrum_is_named_not_silently_dropped(
            self, store_with_series, capsys):
        ds, run_id = store_with_series
        ds.record_thickness(channel=99, thickness_um=175.0, plan_id="geo-1",
                            run_id=run_id, level_um=175.0)
        _run(["fit", "--plan", "geo-1"], ds.project_dir)
        err = capsys.readouterr().err
        assert "no EIS spectrum for channel(s): 99" in err

    def test_a_blank_on_the_same_channel_is_not_regressed_as_a_film(
            self, store_with_series, capsys):
        # Regressing a blank against a film thickness fits the fixture and calls it
        # conductivity. The role filter is what prevents it.
        ds, run_id = store_with_series
        blank = _spectrum(1.0, sigma=0.0, G_fix=5e-9)
        blank.channel = 1
        path = ds.eis_dir(run_id) / "blank_ch1.txt"
        blank.save(path)
        blank.raw_file_path = str(path)
        ds.record_measurement(run_id, blank, role="blank_open",
                              electrode_mode="two")
        code = _run(["fit", "--plan", "geo-1"], ds.project_dir)
        out = capsys.readouterr().out
        assert code == 0                      # the film fit still succeeds
        assert "8 samples" in out             # the blank did not join the series

    def test_an_unusable_series_exits_nonzero_so_a_script_cannot_ignore_it(
            self, tmp_path, capsys):
        ds = DataStore(tmp_path / "p2")
        run_id = ds.start_run("bad_series")
        for i, t_um in enumerate([100, 100, 150, 150]):   # confounded + too few levels
            ch = i + 1
            eis = _spectrum(t_um)
            eis.channel = ch
            path = ds.eis_dir(run_id) / f"eis_ch{ch}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            eis.save(path)
            eis.raw_file_path = str(path)
            ds.record_measurement(run_id, eis, role="sample")
            ds.record_thickness(channel=ch, thickness_um=float(t_um), plan_id="bad",
                                run_id=run_id, level_um=float(t_um))
        code = _run(["fit", "--plan", "bad"], ds.project_dir)
        out = capsys.readouterr().out
        assert code != 0
        assert "DESIGN INADEQUATE" in out or "Issues:" in out
        ds.close()
