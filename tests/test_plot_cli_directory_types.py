import sys

import numpy as np
import pytest

from gwbackpop.plotting import diagnostics


def test_hierarchical_dir_prints_friendly_message(tmp_path, monkeypatch, capsys):
    np.savez(tmp_path / "samples.npz", alpha=np.array([1.0, 2.0]))
    (tmp_path / "summary.csv").write_text("parameter,mean\nalpha,1.5\n", encoding="utf-8")

    called = False

    def fake_load_results(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("hierarchical directories must not load event arrays")

    monkeypatch.setattr(diagnostics, "load_results", fake_load_results)
    monkeypatch.setattr(sys, "argv", ["gwbackpop-plot-event", "--results_dir", str(tmp_path)])

    diagnostics.main()

    output = capsys.readouterr().out
    assert "Detected a hierarchical BackPop output directory" in output
    assert "will not load log_w.npy" in output
    assert "gwbackpop-plot-hierarchical" in output
    assert not called


def test_incomplete_dir_has_clear_missing_files_error(tmp_path, monkeypatch):
    np.save(tmp_path / "points.npy", np.zeros((2, 2)))
    monkeypatch.setattr(sys, "argv", ["gwbackpop-plot-event", "--results_dir", str(tmp_path)])

    with pytest.raises(FileNotFoundError, match=r"Missing required file\(s\): log_w.npy, log_z.npy, blobs.npy, metadata.npz"):
        diagnostics.main()


def test_single_event_dir_calls_load_results(tmp_path, monkeypatch):
    for name in diagnostics.SINGLE_EVENT_REQUIRED_FILES:
        (tmp_path / name).write_bytes(b"placeholder")

    calls = []

    def fake_load_results(results_dir, n_samples):
        calls.append((results_dir, n_samples))
        return {
            "event_name": "event",
            "config_name": "config",
            "mode": "2D",
            "log_z": 0.0,
            "n_eff": 1,
            "params": ["x"],
        }

    monkeypatch.setattr(diagnostics, "load_results", fake_load_results)
    monkeypatch.setattr(diagnostics, "plot_corner_full", lambda res: None)
    monkeypatch.setattr(diagnostics, "plot_corner_zams", lambda res: None)
    monkeypatch.setattr(diagnostics, "plot_corner_physics", lambda res: None)
    monkeypatch.setattr(diagnostics, "plot_corner_kicks", lambda res: None)
    monkeypatch.setattr(diagnostics, "plot_formation_channels", lambda res: None)
    monkeypatch.setattr(diagnostics, "plot_delay_time", lambda res: None)
    monkeypatch.setattr(sys, "argv", ["gwbackpop-plot-event", "--results_dir", str(tmp_path), "--n_samples", "7"])

    diagnostics.main()

    assert calls == [(str(tmp_path), 7)]
