"""CLI entry point for hierarchical BackPop population plots."""
from __future__ import annotations

import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np


PLOT_NAMES = (
    "corner_population",
    "population_distributions_95ci",
    "posteriors_CE_flim",
)


def _load_flat_samples(samples_path: Path) -> tuple[np.ndarray, list[str]]:
    data = np.load(samples_path)
    if "arr_0" in data and len(data.files) == 1:
        samples = np.asarray(data["arr_0"])
        names = [f"param_{i}" for i in range(samples.shape[1])]
        return samples.reshape(-1, samples.shape[-1]), names
    names = list(data.files)
    columns = [np.asarray(data[name]).reshape(-1) for name in names]
    return np.column_stack(columns), names


def _write_summary_copy(summary_path: Path, output_path: Path) -> None:
    with summary_path.open(newline="", encoding="utf-8") as src, output_path.open("w", newline="", encoding="utf-8") as dst:
        rows = list(csv.reader(src))
        csv.writer(dst).writerows(rows)


def plot_hierarchical(results_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str] | None = None, fmt: str = "pdf") -> None:
    results_path = Path(results_dir)
    out_path = Path(output_dir) if output_dir is not None else results_path
    out_path.mkdir(parents=True, exist_ok=True)

    samples_path = results_path / "samples.npz"
    summary_path = results_path / "summary.csv"
    missing = [str(path.name) for path in (samples_path, summary_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete hierarchical BackPop result directory. "
            f"Missing required file(s): {', '.join(missing)}. Expected: samples.npz, summary.csv."
        )

    samples, names = _load_flat_samples(samples_path)
    fig = corner.corner(samples, labels=names, quantiles=[0.05, 0.5, 0.95], show_titles=True)
    corner_path = out_path / f"corner_population.{fmt}"
    fig.savefig(corner_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved: {corner_path}")

    summary_out = out_path / "population_summary.csv"
    _write_summary_copy(summary_path, summary_out)
    print(f"  Saved: {summary_out}")

    for stem in PLOT_NAMES[1:]:
        src = results_path / f"{stem}.pdf"
        if src.is_file() and out_path != results_path:
            dst = out_path / src.name
            dst.write_bytes(src.read_bytes())
            print(f"  Copied: {dst}")


def parse_args():
    parser = ArgumentParser(description="Plot hierarchical BackPop population results.")
    parser.add_argument("--results_dir", required=True, help="Path to a hierarchical BackPop output directory containing samples.npz and summary.csv.")
    parser.add_argument("--output_dir", default=None, help="Directory to save regenerated plots (default: same as results_dir).")
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"], help="Output figure format.")
    return parser.parse_args()


def main():
    opts = parse_args()
    plot_hierarchical(opts.results_dir, opts.output_dir, opts.fmt)


__all__ = ["main", "plot_hierarchical"]
