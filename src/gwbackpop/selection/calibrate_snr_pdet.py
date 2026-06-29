"""Lightweight diagnostic comparison for semi-analytic SNR-proxy pdet."""
from __future__ import annotations
import argparse, csv, json
import numpy as np
import jax.numpy as jnp
from scipy.special import logsumexp

from gwbackpop.inference.hierarchical import POP_PARAM_NAMES, compute_log_wr_injections_numpy, load_cosmic_merger_catalog_for_selection
from gwbackpop.selection.snr_pdet import make_snr_proxy_pdet_callable


def _hyperpoint_to_vec(item):
    if isinstance(item, dict):
        return np.array([item[n] for n in POP_PARAM_NAMES], dtype=float)
    arr = np.asarray(item, dtype=float)
    if arr.size != len(POP_PARAM_NAMES):
        raise ValueError(f"Hyperpoint lists must have {len(POP_PARAM_NAMES)} entries")
    return arr


def log_alpha_snr_proxy(cosmic, lp_vec, pdet_callable):
    pdet = pdet_callable(cosmic["m1_src"], cosmic["m2_src"], cosmic["z_merger"])
    log_wr = compute_log_wr_injections_numpy(lp_vec, cosmic["theta"], cosmic["params"], cosmic["lo"], cosmic["hi"], cosmic["kick_sigma"], cosmic.get("log_q_proposal"), cosmic.get("log_pop_static"))
    return float(logsumexp(np.where(pdet > 0.0, np.log(pdet), -np.inf) + log_wr) - np.log(cosmic["N_inj"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnostic SNR-proxy pdet alpha comparison/threshold scan.")
    ap.add_argument("--injections_path", required=True)
    ap.add_argument("--hyperpoints_json", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--method", default="orientation_monte_carlo", choices=["hard_threshold", "orientation_monte_carlo", "logistic"])
    ap.add_argument("--rho_threshold", type=float, default=10.0)
    ap.add_argument("--sensitivity_scale", type=float, default=1.0)
    ap.add_argument("--threshold_scan", default=None, help="Optional comma-separated thresholds; writes rows for each.")
    args = ap.parse_args(argv)
    cosmic, *_ = load_cosmic_merger_catalog_for_selection(args.injections_path, [], True)
    hyperpoints = [_hyperpoint_to_vec(x) for x in json.load(open(args.hyperpoints_json))]
    thresholds = [float(x) for x in args.threshold_scan.split(",")] if args.threshold_scan else [args.rho_threshold]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["hyperpoint", "log_alpha_snr_proxy", "log_alpha_lvk_farr", "delta_log_alpha", "rho_threshold", "sensitivity_scale", "method"])
        writer.writeheader()
        for th in thresholds:
            pdet = make_snr_proxy_pdet_callable(method=args.method, rho_threshold=th, sensitivity_scale=args.sensitivity_scale)
            for i, hp in enumerate(hyperpoints):
                writer.writerow(dict(hyperpoint=i, log_alpha_snr_proxy=log_alpha_snr_proxy(cosmic, hp, pdet), log_alpha_lvk_farr=np.nan, delta_log_alpha=np.nan, rho_threshold=th, sensitivity_scale=args.sensitivity_scale, method=args.method))

if __name__ == "__main__":
    main()
