#!/bin/bash
# =============================================================================
# run_hierarchical_2d.sh
#
# Hierarchical BackPop population inference — 2D likelihood mode.
# Uses lucky_strikes posteriors (KDE over mc, q — flat logZ prior).
# Sampler: NumPyro NUTS (HMC), implemented in hierarchical_backpop_jax.py.
#
# Prerequisites:
#   1. SLURM catalog runs finished:  results/*/lucky_strikes/log_z.npy exist
#   2. COSMIC merger catalog built:  injections/gwtc3_cosmic_mergers.npz exists
#                                    (run_injections.py --config_name lucky_strikes without --pdet_path)
#   3. LVK found injection file:     $LVK_FOUND_PATH exists
#
# Selection effects: Farr (2019) estimator using raw LVK found injections.
# K matrix is kept in system RAM (not GPU VRAM) via jax.pure_callback.
#
# Output: results/hierarchical/lucky_strikes/nuts/lvk_farr/
#   points.npy          flat posterior samples (n_chains*n_samples, 10)
#   samples.npz         per-chain samples (n_chains, n_samples) for R-hat
#   summary.csv         mean, std, 5/50/95 pct, R-hat, N_eff per parameter
#   corner_population.pdf  corner plot of all 10 Λ_pop parameters
#   ppd_masses.pdf         PPD for chirp mass and mass ratio
#   ppd_kicks.pdf          PPD for natal kick velocities vk1, vk2
#   posteriors_CE_flim.pdf marginal posteriors for alpha and flim
#   metadata.npz        run config, convergence stats, wall time
#   run.log             full stdout log
#
# Usage:
#   chmod +x run_hierarchical_2d.sh
#   ./run_hierarchical_2d.sh
#
# Overrides (environment variables):
#   RESULTS_ROOT=/path
#   INJECTIONS_PATH=/path/to/gwtc3_cosmic_mergers.npz
#   LVK_FOUND_PATH=/path/to/endo3_bbhpop.hdf5
#   NUM_WARMUP=500        NUTS warmup (adaptation) steps
#   NUM_SAMPLES=1000      NUTS posterior samples per chain
#   NUM_CHAINS=4          independent chains (need >= 2 for R-hat)
#   N_SAMPLES=10000       per-event importance sampling draws
#   LVK_N_FOUND_MAX=5000  LVK injections to use for K matrix (speed vs variance)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT="${RESULTS_ROOT:-./results}"
CONFIG_NAME="lucky_strikes"

INJECTIONS_PATH="${INJECTIONS_PATH:-./injections/gwtc3_cosmic_mergers.npz}"
# Build with: python run_injections.py --config_name lucky_strikes --output_path "$INJECTIONS_PATH" ...
INJECTION_CONFIG_NAME="${INJECTION_CONFIG_NAME:-lucky_strikes}"
LVK_FOUND_PATH="${LVK_FOUND_PATH:-./injections/endo3_bbhpop-LIGO-T2100113-v12.hdf5}"

# NUTS sampler settings
NUM_WARMUP="${NUM_WARMUP:-1000}"      # adaptation steps (no live points — this is HMC)
NUM_SAMPLES="${NUM_SAMPLES:-2000}"   # posterior samples per chain
NUM_CHAINS="${NUM_CHAINS:-2}"        # must be >= 2 for R-hat diagnostics

# Data settings
N_SAMPLES="${N_SAMPLES:-5000}"          # per-event importance sampling draws
LVK_N_FOUND_MAX="${LVK_N_FOUND_MAX:-30000}"  # subsample LVK found injections for speed
                                             # Farr variance ~ 1/N_found; 5000 >> needed

OUTPUT_DIR="${RESULTS_ROOT}/hierarchical/${CONFIG_NAME}/nuts/lvk_farr"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "============================================================"
echo " BackPop Hierarchical — 2D (NUTS/HMC)"
echo " Config:       ${CONFIG_NAME}"
echo " Results root: ${RESULTS_ROOT}"
echo " Injections:   ${INJECTIONS_PATH}"
echo " LVK found:    ${LVK_FOUND_PATH}"
echo " Output:       ${OUTPUT_DIR}"
echo " NUTS: ${NUM_CHAINS} chains x ${NUM_WARMUP} warmup + ${NUM_SAMPLES} samples"
echo " K matrix subsampled to N_found=${LVK_N_FOUND_MAX}"
echo "============================================================"

N_EVENTS=$(find "${RESULTS_ROOT}" -path "*/${CONFIG_NAME}/log_z.npy" 2>/dev/null | wc -l)
echo ""
echo "Completed ${CONFIG_NAME} events found: ${N_EVENTS}"
if [ "${N_EVENTS}" -lt 2 ]; then
    echo "ERROR: Need at least 2 completed events."
    exit 1
fi

for f in "${INJECTIONS_PATH}" "${LVK_FOUND_PATH}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: File not found: ${f}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "Starting hierarchical inference..."
START=$(date +%s)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

python hierarchical_backpop_jax.py \
    --results_root      "${RESULTS_ROOT}" \
    --config_name       "${CONFIG_NAME}" \
    --injections_path   "${INJECTIONS_PATH}" \
    --lvk_found_path    "${LVK_FOUND_PATH}" \
    --output_dir        "${OUTPUT_DIR}" \
    --n_samples         "${N_SAMPLES}" \
    --num_warmup        "${NUM_WARMUP}" \
    --num_samples       "${NUM_SAMPLES}" \
    --num_chains        "${NUM_CHAINS}" \
    --lvk_n_found_max   "${LVK_N_FOUND_MAX}" \
    2>&1 | tee "${OUTPUT_DIR}/run.log"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

ELAPSED=$(( $(date +%s) - START ))
printf "\nWall time: %dh %02dm %02ds\n" \
    $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 ))

echo ""
echo "============================================================"
echo " 2D hierarchical inference complete."
echo ""
echo " Outputs in: ${OUTPUT_DIR}/"
echo "   points.npy              flat posterior (n_chains*n_samples, 10)"
echo "   samples.npz             per-chain samples for R-hat"
echo "   summary.csv             convergence table"
echo "   corner_population.pdf   Λ_pop corner plot"
echo "   ppd_masses.pdf          mc and q PPDs"
echo "   ppd_kicks.pdf           vk1, vk2 PPDs"
echo "   posteriors_CE_flim.pdf  alpha and flim marginals"
echo "   run.log                 full log"
echo ""
echo " Quick check:"
python -c "
import numpy as np, os
csv_path = '${OUTPUT_DIR}/summary.csv'
if os.path.exists(csv_path):
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f'  {\"Parameter\":<20s}  {\"Median\":>10s}  {\"90% CI\":>20s}  {\"R-hat\":>8s}')
    print('  ' + '-'*62)
    for row in rows:
        lo = float(row[\"q05\"]); hi = float(row[\"q95\"]); med = float(row[\"q50\"])
        rh = row[\"r_hat\"]
        print(f'  {row[\"parameter\"]:<20s}  {med:10.3f}  [{lo:.3f}, {hi:.3f}]  {rh:>8s}')
else:
    print('  summary.csv not found — check run.log for errors')
" 2>/dev/null || true
echo "============================================================"