#!/bin/bash
# =============================================================================
# run_hierarchical_3d.sh
#
# Hierarchical BackPop population inference — 3D likelihood mode.
# Uses lucky_strikes_zform posteriors (KDE over mc, q, z_merger +
# Andrews+2021 cosmological priors on z_form and logZ).
# Sampler: NumPyro NUTS (HMC), implemented in hierarchical_backpop_jax.py.
#
# Prerequisites:
#   1. cosmo_prior.py fix deployed (z_merger_from_t_delay bug fixed).
#   2. SLURM catalog re-runs finished: results/*/lucky_strikes_zform/log_z.npy
#   3. COSMIC merger catalog re-run with fixed cosmo_prior.py:
#      injections/gwtc3_cosmic_mergers.npz  (old pre-fix NPZ is invalid)
#   4. LVK found injection file available.
#   5. run_hierarchical_2d.sh completed — 3D result is compared against 2D.
#
# Output: results/hierarchical/lucky_strikes_zform/nuts/lvk_farr/
#   (same structure as 2D output — see run_hierarchical_2d.sh)
#
# Usage:
#   chmod +x run_hierarchical_3d.sh
#   ./run_hierarchical_3d.sh
#
# Overrides (environment variables):
#   RESULTS_ROOT=/path
#   INJECTIONS_PATH=/path/to/gwtc3_cosmic_mergers.npz
#   LVK_FOUND_PATH=/path/to/endo3_bbhpop.hdf5
#   NUM_WARMUP=500        NUTS warmup (adaptation) steps
#   NUM_SAMPLES=1000      NUTS posterior samples per chain
#   NUM_CHAINS=4          independent chains (need >= 2 for R-hat)
#   N_SAMPLES=10000       per-event importance sampling draws
#   LVK_N_FOUND_MAX=5000  LVK injections for K matrix
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT="${RESULTS_ROOT:-./results}"
CONFIG_NAME="lucky_strikes_zform"

INJECTIONS_PATH="${INJECTIONS_PATH:-./injections/gwtc3_cosmic_mergers.npz}"
# Build with: python run_injections.py --config_name lucky_strikes_zform --likelihood_mode 3D --output_path "$INJECTIONS_PATH" ...
# The 3D injection metadata must say likelihood_mode=3D, uses_z_form=True,
# uses_sfr_prior=True, and uses_logZ_given_z_prior=True to match 3D events.
INJECTION_CONFIG_NAME="${INJECTION_CONFIG_NAME:-lucky_strikes_zform}"
LVK_FOUND_PATH="${LVK_FOUND_PATH:-./injections/endo3_bbhpop-LIGO-T2100113-v12.hdf5}"

NUM_WARMUP="${NUM_WARMUP:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_CHAINS="${NUM_CHAINS:-4}"
N_SAMPLES="${N_SAMPLES:-10000}"
LVK_N_FOUND_MAX="${LVK_N_FOUND_MAX:-5000}"

OUTPUT_DIR="${RESULTS_ROOT}/hierarchical/${CONFIG_NAME}/nuts/lvk_farr"
DIR_2D="${RESULTS_ROOT}/hierarchical/lucky_strikes/nuts/lvk_farr"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "============================================================"
echo " BackPop Hierarchical — 3D (NUTS/HMC)"
echo " Config:       ${CONFIG_NAME}"
echo " Results root: ${RESULTS_ROOT}"
echo " Injections:   ${INJECTIONS_PATH}"
echo " LVK found:    ${LVK_FOUND_PATH}"
echo " Output:       ${OUTPUT_DIR}"
echo " NUTS: ${NUM_CHAINS} chains x ${NUM_WARMUP} warmup + ${NUM_SAMPLES} samples"
echo " K matrix subsampled to N_found=${LVK_N_FOUND_MAX}"
echo "============================================================"

# Validate injection campaign was built with fixed cosmo_prior.py
echo ""
echo ">>> Checking injection campaign integrity..."
python -c "
import numpy as np, sys
data  = np.load('${INJECTIONS_PATH}', allow_pickle=True)
z_max = float(data['z_merger'].max())
if z_max > 10:
    print(f'ERROR: z_merger max = {z_max:.1f} in injection NPZ.')
    print('The pre-fix cosmo_prior.py was used — z_merger values are wrong.')
    print('Delete the NPZ and re-run run_injections.py --config_name lucky_strikes with the fixed cosmo_prior.py.')
    sys.exit(1)
print(f'Injection NPZ valid: z_merger max = {z_max:.3f}')
" || exit 1

N_EVENTS=$(find "${RESULTS_ROOT}" -path "*/${CONFIG_NAME}/log_z.npy" 2>/dev/null | wc -l)
echo ""
echo "Completed ${CONFIG_NAME} events found: ${N_EVENTS}"
if [ "${N_EVENTS}" -lt 2 ]; then
    echo "ERROR: Need at least 2 completed ${CONFIG_NAME} events."
    echo "Submit: sbatch run_catalog_gwtc3.slurm"
    exit 1
fi

for f in "${INJECTIONS_PATH}" "${LVK_FOUND_PATH}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: File not found: ${f}"
        exit 1
    fi
done

# Check for 2D results to compare against
echo ""
if [ -f "${DIR_2D}/summary.csv" ]; then
    echo "2D results found at ${DIR_2D} — will compute Bayes factor after run."
else
    echo "No 2D results found yet. Run run_hierarchical_2d.sh first for comparison."
fi

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
# Summary and model comparison
# ---------------------------------------------------------------------------

ELAPSED=$(( $(date +%s) - START ))
printf "\nWall time: %dh %02dm %02ds\n" \
    $(( ELAPSED/3600 )) $(( (ELAPSED%3600)/60 )) $(( ELAPSED%60 ))

echo ""
echo "============================================================"
echo " 3D hierarchical inference complete."
echo ""
echo " Outputs in: ${OUTPUT_DIR}/"

# NUTS does not compute log Z — no Bayes factor from posterior samples alone.
# Use harmonic mean estimator or bridge sampling if model comparison is needed.
echo ""
echo " NOTE: NUTS does not compute log Z (marginal likelihood)."
echo " For 2D vs 3D model comparison, either:"
echo "   a) Re-run with hierarchical_backpop.py (Nautilus — gives log Z)"
echo "   b) Use ArviZ bridge sampling: arviz.waic() or loo() on both runs"
echo ""

# Print 3D posterior summary
python -c "
import numpy as np, os, csv
csv_path = '${OUTPUT_DIR}/summary.csv'
if os.path.exists(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f'  {\"Parameter\":<20s}  {\"Median\":>10s}  {\"90% CI\":>20s}  {\"R-hat\":>8s}')
    print('  ' + '-'*62)
    for row in rows:
        lo = float(row[\"q05\"]); hi = float(row[\"q95\"]); med = float(row[\"q50\"])
        print(f'  {row[\"parameter\"]:<20s}  {med:10.3f}  [{lo:.3f}, {hi:.3f}]  {row[\"r_hat\"]:>8s}')
" 2>/dev/null || true

# Parameter shift 2D → 3D as a proxy for z_merger information content
if [ -f "${DIR_2D}/summary.csv" ] && [ -f "${OUTPUT_DIR}/summary.csv" ]; then
    echo ""
    echo " 2D vs 3D parameter shifts (proxy for z_merger information):"
    python -c "
import numpy as np, csv

def read_summary(path):
    with open(path) as f:
        return {r['parameter']: float(r['q50']) for r in csv.DictReader(f)}

s2d = read_summary('${DIR_2D}/summary.csv')
s3d = read_summary('${OUTPUT_DIR}/summary.csv')

print(f'  {\"Parameter\":<20s}  {\"2D median\":>10s}  {\"3D median\":>10s}  {\"shift\":>10s}')
print('  ' + '-'*54)
for k in s2d:
    v2, v3 = s2d[k], s3d.get(k, float('nan'))
    shift = v3 - v2
    print(f'  {k:<20s}  {v2:10.3f}  {v3:10.3f}  {shift:+10.3f}')
" 2>/dev/null || true
fi

echo "============================================================"