#!/bin/bash
# =============================================================================
# run_hierarchical_2d.sh
#
# Hierarchical BackPop population inference — 2D likelihood mode.
# DIAGNOSTIC/COMPARISON WORKFLOW, not the recommended production path.
# Use 2D selection-corrected inference only with a self-consistent 2D
# injection campaign; production selection-corrected inference should use
# run_hierarchical_3d.sh because selection effects are mass-redshift dependent.
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
#   ALLOW_INCONSISTENT_SELECTION_MODEL=False  set True only for intentional legacy diagnostics
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT="${RESULTS_ROOT:-./results}"
CONFIG_NAME="lucky_strikes"

INJECTIONS_PATH="${INJECTIONS_PATH:-./injections/gwtc3_cosmic_mergers.npz}"
# Build with: python run_injections.py --config_name lucky_strikes --likelihood_mode 2D --output_path "$INJECTIONS_PATH" ...
# The 2D injection metadata must say likelihood_mode=2D, uses_z_form=False,
# uses_sfr_prior=False, and uses_logZ_given_z_prior=False so selection uses
# the same flat-logZ/no-population-z_form base measure as the 2D events.
INJECTION_CONFIG_NAME="${INJECTION_CONFIG_NAME:-lucky_strikes}"
LVK_FOUND_PATH="${LVK_FOUND_PATH:-./injections/endo3_bbhpop-LIGO-T2100113-v12.hdf5}"

# NUTS sampler settings
NUM_WARMUP="${NUM_WARMUP:-1000}"      # adaptation steps (no live points — this is HMC)
NUM_SAMPLES="${NUM_SAMPLES:-2000}"   # posterior samples per chain
NUM_CHAINS="${NUM_CHAINS:-2}"        # must be >= 2 for R-hat diagnostics

# Data settings
N_SAMPLES="${N_SAMPLES:-5000}"          # per-event importance sampling draws
LVK_N_FOUND_MAX="${LVK_N_FOUND_MAX:-30000}"  # subsample LVK found injections for speed
ALLOW_INCONSISTENT_SELECTION_MODEL="${ALLOW_INCONSISTENT_SELECTION_MODEL:-False}"  # explicit legacy/diagnostic override
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

# Validate mode metadata and required fields before launching an expensive run.
echo ""
echo ">>> Checking event/injection metadata consistency..."
python - "${RESULTS_ROOT}" "${CONFIG_NAME}" "${INJECTIONS_PATH}" "${LVK_FOUND_PATH}" "${ALLOW_INCONSISTENT_SELECTION_MODEL}" "2D" <<'PY'
import glob
import os
import sys
import numpy as np

results_root, config_name, injections_path, lvk_found_path, allow_raw, expected_mode = sys.argv[1:7]
allow = str(allow_raw).strip().lower() in {"1", "true", "yes", "y"}
expected_3d = expected_mode == "3D"
required_event_keys = {"params_in", "lower_bound", "upper_bound", "event_name", "likelihood_mode", "uses_z_form", "uses_sfr_prior", "uses_logZ_given_z_prior"}
required_event_files = {"points.npy", "log_w.npy", "log_z.npy", "metadata.npz"}
required_injection_keys = {"theta", "m1_src", "m2_src", "z_merger", "params", "lower_bound", "upper_bound", "N_inj", "N_merge", "likelihood_mode", "uses_z_form", "uses_sfr_prior", "uses_logZ_given_z_prior"}

def fail(msg):
    if allow:
        print(f"OVERRIDE WARNING: {msg}")
    else:
        print(f"ERROR: {msg}")
        print("Set ALLOW_INCONSISTENT_SELECTION_MODEL=True only for an intentional legacy/diagnostic override.")
        sys.exit(1)

def scalar(npz, key):
    val = npz[key]
    if getattr(val, "shape", ()) == ():
        return val.item()
    if val.size == 1:
        item = val.ravel()[0]
        return item.item() if hasattr(item, "item") else item
    return val

def as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y"}

if not injections_path or not lvk_found_path:
    fail("Selection correction is not enabled: both INJECTIONS_PATH and LVK_FOUND_PATH are required by this workflow script.")
for path, label in [(injections_path, "injection NPZ"), (lvk_found_path, "LVK found injection file")]:
    if not os.path.isfile(path):
        print(f"ERROR: Missing {label}: {path}")
        sys.exit(1)

event_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(results_root, "*", config_name, "log_z.npy")))
if len(event_dirs) < 2:
    print(f"ERROR: Need at least 2 completed {config_name} events; found {len(event_dirs)}.")
    sys.exit(1)

for d in event_dirs:
    missing_files = sorted(f for f in required_event_files if not os.path.isfile(os.path.join(d, f)))
    if missing_files:
        print(f"ERROR: Event directory {d} is missing required files: {missing_files}")
        sys.exit(1)
    meta = np.load(os.path.join(d, "metadata.npz"), allow_pickle=True)
    missing_keys = sorted(required_event_keys - set(meta.files))
    if missing_keys:
        fail(f"Event metadata {d}/metadata.npz is missing mode/required keys: {missing_keys}")
        continue
    mode = str(scalar(meta, "likelihood_mode")).upper()
    flags = {k: as_bool(scalar(meta, k)) for k in ["uses_z_form", "uses_sfr_prior", "uses_logZ_given_z_prior"]}
    if mode != expected_mode or any(v != expected_3d for v in flags.values()):
        fail(f"Event metadata in {d} does not match {expected_mode}: likelihood_mode={mode}, flags={flags}")

inj = np.load(injections_path, allow_pickle=True)
missing_inj = sorted(required_injection_keys - set(inj.files))
if missing_inj:
    fail(f"Injection metadata/file is missing required keys: {missing_inj}")
else:
    inj_mode = str(scalar(inj, "likelihood_mode")).upper()
    inj_flags = {k: as_bool(scalar(inj, k)) for k in ["uses_z_form", "uses_sfr_prior", "uses_logZ_given_z_prior"]}
    if inj_mode != expected_mode or any(v != expected_3d for v in inj_flags.values()):
        fail(f"Injection metadata does not match {expected_mode}: likelihood_mode={inj_mode}, flags={inj_flags}")
    if int(np.ravel(inj["N_merge"])[0]) <= 0 or len(inj["z_merger"]) == 0:
        print("ERROR: Injection catalog has no merging/redshift samples for selection correction.")
        sys.exit(1)

print(f"Metadata preflight OK: {len(event_dirs)} {expected_mode} events, matching {expected_mode} injections, LVK/Farr selection enabled.")
PY

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
    --allow_inconsistent_selection_model "${ALLOW_INCONSISTENT_SELECTION_MODEL}" \
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