#!/bin/bash
# =============================================================================
# run_hierarchical_3d.sh
#
# Hierarchical BackPop population inference — 3D likelihood mode.
# RECOMMENDED PRODUCTION SCRIPT for selection-corrected hierarchical inference.
# Uses lucky_strikes_zform posteriors (KDE over mc, q, z_merger +
# Andrews+2021 cosmological priors on z_form and logZ).
# Sampler: NumPyro NUTS (HMC), implemented in gwbackpop.inference.hierarchical.
#
# Prerequisites:
#   1. gwbackpop.cosmology fix deployed (z_merger_from_t_delay bug fixed).
#   2. SLURM catalog re-runs finished: results/*/lucky_strikes_zform/log_z.npy
#   3. COSMIC merger catalog re-run with fixed gwbackpop.cosmology:
#      injections/gwtc3_cosmic_mergers.npz  (old pre-fix NPZ is invalid)
#   4. LVK found injection file available.
#   5. Optional: run_hierarchical_2d.sh completed for a diagnostic comparison.
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
#   ALLOW_INCONSISTENT_SELECTION_MODEL=False  set True only for intentional legacy diagnostics
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_ROOT="${RESULTS_ROOT:-./results}"
CONFIG_NAME="lucky_strikes_zform"

INJECTIONS_PATH="${INJECTIONS_PATH:-./injections/gwtc3_cosmic_mergers.npz}"
# Build with: gwbackpop-run-injections --config_name lucky_strikes_zform --likelihood_mode 3D --output_path "$INJECTIONS_PATH" ...
# The 3D injection metadata must say likelihood_mode=3D, uses_z_form=True,
# uses_sfr_prior=True, and uses_logZ_given_z_prior=True to match 3D events.
INJECTION_CONFIG_NAME="${INJECTION_CONFIG_NAME:-lucky_strikes_zform}"
LVK_FOUND_PATH="${LVK_FOUND_PATH:-./injections/endo3_bbhpop-LIGO-T2100113-v12.hdf5}"

NUM_WARMUP="${NUM_WARMUP:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_CHAINS="${NUM_CHAINS:-4}"
N_SAMPLES="${N_SAMPLES:-10000}"
LVK_N_FOUND_MAX="${LVK_N_FOUND_MAX:-5000}"
ALLOW_INCONSISTENT_SELECTION_MODEL="${ALLOW_INCONSISTENT_SELECTION_MODEL:-False}"  # explicit legacy/diagnostic override

OUTPUT_DIR="${RESULTS_ROOT}/hierarchical/${CONFIG_NAME}/nuts/lvk_farr"
DIR_2D="${RESULTS_ROOT}/hierarchical/lucky_strikes/nuts/lvk_farr"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

for cmd in gwbackpop-run-hierarchical; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: ${cmd} not found. Install the package with: python -m pip install -e '.[test]'"
        exit 1
    fi
done

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

# Validate injection campaign was built with fixed gwbackpop.cosmology
echo ""
echo ">>> Checking injection campaign integrity..."
python -c "
import numpy as np, sys
data  = np.load('${INJECTIONS_PATH}', allow_pickle=True)
z_max = float(data['z_merger'].max())
if z_max > 10:
    print(f'ERROR: z_merger max = {z_max:.1f} in injection NPZ.')
    print('The pre-fix gwbackpop.cosmology was used — z_merger values are wrong.')
    print('Delete the NPZ and re-run gwbackpop-run-injections --config_name lucky_strikes with the fixed gwbackpop.cosmology.')
    sys.exit(1)
print(f'Injection NPZ valid: z_merger max = {z_max:.3f}')
" || exit 1

N_EVENTS=$(find "${RESULTS_ROOT}" -path "*/${CONFIG_NAME}/log_z.npy" 2>/dev/null | wc -l)
echo ""
echo "Completed ${CONFIG_NAME} events found: ${N_EVENTS}"
if [ "${N_EVENTS}" -lt 2 ]; then
    echo "ERROR: Need at least 2 completed ${CONFIG_NAME} events."
    echo "Submit: sbatch workflows/slurm/run_catalog_gwtc3.slurm"
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
python - "${RESULTS_ROOT}" "${CONFIG_NAME}" "${INJECTIONS_PATH}" "${LVK_FOUND_PATH}" "${ALLOW_INCONSISTENT_SELECTION_MODEL}" "3D" <<'PY'
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

gwbackpop-run-hierarchical \
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
echo "   a) Re-run with gwbackpop-run-hierarchical using a Nautilus backend when available — gives log Z"
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