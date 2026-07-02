#!/bin/bash
# =============================================================================
# run_catalog_gwtc3_cosmic410_9ev.slurm
#
# Regenerate the 9 O1/O2 single-event BackPop runs with cosmic-popsynth==4.1.0.
#
# IMPORTANT:
#   gwbackpop-run-event writes to:
#       results/<EVENT>/<CONFIG>
#   relative to the current working directory.
#
# This script uses a small work directory with a symlink:
#       ${WORKDIR}/results -> ${RUN_ROOT}/${RESULTS_SUBDIR}
#
# Default output:
#       /hildafs/projects/phy230054p/share/backpop_results/results_cosmic410/<EVENT>/lucky_strikes
#
# To overwrite/use the old results directory instead:
#       sbatch --export=ALL,RESULTS_SUBDIR=results workflows/slurm/run_catalog_gwtc3_cosmic410_9ev.slurm
#
# Submit:
#       mkdir -p /hildafs/projects/phy230054p/share/backpop_results/logs
#       sbatch workflows/slurm/run_catalog_gwtc3_cosmic410_9ev.slurm
#
# Run subset:
#       sbatch --array=0-2 workflows/slurm/run_catalog_gwtc3_cosmic410_9ev.slurm
# =============================================================================

#SBATCH --job-name=backpop_c410
#SBATCH --output=/hildafs/projects/phy230054p/share/backpop_results/logs/backpop_c410_%A_%a.out
#SBATCH --error=/hildafs/projects/phy230054p/share/backpop_results/logs/backpop_c410_%A_%a.err
#SBATCH --array=0-50%10
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=64:00:00
#SBATCH --partition=SIRIUS
#SBATCH --account=phy230054p


# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------

CONDA_ENV="${CONDA_ENV:-gwbackpop-cosmic410}"

CODE_DIR="${CODE_DIR:-/hildafs/projects/phy230014p/magana/src/gwbackpop}"
RUN_ROOT="${RUN_ROOT:-/hildafs/projects/phy230054p/share/backpop_results}"

# Default keeps COSMIC-4.1 products separate from the old scalar-COSMIC products.
# Set RESULTS_SUBDIR=results only if you intentionally want to overwrite/reuse old paths.
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results_cosmic410}"
RESULTS_ROOT="${RUN_ROOT}/${RESULTS_SUBDIR}"

# Workdir only exists so gwbackpop-run-event can write to ./results/...
WORKDIR="${RUN_ROOT}/work_${RESULTS_SUBDIR}"

LOG_DIR="${RUN_ROOT}/logs"

PE_DIR="${PE_DIR:-/hildafs/projects/phy220048p/share/GWTC-PESamples/gwtc3_bbh_1peryr}"

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}" "${WORKDIR}"
ln -sfn "${RESULTS_ROOT}" "${WORKDIR}/results"

# ---------------------------------------------------------------------------
# Nautilus / plotting settings
# ---------------------------------------------------------------------------

NLIVE="${NLIVE:-2000}"
NEFF="${NEFF:-10000}"
FMT="${FMT:-pdf}"
APPROXIMANT="${APPROXIMANT:-C01:Mixed}"
SUPPORT_GATE="${SUPPORT_GATE:-none}"
SUPPORT_HDI="${SUPPORT_HDI:-0.999}"

# ---------------------------------------------------------------------------
# Clean GWTC-3 BBH event list
# Ordered roughly by network SNR (high to low) within O1/O2/O3a/O3b.
# Events excluded from default run listed at the bottom with reasons.
# ---------------------------------------------------------------------------

EVENTS=(
    # ---- O1 ----
    "GW150914"          # reference; ~30+35 M_sun, high SNR
    "GW151226"          # ~14+8 M_sun
    # ---- O2 ----
    "GW170104"          # ~31+20 M_sun
    "GW170608"          # ~12+7 M_sun; lowest mass clean O2 BBH
    "GW170729"          # ~51+34 M_sun; near-PISN but included — check posterior
    "GW170809"          # ~35+24 M_sun
    "GW170814"          # triple-detector; ~31+25 M_sun
    "GW170818"          # ~36+27 M_sun
    "GW170823"          # ~40+29 M_sun
    # ---- O3a ----
    "GW190408_181802"
    "GW190412"          # asymmetric mass ratio; m1~30, m2~8
    "GW190413_052954"
    "GW190413_134308"
    "GW190421_213856"
    "GW190503_185404"
    "GW190512_180714"
    "GW190513_205428"
    "GW190514_065416"
    "GW190517_055101"
    "GW190519_153544"
    "GW190521_074359"   # near PISN boundary — expect poor COSMIC coverage
    "GW190527_092055"
    "GW190602_175927"
    "GW190620_030421"
    "GW190630_185205"
    "GW190701_203306"
    "GW190706_222641"
    "GW190707_093326"
    "GW190708_232457"
    "GW190719_215514"
    "GW190720_000836"
    "GW190727_060333"
    "GW190728_064510"
    "GW190731_140936"
    "GW190803_022701"
    "GW190814"          # Lucky Strikes reference; NSBH / mass-gap secondary
    "GW190828_063405"
    "GW190828_065509"
    "GW190910_112807"
    "GW190915_235702"
    "GW190924_021846"
    "GW190929_012149"
    # ---- O3b ----
    "GW191105_143521"
    "GW191109_010717"
    "GW191129_134029"
    "GW191204_171526"
    "GW191215_223052"
    "GW191222_033537"
    "GW200202_154313"
    "GW200225_060421"
    "GW200311_115853"
)
# Excluded events (add back with care):
#   GW190521         — primary > 85 M_sun, above PISN gap for COSMIC
#   GW190425         — BNS, COSMIC kick stochasticity too high for Nautilus
#   GW170817         — BNS, same reason
#   GW190814_211741  — duplicate entry; GW190814 above is the correct ID

N_EVENTS=${#EVENTS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_EVENTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} exceeds event list length ${N_EVENTS}. Exiting."
    exit 0
fi

EVENT="${EVENTS[${SLURM_ARRAY_TASK_ID}]}"

N_EVENTS=${#EVENTS[@]}
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N_EVENTS}" ]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} exceeds event list length ${N_EVENTS}. Exiting."
    exit 0
fi

EVENT="${EVENTS[${SLURM_ARRAY_TASK_ID}]}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

for cmd in gwbackpop-run-event gwbackpop-plot-event; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: ${cmd} not found after activating ${CONDA_ENV}."
        echo "Install with: python -m pip install -e '.[cosmic]' or activate the correct environment."
        exit 1
    fi
done

echo "============================================================"
echo " Task:    ${SLURM_ARRAY_TASK_ID} / $((N_EVENTS-1))"
echo " Event:   ${EVENT}"
echo " Node:    $(hostname)"
echo " CPUs:    ${SLURM_CPUS_PER_TASK}"
echo " Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Locate PE samples
# ---------------------------------------------------------------------------

find_samples() {
    local event="$1"
    local base_event="${event%%_*}"
    local f=""

    f=$(find "${PE_DIR}" -maxdepth 1 \( \
        -name "*${event}*PEDataRelease*mixed*nocosmo*.h5" -o \
        -name "*${event}*PEDataRelease*mixed*.h5" -o \
        -name "*${event}*.h5" \
        \) 2>/dev/null | head -1)

    if [ -z "${f}" ]; then
        f=$(find "${PE_DIR}" -maxdepth 1 \( \
            -name "*${base_event}*PEDataRelease*mixed*nocosmo*.h5" -o \
            -name "*${base_event}*PEDataRelease*mixed*.h5" -o \
            -name "*${base_event}*.h5" \
            \) 2>/dev/null | head -1)
    fi

    echo "${f}"
}

SAMPLES="$(find_samples "${EVENT}")"

if [ -z "${SAMPLES}" ]; then
    echo "ERROR: PE samples file not found for ${EVENT} in ${PE_DIR}"
    echo "Available HDF5 files:"
    find "${PE_DIR}" -maxdepth 1 -name "*.h5" 2>/dev/null | head -40
    exit 1
fi

echo "Samples: ${SAMPLES}"

elapsed() {
    local start="$1"
    local dt=$(( $(date +%s) - start ))
    printf "%dh %02dm %02ds" $(( dt/3600 )) $(( (dt%3600)/60 )) $(( dt%60 ))
}

T0=$(date +%s)

# ---------------------------------------------------------------------------
# 2D inference — lucky_strikes
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [1/3] 2D inference — lucky_strikes, COSMIC 4.1"
echo "─────────────────────────────────────────────────────────────"
T1=$(date +%s)

gwbackpop-run-event \
    --samples_path             "${SAMPLES}" \
    --event_name               "${EVENT}" \
    --config_name              lucky_strikes \
    --use_redshift_likelihood  False \
    --approximant              "${APPROXIMANT}" \
    --use_pe_weights           True \
    --mass_frame               auto \
    --nlive                    "${NLIVE}" \
    --neff                     "${NEFF}" \
    --support_gate             "${SUPPORT_GATE}" \
    --support_hdi              "${SUPPORT_HDI}" \
    --resume                   False \
    --max_threads              "${SLURM_CPUS_PER_TASK}"

echo ">>> 2D done ($(elapsed "${T1}"))"

RESULT_DIR_2D="${RESULTS_ROOT}/${EVENT}/lucky_strikes"

# ---------------------------------------------------------------------------
# 3D inference — lucky_strikes_zform
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [2/3] 3D inference — lucky_strikes_zform, COSMIC 4.1"
echo "─────────────────────────────────────────────────────────────"
T2=$(date +%s)

gwbackpop-run-event \
    --samples_path             "${SAMPLES}" \
    --event_name               "${EVENT}" \
    --config_name              lucky_strikes_zform \
    --use_redshift_likelihood  True \
    --approximant              "${APPROXIMANT}" \
    --use_pe_weights           True \
    --mass_frame               auto \
    --nlive                    "${NLIVE}" \
    --neff                     "${NEFF}" \
    --support_gate             "${SUPPORT_GATE}" \
    --support_hdi              "${SUPPORT_HDI}" \
    --resume                   False \
    --max_threads              "${SLURM_CPUS_PER_TASK}"

echo ">>> 3D done ($(elapsed "${T2}"))"

RESULT_DIR_3D="${RESULTS_ROOT}/${EVENT}/lucky_strikes_zform"

# ---------------------------------------------------------------------------
# Plot: 2D, 3D comparison
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [3/3] Plotting — lucky_strikes 2D vs 3D"
echo "─────────────────────────────────────────────────────────────"
T3=$(date +%s)

gwbackpop-plot-event \
    --results_dir  "${RESULT_DIR_2D}" \
    --compare_dir  "${RESULT_DIR_3D}" \
    --samples_path "${SAMPLES}" \
    --approximant  "${APPROXIMANT}" \
    --n_samples    5000 \
    --fmt          "${FMT}"

gwbackpop-plot-event \
    --results_dir  "${RESULT_DIR_3D}" \
    --samples_path "${SAMPLES}" \
    --approximant  "${APPROXIMANT}" \
    --n_samples    5000 \
    --fmt          "${FMT}"

echo ">>> Plots done ($(elapsed "${T3}"))"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo " ${EVENT} complete."
echo " Total wall time: $(elapsed "${T0}")"
echo ""
echo " Results:"
echo "   ${RESULT_DIR_2D}/"
echo "   ${RESULT_DIR_3D}/"
echo "============================================================"