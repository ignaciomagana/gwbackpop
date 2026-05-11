#!/bin/bash
# =============================================================================
# run_catalog_gwtc3.slurm
#
# SLURM array job to run BackPop single-event inference over the clean GWTC-3
# BBH catalog.  One SLURM task per event, all tasks run in parallel.
#
# Event selection criteria (following Wong+2023):
#   - Confident BBH detections from GWTC-2.1 / GWTC-3
#   - Primary mass below the PISN gap (m1 < ~45 M_sun detector frame) so
#     COSMIC can produce consistent progenitors
#   - No confident BNS events (GW170817, GW190425)
#   - GW190814 included separately (the Lucky Strikes reference event)
#   - Events flagged as marginal or uncertain classification excluded
#
# Config: lucky_strikes (full 17D — both kicks free)
# Both 2D and 3D likelihood modes are run per event.
#
# Submit:
#   sbatch run_catalog_gwtc3.slurm
#
# Override PE samples directory:
#   sbatch --export=ALL,PE_DIR=/path/to/samples run_catalog_gwtc3.slurm
#
# Run a specific subset (e.g. tasks 0–4):
#   sbatch --array=0-4 run_catalog_gwtc3.slurm
# =============================================================================

#SBATCH --job-name=backpop_gwtc3
#SBATCH --output=logs/backpop_%A_%a.out
#SBATCH --error=logs/backpop_%A_%a.err
#SBATCH --array=0-8          # 47 events total (0-indexed); adjust if list changes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # Nautilus worker threads — matches lucky_strikes runtime
#SBATCH --mem=32G
#SBATCH --time=48:00:00       # 4hr wall time; GW190814-like hard events may need 6hr
#SBATCH --partition=RM # adjust to your cluster partition
#SBATCH --account=phy230054p                                          

# ---------------------------------------------------------------------------
# Cluster-specific setup — edit for your system
# ---------------------------------------------------------------------------

# Conda environment with COSMIC (array alpha1), Nautilus, pesummary
CONDA_ENV="backpop"

# Root directory of the BackPop codebase
BACKPOP_DIR="/hildafs/projects/phy230014p/magana/backpop_claude"

# Root directory containing all PE samples HDF5 files
# Expected layout: ${PE_DIR}/IGWN-GWTC*-v*-<EVENT>_*_PEDataRelease_mixed_nocosmo.h5
PE_DIR="${PE_DIR:-/hildafs/projects/phy220048p/share/GWTC-PESamples/gwtc3_bbh_1peryr}"

# Results root
RESULTS_ROOT="${BACKPOP_DIR}/results"

cd "${BACKPOP_DIR}"
mkdir -p logs

# Nautilus settings — conservative; reduce for faster turnaround
NLIVE=2000
NEFF=10000
FMT="pdf"

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

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "============================================================"
echo " Task:    ${SLURM_ARRAY_TASK_ID} / $((N_EVENTS-1))"
echo " Event:   ${EVENT}"
echo " Node:    $(hostname)"
echo " CPUs:    ${SLURM_CPUS_PER_TASK}"
echo " Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Locate PE samples file
# Tries common GWTC naming patterns; fails clearly if not found.
# ---------------------------------------------------------------------------

find_samples() {
    local event="$1"
    local base_event
    # Strip sub-second suffix for search (some files use truncated names)
    base_event="${event%%_*}"
    # Try exact match first, then glob
    local f
    f=$(find "${PE_DIR}" -maxdepth 1 \
        -name "*${event}*PEDataRelease*mixed*nocosmo*.h5" \
        -o -name "*${event}*PEDataRelease*mixed*.h5" 2>/dev/null | head -1)
    if [ -z "$f" ]; then
        f=$(find "${PE_DIR}" -maxdepth 1 \
            -name "*${base_event}*PEDataRelease*mixed*.h5" 2>/dev/null | head -1)
    fi
    echo "$f"
}

SAMPLES=$(find_samples "${EVENT}")

if [ -z "${SAMPLES}" ]; then
    echo "ERROR: PE samples file not found for ${EVENT} in ${PE_DIR}"
    echo "  Searched for: *${EVENT}*PEDataRelease*mixed*.h5"
    echo "  Available files:"
    ls "${PE_DIR}"/*.h5 2>/dev/null | head -20
    exit 1
fi

echo "Samples: ${SAMPLES}"

# ---------------------------------------------------------------------------
# Helper: elapsed time since epoch second
# ---------------------------------------------------------------------------
elapsed() {
    local dt=$(( $(date +%s) - $1 ))
    printf "%dh %02dm %02ds" $(( dt/3600 )) $(( (dt%3600)/60 )) $(( dt%60 ))
}

# ---------------------------------------------------------------------------
# [1/3] 2D inference — lucky_strikes (16D, flat logZ prior)
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [1/3]  2D inference — lucky_strikes"
echo "─────────────────────────────────────────────────────────────"
T1=$(date +%s)

python run_backpop.py \
    --samples_path             "${SAMPLES}" \
    --event_name               "${EVENT}" \
    --config_name              lucky_strikes \
    --use_redshift_likelihood  False \
    --approximant              "C01:Mixed" \
    --use_pe_weights           True \
    --nlive                    "${NLIVE}" \
    --neff                     "${NEFF}" \
    --resume                   False

echo ">>> 2D done ($(elapsed ${T1}))"

# ---------------------------------------------------------------------------
# [2/3] 3D inference — lucky_strikes_zform (17D, cosmological prior)
# ---------------------------------------------------------------------------

#echo ""
#echo "─────────────────────────────────────────────────────────────"
#echo " [2/3]  3D inference — lucky_strikes_zform"
#echo "─────────────────────────────────────────────────────────────"
#T2=$(date +%s)
#
#python run_backpop.py \
#    --samples_path             "${SAMPLES}" \
#    --event_name               "${EVENT}" \
#    --config_name              lucky_strikes_zform \
#    --use_redshift_likelihood  True \
#    --approximant              "C01:Mixed" \
#    --use_pe_weights           True \
#    --nlive                    "${NLIVE}" \
#    --neff                     "${NEFF}" \
#    --resume                   False
#
#echo ">>> 3D done ($(elapsed ${T2}))"

# ---------------------------------------------------------------------------
# [3/3] Plots
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [3/3]  Plotting"
echo "─────────────────────────────────────────────────────────────"
T3=$(date +%s)

python plot_backpop.py \
    --results_dir  "${RESULTS_ROOT}/${EVENT}/lucky_strikes" \
    --samples_path "${SAMPLES}" \
    --n_samples    5000 \
    --fmt          "${FMT}"

#    --compare_dir  "${RESULTS_ROOT}/${EVENT}/lucky_strikes_zform" \

echo ">>> Plots done ($(elapsed ${T3}))"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo " ${EVENT} complete."
echo " Total wall time: $(elapsed ${SLURM_TASK_T0:-${T1}})"
echo ""
echo " Results:"
echo "   ${RESULTS_ROOT}/${EVENT}/lucky_strikes/"
#echo "   ${RESULTS_ROOT}/${EVENT}/lucky_strikes_zform/"
echo "============================================================"
