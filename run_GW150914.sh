#!/bin/bash
# =============================================================================
# run_GW150914.sh
#
# Run BackPop inference for GW150914 over the full 17D/18D parameter space
# (lucky_strikes config — both natal kicks free) in 2D and 3D likelihood modes,
# then generate all diagnostic plots for each run and a side-by-side comparison.
#
# Runs (in order):
#   [1] lucky_strikes         (17D, 2D KDE over mc/q)
#   [2] lucky_strikes_zform   (18D, 3D KDE over mc/q/z + Andrews+2021 prior)
#   [3] plot each run individually
#   [4] plot 2D vs 3D comparison
#
# Why full kicks for GW150914?
#   Although the first BH (~36 Msun) is expected to receive a near-zero kick
#   via fallback, we run the full space to:
#     a) validate that vk1 is unconstrained (expected flat posterior)
#     b) keep the catalog runs homogeneous — one config for all events
#     c) provide a reference posterior for the hierarchical sigma_v inference
#   The extra 4 dimensions add ~20-30 min vs lucky_strikes_fixed_vk1.
#
# Runtime estimates (single node, abundant compute):
#   [1] lucky_strikes         ~45-60 min
#   [2] lucky_strikes_zform   ~60-90 min  (one extra dim + cosmo prior eval)
#   [3] plotting              ~2-5 min per run
#   Total wall time:          ~2-3 hr
#
# Usage:
#   chmod +x run_GW150914.sh
#   ./run_GW150914.sh
#
# Overrides (environment variables):
#   SAMPLES=/path/to/file.h5   — PE samples file
#   NLIVE=3000                  — Nautilus n_live
#   NEFF=30000                  — Nautilus n_eff target
#   PLOT_NSAMPLES=10000         — posterior draws for plotting
#   FMT=png                     — figure format (pdf, png, svg)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — all overridable via environment
# ---------------------------------------------------------------------------

EVENT="GW150914"

# GWTC-2.1 public PE samples (mixed approximant, no cosmological prior applied)
SAMPLES="${SAMPLES:-/hildafs/projects/phy220048p/share/GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5}"

APPROXIMANT="C01:Mixed"

# Nautilus settings.
# GW150914 is well-behaved at 17D — nlive=2000 gives clean posteriors.
# Increase to 3000 if posteriors look poorly sampled.
NLIVE="${NLIVE:-2000}"
NEFF="${NEFF:-10000}"

# Plotting
PLOT_NSAMPLES="${PLOT_NSAMPLES:-10000}"
FMT="${FMT:-pdf}"

# Result directories (derived — do not override)
DIR_2D="results/${EVENT}/lucky_strikes"
DIR_3D="results/${EVENT}/lucky_strikes_zform"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "============================================================"
echo " BackPop: ${EVENT}  (full lucky_strikes — 17D/18D)"
echo " Samples: ${SAMPLES}"
echo " nlive=${NLIVE}  neff=${NEFF}"
echo " Figure format: ${FMT}"
echo "============================================================"

if [ ! -f "${SAMPLES}" ]; then
    echo ""
    echo "ERROR: PE samples file not found:"
    echo "  ${SAMPLES}"
    echo ""
    echo "Download from GWOSC (GWTC-2.1 confident):"
    echo "  https://gwosc.org/eventapi/json/GWTC-2.1-confident/GW150914/v4/"
    echo ""
    echo "Or override the path:"
    echo "  SAMPLES=/path/to/file.h5 ./run_GW150914.sh"
    exit 1
fi

for cmd in gwbackpop-run-event gwbackpop-plot; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "ERROR: ${cmd} not found. Install the package with: python -m pip install -e '.[test]'"
        exit 1
    fi
done

START_TOTAL=$(date +%s)

# ---------------------------------------------------------------------------
# Helper: print elapsed time since a given epoch second
# ---------------------------------------------------------------------------
elapsed() {
    local t0=$1
    local t1
    t1=$(date +%s)
    local dt=$(( t1 - t0 ))
    printf "%dh %02dm %02ds" $(( dt/3600 )) $(( (dt%3600)/60 )) $(( dt%60 ))
}

# ---------------------------------------------------------------------------
# [1/4] 2D run — lucky_strikes (17D)
#        KDE over (mc, q), flat logZ prior
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [1/4]  2D inference — lucky_strikes (17D)"
echo "        KDE: (mc, q)  |  logZ prior: flat"
echo "─────────────────────────────────────────────────────────────"
echo ""

T1=$(date +%s)

gwbackpop-run-event \
    --samples_path             "${SAMPLES}" \
    --event_name               "${EVENT}" \
    --config_name              lucky_strikes \
    --use_redshift_likelihood  False \
    --approximant              "${APPROXIMANT}" \
    --use_pe_weights           True \
    --nlive                    "${NLIVE}" \
    --neff                     "${NEFF}" \
    --resume                   False

echo ""
echo ">>> [1/4] Done ($(elapsed ${T1})).  Results: ${DIR_2D}/"

# ---------------------------------------------------------------------------
# [2/4] 3D run — lucky_strikes_zform (18D)
#        KDE over (mc, q, z_merger) + Andrews+2021 cosmological prior
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [2/4]  3D inference — lucky_strikes_zform (18D)"
echo "        KDE: (mc, q, z_merger)  |  P(z_form) x P(logZ|z_form)"
echo "─────────────────────────────────────────────────────────────"
echo ""

T2=$(date +%s)

gwbackpop-run-event \
    --samples_path             "${SAMPLES}" \
    --event_name               "${EVENT}" \
    --config_name              lucky_strikes_zform \
    --use_redshift_likelihood  True \
    --approximant              "${APPROXIMANT}" \
    --use_pe_weights           True \
    --nlive                    "${NLIVE}" \
    --neff                     "${NEFF}" \
    --resume                   False

echo ""
echo ">>> [2/4] Done ($(elapsed ${T2})).  Results: ${DIR_3D}/"

# ---------------------------------------------------------------------------
# [3/4] Individual plots for each run
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [3/4]  Plotting individual runs"
echo "─────────────────────────────────────────────────────────────"

echo ""
echo "  Plotting 2D run (${DIR_2D})..."
T3=$(date +%s)

gwbackpop-plot \
    --results_dir  "${DIR_2D}" \
    --samples_path "${SAMPLES}" \
    --approximant  "${APPROXIMANT}" \
    --n_samples    "${PLOT_NSAMPLES}" \
    --fmt          "${FMT}"

echo "  2D plots done ($(elapsed ${T3}))."

echo ""
echo "  Plotting 3D run (${DIR_3D})..."
T3b=$(date +%s)

gwbackpop-plot \
    --results_dir  "${DIR_3D}" \
    --samples_path "${SAMPLES}" \
    --approximant  "${APPROXIMANT}" \
    --n_samples    "${PLOT_NSAMPLES}" \
    --fmt          "${FMT}"

echo "  3D plots done ($(elapsed ${T3b}))."

# ---------------------------------------------------------------------------
# [4/4] 2D vs 3D comparison plot
# ---------------------------------------------------------------------------

echo ""
echo "─────────────────────────────────────────────────────────────"
echo " [4/4]  2D vs 3D comparison"
echo "─────────────────────────────────────────────────────────────"
echo ""

T4=$(date +%s)

# Write comparison into the 2D directory so both share a common output
gwbackpop-plot \
    --results_dir  "${DIR_2D}" \
    --compare_dir  "${DIR_3D}" \
    --samples_path "${SAMPLES}" \
    --approximant  "${APPROXIMANT}" \
    --n_samples    "${PLOT_NSAMPLES}" \
    --output_dir   "results/${EVENT}/comparison" \
    --fmt          "${FMT}"

echo ""
echo "  Comparison plots done ($(elapsed ${T4}))."

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "============================================================"
echo " ${EVENT} — all runs and plots complete."
echo " Total wall time: $(elapsed ${START_TOTAL})"
echo ""
echo " Inference outputs:"
echo "   ${DIR_2D}/"
echo "     points.npy      posterior samples (17D)"
echo "     log_w.npy       log importance weights"
echo "     log_z.npy       log evidence  ← hierarchical step input"
echo "     blobs.npy       COSMIC bpp + kick tracks"
echo "     metadata.npz    self-describing run config"
echo ""
echo "   ${DIR_3D}/"
echo "     (same structure, 18D — includes z_form)"
echo ""
echo " Figures (${FMT}):"
echo "   ${DIR_2D}/"
echo "     corner_full.${FMT}        all 17 parameters"
echo "     corner_zams.${FMT}        ZAMS initial conditions"
echo "     corner_physics.${FMT}     alpha, flim hyperparams"
echo "     corner_kicks.${FMT}       vk1/vk2 + angles"
echo "     gw_comparison.${FMT}      BackPop mc/q vs LVK PE"
echo "     formation_channels.${FMT} CE vs stable-MT fractions"
echo "     delay_time.${FMT}         t_merge distribution"
echo ""
echo "   ${DIR_3D}/  (same figures + z_form in corners)"
echo ""
echo "   results/${EVENT}/comparison/"
echo "     comparison_2d_vs_3d.${FMT}  shared params overlaid"
echo "                                 Δ log Z reported in title"
echo ""
echo " Key diagnostic:"
echo "   Δ log Z = log Z(3D) - log Z(2D)"
echo "   |Δ log Z| < 1 → redshift adds little constraint for this event"
echo "   |Δ log Z| > 3 → redshift meaningfully tightens the posterior"
echo "============================================================"
