"""
plot_gwbackpop.evolution.cosmic
---------------
Visualise BackPop posterior results from run_gwbackpop.evolution.cosmic.

Reads metadata.npz for all config/param info — no manual config flags needed.

Figures produced (all saved to the results directory):
  corner_full.pdf        — corner plot of every sampled parameter
  corner_zams.pdf        — ZAMS initial conditions only
  corner_physics.pdf     — binary evolution hyperparams (alpha, flim)
  corner_kicks.pdf       — natal kick parameters (if present in config)
  gw_comparison.pdf      — BackPop predicted (mc, q) overlaid on LVK PE samples
  formation_channels.pdf — CE vs stable-MT channel fractions from bpp blobs
  delay_time.pdf         — delay time distribution from COSMIC
  [optional] comparison_2d_vs_3d.pdf — if --compare_dir supplied

Usage
-----
  # Single run
  python plot_gwbackpop.evolution.cosmic \\
      --results_dir results/GW150914/lucky_strikes_fixed_vk1

  # With LVK PE samples overlay on gw_comparison.pdf
  python plot_gwbackpop.evolution.cosmic \\
      --results_dir results/GW150914/lucky_strikes_fixed_vk1 \\
      --samples_path data/GW150914_posterior.h5

  # 2D vs 3D comparison
  python plot_gwbackpop.evolution.cosmic \\
      --results_dir        results/GW150914/lucky_strikes_fixed_vk1 \\
      --compare_dir        results/GW150914/lucky_strikes_fixed_vk1_zform \\
      --samples_path       data/GW150914_posterior.h5

  # Fewer posterior samples for quick preview
  python plot_gwbackpop.evolution.cosmic \\
      --results_dir results/GW150914/lucky_strikes_fixed_vk1 \\
      --n_samples 2000
"""

from __future__ import annotations

import os
import sys
import warnings
from argparse import ArgumentParser

import numpy as np
from gwbackpop.metadata import load_metadata_prefer_json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import corner
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    'font.family':           'serif',
    'font.serif':            ['Times New Roman', 'DejaVu Serif'],
    'text.usetex':           False,
    'mathtext.fontset':      'cm',
    'axes.labelsize':        13,
    'axes.titlesize':        13,
    'xtick.labelsize':       11,
    'ytick.labelsize':       11,
    'legend.fontsize':       11,
    'figure.titlesize':      14,
    'figure.dpi':            150,
    'savefig.bbox':          'tight',
    'savefig.dpi':           200,
})

sns.set_style('ticks')
PALETTE = sns.color_palette('colorblind', 8)
C_ZAMS    = PALETTE[0]   # blue   — ZAMS parameters
C_PHYSICS = PALETTE[1]   # orange — binary evolution hyperparams
C_KICKS   = PALETTE[2]   # green  — natal kicks
C_GW      = PALETTE[3]   # red    — GW PE samples / observables
C_3D      = PALETTE[4]   # purple — 3D / z_form parameters
C_CE      = PALETTE[5]   # brown  — CE channel
C_SMT     = PALETTE[6]   # pink   — stable MT channel
C2        = PALETTE[7]   # grey   — secondary / compare run

CORNER_KWARGS = dict(
    levels      = [0.68, 0.95],
    quantiles   = [0.05, 0.5, 0.95],
    show_titles = True,
    title_fmt   = '.3f',
    title_kwargs= {"fontsize": 11},
    label_kwargs= {"fontsize": 13},
    hist_kwargs = {"linewidth": 1.5, "density": True},
    smooth      = 1.0,
    plot_density= False,
    plot_datapoints=False,
)

# ---------------------------------------------------------------------------
# Parameter groupings
# ---------------------------------------------------------------------------

ZAMS_PARAMS    = {'m1', 'q', 'logtb', 'logZ'}
PHYSICS_PARAMS = {'alpha_1', 'alpha_2', 'flim_1', 'flim_2'}
KICK1_PARAMS   = {'vk1', 'theta1', 'phi1', 'omega1'}
KICK2_PARAMS   = {'vk2', 'theta2', 'phi2', 'omega2'}
KICK_PARAMS    = KICK1_PARAMS | KICK2_PARAMS
COSMO_PARAMS   = {'z_form'}

# Human-readable labels — import from gwbackpop.evolution.cosmic, fall back gracefully
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gwbackpop.evolution.cosmic import PARAM_LABELS, COLS_KEEP, KICK_COLUMNS, BPP_SHAPE, KICK_SHAPE
except ImportError:
    # Fallback labels if gwbackpop.evolution.cosmic is not on the path
    PARAM_LABELS = {
        'm1':      r'$m_1\ [M_\odot]$',
        'q':       r'$q_\mathrm{ZAMS}$',
        'logtb':   r'$\log_{10}(t_b/\mathrm{day})$',
        'logZ':    r'$\log_{10}Z$',
        'alpha_1': r'$\alpha_1$',
        'alpha_2': r'$\alpha_2$',
        'flim_1':  r'$f_\mathrm{lim,1}$',
        'flim_2':  r'$f_\mathrm{lim,2}$',
        'vk1':     r'$v_{k,1}\ [\mathrm{km\,s^{-1}}]$',
        'theta1':  r'$\theta_1$',
        'phi1':    r'$\phi_1$',
        'omega1':  r'$\omega_1$',
        'vk2':     r'$v_{k,2}\ [\mathrm{km\,s^{-1}}]$',
        'theta2':  r'$\theta_2$',
        'phi2':    r'$\phi_2$',
        'omega2':  r'$\omega_2$',
        'z_form':  r'$z_\mathrm{form}$',
    }
    COLS_KEEP = [
        'tphys', 'mass_1', 'mass_2', 'massc_1', 'massc_2',
        'menv_1', 'menv_2', 'kstar_1', 'kstar_2',
        'porb', 'ecc', 'evol_type', 'rad_1', 'rad_2', 'lum_1', 'lum_2',
    ]
    KICK_COLUMNS = [
        'star', 'disrupted', 'natal_kick', 'phi', 'theta', 'mean_anomaly',
        'delta_vsysx_1', 'delta_vsysy_1', 'delta_vsysz_1', 'vsys_1_total',
        'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2', 'vsys_2_total',
        'theta_euler', 'phi_euler', 'psi_euler', 'randomseed',
    ]
    BPP_SHAPE  = (25, len(COLS_KEEP))
    KICK_SHAPE = (2, len(KICK_COLUMNS))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: str, n_samples: int = 10_000) -> dict:
    """Load and process posterior samples from a BackPop results directory.

    Reads points.npy, log_w.npy, log_z.npy, blobs.npy, and metadata.npz.
    Returns a dict with posterior samples, weights, blobs, and metadata.

    Parameters
    ----------
    results_dir : str
        Path to the results directory (e.g. results/GW150914/lucky_strikes_fixed_vk1).
    n_samples : int
        Number of weighted posterior samples to draw for plotting.

    Returns
    -------
    dict with keys:
        samples_df   — pd.DataFrame of posterior draws, shape (n_samples, n_params)
        weights      — normalised posterior weights (full posterior, not just n_samples)
        log_z        — scalar log evidence
        bpp_df       — pd.DataFrame of COSMIC bpp tracks for the n_samples draws
        kick_df      — pd.DataFrame of COSMIC kick info for the n_samples draws
        metadata     — dict loaded from metadata.npz
        params       — list of sampled parameter names
        labels       — list of LaTeX labels for sampled parameters
        event_name   — str
        config_name  — str
        mode         — '2D' or '3D+cosmo'
    """
    def _path(fname):
        return os.path.join(results_dir, fname)

    # ---- Core arrays ----
    points = np.load(_path("points.npy"))
    log_w  = np.load(_path("log_w.npy"))
    log_z  = float(np.load(_path("log_z.npy")).ravel()[0])
    blobs  = np.load(_path("blobs.npy"), allow_pickle=True)

    # ---- Normalised weights and draws ----
    weights = np.exp(log_w - log_z)
    weights = weights / weights.sum()
    n_eff   = int(1.0 / np.sum(weights**2))

    idx = np.random.choice(len(points), size=n_samples, replace=True, p=weights)

    # ---- Metadata ----
    metadata    = load_metadata_prefer_json(_path("metadata.npz"))
    params      = list(metadata.get('params_in', []))
    event_name  = str(metadata.get('event_name', 'unknown'))
    config_name = str(metadata.get('config_name', 'unknown'))
    mode        = str(metadata.get('likelihood_mode', '2D'))

    labels = [PARAM_LABELS.get(p, p) for p in params]

    # ---- Posterior samples ----
    samples_df = pd.DataFrame(points[idx], columns=params)

    # ---- Bpp + kick blobs ----
    blobs_sel = blobs[idx]
    bpp_list  = []
    kick_list = []
    bin_nums  = []

    for i, b in enumerate(blobs_sel):
        bpp_raw  = b['bpp'].reshape(BPP_SHAPE)
        kick_raw = b['kick_info'].reshape(KICK_SHAPE)
        bpp_df_i = pd.DataFrame(bpp_raw, columns=COLS_KEEP)
        # Drop placeholder rows (mass_1 == 0 means COSMIC didn't write there)
        bpp_df_i = bpp_df_i.loc[bpp_df_i.mass_1 > 0]
        bpp_list.append(bpp_df_i)
        kick_list.append(pd.DataFrame(kick_raw, columns=KICK_COLUMNS))
        bin_nums.extend([i] * len(bpp_df_i))

    bpp_df  = pd.concat(bpp_list,  ignore_index=True)
    kick_df = pd.concat(kick_list, ignore_index=True)
    bpp_df['bin_num'] = bin_nums

    return dict(
        samples_df  = samples_df,
        weights     = weights,
        log_z       = log_z,
        n_eff       = n_eff,
        bpp_df      = bpp_df,
        kick_df     = kick_df,
        metadata    = metadata,
        params      = params,
        labels      = labels,
        event_name  = event_name,
        config_name = config_name,
        mode        = mode,
    )


def _figure_title(res: dict) -> str:
    return f"{res['event_name']}  |  {res['config_name']}  |  {res['mode']}  |  log Z = {res['log_z']:.2f}  |  N_eff = {res['n_eff']}"


# ---------------------------------------------------------------------------
# Formation channel classification
# ---------------------------------------------------------------------------

# COSMIC evol_type values for CE events
_CE_EVOL_TYPES = {7, 8, 9, 14, 15}

def classify_channels(bpp_df: pd.DataFrame) -> pd.Series:
    """Classify each binary (by bin_num) as 'CE' or 'Stable MT'.

    Returns a Series indexed by bin_num with values 'CE' or 'Stable MT'.
    CE is identified by the presence of evol_type ∈ {7,8,9,14,15} in the
    binary's evolution track.
    """
    def _channel(grp):
        return 'CE' if grp['evol_type'].isin(_CE_EVOL_TYPES).any() else 'Stable MT'

    return bpp_df.groupby('bin_num').apply(_channel)


# ---------------------------------------------------------------------------
# Figure 1: Full corner plot
# ---------------------------------------------------------------------------

def plot_corner_full(res: dict, color=None, fig=None, label: str = None) -> plt.Figure:
    """Corner plot of all sampled parameters."""
    data   = res['samples_df'].to_numpy()
    labels = res['labels']
    color  = color or C_ZAMS

    kw = dict(CORNER_KWARGS, color=color)
    if label:
        kw['labels'] = labels

    if fig is None:
        fig = corner.corner(data, labels=labels, **kw)
    else:
        corner.corner(data, labels=labels, fig=fig, **kw)

    fig.suptitle(_figure_title(res), fontsize=10, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Figure 2: ZAMS parameters corner
# ---------------------------------------------------------------------------

def plot_corner_zams(res: dict, color=None) -> plt.Figure | None:
    """Corner plot of ZAMS initial conditions: m1, m2 (=q*m1), log tb, log Z."""
    params     = res['params']
    samples_df = res['samples_df'].copy()

    # Convert q_ZAMS → m2_ZAMS for more physical interpretation
    if 'm1' in params and 'q' in params:
        samples_df['m2_zams'] = samples_df['m1'] * samples_df['q']
        zams_cols  = ['m1', 'm2_zams', 'logtb', 'logZ']
        zams_label = [
            r'$m_{1,\mathrm{ZAMS}}\ [M_\odot]$',
            r'$m_{2,\mathrm{ZAMS}}\ [M_\odot]$',
            r'$\log_{10}(t_b/\mathrm{day})$',
            r'$\log_{10}Z$',
        ]
    else:
        zams_cols  = [p for p in params if p in ZAMS_PARAMS]
        zams_label = [PARAM_LABELS.get(p, p) for p in zams_cols]

    if not zams_cols:
        return None

    # Colour by formation channel if bpp available
    channels = classify_channels(res['bpp_df'])
    ch_arr   = channels.values
    n        = len(samples_df)

    data = samples_df[zams_cols].to_numpy()

    fig = corner.corner(
        data, labels=zams_label, color=C_ZAMS,
        **CORNER_KWARGS,
    )

    # Overlay CE channel in a contrasting colour
    ce_mask  = (ch_arr == 'CE')[:n]
    smt_mask = ~ce_mask
    n_ce  = ce_mask.sum()
    n_smt = smt_mask.sum()

    if n_ce > 10:
        corner.corner(
            data[ce_mask], fig=fig, color=C_CE,
            **{**CORNER_KWARGS, 'show_titles': False, 'quantiles': []},
        )

    # Legend
    patches = [
        mpatches.Patch(color=C_ZAMS, label=f'All  (N={n})'),
        mpatches.Patch(color=C_CE,   label=f'CE   ({n_ce}/{n}, {100*n_ce/n:.0f}%)'),
        mpatches.Patch(color=C_SMT,  label=f'Stable MT ({n_smt}/{n}, {100*n_smt/n:.0f}%)'),
    ]
    fig.legend(handles=patches, loc='upper right',
               bbox_to_anchor=(1.0, 1.0), fontsize=11)
    fig.suptitle(f"ZAMS Parameters  —  {_figure_title(res)}", fontsize=9, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Binary physics hyperparameters
# ---------------------------------------------------------------------------

def plot_corner_physics(res: dict) -> plt.Figure | None:
    """Corner plot of binary evolution hyperparams: alpha_1, alpha_2, flim_1, flim_2."""
    params   = res['params']
    phys_cols = [p for p in params if p in PHYSICS_PARAMS]
    if not phys_cols:
        return None

    data   = res['samples_df'][phys_cols].to_numpy()
    labels = [PARAM_LABELS.get(p, p) for p in phys_cols]

    fig = corner.corner(data, labels=labels, color=C_PHYSICS, **CORNER_KWARGS)
    fig.suptitle(f"Binary Evolution Hyperparameters  —  {_figure_title(res)}", fontsize=9, y=1.01)
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Natal kick parameters
# ---------------------------------------------------------------------------

def plot_corner_kicks(res: dict) -> plt.Figure | None:
    """Corner plot of natal kick parameters (only those present in config)."""
    params     = res['params']
    kick_cols  = [p for p in params if p in KICK_PARAMS]
    if not kick_cols:
        return None

    # Split into kick-1 and kick-2 subsets for colour coding
    k1 = [p for p in kick_cols if p in KICK1_PARAMS]
    k2 = [p for p in kick_cols if p in KICK2_PARAMS]

    data   = res['samples_df'][kick_cols].to_numpy()
    labels = [PARAM_LABELS.get(p, p) for p in kick_cols]

    fig = corner.corner(data, labels=labels, color=C_KICKS, **CORNER_KWARGS)
    fig.suptitle(f"Natal Kick Parameters  —  {_figure_title(res)}", fontsize=9, y=1.01)

    # Annotation
    note = ""
    if k1:
        note += f"First SN kick params: {k1}\n"
    if k2:
        note += f"Second SN kick params: {k2}"
    if note:
        fig.text(0.98, 0.98, note.strip(), ha='right', va='top',
                 fontsize=9, transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    return fig


# ---------------------------------------------------------------------------
# Figure 5: GW observable comparison (mc, q)
# ---------------------------------------------------------------------------

def plot_gw_comparison(
    res: dict,
    samples_path: str | None = None,
    approximant: str = "C01:Mixed",
) -> plt.Figure:
    """BackPop predicted (mc, q) overlaid on LVK PE samples.

    If samples_path is None, shows only the BackPop posterior prediction.
    """
    samples_df = res['samples_df']

    # Compute BackPop predicted merger masses from samples_df
    # These are ZAMS masses — need to look in blobs for merger masses
    bpp_df = res['bpp_df']

    # Find the merger row per binary (kstar=14/14, evol_type=3)
    merger_mask = (
        (bpp_df.kstar_1 == 14) & (bpp_df.kstar_2 == 14) & (bpp_df.evol_type == 3)
    )
    mergers = bpp_df.loc[merger_mask].drop_duplicates(subset='bin_num', keep='first')

    m1_bp = mergers['mass_1'].values
    m2_bp = mergers['mass_2'].values
    # Ensure m1 >= m2
    swap  = m1_bp < m2_bp
    m1_bp[swap], m2_bp[swap] = m2_bp[swap], m1_bp[swap]
    mc_bp = (m1_bp * m2_bp)**(3.0/5.0) / (m1_bp + m2_bp)**(1.0/5.0)
    q_bp  = m2_bp / m1_bp

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: mass plane ----
    ax = axes[0]
    ax.scatter(m1_bp, m2_bp, s=4, alpha=0.3, color=C_ZAMS, label='BackPop')
    ax.set_xlabel(r'$m_1\ [M_\odot]$')
    ax.set_ylabel(r'$m_2\ [M_\odot]$')
    ax.set_title('Merger mass plane')

    # ---- Right: (mc, q) plane ----
    ax = axes[1]
    ax.scatter(mc_bp, q_bp, s=4, alpha=0.3, color=C_ZAMS, label='BackPop', zorder=2)

    # Overlay LVK PE samples if provided
    if samples_path is not None and os.path.exists(samples_path):
        try:
            from gwbackpop.evolution.cosmic import zofdL, ddLdz
            from pesummary.io import read
            data    = read(samples_path, package="gw")
            avail   = list(data.samples_dict.keys())
            approx  = approximant if approximant in avail else avail[0]
            s       = data.samples_dict[approx]
            m1d     = np.asarray(s['mass_1'])
            m2d     = np.asarray(s['mass_2'])
            dL      = np.asarray(s['luminosity_distance'])
            z_src   = zofdL(dL)
            m1s     = m1d / (1 + z_src)
            m2s     = m2d / (1 + z_src)
            mc_gw   = (m1s * m2s)**(3/5) / (m1s + m2s)**(1/5)
            q_gw    = np.where(m2s <= m1s, m2s/m1s, m1s/m2s)
            ax.scatter(mc_gw, q_gw, s=6, alpha=0.4, color=C_GW, label='LVK PE', zorder=1)
            axes[0].scatter(m1s, m2s, s=6, alpha=0.4, color=C_GW, label='LVK PE', zorder=1)
        except Exception as e:
            print(f"[plot] Could not load LVK PE samples: {e}")

    ax.set_xlabel(r'$\mathcal{M}_c\ [M_\odot]$')
    ax.set_ylabel(r'$q = m_2/m_1$')
    ax.set_title(r'$(m_c,\ q)$ comparison')
    ax.legend(markerscale=3, framealpha=0.8)
    axes[0].legend(markerscale=3, framealpha=0.8)

    fig.suptitle(f"GW Observable Comparison  —  {_figure_title(res)}", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Formation channels
# ---------------------------------------------------------------------------

def plot_formation_channels(res: dict) -> plt.Figure:
    """Bar chart of CE vs Stable MT fraction with 68% binomial uncertainty."""
    channels = classify_channels(res['bpp_df'])
    n_total  = len(channels)
    n_ce     = (channels == 'CE').sum()
    n_smt    = n_total - n_ce
    f_ce     = n_ce  / n_total
    f_smt    = n_smt / n_total

    # Wilson score 68% CI
    def _wilson(k, n, z=1.0):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2*n)) / denom
        half   = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        return max(centre - half, 0), min(centre + half, 1)

    lo_ce,  hi_ce  = _wilson(n_ce,  n_total)
    lo_smt, hi_smt = _wilson(n_smt, n_total)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # ---- Bar chart ----
    ax = axes[0]
    bars = ax.bar(
        ['CE', 'Stable MT'],
        [f_ce, f_smt],
        color=[C_CE, C_SMT],
        edgecolor='white', linewidth=1.5, width=0.5,
    )
    ax.errorbar(
        [0, 1], [f_ce, f_smt],
        yerr=[[f_ce - lo_ce, f_smt - lo_smt],
              [hi_ce - f_ce, hi_smt - f_smt]],
        fmt='none', color='black', capsize=5, linewidth=2,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Fraction')
    ax.set_title(f'Formation channels  (N={n_total})')
    for bar, frac in zip(bars, [f_ce, f_smt]):
        ax.text(bar.get_x() + bar.get_width()/2, frac + 0.03,
                f'{frac:.2f}', ha='center', va='bottom', fontsize=12)

    # ---- Pie chart ----
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        [n_ce, n_smt],
        labels=[f'CE\n({n_ce})', f'Stable MT\n({n_smt})'],
        colors=[C_CE, C_SMT],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops=dict(fontsize=12),
    )
    ax.set_title('Channel fractions')

    fig.suptitle(f"Formation Channels  —  {_figure_title(res)}", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7: Delay time distribution
# ---------------------------------------------------------------------------

def plot_delay_time(res: dict) -> plt.Figure:
    """Histogram of binary merger delay times from COSMIC bpp tracks."""
    bpp_df = res['bpp_df']
    merger_mask = (
        (bpp_df.kstar_1 == 14) & (bpp_df.kstar_2 == 14) & (bpp_df.evol_type == 3)
    )
    mergers = bpp_df.loc[merger_mask].drop_duplicates(subset='bin_num', keep='first')
    t_merge_gyr = mergers['tphys'].values / 1e3   # Myr → Gyr

    channels = classify_channels(bpp_df)
    bin_nums = mergers['bin_num'].values
    ch_arr   = channels.reindex(bin_nums).values

    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.logspace(np.log10(max(t_merge_gyr.min(), 1e-3)), np.log10(14.0), 30)

    ax.hist(t_merge_gyr, bins=bins, color=C_ZAMS, alpha=0.8,
            label='All', density=True, histtype='stepfilled')

    ce_mask  = ch_arr == 'CE'
    smt_mask = ~ce_mask

    if ce_mask.sum() > 5:
        ax.hist(t_merge_gyr[ce_mask],  bins=bins, color=C_CE,  alpha=0.6,
                label='CE', density=True, histtype='step', linewidth=2)
    if smt_mask.sum() > 5:
        ax.hist(t_merge_gyr[smt_mask], bins=bins, color=C_SMT, alpha=0.6,
                label='Stable MT', density=True, histtype='step', linewidth=2, linestyle='--')

    ax.axvline(13.7, color='grey', linestyle=':', linewidth=1.5, label='Hubble time')
    ax.set_xscale('log')
    ax.set_xlabel(r'$t_\mathrm{delay}\ [\mathrm{Gyr}]$')
    ax.set_ylabel('PDF')
    ax.set_title(f"Delay Time Distribution  —  {_figure_title(res)}", fontsize=9)
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 8: 2D vs 3D comparison
# ---------------------------------------------------------------------------

def plot_2d_vs_3d_comparison(res_2d: dict, res_3d: dict, params_compare: list[str] | None = None) -> plt.Figure:
    """Overlay posteriors from 2D and 3D runs for shared parameters.

    Focuses on parameters most likely to shift: logZ and the binary
    physics hyperparams, since the 3D run applies an informative
    P(logZ|z_form) prior that the 2D run does not.
    """
    # Find shared parameters between both runs
    p2 = set(res_2d['params'])
    p3 = set(res_3d['params'])
    if params_compare is None:
        # Default: show ZAMS + physics, exclude kick angles and z_form
        shared = (p2 & p3) - COSMO_PARAMS - {'theta1','phi1','omega1','theta2','phi2','omega2'}
        params_compare = [p for p in res_2d['params'] if p in shared]

    if not params_compare:
        print("[plot] No shared parameters to compare — skipping 2D vs 3D plot.")
        return None

    data_2d = res_2d['samples_df'][params_compare].to_numpy()
    data_3d = res_3d['samples_df'][params_compare].to_numpy()
    labels  = [PARAM_LABELS.get(p, p) for p in params_compare]

    fig = corner.corner(data_2d, labels=labels, color=C_ZAMS, **CORNER_KWARGS)
    corner.corner(
        data_3d, labels=labels, fig=fig, color=C_3D,
        **{**CORNER_KWARGS, 'show_titles': False, 'quantiles': []},
    )

    patches = [
        mpatches.Patch(color=C_ZAMS, label=f"2D: {res_2d['config_name']}  log Z={res_2d['log_z']:.2f}"),
        mpatches.Patch(color=C_3D,   label=f"3D: {res_3d['config_name']}  log Z={res_3d['log_z']:.2f}"),
    ]
    fig.legend(handles=patches, loc='upper right',
               bbox_to_anchor=(0.98, 0.98), fontsize=11,
               framealpha=0.9)

    delta_logz = res_3d['log_z'] - res_2d['log_z']
    fig.suptitle(
        f"{res_2d['event_name']}  —  2D vs 3D comparison  |  "
        f"Δ log Z = {delta_logz:+.2f}",
        fontsize=10, y=1.01,
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = ArgumentParser(description="Plot BackPop posterior results.")
    p.add_argument("--results_dir",  required=True,
                   help="Path to BackPop results directory.")
    p.add_argument("--compare_dir",  default=None,
                   help="Optional second results directory for 2D vs 3D comparison.")
    p.add_argument("--samples_path", default=None,
                   help="Path to LVK PE samples HDF5 for GW observable overlay.")
    p.add_argument("--approximant",  default="C01:Mixed",
                   help="Posterior approximant label in the PE file.")
    p.add_argument("--n_samples",    type=int, default=10_000,
                   help="Posterior draws for plotting (default 10000).")
    p.add_argument("--output_dir",   default=None,
                   help="Directory to save figures (default: same as results_dir).")
    p.add_argument("--fmt",          default="pdf",
                   choices=["pdf", "png", "svg"],
                   help="Output figure format.")
    return p.parse_args()


def save(fig: plt.Figure, path: str, fmt: str) -> None:
    if fig is None:
        return
    out = f"{path}.{fmt}"
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    opts     = parse_args()
    out_dir  = opts.output_dir or opts.results_dir
    os.makedirs(out_dir, exist_ok=True)
    fmt      = opts.fmt
    np.random.seed(42)

    print(f"[plot_backpop] Loading: {opts.results_dir}")
    res = load_results(opts.results_dir, n_samples=opts.n_samples)
    print(f"  Event:  {res['event_name']}")
    print(f"  Config: {res['config_name']}")
    print(f"  Mode:   {res['mode']}")
    print(f"  log Z:  {res['log_z']:.3f}")
    print(f"  N_eff:  {res['n_eff']}")
    print(f"  Params: {res['params']}")
    print()

    def _save(fig, name):
        save(fig, os.path.join(out_dir, name), fmt)

    print("[plot_backpop] Generating figures...")

    # 1. Full corner
    print("  [1/7] Full corner plot...")
    _save(plot_corner_full(res), "corner_full")

    # 2. ZAMS corner (coloured by channel)
    print("  [2/7] ZAMS corner plot...")
    _save(plot_corner_zams(res), "corner_zams")

    # 3. Binary physics corner
    print("  [3/7] Binary physics corner plot...")
    _save(plot_corner_physics(res), "corner_physics")

    # 4. Kick parameters corner
    print("  [4/7] Kick parameter corner plot...")
    _save(plot_corner_kicks(res), "corner_kicks")

#     # 5. GW observable comparison
#     print("  [5/7] GW observable comparison...")
#     _save(
#         plot_gw_comparison(res, opts.samples_path, opts.approximant),
#         "gw_comparison",
#     )

    # 6. Formation channels
    print("  [6/7] Formation channel fractions...")
    _save(plot_formation_channels(res), "formation_channels")

    # 7. Delay time
    print("  [7/7] Delay time distribution...")
    _save(plot_delay_time(res), "delay_time")

    # 8. Optional 2D vs 3D comparison
    if opts.compare_dir:
        print(f"  [8/?] 2D vs 3D comparison: {opts.compare_dir}")
        res2 = load_results(opts.compare_dir, n_samples=opts.n_samples)

        # Determine which is 2D and which is 3D
        if '3D' in res['mode'] and '3D' not in res2['mode']:
            r2d, r3d = res2, res
        elif '3D' in res2['mode'] and '3D' not in res['mode']:
            r2d, r3d = res, res2
        else:
            r2d, r3d = res, res2   # both same mode — compare anyway

        _save(plot_2d_vs_3d_comparison(r2d, r3d), "comparison_2d_vs_3d")

    print()
    print(f"[plot_backpop] All figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
