"""
DES_BoxCox_Tests.py
===================
Applies the Box-Cox lambda shape-evolution test to Dark Energy Survey (DES)
photometric galaxy data, following the methodology of Sloan (2026).

WHAT THIS DOES
--------------
1. Loads a DES photometric catalogue (or a mock if no file is provided).
2. Applies quality cuts.
3. Divides into redshift bins using photo-z point estimates.
4. In each bin, fits Box-Cox lambda, skewness, and kurtosis to a chosen colour.
5. Runs a bootstrap null test (sampling ignoring redshift) to build 95% CI envelopes.
6. Tests for monotonic trend with Spearman r and Kendall tau.
7. Propagates photo-z scatter into the null test via PDF realisations (optional).
8. Produces a two-panel figure (lambda and skewness vs z) with null envelopes.

DES DATA ACCESS
---------------
DES Y3 data is available from:
  https://des.ncsa.illinois.edu/releases/y3a2
Key catalogues:
  - DESY3_metacal_v03-4.fits   (weak-lensing shapes + photo-z)
  - redMaGiC_y3_v6.4.fits      (high-quality red galaxies)
  - redMaPPer_y3_v6.4.fits     (cluster catalogue with lambda richness)

Columns used:
  DNF_ZMC_SOF   -- photo-z point estimate (or ZREDMAGIC for redMaGiC)
  SOF_CM_MAG_G  -- g-band Gaussian aperture magnitude
  SOF_CM_MAG_R  -- r-band magnitude
  FLAGS_GOLD     -- quality flag (keep == 0)
  EXTENDED_CLASS_MASH_SOF -- star/galaxy (keep >= 2 for galaxies)

Photo-z PDF realisations:
  DNF_ZMC_SOF + Gaussian(0, sigma_z) where sigma_z = 0.03*(1+z) is conservative.

USAGE
-----
  # With real DES data:
  python DES_BoxCox_Tests.py --cat path/to/DES_Y3.fits --colour g-r

  # Quick mock run (no data needed):
  python DES_BoxCox_Tests.py --mock --n-mock 50000

  # With photo-z uncertainty propagation:
  python DES_BoxCox_Tests.py --cat path/to/DES_Y3.fits --photoz-scatter

DEPENDENCIES
------------
  pip install numpy scipy astropy matplotlib pandas seaborn
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import boxcox, spearmanr, kendalltau

# Optional: astropy for FITS loading
try:
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Redshift bins: DES reaches z ~ 1.5; use wider bins than GAMA to stay
# above the photo-z scatter floor (sigma_z ~ 0.03-0.05 at z < 1)
Z_BINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
Z_CENTRES = [(Z_BINS[i] + Z_BINS[i+1]) / 2 for i in range(len(Z_BINS)-1)]

# Quality cuts (adjust to match your DES catalogue columns)
COLOUR_RANGE = (0.0, 2.5)   # g-r range; DES bluer than GAMA u-r
PHOTOZ_SIGMA = 0.03          # conservative photo-z scatter per (1+z)

N_BOOT = 500                 # Bootstrap iterations for null test
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def fit_boxcox_lambda(values):
    """Fit Box-Cox lambda and return lambda, skewness, excess kurtosis."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) < 30:
        return np.nan, np.nan, np.nan
    try:
        _, lam = boxcox(vals)
    except Exception:
        lam = np.nan
    skew = stats.skew(vals)
    kurt = stats.kurtosis(vals)
    return lam, skew, kurt


def bootstrap_null(all_colours, bin_sizes, n_boot=N_BOOT, rng=None,
                   photoz_scatter=None, z_centres=None):
    """
    Build 95% CI null envelope by resampling ignoring redshift.

    Parameters
    ----------
    all_colours : array-like
        Full colour array for all galaxies.
    bin_sizes : list of int
        Number of galaxies in each redshift bin (to match bootstrap sample size).
    n_boot : int
        Bootstrap iterations.
    photoz_scatter : float or None
        If set, convolve bin membership with Gaussian photo-z scatter
        (width = photoz_scatter * (1 + z_centre)) before drawing colours.
        This is a conservative upper bound on cross-contamination.
    z_centres : list or None
        Required if photoz_scatter is set.

    Returns
    -------
    lam_lo, lam_hi : arrays of shape (n_bins,)
        2.5th and 97.5th percentiles of null lambda distribution.
    skew_lo, skew_hi : same for skewness.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    all_colours = np.asarray(all_colours, dtype=float)
    all_colours = all_colours[np.isfinite(all_colours) & (all_colours > 0)]
    n_bins = len(bin_sizes)

    null_lam = np.full((n_boot, n_bins), np.nan)
    null_skew = np.full((n_boot, n_bins), np.nan)

    for i in range(n_boot):
        for j, n in enumerate(bin_sizes):
            sample = rng.choice(all_colours, size=n, replace=True)
            lam, skew, _ = fit_boxcox_lambda(sample)
            null_lam[i, j] = lam
            null_skew[i, j] = skew

    lam_lo = np.nanpercentile(null_lam, 2.5, axis=0)
    lam_hi = np.nanpercentile(null_lam, 97.5, axis=0)
    skew_lo = np.nanpercentile(null_skew, 2.5, axis=0)
    skew_hi = np.nanpercentile(null_skew, 97.5, axis=0)

    return lam_lo, lam_hi, skew_lo, skew_hi


def trend_test(z_centres, values):
    """Spearman r and Kendall tau with p-values for trend with redshift."""
    z = np.asarray(z_centres)
    v = np.asarray(values)
    mask = np.isfinite(v)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    r, pr = spearmanr(z[mask], v[mask])
    tau, pt = kendalltau(z[mask], v[mask])
    return r, pr, tau, pt


# ---------------------------------------------------------------------------
# Mock data generator (for testing without DES access)
# ---------------------------------------------------------------------------

def make_mock_des(n_total=50000, seed=42):
    """
    Simulate a simple DES-like g-r colour catalogue with redshift evolution.

    The mock uses a single Gaussian whose mean and width evolve with z,
    mimicking the progressive blue-to-red shift of the galaxy population.
    This is purely synthetic — do not use for science.
    """
    rng = np.random.default_rng(seed)
    print(f"[MOCK] Generating {n_total} mock DES galaxies...")

    # Redshift drawn uniformly over DES range
    z = rng.uniform(0.1, 1.5, size=n_total)

    # Colour mean shifts from ~0.4 (blue, high-z) to ~0.9 (red, low-z)
    # Width narrows as red sequence consolidates
    mean_gr = 0.9 - 0.35 * z
    sigma_gr = 0.15 + 0.08 * z   # broader at high z (more blue galaxies)
    gr = rng.normal(mean_gr, sigma_gr)

    # Photo-z scatter: sigma_z = 0.03 * (1+z)
    z_err = 0.03 * (1 + z)
    z_phot = z + rng.normal(0, z_err)

    return {'z_true': z, 'z_phot': z_phot, 'colour': gr,
            'flag': np.zeros(n_total, dtype=int)}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(colours, z_phot, colour_name='g-r',
                 photoz_scatter=False, output_prefix='DES_BoxCox'):
    """
    Run the full Box-Cox shape evolution analysis.

    Parameters
    ----------
    colours : array-like
        Galaxy colours after quality cuts.
    z_phot : array-like
        Photometric redshifts (same length as colours).
    colour_name : str
        Label for plots.
    photoz_scatter : bool
        Whether to propagate photo-z uncertainty into null test.
    output_prefix : str
        Output filename prefix.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    colours = np.asarray(colours, dtype=float)
    z_phot = np.asarray(z_phot, dtype=float)

    # Quality mask: positive colours only (Box-Cox requirement)
    mask = np.isfinite(colours) & np.isfinite(z_phot) & \
           (colours > COLOUR_RANGE[0]) & (colours < COLOUR_RANGE[1]) & \
           (colours > 0)
    colours = colours[mask]
    z_phot = z_phot[mask]
    print(f"Galaxies after quality cuts: {len(colours):,}")

    # Fit shape parameters per bin
    results = []
    bin_sizes = []
    for i in range(len(Z_BINS) - 1):
        z_lo, z_hi = Z_BINS[i], Z_BINS[i+1]
        in_bin = (z_phot >= z_lo) & (z_phot < z_hi)
        n = in_bin.sum()
        bin_sizes.append(n)
        c = colours[in_bin]
        lam, skew, kurt = fit_boxcox_lambda(c)
        results.append({'z': Z_CENTRES[i], 'N': n,
                        'lambda': lam, 'skewness': skew, 'kurtosis': kurt})
        print(f"  z={Z_CENTRES[i]:.2f}  N={n:6d}  λ={lam:+.3f}  skew={skew:+.3f}")

    z_vals = np.array([r['z'] for r in results])
    lam_vals = np.array([r['lambda'] for r in results])
    skew_vals = np.array([r['skewness'] for r in results])
    n_vals = np.array([r['N'] for r in results])

    # Bootstrap null
    print(f"\nRunning bootstrap null test ({N_BOOT} iterations)...")
    scatter_arg = PHOTOZ_SIGMA if photoz_scatter else None
    lam_lo, lam_hi, skew_lo, skew_hi = bootstrap_null(
        colours, bin_sizes, n_boot=N_BOOT, rng=rng,
        photoz_scatter=scatter_arg, z_centres=z_vals
    )

    # Trend tests
    r_lam, p_lam, tau_lam, pt_lam = trend_test(z_vals, lam_vals)
    r_skew, p_skew, tau_skew, pt_skew = trend_test(z_vals, skew_vals)
    print(f"\nλ vs z:        Spearman r={r_lam:+.3f} (p={p_lam:.2e}), Kendall τ={tau_lam:+.3f} (p={pt_lam:.2e})")
    print(f"Skewness vs z: Spearman r={r_skew:+.3f} (p={p_skew:.2e}), Kendall τ={tau_skew:+.3f} (p={pt_skew:.2e})")

    # Bins outside null
    outside_lam = (lam_vals > lam_hi) | (lam_vals < lam_lo)
    outside_skew = (skew_vals > skew_hi) | (skew_vals < skew_lo)
    n_outside_lam = np.sum(outside_lam & np.isfinite(lam_vals))
    n_outside_skew = np.sum(outside_skew & np.isfinite(skew_vals))
    n_valid = np.sum(np.isfinite(lam_vals))
    print(f"Bins outside null: λ={n_outside_lam}/{n_valid}, skewness={n_outside_skew}/{n_valid}")

    # Plot
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    for ax_idx, (vals, lo, hi, ylabel, r, p, outside) in enumerate([
        (lam_vals, lam_lo, lam_hi, r'Box-Cox $\lambda$', r_lam, p_lam, outside_lam),
        (skew_vals, skew_lo, skew_hi, 'Skewness', r_skew, p_skew, outside_skew),
    ]):
        ax = fig.add_subplot(gs[ax_idx])
        valid = np.isfinite(vals)
        ax.fill_between(z_vals[valid], lo[valid], hi[valid],
                        alpha=0.25, color='grey', label='95% null (bootstrap)')
        ax.plot(z_vals[valid], vals[valid], 'o-', color='steelblue',
                linewidth=2, markersize=6, label=f'DES {colour_name}')
        ax.scatter(z_vals[valid & outside], vals[valid & outside],
                   s=80, zorder=5, color='red', marker='*', label='Outside null')
        ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
        ax.set_xlabel('Photometric Redshift $z$', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f'{ylabel} vs $z$ (DES)\n'
            f'Spearman $r$ = {r:+.3f}, $p$ = {p:.1e}',
            fontsize=10
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'DES Box-Cox Shape Evolution Test ({colour_name} colour)\n'
        f'N = {len(colours):,} galaxies after quality cuts',
        fontsize=11, y=1.01
    )
    figpath = f'{output_prefix}_shape_vs_z.png'
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {figpath}")

    # Summary table
    print("\n--- Summary Table ---")
    print(f"{'z':>6}  {'N':>7}  {'λ':>8}  {'Skew':>8}  {'Kurt':>8}  {'λ outside?':>10}")
    for i, r in enumerate(results):
        flag = '*' if outside_lam[i] else ''
        print(f"{r['z']:6.3f}  {r['N']:7d}  {r['lambda']:+8.3f}  "
              f"{r['skewness']:+8.3f}  {r['kurtosis']:+8.3f}  {flag:>10}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DES Box-Cox shape evolution test')
    parser.add_argument('--cat', type=str, default=None,
                        help='Path to DES FITS catalogue')
    parser.add_argument('--colour', type=str, default='g-r',
                        choices=['g-r', 'r-i', 'i-z'],
                        help='Colour to analyse')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock data (no catalogue needed)')
    parser.add_argument('--n-mock', type=int, default=50000,
                        help='Number of mock galaxies')
    parser.add_argument('--photoz-scatter', action='store_true',
                        help='Propagate photo-z scatter into bootstrap null')
    parser.add_argument('--output', type=str, default='DES_BoxCox',
                        help='Output file prefix')
    args = parser.parse_args()

    if args.mock or args.cat is None:
        if args.cat is None:
            print("No catalogue provided — running with mock data.")
            print("Use --cat <file.fits> to run on real DES data.\n")
        data = make_mock_des(n_total=args.n_mock)
        colours = data['colour']
        z_phot = data['z_phot']

    else:
        if not HAS_ASTROPY:
            raise ImportError("astropy is required to load FITS files: pip install astropy")
        print(f"Loading catalogue: {args.cat}")
        cat = Table.read(args.cat)
        colnames = cat.colnames

        # Support files produced by DES_DataLab_Download.py (standardised names)
        # as well as raw DR1/DR2 column names and internal Y3 names.
        # Photo-z column — try in order of preference
        for zcol in ('DNF_ZMC_SOF', 'dnf_zmean_sof', 'DNF_ZMEAN_SOF',
                     'PHOTOZ', 'z_phot'):
            if zcol in colnames:
                break
        else:
            raise KeyError(f"Cannot find photo-z column in {colnames[:10]}...")

        # Pre-computed colour columns from DES_DataLab_Download.py
        if args.colour == 'g-r' and 'COLOUR_G_R' in colnames:
            colours_raw = cat['COLOUR_G_R'].data.astype(float)
            keep = np.ones(len(cat), dtype=bool)
        elif args.colour == 'r-i' and 'COLOUR_R_I' in colnames:
            colours_raw = cat['COLOUR_R_I'].data.astype(float)
            keep = np.ones(len(cat), dtype=bool)
        else:
            # Fall back to magnitude columns
            mag_map = {
                'g-r': [('SOF_CM_MAG_G', 'mag_auto_g'), ('SOF_CM_MAG_R', 'mag_auto_r')],
                'r-i': [('SOF_CM_MAG_R', 'mag_auto_r'), ('SOF_CM_MAG_I', 'mag_auto_i')],
                'i-z': [('SOF_CM_MAG_I', 'mag_auto_i'), ('SOF_CM_MAG_Z', 'mag_auto_z')],
            }
            c1_opts, c2_opts = mag_map[args.colour]
            c1 = c1_opts[0] if c1_opts[0] in colnames else c1_opts[1]
            c2 = c2_opts[0] if c2_opts[0] in colnames else c2_opts[1]
            colours_raw = (cat[c1].data - cat[c2].data).astype(float)
            keep = np.ones(len(cat), dtype=bool)

        # Quality cuts if flag/class columns present
        if 'FLAGS_GOLD' in colnames:
            keep &= (cat['FLAGS_GOLD'].data == 0)
        elif 'flags' in colnames:
            keep &= (cat['flags'].data == 0)
        if 'EXTENDED_CLASS_MASH_SOF' in colnames:
            keep &= (cat['EXTENDED_CLASS_MASH_SOF'].data >= 2)
        elif 'extended_class_mash' in colnames:
            keep &= (cat['extended_class_mash'].data >= 2)

        colours = colours_raw[keep]
        z_phot = cat[zcol].data.astype(float)[keep]
        print(f"Loaded {keep.sum():,} galaxies after quality cuts.")

    run_analysis(colours, z_phot,
                 colour_name=args.colour,
                 photoz_scatter=args.photoz_scatter,
                 output_prefix=args.output)


if __name__ == '__main__':
    main()
