"""
Run this script from /Users/ksloan/github/MSc_Project/
  python GenerateFigures.py

Generates publication-quality figures:
  ChartsPlots/Fig1_JSU_vs_BoxCox_Instability.png   (Methods paper Fig 1)
  ChartsPlots/Fig2_Shape_vs_Redshift.png            (GAMA paper Fig 1 / Methods paper Fig 2)

Requires: numpy, matplotlib, scipy, astropy (already in your conda env)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from astropy.table import Table
import warnings
warnings.filterwarnings('ignore')

# ── pre-computed values from executed notebook ──────────────────────────────
z_centres = np.array([0.026, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325])
N_per_bin  = np.array([3932, 16579, 29589, 32999, 27116, 33685, 21588])
jsu_a  = np.array([-2.763, -11.677,  20.679,  9.776,  3.431,  2.802,  2.836])
jsu_b  = np.array([ 2.210,   5.925,   7.171,  3.566,  2.112,  1.752,  1.802])
lam    = np.array([-0.200,   0.431,   1.125,  1.694,  1.995,  2.532,  2.701])
skew   = np.array([ 0.899,   0.263,  -0.148, -0.387, -0.523, -0.745, -0.821])
kurt   = np.array([ 0.824,  -0.750,  -1.046, -0.897, -0.623, -0.188,  0.069])

# ── bootstrap null (re-derive from data) ────────────────────────────────────
print("Loading data and running bootstrap null (500 iterations)...")
t = Table.read('GAMA_DATA/StellarMassesLambdarv24.fits')
t = t[(t['uminusr'] > 0.5) & (t['uminusr'] < 4.0) &
      (t['logmstar'] > 8.0) & (t['Z'] > 0.002) & (t['Z'] < 0.35)]
uminusr = np.array(t['uminusr'], dtype=float)

rng = np.random.default_rng(42)
N_BOOT = 500
null_lam  = {zc: [] for zc in z_centres}
null_skew = {zc: [] for zc in z_centres}

for boot_i in range(N_BOOT):
    for zc, n in zip(z_centres, N_per_bin):
        x = uminusr[rng.choice(len(uminusr), size=n, replace=True)]
        shift = max(0, -x.min() + 0.01)
        try:
            _, l = stats.boxcox(x + shift)
        except Exception:
            l = np.nan
        null_lam[zc].append(l)
        null_skew[zc].append(stats.skew(x))
    if (boot_i + 1) % 100 == 0:
        print(f"  {boot_i+1}/{N_BOOT}")

print("Bootstrap complete.")

def null_band(null_dict):
    lo  = np.array([np.percentile(null_dict[z],  2.5) for z in z_centres])
    hi  = np.array([np.percentile(null_dict[z], 97.5) for z in z_centres])
    med = np.array([np.percentile(null_dict[z], 50  ) for z in z_centres])
    return lo, hi, med

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — JSU instability vs Box-Cox smoothness (Methods paper)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)
fig.suptitle('Figure 1: Johnson SU parameter instability vs Box-Cox λ smoothness\n'
             'All three statistics track the same underlying distributional change',
             fontsize=10)

# Panel A — JSU a (unstable)
ax = axes[0]
ax.plot(z_centres, jsu_a, 'o-', color='crimson', lw=2, ms=7)
ax.axhline(0, color='grey', lw=0.7, linestyle='--')
ax.set_xlabel('Redshift  z', fontsize=10)
ax.set_ylabel('Johnson SU  a', fontsize=10)
ax.set_title('(a)  JSU shape param a\n[non-monotonic, numerically unstable]', fontsize=9)
ax.grid(True, alpha=0.3)
for spine in ax.spines.values():
    spine.set_edgecolor('#AAAAAA')

# Panel B — Box-Cox λ (smooth)
ax = axes[1]
lo, hi, med = null_band(null_lam)
ax.fill_between(z_centres, lo, hi, alpha=0.2, color='grey', label='Null 95% CI')
ax.plot(z_centres, med, '--', color='grey', lw=1, label='Null median')
ax.plot(z_centres, lam, 'o-', color='darkorange', lw=2, ms=7, label='Observed λ')
outside = (lam < lo) | (lam > hi)
if outside.any():
    ax.plot(z_centres[outside], lam[outside], '*', color='red', ms=12, zorder=5)
ax.set_xlabel('Redshift  z', fontsize=10)
ax.set_ylabel('Box-Cox  λ', fontsize=10)
ax.set_title('(b)  Box-Cox λ\n[monotonic, Spearman r = +1.00, p < 0.001]', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel C — Skewness (smooth, for reference)
ax = axes[2]
lo_s, hi_s, med_s = null_band(null_skew)
ax.fill_between(z_centres, lo_s, hi_s, alpha=0.2, color='grey', label='Null 95% CI')
ax.plot(z_centres, med_s, '--', color='grey', lw=1)
ax.plot(z_centres, skew, 'o-', color='seagreen', lw=2, ms=7, label='Observed skewness')
outside_s = (skew < lo_s) | (skew > hi_s)
if outside_s.any():
    ax.plot(z_centres[outside_s], skew[outside_s], '*', color='red', ms=12, zorder=5)
ax.set_xlabel('Redshift  z', fontsize=10)
ax.set_ylabel('Skewness', fontsize=10)
ax.set_title('(c)  Skewness\n[monotonic, Spearman r = −1.00, p < 0.001]', fontsize=9)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ChartsPlots/Fig1_JSU_vs_BoxCox_Instability.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved Fig1_JSU_vs_BoxCox_Instability.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Clean 2-panel: λ and skewness vs z (GAMA paper)
# ════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
fig.suptitle('u−r colour distribution shape vs redshift (N = 165,488 GAMA galaxies)\n'
             'Grey band = 95% bootstrap null envelope  |  Red stars = outside null',
             fontsize=10)

lo, hi, med = null_band(null_lam)
ax1.fill_between(z_centres, lo, hi, alpha=0.22, color='grey', label='Null 95% CI')
ax1.plot(z_centres, med, '--', color='grey', lw=1, label='Null median')
ax1.plot(z_centres, lam, 'o-', color='darkorange', lw=2.2, ms=8, label='Observed λ')
outside = (lam < lo) | (lam > hi)
if outside.any():
    ax1.plot(z_centres[outside], lam[outside], '*', color='red', ms=14, zorder=5,
             label='Outside null 95% CI')
ax1.axhline(0, color='grey', lw=0.7, linestyle=':')
ax1.axhline(1, color='grey', lw=0.7, linestyle=':')
ax1.text(0.27, 0.12,  'log-Normal (λ=0)', fontsize=7.5, color='grey', ha='center')
ax1.text(0.27, 1.12,  'Normal (λ=1)',      fontsize=7.5, color='grey', ha='center')
ax1.set_xlabel('Redshift  z', fontsize=11)
ax1.set_ylabel('Box-Cox  λ', fontsize=11)
ax1.set_title('(a)  Box-Cox λ\nSpearman r = +1.000,  p = 4.0 × 10⁻⁴', fontsize=9.5)
ax1.legend(fontsize=8.5)
ax1.grid(True, alpha=0.3)

lo_s, hi_s, med_s = null_band(null_skew)
ax2.fill_between(z_centres, lo_s, hi_s, alpha=0.22, color='grey', label='Null 95% CI')
ax2.plot(z_centres, med_s, '--', color='grey', lw=1, label='Null median')
ax2.plot(z_centres, skew, 'o-', color='steelblue', lw=2.2, ms=8, label='Observed skewness')
outside_s = (skew < lo_s) | (skew > hi_s)
if outside_s.any():
    ax2.plot(z_centres[outside_s], skew[outside_s], '*', color='red', ms=14, zorder=5,
             label='Outside null 95% CI')
ax2.axhline(0, color='grey', lw=0.7, linestyle=':')
ax2.set_xlabel('Redshift  z', fontsize=11)
ax2.set_ylabel('Skewness', fontsize=11)
ax2.set_title('(b)  Skewness\nSpearman r = −1.000,  p = 4.0 × 10⁻⁴', fontsize=9.5)
ax2.legend(fontsize=8.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ChartsPlots/Fig2_Shape_vs_Redshift.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved Fig2_Shape_vs_Redshift.png")
print("\nAll figures done. Find them in ChartsPlots/")
