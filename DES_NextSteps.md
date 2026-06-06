# DES Box-Cox Analysis — Next Steps

When the external hard disk arrives, follow these steps to download DES data and run the Box-Cox shape evolution analysis.

---

## 1. Prepare the External Disk

Format as exFAT (readable on both Mac and Linux) or APFS if Mac-only.
Create a dedicated folder:

```
/Volumes/<DiskName>/DES/
```

All DES downloads go here — the full 500k-row FITS file is ~50 MB, but a deeper or multi-band download can reach several hundred MB.

---

## 2. Install Dependencies

```bash
pip install astro-datalab astropy
```

No NOIRLab account is required for anonymous read-only access to public DES DR1/DR2 tables.
For queries > 300k rows the script auto-switches to async mode; no extra setup needed.

---

## 3. Download the Data

Run from the `MSc_Project` directory (or wherever the scripts live):

```bash
# Default: 500k galaxies from DES DR2, r < 22.5, z = 0.05–1.5
python DES_DataLab_Download.py --out /Volumes/<DiskName>/DES/des_dr2_500k.fits

# Smaller test run first (recommended — takes ~1 min, confirms the pipeline works)
python DES_DataLab_Download.py --limit 50000 --out /Volumes/<DiskName>/DES/des_dr2_test_50k.fits
```

**Output columns** (renamed by the script for consistency with the analysis pipeline):

| Column | Content |
|---|---|
| `SOF_CM_MAG_G` | g-band magnitude |
| `SOF_CM_MAG_R` | r-band magnitude |
| `SOF_CM_MAG_I` | i-band magnitude |
| `DNF_ZMC_SOF` | photometric redshift |
| `COLOUR_G_R` | g−r (added by script) |
| `COLOUR_R_I` | r−i (added by script) |

---

## 4. Run the Box-Cox Analysis

```bash
# Test run first
python DES_BoxCox_Tests.py \
    --cat /Volumes/<DiskName>/DES/des_dr2_test_50k.fits \
    --colour g-r \
    --photoz-scatter

# Full 500k run
python DES_BoxCox_Tests.py \
    --cat /Volumes/<DiskName>/DES/des_dr2_500k.fits \
    --colour g-r \
    --photoz-scatter
```

The `--photoz-scatter` flag applies a Gaussian smear (σ = 0.03(1+z)) to simulate photometric redshift uncertainty in the null bootstrap, making the null test realistic for photo-z data.

Output: a 2-panel figure (`DES_BoxCox_shape_evolution.png`) showing λ and skewness vs redshift with bootstrap null envelopes.

---

## 5. What to Look For

Compare DES results against the GAMA findings (PaperDraft_v8):

| Property | GAMA (u−r, z < 0.35) | DES expectation (g−r, z < 1.5) |
|---|---|---|
| λ trend with z | +1.00, p = 4×10⁻⁴ | Expect monotonic but check sign |
| Skewness trend | −1.00, p = 4×10⁻⁴ | Expect similar direction |
| Bootstrap null | All 7 bins outside | More bins; photo-z scatter widens null |
| Red fraction | Increases with z | Same expected direction |

Key differences to bear in mind:
- DES uses **g−r**, not **u−r** — the red/blue threshold will differ (~0.6 in g−r vs 2.1 in u−r)
- DES photo-z scatter (σ ≈ 0.03(1+z)) smears redshift bins; the null test accounts for this
- DES is ~4 mag deeper than GAMA so Malmquist selection is much less severe — trends at z > 0.5 are more representative of the true population

---

## 6. Selection Bias Caveat (carry forward from GAMA work)

The GAMA analysis showed that trends with z in a flux-limited sample conflate evolution with Malmquist selection. DES mitigates this significantly (r < 24 vs r < 19.8) but does not eliminate it. For a clean comparison:

- Consider applying a **stellar mass completeness cut** if photometric stellar masses are available in the catalogue
- At minimum, check that the N per bin does not drop sharply at high z (sharp drop = incompleteness setting in)
- The `--photoz-scatter` null test controls for photo-z broadening but not for sample incompleteness

---

## 7. Files

| File | Purpose |
|---|---|
| `DES_DataLab_Download.py` | Downloads DES DR2 catalogue from NOIRLab Astro Data Lab |
| `DES_BoxCox_Tests.py` | Runs Box-Cox shape evolution pipeline; supports `--mock` for offline testing |
| `DES_NextSteps.md` | This file |

---

## 8. Offline Testing (while waiting for disk / data)

The `--mock` flag generates synthetic DES-like data with no download required:

```bash
python DES_BoxCox_Tests.py --mock
```

This produces a 2-panel figure using synthetic galaxies drawn from a model colour distribution, allowing the pipeline to be tested end-to-end before any real data arrives.
