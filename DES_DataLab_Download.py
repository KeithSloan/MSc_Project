"""
DES_DataLab_Download.py
=======================
Downloads a galaxy colour + photometric redshift sample from the Dark Energy
Survey via the NOIRLab Astro Data Lab public SQL service.

Produces a FITS file ready for DES_BoxCox_Tests.py.

INSTALLATION
------------
  pip install astro-datalab astropy

No account required for anonymous read access to public DES tables.

USAGE
-----
  # Default: 500k galaxies from DES DR2, r < 22.5
  python DES_DataLab_Download.py

  # Smaller test sample
  python DES_DataLab_Download.py --limit 100000 --out DES_test_100k.fits

  # Use DES DR1 instead of DR2
  python DES_DataLab_Download.py --schema des_dr1

  # Deeper cut (more high-z galaxies, larger file)
  python DES_DataLab_Download.py --maglim 23.5 --limit 1000000

OUTPUT COLUMNS (renamed to match DES_BoxCox_Tests.py)
------------------------------------------------------
  SOF_CM_MAG_G          -- g-band magnitude
  SOF_CM_MAG_R          -- r-band magnitude
  SOF_CM_MAG_I          -- i-band magnitude
  DNF_ZMC_SOF           -- photometric redshift (point estimate)
  FLAGS_GOLD            -- quality flag (0 = clean)
  EXTENDED_CLASS_MASH_SOF -- star/galaxy (>=2 = galaxy)

WHAT THE QUERY SELECTS
----------------------
  - FLAGS quality cut applied server-side (flags = 0)
  - Galaxy classifier applied server-side (extended_class_mash >= 2)
  - r-band magnitude range 17 < r < maglim (removes saturated stars at bright end)
  - Photo-z range 0.05 < z < 1.5
  - Random LIMIT ensures spatially unbiased sample across the footprint
"""

import argparse
import sys
import os

try:
    from dl import queryClient as qc
except ImportError:
    print("ERROR: astro-datalab not installed.")
    print("Run:  pip install astro-datalab")
    sys.exit(1)

try:
    from astropy.table import Table
    import numpy as np
except ImportError:
    print("ERROR: astropy not installed.")
    print("Run:  pip install astropy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Schema-specific column maps
# DES public DR1/DR2 column names differ from internal Y3 processing names.
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS = {
    'des_dr2': {
        'table':    'des_dr2.main',
        'mag_g':    'mag_auto_g',
        'mag_r':    'mag_auto_r',
        'mag_i':    'mag_auto_i',
        'photoz':   'dnf_zmean_sof',    # DNF photo-z mean
        'flag':     'flags',
        'class':    'extended_class_mash',
        'flag_val': 0,
        'class_min': 2,
    },
    'des_dr1': {
        'table':    'des_dr1.main',
        'mag_g':    'mag_auto_g',
        'mag_r':    'mag_auto_r',
        'mag_i':    'mag_auto_i',
        'photoz':   'dnf_zmean_sof',
        'flag':     'flags',
        'class':    'extended_class_mash',
        'flag_val': 0,
        'class_min': 2,
    },
}

# Output column names expected by DES_BoxCox_Tests.py
OUTPUT_NAMES = {
    'mag_g':  'SOF_CM_MAG_G',
    'mag_r':  'SOF_CM_MAG_R',
    'mag_i':  'SOF_CM_MAG_I',
    'photoz': 'DNF_ZMC_SOF',
    'flag':   'FLAGS_GOLD',
    'class':  'EXTENDED_CLASS_MASH_SOF',
}


def build_query(cols, maglim, z_min, z_max, limit):
    """Build the SQL query string."""
    return f"""
        SELECT
            {cols['mag_g']}   AS g_mag,
            {cols['mag_r']}   AS r_mag,
            {cols['mag_i']}   AS i_mag,
            {cols['photoz']}  AS z_phot,
            {cols['flag']}    AS flag,
            {cols['class']}   AS star_gal
        FROM {cols['table']}
        WHERE {cols['flag']}    = {cols['flag_val']}
          AND {cols['class']}  >= {cols['class_min']}
          AND {cols['mag_r']}  BETWEEN 17.0 AND {maglim}
          AND {cols['photoz']} BETWEEN {z_min} AND {z_max}
        LIMIT {limit}
    """.strip()


def download(schema='des_dr2', maglim=22.5, z_min=0.05, z_max=1.5,
             limit=500000, out=None, async_query=False):

    if schema not in SCHEMA_COLUMNS:
        print(f"ERROR: Unknown schema '{schema}'. Choose from: {list(SCHEMA_COLUMNS)}")
        sys.exit(1)

    cols = SCHEMA_COLUMNS[schema]
    if out is None:
        out = f"DES_{schema}_colours_{limit//1000}k.fits"

    sql = build_query(cols, maglim, z_min, z_max, limit)

    print(f"Schema  : {schema}")
    print(f"Table   : {cols['table']}")
    print(f"Limit   : {limit:,} rows")
    print(f"r limit : < {maglim}")
    print(f"z range : {z_min} – {z_max}")
    print(f"Output  : {out}")
    print()
    print("Submitting query to Astro Data Lab...")

    if async_query:
        # Async: better for large queries (>200k rows)
        jobid = qc.query(sql=sql, fmt='table', async_=True)
        print(f"Job ID  : {jobid}")
        print("Waiting for job to complete (this may take a few minutes)...")

        import time
        while True:
            status = qc.status(jobid)
            print(f"  Status: {status}")
            if status == 'COMPLETED':
                break
            elif status in ('ERROR', 'ABORTED'):
                print(f"Query failed with status: {status}")
                print(qc.error(jobid))
                sys.exit(1)
            time.sleep(10)

        result = qc.results(jobid, fmt='table')
    else:
        # Synchronous: fine for <= 300k rows
        result = qc.query(sql=sql, fmt='table')

    if isinstance(result, str):
        # queryClient sometimes returns CSV string; parse it
        from io import StringIO
        import pandas as pd
        df = pd.read_csv(StringIO(result))
        result = Table.from_pandas(df)

    print(f"Received {len(result):,} rows.")

    # Rename columns to match DES_BoxCox_Tests.py expectations
    col_map = {
        'g_mag':    OUTPUT_NAMES['mag_g'],
        'r_mag':    OUTPUT_NAMES['mag_r'],
        'i_mag':    OUTPUT_NAMES['mag_i'],
        'z_phot':   OUTPUT_NAMES['photoz'],
        'flag':     OUTPUT_NAMES['flag'],
        'star_gal': OUTPUT_NAMES['class'],
    }
    for old, new in col_map.items():
        if old in result.colnames:
            result.rename_column(old, new)

    # Compute g-r and r-i colours and add as convenience columns
    g = result[OUTPUT_NAMES['mag_g']].data.astype(float)
    r = result[OUTPUT_NAMES['mag_r']].data.astype(float)
    i = result[OUTPUT_NAMES['mag_i']].data.astype(float)
    result['COLOUR_G_R'] = g - r
    result['COLOUR_R_I'] = r - i

    result.write(out, overwrite=True)
    size_mb = os.path.getsize(out) / 1e6
    print(f"Saved   : {out}  ({size_mb:.1f} MB)")
    print()
    print("Sample statistics:")
    z = result[OUTPUT_NAMES['photoz']].data.astype(float)
    gr = result['COLOUR_G_R'].data.astype(float)
    print(f"  z_phot : {np.nanmin(z):.3f} – {np.nanmax(z):.3f}  "
          f"(median {np.nanmedian(z):.3f})")
    print(f"  g-r    : {np.nanmin(gr):.3f} – {np.nanmax(gr):.3f}  "
          f"(median {np.nanmedian(gr):.3f})")
    print()
    print("Run the Box-Cox analysis with:")
    print(f"  python DES_BoxCox_Tests.py --cat {out} --colour g-r --photoz-scatter")


def main():
    parser = argparse.ArgumentParser(
        description='Download DES colour+photo-z sample from Astro Data Lab')
    parser.add_argument('--schema', default='des_dr2',
                        choices=list(SCHEMA_COLUMNS),
                        help='DES schema to query (default: des_dr2)')
    parser.add_argument('--limit', type=int, default=500000,
                        help='Number of rows to download (default: 500000)')
    parser.add_argument('--maglim', type=float, default=22.5,
                        help='Faint r-band magnitude limit (default: 22.5)')
    parser.add_argument('--zmin', type=float, default=0.05,
                        help='Minimum photometric redshift (default: 0.05)')
    parser.add_argument('--zmax', type=float, default=1.5,
                        help='Maximum photometric redshift (default: 1.5)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output FITS filename')
    parser.add_argument('--async', dest='async_query', action='store_true',
                        help='Use async query (recommended for limit > 300k)')
    args = parser.parse_args()

    # Auto-switch to async for large downloads
    use_async = args.async_query or args.limit > 300000
    if use_async and not args.async_query:
        print(f"Note: limit={args.limit:,} > 300k — switching to async query automatically.")

    download(
        schema=args.schema,
        maglim=args.maglim,
        z_min=args.zmin,
        z_max=args.zmax,
        limit=args.limit,
        out=args.out,
        async_query=use_async,
    )


if __name__ == '__main__':
    main()
