"""Combine and audit APOGEE XP FITS batches after DataLink becomes available.

Column names are command-line options because the real DATAMODEL_GAIA FITS schema
must be confirmed from a successful small download before production use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.table import Table, vstack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", default="data/apogee_xp_batches")
    parser.add_argument("--pattern", default="*.fits")
    parser.add_argument("--requested", default="data/apogee_training_xp_source_ids.csv")
    parser.add_argument("--output", default="data/apogee_dr17_xp_training_combined.fits")
    parser.add_argument("--source-col", default="source_id")
    parser.add_argument("--bp-col", default="bp_coefficients")
    parser.add_argument("--rp-col", default="rp_coefficients")
    args = parser.parse_args()

    files = sorted(Path(args.batch_dir).glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No batch files match {args.batch_dir}/{args.pattern}")
    tables = [Table.read(path) for path in files]
    required = {args.source_col, args.bp_col, args.rp_col}
    for path, table in zip(files, tables):
        missing = required - set(table.colnames)
        if missing:
            raise KeyError(f"{path} lacks columns: {sorted(missing)}")
    combined = vstack(tables, metadata_conflicts="silent")
    source_ids = np.asarray(combined[args.source_col], dtype=np.int64)
    unique_ids, first_indices, counts = np.unique(source_ids, return_index=True, return_counts=True)
    duplicate_ids = unique_ids[counts > 1]
    combined = combined[np.sort(first_indices)]

    bp_lengths = np.array([len(np.asarray(v)) for v in combined[args.bp_col]])
    rp_lengths = np.array([len(np.asarray(v)) for v in combined[args.rp_col]])
    valid_lengths = (bp_lengths == 55) & (rp_lengths == 55)

    requested = pd.read_csv(args.requested, usecols=["source_id"], dtype={"source_id": "int64"})
    requested_ids = set(requested["source_id"])
    returned_ids = set(np.asarray(combined[args.source_col], dtype=np.int64))
    missing_ids = sorted(requested_ids - returned_ids)
    unexpected_ids = sorted(returned_ids - requested_ids)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write(output, overwrite=True)
    pd.DataFrame({"source_id": missing_ids}).to_csv(output.with_name("apogee_xp_missing_ids.csv"), index=False)
    pd.DataFrame({"source_id": unexpected_ids}).to_csv(output.with_name("apogee_xp_unexpected_ids.csv"), index=False)
    pd.DataFrame({"source_id": duplicate_ids}).to_csv(output.with_name("apogee_xp_duplicate_ids.csv"), index=False)
    summary = {
        "batch_files": len(files),
        "rows_before_deduplication": int(len(source_ids)),
        "unique_rows": int(len(combined)),
        "duplicate_source_ids": int(len(duplicate_ids)),
        "requested_source_ids": int(len(requested_ids)),
        "missing_source_ids": int(len(missing_ids)),
        "unexpected_source_ids": int(len(unexpected_ids)),
        "rows_with_valid_55_55_lengths": int(valid_lengths.sum()),
        "rows_with_invalid_lengths": int((~valid_lengths).sum()),
    }
    output.with_name("apogee_xp_download_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
