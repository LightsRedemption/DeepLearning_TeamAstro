"""Create deterministic source-ID splits after XP validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(source_ids, seed=42):
    ids = pd.Series(source_ids, name="source_id").drop_duplicates().reset_index(drop=True)
    train, remainder = train_test_split(ids, test_size=0.30, random_state=seed, shuffle=True)
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=seed, shuffle=True
    )
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert len(train) + len(validation) + len(test) == len(ids)
    return {
        "train": train.sort_values().reset_index(drop=True),
        "validation": validation.sort_values().reset_index(drop=True),
        "test": test.sort_values().reset_index(drop=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV containing final valid source_id rows")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, usecols=["source_id"], dtype={"source_id": "int64"})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = create_splits(frame["source_id"], args.seed)
    for name, values in splits.items():
        values.to_frame().to_csv(output / f"{name}_source_ids.csv", index=False)
        print(f"{name}: {len(values):,}")

if __name__ == "__main__":
    main()
