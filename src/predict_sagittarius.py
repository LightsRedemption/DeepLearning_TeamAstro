"""Predict stellar labels for Sagittarius Gaia XP sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from astropy.table import Table

from models import make_model
from preprocessing import TARGET_NAMES, preprocess_many


def choose_device(requested: str) -> torch.device:
    """Choose the requested inference device."""

    if requested != "auto":
        device = torch.device(requested)

        if device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS was requested but is unavailable."
                )

        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable."
                )

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def predict_batches(
    model,
    features,
    device,
    batch_size
) -> np.ndarray:
    """Run model inference in batches."""

    predictions = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size

            batch = torch.from_numpy(
                features[start:stop].astype(
                    np.float32,
                    copy=False
                )
            ).to(device)

            predictions.append(
                model(batch).cpu().numpy()
            )

    if not predictions:
        return np.empty(
            (0, len(TARGET_NAMES)),
            dtype=np.float32
        )

    return np.vstack(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/vasiliev_sgr_xp_combined.fits"
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--feature-scaler",
        required=True
    )

    parser.add_argument(
        "--target-scaler",
        required=True
    )

    parser.add_argument(
        "--output",
        default=(
            "results/sgr_preliminary_predictions.csv"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto"
    )

    parser.add_argument(
        "--source-col",
        default="source_id"
    )

    parser.add_argument(
        "--bp-col",
        default="bp_coefficients"
    )

    parser.add_argument(
        "--rp-col",
        default="rp_coefficients"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)
    feature_scaler_path = Path(args.feature_scaler)
    target_scaler_path = Path(args.target_scaler)
    output_path = Path(args.output)

    for path in [
        input_path,
        model_path,
        feature_scaler_path,
        target_scaler_path
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    table = Table.read(input_path)

    required_columns = {
        args.source_col,
        args.bp_col,
        args.rp_col
    }

    missing_columns = (
        required_columns - set(table.colnames)
    )

    if missing_columns:
        raise KeyError(
            "Input FITS file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if len(table) == 0:
        raise ValueError(
            "The Sagittarius XP input contains no rows."
        )

    preprocessing = preprocess_many(
        table[args.bp_col],
        table[args.rp_col]
    )

    source_ids = np.asarray(
        table[args.source_col],
        dtype=np.int64
    )

    valid_source_ids = source_ids[
        preprocessing.valid_mask
    ]

    rejected_source_ids = source_ids[
        ~preprocessing.valid_mask
    ]

    rejected_reasons = np.asarray(
        preprocessing.rejection_reasons,
        dtype=object
    )[~preprocessing.valid_mask]

    if len(preprocessing.features) == 0:
        raise ValueError(
            "No Sagittarius rows passed preprocessing."
        )

    feature_scaler = joblib.load(
        feature_scaler_path
    )

    target_scaler = joblib.load(
        target_scaler_path
    )

    scaled_features = feature_scaler.transform(
        preprocessing.features
    ).astype(np.float32)

    device = choose_device(args.device)

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model_name = checkpoint.get(
        "model_name",
        "baseline_mlp"
    )

    model = make_model(model_name).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    scaled_predictions = predict_batches(
        model,
        scaled_features,
        device,
        args.batch_size
    )

    predictions = target_scaler.inverse_transform(
        scaled_predictions
    )

    output_frame = pd.DataFrame({
        "source_id": valid_source_ids
    })

    for index, target in enumerate(TARGET_NAMES):
        output_frame[f"pred_{target}"] = (
            predictions[:, index]
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_frame.to_csv(
        output_path,
        index=False
    )

    rejection_path = output_path.with_name(
        output_path.stem
        + "_rejected.csv"
    )

    rejected_frame = pd.DataFrame({
        "source_id": rejected_source_ids,
        "rejection_reason": rejected_reasons
    })

    rejected_frame.to_csv(
        rejection_path,
        index=False
    )

    summary = {
        "input": str(input_path),
        "model": str(model_path),
        "model_name": model_name,
        "device": str(device),
        "input_rows": len(table),
        "accepted_rows": len(output_frame),
        "rejected_rows": len(rejected_frame),
        "input_features": preprocessing.features.shape[1],
        "targets": list(TARGET_NAMES),
        "prediction_output": str(output_path),
        "rejection_output": str(rejection_path)
    }

    summary_path = output_path.with_name(
        output_path.stem
        + "_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2)
    )

    print("Model:", model_name)
    print("Device:", device)
    print("Input rows:", len(table))
    print("Accepted rows:", len(output_frame))
    print("Rejected rows:", len(rejected_frame))
    print("Saved predictions:", output_path)
    print("Saved rejections:", rejection_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()