"""Shared regression metrics and prediction exports."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

TARGET_NAMES = ("TEFF", "LOGG", "FE_H", "N_O", "AL_FE")


def regression_metrics(y_true, y_pred, target_names=TARGET_NAMES):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_pred must have the same two-dimensional shape")
    return {
        name: {
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "rmse": float(mean_squared_error(y_true[:, i], y_pred[:, i]) ** 0.5),
        }
        for i, name in enumerate(target_names)
    }


def save_evaluation(output_dir, source_ids, y_true, y_pred, target_names=TARGET_NAMES):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = regression_metrics(y_true, y_pred, target_names)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    frame = pd.DataFrame({"source_id": np.asarray(source_ids, dtype=np.int64)})
    for index, name in enumerate(target_names):
        frame[f"true_{name}"] = np.asarray(y_true)[:, index]
        frame[f"pred_{name}"] = np.asarray(y_pred)[:, index]
        frame[f"residual_{name}"] = frame[f"pred_{name}"] - frame[f"true_{name}"]
    frame.to_csv(output / "predictions.csv", index=False)
    return metrics
