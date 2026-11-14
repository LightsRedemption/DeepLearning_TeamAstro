"""Train Ridge and PyTorch MLP regression models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_model import save_evaluation
from models import make_model


TARGET_NAMES = ("TEFF", "LOGG", "FE_H", "N_O", "AL_FE")


def parse_checkpoint_epochs(value: str) -> list[int]:
    """Convert a comma-separated list such as 10,25,50 into integers."""
    epochs = []

    for item in value.split(","):
        item = item.strip()

        if item:
            epochs.append(int(item))

    return sorted(set(epochs))


def choose_device(requested: str) -> torch.device:
    """Choose MPS, CUDA, or CPU."""

    if requested != "auto":
        device = torch.device(requested)

        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for repeatable training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_model(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int
) -> np.ndarray:
    """Generate predictions without storing gradients."""

    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32, copy=False))
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    predictions = []
    model.eval()

    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device)
            output = model(features)
            predictions.append(output.cpu().numpy())

    return np.vstack(predictions)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    model_name: str,
    n_inputs: int,
    n_outputs: int
) -> None:
    """Save a resumable PyTorch checkpoint."""

    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "n_inputs": n_inputs,
            "n_outputs": n_outputs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        path
    )


def train_ridge(args, data, output_dir: Path) -> None:
    """Train the Ridge baseline."""

    X_train = data["X_train"].astype(np.float64)
    y_train = data["y_train"].astype(np.float64)
    X_validation = data["X_validation"].astype(np.float64)
    X_test = data["X_test"].astype(np.float64)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_validation_scaled = feature_scaler.transform(X_validation)
    X_test_scaled = feature_scaler.transform(X_test)

    y_train_scaled = target_scaler.fit_transform(y_train)

    model = Ridge(alpha=args.ridge_alpha)

    start_time = time.perf_counter()
    model.fit(X_train_scaled, y_train_scaled)
    training_seconds = time.perf_counter() - start_time

    validation_scaled = model.predict(X_validation_scaled)
    test_scaled = model.predict(X_test_scaled)

    validation_predictions = target_scaler.inverse_transform(
        validation_scaled
    )
    test_predictions = target_scaler.inverse_transform(test_scaled)

    joblib.dump(model, output_dir / "ridge.joblib")
    joblib.dump(feature_scaler, output_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, output_dir / "target_scaler.joblib")

    np.save(
        output_dir / "validation_predictions.npy",
        validation_predictions
    )
    np.save(
        output_dir / "test_predictions.npy",
        test_predictions
    )

    save_evaluation(
        output_dir / "validation",
        data["validation_source_ids"],
        data["y_validation"],
        validation_predictions,
        TARGET_NAMES
    )

    save_evaluation(
        output_dir / "test",
        data["test_source_ids"],
        data["y_test"],
        test_predictions,
        TARGET_NAMES
    )

    timing = {
        "training_seconds": training_seconds
    }

    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2)
    )

    print(f"Ridge training time: {training_seconds:.2f} seconds")
    print(f"Saved Ridge outputs to {output_dir}")

def train_extra_trees(args, data, output_dir: Path) -> None:
    """Train an Extra Trees multi-output regression model."""

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)

    X_validation = data["X_validation"].astype(np.float32)
    y_validation = data["y_validation"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    model = ExtraTreesRegressor(
        n_estimators=args.extra_trees_estimators,
        max_features=args.extra_trees_max_features,
        min_samples_leaf=args.extra_trees_min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.extra_trees_jobs,
        verbose=1
    )

    print("Model: extra_trees")
    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Trees: {args.extra_trees_estimators}")
    print(f"Maximum features: {args.extra_trees_max_features}")
    print(
        "Minimum samples per leaf: "
        f"{args.extra_trees_min_samples_leaf}"
    )
    print()

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_seconds = time.perf_counter() - start_time

    prediction_start = time.perf_counter()

    validation_predictions = model.predict(
        X_validation
    ).astype(np.float32)

    test_predictions = model.predict(
        X_test
    ).astype(np.float32)

    prediction_seconds = (
        time.perf_counter() - prediction_start
    )

    joblib.dump(
        model,
        output_dir / "extra_trees.joblib",
        compress=3
    )

    np.save(
        output_dir / "validation_predictions.npy",
        validation_predictions
    )

    np.save(
        output_dir / "test_predictions.npy",
        test_predictions
    )

    save_evaluation(
        output_dir / "validation",
        data["validation_source_ids"],
        y_validation,
        validation_predictions,
        TARGET_NAMES
    )

    save_evaluation(
        output_dir / "test",
        data["test_source_ids"],
        y_test,
        test_predictions,
        TARGET_NAMES
    )

    importances = {
        "feature_index": list(
            range(len(model.feature_importances_))
        ),
        "importance": [
            float(value)
            for value in model.feature_importances_
        ]
    }

    (output_dir / "feature_importances.json").write_text(
        json.dumps(importances, indent=2)
    )

    timing = {
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "n_estimators": args.extra_trees_estimators,
        "max_features": args.extra_trees_max_features,
        "min_samples_leaf": (
            args.extra_trees_min_samples_leaf
        ),
        "n_jobs": args.extra_trees_jobs
    }

    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2)
    )

    print()
    print(
        f"Extra Trees training time: "
        f"{training_seconds:.2f} seconds"
    )
    print(
        f"Prediction time: "
        f"{prediction_seconds:.2f} seconds"
    )
    print(f"Saved Extra Trees outputs to {output_dir}")


def train_mlp(
    args,
    data,
    output_dir: Path,
    device: torch.device
) -> None:
    """Train one of the PyTorch MLP models."""

    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.float32)

    X_validation = data["X_validation"].astype(np.float32)
    y_validation = data["y_validation"].astype(np.float32)

    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train
    ).astype(np.float32)

    X_validation_scaled = feature_scaler.transform(
        X_validation
    ).astype(np.float32)

    X_test_scaled = feature_scaler.transform(
        X_test
    ).astype(np.float32)

    y_train_scaled = target_scaler.fit_transform(
        y_train
    ).astype(np.float32)

    y_validation_scaled = target_scaler.transform(
        y_validation
    ).astype(np.float32)

    joblib.dump(
        feature_scaler,
        output_dir / "feature_scaler.joblib"
    )

    joblib.dump(
        target_scaler,
        output_dir / "target_scaler.joblib"
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_scaled),
        torch.from_numpy(y_train_scaled)
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0
    )

    validation_X_tensor = torch.from_numpy(
        X_validation_scaled
    ).to(device)

    validation_y_tensor = torch.from_numpy(
        y_validation_scaled
    ).to(device)

    model = make_model(args.model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    loss_function = nn.MSELoss()

    checkpoint_epochs = parse_checkpoint_epochs(
        args.checkpoint_epochs
    )

    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Training rows: {len(X_train_scaled):,}")
    print(f"Validation rows: {len(X_validation_scaled):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Maximum epochs: {args.max_epochs}")
    print()

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.perf_counter()

        model.train()
        training_loss_sum = 0.0
        training_rows = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)

            loss.backward()
            optimizer.step()

            current_batch_size = batch_X.shape[0]

            training_loss_sum += (
                loss.item() * current_batch_size
            )

            training_rows += current_batch_size

        training_loss = training_loss_sum / training_rows

        model.eval()

        with torch.no_grad():
            validation_predictions = model(
                validation_X_tensor
            )

            validation_loss = loss_function(
                validation_predictions,
                validation_y_tensor
            ).item()

        epoch_seconds = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "epoch_seconds": epoch_seconds
            }
        )

        improved = (
            validation_loss
            < best_validation_loss - args.minimum_delta
        )

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                output_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                validation_loss,
                args.model,
                X_train_scaled.shape[1],
                y_train_scaled.shape[1]
            )
        else:
            epochs_without_improvement += 1

        if epoch in checkpoint_epochs:
            checkpoint_path = (
                output_dir
                / f"checkpoint_epoch_{epoch:03d}.pt"
            )

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_loss,
                args.model,
                X_train_scaled.shape[1],
                y_train_scaled.shape[1]
            )

        print(
            f"Epoch {epoch:03d}/{args.max_epochs:03d} | "
            f"train={training_loss:.6f} | "
            f"validation={validation_loss:.6f} | "
            f"time={epoch_seconds:.2f}s"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
     f"best epoch was {best_epoch}."
            )
            break

    total_training_seconds = (
        time.perf_counter() - training_start
    )

    checkpoint = torch.load(
        output_dir / "best_model.pt",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    validation_predictions_scaled = predict_model(
        model,
        X_validation_scaled,
        device,
        args.batch_size
    )

    test_predictions_scaled = predict_model(
        model,
        X_test_scaled,
        device,
        args.batch_size
    )

    validation_predictions = target_scaler.inverse_transform(
        validation_predictions_scaled
    )

    test_predictions = target_scaler.inverse_transform(
        test_predictions_scaled
    )

    np.save(
        output_dir / "validation_predictions.npy",
        validation_predictions
    )

    np.save(
        output_dir / "test_predictions.npy",
        test_predictions
    )

    save_evaluation(
        output_dir / "validation",
        data["validation_source_ids"],
        y_validation,
        validation_predictions,
        TARGET_NAMES
    )

    save_evaluation(
        output_dir / "test",
        data["test_source_ids"],
        y_test,
        test_predictions,
        TARGET_NAMES
    )

    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2)
    )

    completed_epochs = len(history)
    average_epoch_seconds = (
        total_training_seconds / completed_epochs
    )

    timing = {
        "device": str(device),
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "total_training_seconds": total_training_seconds,
        "average_epoch_seconds": average_epoch_seconds,
        "estimated_200_epoch_seconds": (
            average_epoch_seconds * 200
        )
    }

    (output_dir / "timing.json").write_text(
        json.dumps(timing, indent=2)
    )

    print()
    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation loss: "
        f"{best_validation_loss:.6f}"
    )
    print(
        "Total training time: "
        f"{total_training_seconds:.2f} seconds"
    )
    print(
        "Average time per epoch: "
        f"{average_epoch_seconds:.2f} seconds"
    )
    print(
        "Estimated time for 200 epochs: "
        f"{average_epoch_seconds * 200 / 60:.2f} minutes"
    )
    print(f"Saved outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "ridge",
            "extra_trees",
            "baseline_mlp",
            "deeper_mlp"
        ],
        required=True
    )

    parser.add_argument(
        "--data",
        default="data/model_ready_splits.npz"
    )

    parser.add_argument(
        "--output-root",
        default="results"
    )

    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--extra-trees-estimators",
        type=int,
        default=300
    )

    parser.add_argument(
        "--extra-trees-max-features",
        default="sqrt"
    )

    parser.add_argument(
        "--extra-trees-min-samples-leaf",
        type=int,
        default=2
    )

    parser.add_argument(
        "--extra-trees-jobs",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.00001
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25
    )

    parser.add_argument(
        "--minimum-delta",
        type=float,
        default=0.000001
    )

    parser.add_argument(
        "--checkpoint-epochs",
        default="10,25,50,100,200"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto"
    )

    args = parser.parse_args()

    set_random_seeds(args.seed)

    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} does not exist."
        )

    data = np.load(data_path)

    required = {
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
        "X_test",
        "y_test",
        "train_source_ids",
        "validation_source_ids",
        "test_source_ids"
    }

    missing = required - set(data.files)

    if missing:
        raise KeyError(
            f"Model-ready NPZ lacks keys: {sorted(missing)}"
        )

    output_dir = Path(args.output_root) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()

    if args.model in {"ridge", "extra_trees"}:
        config["device"] = "cpu"
    else:
        config["device"] = str(
            choose_device(args.device)
        )

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    if args.model == "ridge":
        train_ridge(args, data, output_dir)
        return

    if args.model == "extra_trees":
        train_extra_trees(args, data, output_dir)
        return

    device = choose_device(args.device)

    train_mlp(
        args,
        data,
        output_dir,
        device
    )


if __name__ == "__main__":
    main()