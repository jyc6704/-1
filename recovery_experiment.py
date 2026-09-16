"""Paired 3/8 Colored MNIST: one exposure epoch, then neutral recovery.

Run: python recovery_experiment.py
Smoke: python recovery_experiment.py --seeds 42 --p-values .99 --recovery-epochs 2
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    COLOR_IDS, HUE_JITTER, HUE_STEP, LABEL_TO_COLOR, OTHER_COLOR, TARGET_DIGITS,
    SpuriousColoredMNIST, load_digit38_mnist, make_biased_color_ids,
    make_hue_offsets, make_recovery_color_ids, recovery_neutral_seed,
    split_exposure_recovery_validation, subset_labels,
)
from metrics import recovery_score
from model import BinarySmallCNN
from pilot import seed_everything, unique_run_dir
from train import evaluate_binary, train_one_epoch

LEARNING_RATE = 0.001
METRICS = (
    "aligned_accuracy", "conflict_accuracy", "neutral_accuracy", "shortcut_gap",
    "abs_shortcut_gap", "flip_rate", "recovery_score",
)
FIELDS = (
    "seed", "p", "exposure_epochs", "recovery_epoch", "exposure_train_loss",
    "exposure_train_accuracy", "recovery_train_loss", "recovery_train_accuracy",
    *METRICS, "num_exposure", "num_recovery", "num_validation", "learning_rate",
    "batch_size", "initial_model_sha256",
)


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_plan(seed: int, epochs: int) -> dict:
    return {
        "base_seed": seed, "split_seed": seed, "model_initialization_seed": seed,
        "hue_seed": seed, "exposure_color_seed": seed,
        "exposure_shuffle_seed": seed, "validation_loader_seed": seed,
        "recovery_neutral_seeds": {
            str(epoch): recovery_neutral_seed(seed, epoch)
            for epoch in range(1, epochs + 1)
        },
        "recovery_shuffle_seeds": {
            str(epoch): seed + 20_000 + epoch for epoch in range(1, epochs + 1)
        },
    }


def make_loader(dataset, batch_size: int, workers: int, seed: int, shuffle: bool):
    # Each invocation gets a fresh local RNG. Consuming one p run cannot affect another.
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def validate_colors(labels, colors, p: float) -> None:
    if labels.shape != colors.shape or not set(colors.tolist()) <= set(COLOR_IDS):
        raise ValueError("Invalid color assignment shape or color IDs.")
    for digit in TARGET_DIGITS:
        mask = labels == digit
        count = int(mask.sum())
        if count == 0:
            raise ValueError(f"Missing digit {digit}.")
        aligned = int((colors[mask] == LABEL_TO_COLOR[digit]).sum())
        # Nearest-integer allocation permits at most half an image of rounding.
        if abs(aligned - count * p) > 0.5 + 1e-9:
            raise ValueError(f"Digit {digit}: actual alignment does not match p={p}.")


def prepare_seed(digit_train, seed: int, epochs: int) -> dict:
    splits = split_exposure_recovery_validation(digit_train, seed)
    repeated = split_exposure_recovery_validation(digit_train, seed)
    if any(a.indices != b.indices for a, b in zip(splits, repeated)):
        raise ValueError("Source split is not reproducible.")
    exposure, recovery, validation = splits
    labels = subset_labels(recovery)
    neutral = {}
    for epoch in range(1, epochs + 1):
        colors = make_recovery_color_ids(labels, seed, epoch)
        validate_colors(labels, colors, 0.5)
        if not torch.equal(colors, make_recovery_color_ids(labels, seed, epoch)):
            raise ValueError("Neutral assignment is not reproducible.")
        if epoch > 1 and torch.equal(colors, neutral[epoch - 1]):
            raise ValueError("Consecutive recovery epochs have identical colors.")
        neutral[epoch] = colors
    seed_everything(seed)
    initial_state = copy.deepcopy(BinarySmallCNN().state_dict())
    val_labels = subset_labels(validation)
    hue = make_hue_offsets(len(digit_train.mnist), seed)
    if not torch.isfinite(hue).all() or (hue.abs() > HUE_JITTER).any():
        raise ValueError("Invalid hue offsets.")
    return {
        "splits": splits, "initial_state": initial_state,
        "initial_model_sha256": state_hash(initial_state),
        "assignments": {
            "seeds": seed_plan(seed, epochs), "hue_offsets": hue,
            "recovery_color_ids": neutral,
            "validation_aligned_color_ids": val_labels.clone(),
            "validation_conflict_color_ids": torch.tensor(
                [OTHER_COLOR[int(label)] for label in val_labels], dtype=torch.long
            ),
        },
        "indices": {
            f"{name}_original_indices": digit_train.original_indices[subset.indices]
            for name, subset in zip(("exposure", "recovery", "validation"), splits)
        },
    }


def reset_adam(model, exposure_optimizer):
    """Discard all accumulated Adam moments and steps, preserving model parameters."""
    before = state_hash(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if optimizer is exposure_optimizer or optimizer.state:
        raise ValueError("Recovery Adam must have empty state.")
    if state_hash(model.state_dict()) != before:
        raise ValueError("Optimizer reset changed model weights.")
    return optimizer


def evaluate(model, aligned, conflict, device) -> dict:
    values = evaluate_binary(model, aligned, conflict, device)
    values["abs_shortcut_gap"] = abs(values["shortcut_gap"])
    values["recovery_score"] = recovery_score(
        values["neutral_accuracy"], values["shortcut_gap"], values["flip_rate"]
    )
    for name in METRICS:
        lower = -1.0 if name == "shortcut_gap" else 0.0
        if not math.isfinite(values[name]) or not lower <= values[name] <= 1.0:
            raise ValueError(f"Metric out of range: {name}={values[name]}")
    return values


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["p"], row["recovery_epoch"])].append(row)
    summaries = []
    for (p, epoch), group in sorted(groups.items()):
        if len({r["seed"] for r in group}) != len(group):
            raise ValueError("Duplicate seed in summary group.")
        result = {"p": p, "recovery_epoch": epoch, "num_seeds": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group]
            result[f"{metric}_mean"] = statistics.mean(values)
            # Sample SD (ddof=1); undefined for a single-seed smoke run.
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        summaries.append(result)
    return summaries


def save_summary(rows: list[dict], run_dir: Path) -> list[dict]:
    summary = summarize(rows)
    if summary:
        temporary = run_dir / "summary_results.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        temporary.replace(run_dir / "summary_results.csv")
    return summary


def plot_results(summary: list[dict], run_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for metric, filename, label in (
        ("recovery_score", "recovery_score", "Composite Recovery Score"),
        ("abs_shortcut_gap", "shortcut_gap", "Absolute Shortcut Gap"),
        ("flip_rate", "flip_rate", "Flip Rate"),
        ("neutral_accuracy", "neutral_accuracy", "Neutral Accuracy"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for p in sorted({row["p"] for row in summary}):
            group = [row for row in summary if row["p"] == p]
            x = [row["recovery_epoch"] for row in group]
            y = [row[f"{metric}_mean"] for row in group]
            sd = [row[f"{metric}_std"] for row in group]
            if all(value is not None for value in sd):
                ax.errorbar(x, y, yerr=sd, marker="o", capsize=4, label=f"p={p:g}")
            else:
                ax.plot(x, y, marker="o", label=f"p={p:g} (SD unavailable)")
        ax.set(xlabel="Recovery Epoch (0 = after exposure)", ylabel=label)
        # Leave room for full error bars even when mean +/- SD crosses 0 or 1.
        lows = [r[f"{metric}_mean"] - (r[f"{metric}_std"] or 0) for r in summary]
        highs = [r[f"{metric}_mean"] + (r[f"{metric}_std"] or 0) for r in summary]
        ax.set_ylim(min(0.0, min(lows)) - .02, max(1.0, max(highs)) + .02)
        ax.set_xticks(sorted({row["recovery_epoch"] for row in summary}))
        ax.set_title(f"{label}: seed mean +/- 1 sample SD")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{filename}.png", dpi=150)
        plt.close(fig)


def run_condition(args, config, run_dir, seed, p, prepared, device, record):
    exposure, recovery, validation = prepared["splits"]
    assignments = prepared["assignments"]
    plan = assignments["seeds"]
    hue = assignments["hue_offsets"]
    seed_everything(seed)
    model = BinarySmallCNN().to(device)
    model.load_state_dict(prepared["initial_state"])
    if state_hash(model.state_dict()) != prepared["initial_model_sha256"]:
        raise ValueError("p conditions must start from identical model weights.")
    colors = make_biased_color_ids(subset_labels(exposure), p, plan["exposure_color_seed"])
    validate_colors(subset_labels(exposure), colors, p)
    tag = f"seed{seed}_p{p}"
    torch.save({"color_ids": colors, "seed": seed, "p": p},
               run_dir / "assignments" / f"{tag}_exposure.pt")
    exposure_data = SpuriousColoredMNIST(exposure, colors, hue)
    exposure_loader = make_loader(exposure_data, args.batch_size, args.num_workers,
                                  plan["exposure_shuffle_seed"], True)
    val_loaders = [
        make_loader(SpuriousColoredMNIST(validation, assignments[key], hue),
                    args.batch_size, args.num_workers, plan["validation_loader_seed"], False)
        for key in ("validation_aligned_color_ids", "validation_conflict_color_ids")
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"[seed {seed} | p {p:g}] exposure (1 epoch)", flush=True)
    exposure_loss, exposure_accuracy = train_one_epoch(model, exposure_loader, optimizer, device)

    def evaluate_and_record(epoch, loss=None, accuracy=None):
        values = evaluate(model, *val_loaders, device)
        if any(value is not None and not math.isfinite(value)
               for value in (exposure_loss, exposure_accuracy, loss, accuracy)):
            raise ValueError("Non-finite training result.")
        row = {
            "seed": seed, "p": p, "exposure_epochs": 1, "recovery_epoch": epoch,
            "exposure_train_loss": exposure_loss, "exposure_train_accuracy": exposure_accuracy,
            "recovery_train_loss": loss, "recovery_train_accuracy": accuracy, **values,
            "num_exposure": len(exposure), "num_recovery": len(recovery),
            "num_validation": len(validation), "learning_rate": LEARNING_RATE,
            "batch_size": args.batch_size,
            "initial_model_sha256": prepared["initial_model_sha256"],
        }
        record(row)
        if epoch in (0, args.recovery_epochs):
            torch.save({
                "model_state_dict": model.state_dict(), "config": config,
                "seed": seed, "p": p, "recovery_epoch": epoch, "metrics": row,
            }, run_dir / "checkpoints" / f"{tag}_epoch{epoch:02d}.pt")
        print(f"[seed {seed} | p {p:g}] recovery epoch {epoch}/{args.recovery_epochs} "
              f"neutral={values['neutral_accuracy']:.4f} gap={values['shortcut_gap']:.4f} "
              f"flip={values['flip_rate']:.4f} score={values['recovery_score']:.4f}", flush=True)

    evaluate_and_record(0)
    optimizer = reset_adam(model, optimizer)
    for epoch in range(1, args.recovery_epochs + 1):
        # A new dataset consumes that epoch's prevalidated deterministic assignment.
        recovery_data = SpuriousColoredMNIST(
            recovery, assignments["recovery_color_ids"][epoch], hue
        )
        loader = make_loader(recovery_data, args.batch_size, args.num_workers,
                             plan["recovery_shuffle_seeds"][str(epoch)], True)
        loss, accuracy = train_one_epoch(model, loader, optimizer, device)
        evaluate_and_record(epoch, loss, accuracy)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--p-values", type=float, nargs="+", default=[.5, .9, .99])
    parser.add_argument("--recovery-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=None,
                        help="Optional CPU intra-op thread count, recorded in config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("recovery_results_38"))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= s < 2**32 for s in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**32).")
    if len(set(args.p_values)) != len(args.p_values) or any(not .5 <= p <= 1 for p in args.p_values):
        parser.error("p-values must be unique finite numbers in [0.5, 1].")
    if args.recovery_epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive; num-workers must be nonnegative.")
    if args.num_threads is not None and args.num_threads < 1:
        parser.error("num-threads must be positive.")
    return args


def main(argv=None) -> Path:
    args = parse_args(argv)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "digits": list(TARGET_DIGITS), "color_ids": list(COLOR_IDS),
        "target_mapping": {str(d): i for i, d in enumerate(TARGET_DIGITS)},
        "split_ratios": [.4, .4, .2], "exposure_epochs": 1,
        "model_class": "BinarySmallCNN", "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "loss": "CrossEntropyLoss", "reset_optimizer_after_exposure": True,
        "hue_step_degrees": HUE_STEP, "hue_jitter_degrees": HUE_JITTER,
        "recovery_score_definition": "neutral_accuracy * (1 - abs(shortcut_gap)) * (1 - flip_rate)",
        "recovery_score_interpretation": "study-defined composite index, not percent recovered",
        "summary_sd_ddof": 1, "single_seed_sd": "undefined: blank CSV, no error bars",
        "seed_plans": {str(s): seed_plan(s, args.recovery_epochs) for s in args.seeds},
        "device": str(device), "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "num_threads": torch.get_num_threads(), "platform": platform.platform(),
    }
    run_dir = unique_run_dir(args.output, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    for folder in ("assignments", "checkpoints"):
        (run_dir / folder).mkdir()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"saved={run_dir.resolve()} | device={device}", flush=True)
    rows, split_indices = [], {}

    def status(state, **extra):
        (run_dir / "status.json").write_text(json.dumps({
            "status": state, "completed_evaluations": len(rows),
            "expected_evaluations": len(args.seeds) * len(args.p_values) * (args.recovery_epochs + 1),
            **extra,
        }, indent=2), encoding="utf-8")

    status("running")
    try:
        digit_train, _ = load_digit38_mnist(root=str(args.data_root), download=args.download)
        with (run_dir / "raw_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()

            def record(row):
                writer.writerow(row)
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(row)
                save_summary(rows, run_dir)
                status("running")

            for seed in args.seeds:
                prepared = prepare_seed(digit_train, seed, args.recovery_epochs)
                split_indices[seed] = prepared["indices"]
                torch.save(split_indices, run_dir / "split_indices.pt")
                torch.save({**prepared["assignments"],
                            "initial_model_sha256": prepared["initial_model_sha256"]},
                           run_dir / "assignments" / f"seed{seed}_shared.pt")
                for p in args.p_values:
                    run_condition(args, config, run_dir, seed, p, prepared, device, record)
        plot_results(save_summary(rows, run_dir), run_dir)
    except BaseException as error:
        status("failed", error=f"{type(error).__name__}: {error}")
        raise
    status("complete")
    return run_dir


if __name__ == "__main__":
    main()
