"""Binary (3 vs 8) colored-MNIST recovery study with dense-then-sparse evaluations.

Designed for https://github.com/jyc6704/-1 . Place this file in the repository root.

    python binary_step_experiment_v2.py               # full study
    python binary_step_experiment_v2.py --smoke       # quick REAL-data trial
    python binary_step_experiment_v2.py --analyze-only # verify and rebuild reports

The old experiment.py, recovery_experiment.py, binary_step_experiment.py and their directories are untouched.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torchvision
from filelock import FileLock, Timeout

from experiment_data import load_source, prepare_source, palette
from experiment_io import (
    atomic, digest, git, keep_awake, load_checkpoint, model_hash, publish,
    read_json, restore_rng, rng_state, save_checkpoint, save_npz, stable_seed,
    verify_manifest, write_csv, write_json, write_manifest, write_text,
)
from model import BinarySmallCNN

HERE = Path(__file__).resolve().parent
SOURCE_NAME = "mnist_38"  # Legacy constant; run names are selected with --datasets
MODEL_NAME = "BinarySmallCNN"
DIGITS = (3, 8)
COLORS = (3, 8)
BATCH_SIZE = 64
LR = 0.001
METRICS = (
    "mean_color_accuracy", "all_colors_accuracy", "aligned_accuracy",
    "conflict_accuracy", "neutral_accuracy", "shortcut_gap", "abs_shortcut_gap",
    "flip_rate", "recovery_score", "digit3_q", "digit8_q",
)
CODE_FILES = ["binary_step_experiment_v2.py", "experiment_data.py", "experiment_io.py", "model.py"]


class PlannedStop(Exception):
    """A test-controlled graceful stop with a committed checkpoint."""


def now():
    return datetime.now(timezone.utc).isoformat()


def ptag(p):
    return f"p{round(p*100):03d}"


def configure(device, threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")


def binary_ids(source):
    ids = np.asarray(source["ids"], dtype=np.int64)
    test = np.asarray(source["test_ids"], dtype=np.int64)
    return ids[np.isin(source["labels"][ids], DIGITS)], test[np.isin(source["test_labels"][test], DIGITS)]


def stratified_split(ids, labels, seed):
    """One disjoint 40/40/20 split per seed; invariant to p/exposure length."""
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(ids[labels[ids] == d]) for d in DIGITS]
    if min(map(len, groups)) < 5:
        raise ValueError("At least five source images per digit are needed for each split.")
    pieces = {"exposure": [], "recovery": [], "validation": []}
    for group in groups:
        n = len(group)
        a, b = round(n * .4), round(n * .4)
        pieces["exposure"].append(group[:a])
        pieces["recovery"].append(group[a:a+b])
        pieces["validation"].append(group[a+b:])
    out = {k: np.concatenate(v) for k, v in pieces.items()}
    joined = np.concatenate(list(out.values()))
    if len(joined) != len(ids) or len(np.unique(joined)) != len(ids) or set(joined) != set(ids):
        raise ValueError("Train source split overlaps or loses items.")
    return out


def assign_binary_colors(labels, p, seed):
    if not math.isfinite(p) or not .5 <= p <= 1.:
        raise ValueError("Binary aligned-color probability must be in [0.5,1].")
    rng = np.random.default_rng(seed)
    out = np.empty(len(labels), dtype=np.uint8)
    for digit, opposite in ((3, 8), (8, 3)):
        ix = rng.permutation(np.flatnonzero(labels == digit))
        aligned = round(len(ix) * p)
        out[ix[:aligned]] = digit
        out[ix[aligned:]] = opposite
        if abs(int((out[labels == digit] == digit).sum()) - len(ix)*p) > .500001:
            raise ValueError("Biased/neutral color distribution verification failed.")
    return out


def initial_model(seed):
    torch.manual_seed(stable_seed(seed, "binary38_model"))
    return BinarySmallCNN()


def scheduled_recovery_steps(steps_per_epoch: int, recovery_epochs: int,
                             late_eval_interval: int = 5) -> list[int]:
    """Absolute recovery steps, with t=0, every first-epoch step, then sparse checks.

    The later interval restarts at each epoch boundary; the final batch of EVERY
    recovery epoch is always evaluated even when its step count is not a multiple
    of the interval. Do not interpolate missing intermediate evaluations.
    """
    if steps_per_epoch < 1 or recovery_epochs < 1 or late_eval_interval < 1:
        raise ValueError("Invalid recovery measurement schedule parameters.")
    scheduled = [0]
    for epoch in range(1, recovery_epochs + 1):
        for local_step in range(1, steps_per_epoch + 1):
            if (epoch == 1 or local_step % late_eval_interval == 0
                    or local_step == steps_per_epoch):
                scheduled.append((epoch - 1) * steps_per_epoch + local_step)
    return scheduled


def should_evaluate_recovery(epoch: int, local_step: int, steps_per_epoch: int,
                             late_eval_interval: int = 5) -> bool:
    if not (1 <= local_step <= steps_per_epoch and epoch >= 1):
        raise ValueError("Invalid epoch/local optimizer step.")
    return (epoch == 1 or local_step % late_eval_interval == 0
            or local_step == steps_per_epoch)


def make_shared_design(root, source, ids, test_ids, seed, p_values, exposure_epochs, recovery_epochs, name=SOURCE_NAME):
    folder = root / "shared" / name / f"seed{seed}"
    splits = stratified_split(ids, source["labels"], stable_seed(name, seed, "split"))
    hue = np.random.default_rng(stable_seed(name, seed, "hue")).uniform(
        -5., 5., len(source["labels"])
    ).astype(np.float32)
    test_hue = np.random.default_rng(stable_seed(name, seed, "test_hue")).uniform(
        -5., 5., len(source["test_labels"])
    ).astype(np.float32)
    arrays = {**splits, "test_ids": test_ids, "hue": hue, "test_hue": test_hue,
              "digits": np.asarray(DIGITS, dtype=np.int64), "colors": np.asarray(COLORS, dtype=np.int64)}
    frequencies = {}
    for p in p_values:
        k = f"exposure_{ptag(p)}"
        arrays[k] = assign_binary_colors(source["labels"][splits["exposure"]], p,
                                         stable_seed(name, seed, "exposure_color"))
        frequencies[k] = {str(d): {str(c): int((arrays[k][source["labels"][splits["exposure"]] == d] == c).sum())
                                   for c in COLORS} for d in DIGITS}
    for e in range(1, recovery_epochs + 1):
        k = f"recovery_colors_{e}"
        arrays[k] = assign_binary_colors(source["labels"][splits["recovery"]], .5,
                                         stable_seed(name, seed, "recovery_color", e))
        frequencies[k] = {str(d): {str(c): int((arrays[k][source["labels"][splits["recovery"]] == d] == c).sum())
                                   for c in COLORS} for d in DIGITS}
        if e > 1 and np.array_equal(arrays[k], arrays[f"recovery_colors_{e-1}"]):
            raise ValueError("Two consecutive recovery color allocations are identical.")
    for e in range(1, max(exposure_epochs) + 1):
        arrays[f"exposure_order_{e}"] = np.random.default_rng(
            stable_seed(name, seed, "exposure_order", e)
        ).permutation(len(splits["exposure"]))
    for e in range(1, recovery_epochs + 1):
        arrays[f"recovery_order_{e}"] = np.random.default_rng(
            stable_seed(name, seed, "recovery_order", e)
        ).permutation(len(splits["recovery"]))
    path = folder / "design.npz"
    if path.exists():
        with np.load(path) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise ValueError("Shared design differs from existing run; use another --output.")
    else:
        save_npz(path, **arrays)
    write_json(folder / "color_counts.json", frequencies)
    write_json(folder / "manifest.json", {"seed": seed, "source": name,
        "design_sha256": digest(path), "counts": {k: len(v) for k, v in splits.items()},
        "classes": list(DIGITS), "colors": list(COLORS), "exposure_epochs": exposure_epochs,
        "recovery_epochs": recovery_epochs, "same_hue_per_source": True})
    return {**arrays, "palette": palette(hue), "test_palette": palette(test_hue), "hash": digest(path)}


def colored_batch(source_images, source_ids, color_ids, rgb_palette, device):
    ids = torch.as_tensor(source_ids, dtype=torch.long)
    colors = torch.as_tensor(color_ids, dtype=torch.long)
    gray = source_images[ids].to(device=device, dtype=torch.float32) / 255.
    rgb = rgb_palette[ids, colors].to(device)
    return gray[:, None, :, :] * rgb[:, :, None, None]


def binary_metrics(predictions, digits):
    """predictions has columns for colors (3,8); values are class IDs (0,1)."""
    pred = np.asarray(predictions)
    ys = np.asarray(digits)
    if pred.ndim != 2 or pred.shape != (len(ys), 2) or len(ys) == 0:
        raise ValueError("Predictions must be nonempty N x 2 matrix.")
    if not np.isin(pred, (0, 1)).all() or not np.isin(ys, DIGITS).all():
        raise ValueError("Invalid binary predictions or original digit labels.")
    target = (ys == 8).astype(np.uint8)
    right = pred == target[:, None]
    aligned_ix = (ys == 8).astype(np.int64)
    arange = np.arange(len(ys))
    aligned = float(right[arange, aligned_ix].mean())
    conflict = float(right[arange, 1-aligned_ix].mean())
    flip = float((pred[:, 0] != pred[:, 1]).mean())
    mean = float(right.mean())
    q = float(right.all(axis=1).mean())
    gap = aligned - conflict
    if not np.isclose(q, mean-flip/2., atol=1e-12):
        raise AssertionError("Binary all-colors identity Q = M - F/2 failed.")
    return {"mean_color_accuracy": mean, "all_colors_accuracy": q,
            "aligned_accuracy": aligned, "conflict_accuracy": conflict,
            "neutral_accuracy": mean, "shortcut_gap": gap, "abs_shortcut_gap": abs(gap),
            "flip_rate": flip, "recovery_score": mean * (1-abs(gap))*(1-flip),
            "digit3_q": float(right[ys == 3].all(axis=1).mean()) if (ys == 3).any() else None,
            "digit8_q": float(right[ys == 8].all(axis=1).mean()) if (ys == 8).any() else None}


def evaluate(model, images, labels, source_ids, rgb_palette, device, batch_size):
    """Exact paired evaluation of BOTH colors of SAME source; no training or RNG change."""
    saved_rng = rng_state()
    was_training = model.training
    out = np.empty((len(source_ids), 2), dtype=np.uint8)
    started = time.perf_counter()
    max_sources = max(1, batch_size // 2)
    try:
        model.eval()
        with torch.inference_mode():
            for st in range(0, len(source_ids), max_sources):
                ids = source_ids[st:st+max_sources]
                t = torch.as_tensor(ids, dtype=torch.long)
                gray = images[t].to(device=device, dtype=torch.float32)/255.
                rgb = rgb_palette[t][:, COLORS].to(device)
                pair = (gray[:, None, None, :, :] * rgb[:, :, :, None, None]).flatten(0, 1)
                result = torch.cat([model(chunk).argmax(1).cpu() for chunk in pair.split(batch_size)])
                out[st:st+len(ids)] = result.reshape(-1,2).numpy().astype(np.uint8)
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
    return out, binary_metrics(out, labels[source_ids]), time.perf_counter()-started


def run_path(root, protocol, seed, p, exposure_epochs, source, design, args, name=SOURCE_NAME):
    path = root / "runs" / name / f"seed{seed}" / f"exp{exposure_epochs:02d}" / ptag(p)
    path.mkdir(parents=True, exist_ok=True)
    run_id = f"{name}_s{seed}_exp{exposure_epochs}_{ptag(p)}"
    model = initial_model(seed).to(args.device)
    initial_hash = model_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    expected = {"protocol_hash": protocol["protocol_hash"], "design_hash": design["hash"],
                "initial_model_sha256": initial_hash, "run_id": run_id}
    latest = path / "latest.json"
    if latest.exists():
        pointer = read_json(latest)
        checkpoint = path / pointer["file"]
        if digest(checkpoint) != pointer["sha256"]:
            raise ValueError("Latest checkpoint hash mismatch; refusing to resume.")
        state = load_checkpoint(checkpoint)
        if any(state.get(k) != v for k,v in expected.items()):
            raise ValueError("Checkpoint's protocol, seed design, or initialization differs.")
        model.load_state_dict(state.pop("model"))
        optimizer.load_state_dict(state.pop("optimizer"))
        restore_rng(state.pop("rng"))
    else:
        state = {**expected, "epoch_index": 0, "next_batch": 0, "global_step": 0,
                 "recovery_step": 0, "recovery_samples_seen": 0, "exposure_samples_seen": 0,
                 "loss_sum": 0., "correct": 0, "samples": 0,
                 "observations": [], "train_log": [], "prediction_blocks": [],
                 "elapsed_seconds": 0.}
    write_json(path / "config.json", {**expected, "dataset": name, "seed": seed,
        "p": p, "exposure_epochs": exposure_epochs, "recovery_epochs": args.recovery_epochs,
        "num_exposure": len(design["exposure"]), "num_recovery": len(design["recovery"]),
        "num_validation": len(design["validation"]), "num_test": len(design["test_ids"]),
        "batch_size": args.batch_size, "checkpoint_every_steps": args.checkpoint_every_steps,
        "late_eval_interval": args.late_eval_interval})
    initial_path = root / "shared" / name / f"seed{seed}" / "initial_model.pt"
    if not initial_path.exists():
        saved_rng = rng_state()
        save_checkpoint(initial_path, initial_model(seed).state_dict())
        restore_rng(saved_rng)
    if model_hash(model) != initial_hash and not latest.exists():
        raise ValueError("Initial model hash changed unexpectedly.")
    buffer = []
    base_elapsed = state["elapsed_seconds"]
    started = time.perf_counter()

    def checkpoint_now():
        # Persist predictions BEFORE checkpoint pointer; orphan blocks are harmless on crash.
        if buffer:
            steps = np.asarray([step for step, _ in buffer], dtype=np.int32)
            values = np.stack([prediction for _,prediction in buffer])
            name = f"validation_block_{steps[0]:07d}_{steps[-1]:07d}.npz"
            predpath = path / "raw" / "predictions" / name
            save_npz(predpath, steps=steps, predictions=values,
                     targets=source["labels"][design["validation"]],
                     source_ids=design["validation"], color_ids=np.asarray(COLORS))
            relative = predpath.relative_to(root).as_posix()
            checksum = digest(predpath)
            positions = {int(step): i for i,step in enumerate(steps)}
            for row in state["observations"]:
                if row["recovery_step"] in positions and "predictions_file" not in row:
                    row["predictions_file"] = relative
                    row["predictions_index"] = positions[row["recovery_step"]]
                    row["predictions_sha256"] = checksum
            state["prediction_blocks"].append({"file": relative, "sha256": checksum,
                                               "from_step": int(steps[0]), "to_step": int(steps[-1])})
            buffer.clear()
        state["elapsed_seconds"] = base_elapsed + time.perf_counter()-started
        ckpt = path / "checkpoints" / f"step_{state['global_step']:07d}.pt"
        save_checkpoint(ckpt, {**state, "model": model.state_dict(),
                               "optimizer": optimizer.state_dict(), "rng": rng_state()})
        write_json(latest, {"file": ckpt.relative_to(path).as_posix(), "sha256": digest(ckpt)})
        write_json(path / "metrics.json", state["observations"])
        write_csv(path / "train_log.csv", state["train_log"],
                  fields=["phase","epoch","loss","accuracy","num_sources","global_step"])
        write_json(path / "status.json", {"run_id":run_id,
            "training_complete": state["epoch_index"] >= exposure_epochs + args.recovery_epochs,
            "global_step":state["global_step"], "recovery_step":state["recovery_step"],
            "epoch_index":state["epoch_index"],"next_batch":state["next_batch"],"updated_at":now()})

    def observe():
        predictions, values, seconds = evaluate(model, source["images"], source["labels"],
            design["validation"], design["palette"], args.device, args.eval_batch_size)
        t = state["recovery_step"]
        total_batches = math.ceil(len(design["recovery"])/args.batch_size)
        rec_epoch = 0 if not t else math.ceil(t/total_batches)
        epoch_batch = t - (rec_epoch-1)*total_batches if t else 0
        # This is the exact cumulative number of training sources, including
        # the smaller final batch; checkpoints preserve it across resumes.
        actual = state["recovery_samples_seen"] / len(design["recovery"])
        if t and not math.isclose(actual, (rec_epoch-1) + min(
                epoch_batch*args.batch_size, len(design["recovery"])) / len(design["recovery"]),
                abs_tol=1e-10):
            raise ValueError("Recovery step and image count disagree.")
        row = {"run_id":run_id,"dataset":name,"seed":seed,"p":p,
               "exposure_epochs":exposure_epochs,"split":"validation","recovery_epoch":rec_epoch,
               "actual_recovery_epoch":actual,"recovery_step":t,
               "global_step":state["global_step"],
               "recovery_epoch_step":epoch_batch,
               "recovery_samples_seen":state["recovery_samples_seen"],
               "exposure_samples_seen":state["exposure_samples_seen"],
               "num_sources":len(design["validation"]),
               "num_colors":2,"num_exposure":len(design["exposure"]),
               "num_recovery":len(design["recovery"]),"initial_model_sha256":initial_hash,
               "design_sha256":design["hash"],"evaluation_seconds":seconds,
               "elapsed_seconds":base_elapsed+time.perf_counter()-started, **values}
        state["observations"].append(row)
        buffer.append((t, predictions))
        print(f"[{run_id}] step {t:4}/{args.recovery_epochs*total_batches} "
              f"Q={values['all_colors_accuracy']:.4f} M={values['mean_color_accuracy']:.4f} "
              f"G={values['shortcut_gap']:+.4f} F={values['flip_rate']:.4f} "
              f"S={values['recovery_score']:.4f}", flush=True)

    if state["epoch_index"] >= exposure_epochs + args.recovery_epochs:
        return path
    print(f"[{run_id}] resume epoch_index={state['epoch_index']}, "
          f"batch={state['next_batch']}, step={state['recovery_step']}", flush=True)
    while state["epoch_index"] < exposure_epochs + args.recovery_epochs:
        index = state["epoch_index"]
        is_exposure = index < exposure_epochs
        phase = "exposure" if is_exposure else "recovery"
        epoch = index+1 if is_exposure else index-exposure_epochs+1
        ids = design[phase]
        order = design[f"{phase}_order_{epoch}"]
        colors = design[f"exposure_{ptag(p)}"] if is_exposure else design[f"recovery_colors_{epoch}"]
        batches = math.ceil(len(ids)/args.batch_size)
        for bi in range(state["next_batch"], batches):
            positions = order[bi*args.batch_size:(bi+1)*args.batch_size]
            ix = ids[positions]
            model.train()
            x = colored_batch(source["images"], ix, colors[positions], design["palette"], args.device)
            targets = torch.as_tensor((source["labels"][ix] == 8).astype(np.int64), device=args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError(f"{run_id}: nonfinite training loss")
            loss.backward()
            optimizer.step()
            state["global_step"] += 1
            state["next_batch"] = bi+1
            state["loss_sum"] += float(loss.detach()) * len(ix)
            state["correct"] += int((logits.argmax(1)==targets).sum())
            state["samples"] += len(ix)
            if is_exposure:
                state["exposure_samples_seen"] += len(ix)
            else:
                state["recovery_step"] += 1
                state["recovery_samples_seen"] += len(ix)
                if should_evaluate_recovery(epoch, bi + 1, batches, args.late_eval_interval):
                    observe()
            end = bi+1 == batches
            if end:
                state["train_log"].append({"phase":phase,"epoch":epoch,
                    "loss":state["loss_sum"]/state["samples"],
                    "accuracy":state["correct"]/state["samples"],
                    "num_sources":state["samples"],"global_step":state["global_step"]})
                state.update(epoch_index=index+1, next_batch=0, loss_sum=0., correct=0, samples=0)
                if is_exposure and epoch == exposure_epochs:
                    observe()  # Baseline t=0: AFTER exactly E biased exposure epochs.
                    before = model_hash(model)
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                    if optimizer.state or model_hash(model) != before:
                        raise ValueError("Adam reset must preserve model weights and drop Adam moments.")
            stop = args.stop_after_steps is not None and state["global_step"] >= args.stop_after_steps
            if end or stop or (not is_exposure and state["recovery_step"] % args.checkpoint_every_steps == 0) or (
                is_exposure and state["global_step"] % args.checkpoint_every_steps == 0
            ):
                checkpoint_now()
            if stop:
                raise PlannedStop(f"Saved {run_id} at global step {state['global_step']}. Rerun to resume.")
    return path


def evaluate_test_endpoints(root, protocol, path, source, design, args):
    config = read_json(path / "config.json")
    obs = read_json(path / "metrics.json")
    already = read_json(path / "test_metrics.json") if (path / "test_metrics.json").exists() else []
    expected = [0, args.recovery_epochs*math.ceil(len(design["recovery"])/args.batch_size)]
    for t in expected:
        if any(row["recovery_step"] == t for row in already):
            continue
        row = next(row for row in obs if row["recovery_step"] == t)
        checkpoint = root / row["checkpoint_file"] if "checkpoint_file" in row else None
        if checkpoint is None or not checkpoint.exists():
            # Each evaluation is followed by a checkpoint only at t=0 or epoch boundary.
            steps = sorted((int(f.stem.split("_")[-1]), f) for f in (path / "checkpoints").glob("step_*.pt"))
            needed_global = int(row["global_step"])
            matches = [f for step, f in steps if step == needed_global]
            if not matches:
                raise ValueError(f"Missing saved checkpoint for official test step={t}.")
            checkpoint = matches[0]
        state = load_checkpoint(checkpoint)
        model = BinarySmallCNN().to(args.device)
        model.load_state_dict(state["model"])
        pred,metrics,seconds = evaluate(model, source["test_images"], source["test_labels"],
            design["test_ids"], design["test_palette"], args.device, args.eval_batch_size)
        predpath = path / "raw" / "predictions" / f"test_step{t:07d}.npz"
        save_npz(predpath, predictions=pred, source_ids=design["test_ids"],
            targets=source["test_labels"][design["test_ids"]], color_ids=np.asarray(COLORS))
        already.append({**row, "split":"test", "num_sources":len(design["test_ids"]),
            "predictions_file":predpath.relative_to(root).as_posix(),
            "predictions_index":0,"predictions_sha256":digest(predpath),
            "evaluation_seconds":seconds,"elapsed_seconds":None,**metrics})
        write_json(path / "test_metrics.json", already)
        print(f"[{config['run_id']}] official TEST step={t} Q={metrics['all_colors_accuracy']:.4f}", flush=True)


def all_observations(root):
    result = []
    for path in sorted((root / "runs").glob("*/seed*/exp*/p*/metrics.json")):
        result.extend(read_json(path))
        test = path.with_name("test_metrics.json")
        if test.exists():
            result.extend(read_json(test))
    return sorted(result, key=lambda r:(r["dataset"],r["split"],r["exposure_epochs"],
                                        r["p"],r["seed"],r["recovery_step"]))


def derive(rows, protocol):
    """R_Q uses each dataset/seed/exposure's own neutral p=.5 final Q."""
    keys = ("dataset", "split", "seed", "exposure_epochs", "p", "recovery_step")
    by_key = {tuple(r[k] for k in keys): r for r in rows}
    final_control = {}
    for r in rows:
        if r["p"] != .5:
            continue
        k = (r["dataset"], r["split"], r["seed"], r["exposure_epochs"])
        if k not in final_control or r["recovery_step"] > final_control[k]["recovery_step"]:
            final_control[k] = r
    out = []
    for row in rows:
        r = dict(row)
        dataset, split, seed, exp, p = (row[k] for k in keys[:-1])
        same = (dataset, split, seed, exp)
        base = by_key.get((*same, p, 0))
        reference = final_control.get(same)
        control_at_step = by_key.get((*same, .5, row["recovery_step"]))
        q = row["all_colors_accuracy"]
        r["q_initial"] = base["all_colors_accuracy"] if base else None
        r["q_reference"] = reference["all_colors_accuracy"] if reference else None
        r["q_gap_same_step_pp"] = (100 * (control_at_step["all_colors_accuracy"]-q)
                                    if control_at_step else None)
        if p == .5:
            r["q_recovery_percent"] = None
            r["recovery_percent_status"] = "neutral_control"
        elif base is None or reference is None:
            r["q_recovery_percent"] = None
            r["recovery_percent_status"] = "reference_missing"
        elif reference["all_colors_accuracy"] - base["all_colors_accuracy"] <= .001:
            r["q_recovery_percent"] = None
            r["recovery_percent_status"] = "initial_gap_too_small_or_nonpositive"
        else:
            r["q_recovery_percent"] = 100*(q-base["all_colors_accuracy"]) / (
                reference["all_colors_accuracy"] - base["all_colors_accuracy"])
            r["recovery_percent_status"] = "available"
        out.append(r)
    return out


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"],row["split"],row["exposure_epochs"],row["p"],
                row["recovery_step"])].append(row)
    result = []
    for (dataset,split,exposure,p,step),group in sorted(groups.items()):
        r = {"dataset":dataset,"split":split,"exposure_epochs":exposure,"p":p,
             "recovery_step":step,
             "recovery_epoch_mean":statistics.mean(x["actual_recovery_epoch"] for x in group),
             "num_seeds":len(group)}
        for metric in (*METRICS,"q_recovery_percent","q_gap_same_step_pp"):
            values = [row[metric] for row in group if row.get(metric) is not None]
            r[f"{metric}_mean"] = statistics.mean(values) if values else None
            r[f"{metric}_std"] = statistics.stdev(values) if len(values)>1 else None
            r[f"{metric}_n"] = len(values)
        result.append(r)
    return result

def verify_predictions(root, rows):
    """All saved per-step predictions must reproduce every reported base metric."""
    # Keep at most ONE compressed prediction block in memory; EMNIST may contain
    # tens of thousands of evaluated steps, so an unlimited cache would exceed RAM.
    cached_name = None
    cached_value = None
    unique_files = set()
    count = 0
    for row in rows:
        if "predictions_file" not in row:
            raise ValueError(f"Uncommitted or missing predictions in {row['run_id']} step={row['recovery_step']}")
        name = row["predictions_file"]
        if name != cached_name:
            f = root / name
            if digest(f) != row["predictions_sha256"]:
                raise ValueError("Prediction chunk checksum mismatch: " + name)
            with np.load(f) as z:
                cached_value = (z["predictions"], z["targets"].copy())
            cached_name = name
            unique_files.add(name)
        preds, targets = cached_value
        p = preds[int(row["predictions_index"])] if row["split"] == "validation" else preds
        expected = binary_metrics(p, targets)
        for metric in METRICS:
            if not math.isclose(expected[metric], row[metric], abs_tol=1e-10):
                raise ValueError(f"Saved prediction check failed: {name} {metric}")
        count += 1
    return {"evaluations_verified": count, "unique_prediction_files": len(unique_files),
            "binary_q_identity": "Q=M-F/2", "raw_predictions": "verified"}


def plot_results(root, summary, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = root / "figures"
    folder.mkdir(parents=True,exist_ok=True)
    for name in [base+"_38" for base in protocol["datasets"]]:
        for exp in protocol["exposure_epochs"]:
            rows = [r for r in summary if r["dataset"]==name and r["split"]=="validation"
                    and r["exposure_epochs"]==exp]
            if not rows: continue
            panels = [("all_colors_accuracy","Both-colors accuracy Q"),
                      ("q_recovery_percent","Reference-based recovery R_Q (%)"),
                      ("shortcut_gap","Shortcut gap (A-C)"),
                      ("flip_rate","Color flip rate"),
                      ("recovery_score","Composite recovery score S"),
                      ("mean_color_accuracy","Mean color accuracy M")]
            fig,axes=plt.subplots(3,2,figsize=(13,12))
            for ax,(metric,title) in zip(axes.flat,panels):
                for p in protocol["p_values"]:
                    curve=[r for r in rows if r["p"]==p and r[f"{metric}_mean"] is not None]
                    if not curve:continue
                    x=[r["recovery_step"] for r in curve]
                    y=[r[f"{metric}_mean"] for r in curve]
                    sd=[r[f"{metric}_std"] for r in curve]
                    ax.plot(x,y,label=f"p={p:g}")
                    if all(v is not None for v in sd):
                        ax.fill_between(x,[a-b for a,b in zip(y,sd)],
                                        [a+b for a,b in zip(y,sd)],alpha=.12)
                ax.set_title(title)
                ax.set_xlabel("Recovery optimizer step (0 = just after exposure)")
                ax.grid(alpha=.25)
                if ax.lines:ax.legend()
            fig.suptitle(f"{name}, biased exposure={exp} epochs; seed mean +/- 1 sample SD")
            fig.tight_layout()
            for ext in ("png","svg"):
                fig.savefig(folder/f"{name}_exp{exp:02d}_recovery.{ext}",dpi=160)
            plt.close(fig)
        for p in [value for value in protocol["p_values"] if value!=.5]:
            fig,ax=plt.subplots(figsize=(9,5))
            for exp in protocol["exposure_epochs"]:
                curve = [r for r in summary if r["dataset"]==name and r["split"]=="validation"
                         and r["p"]==p and r["exposure_epochs"]==exp
                         and r["q_recovery_percent_mean"] is not None]
                if curve:
                    ax.plot([r["recovery_step"] for r in curve],
                            [r["q_recovery_percent_mean"] for r in curve],label=f"exposure {exp} epochs")
            ax.set(xlabel="Recovery optimizer step",ylabel="R_Q (%)",
                   title=f"{name}: exposure duration comparison p={p:g}; per-length neutral reference")
            ax.grid(alpha=.25)
            if ax.lines:ax.legend()
            fig.tight_layout()
            for ext in ("png","svg"):
                fig.savefig(folder/f"exposure_comparison_{name}_{ptag(p)}.{ext}",dpi=160)
            plt.close(fig)


def build_report(root, protocol, complete=False):
    rows=all_observations(root)
    write_csv(root/"raw"/"observations.csv",rows)
    verification=verify_predictions(root,rows)
    derived=derive(rows,protocol)
    summary=summarize(derived)
    write_csv(root/"processed"/"derived_results.csv",derived)
    write_csv(root/"processed"/"per_digit_results.csv",[
        {k:r[k] for k in ("run_id","dataset","split","seed","p","exposure_epochs",
                         "recovery_step","digit3_q","digit8_q")} for r in derived
    ])
    write_csv(root/"processed"/"validation_summary.csv",[r for r in summary if r["split"]=="validation"])
    write_csv(root/"processed"/"test_summary.csv",[r for r in summary if r["split"]=="test"])
    groups=defaultdict(list)
    for row in summary:
        groups[(row["dataset"],row["split"],row["exposure_epochs"],row["p"])].append(row)
    key=[]
    for group in groups.values():
        limit=max(r["recovery_step"] for r in group)
        key.extend(r for r in group if r["recovery_step"] in (0,limit))
    write_csv(root/"processed"/"key_results.csv",key)
    recovery_times=[]
    by_path=defaultdict(list)
    for r in derived:
        by_path[(r["dataset"],r["split"],r["seed"],r["exposure_epochs"],r["p"])].append(r)
    for k,group in sorted(by_path.items()):
        if k[-1]==.5:continue
        group.sort(key=lambda r:r["recovery_step"])
        first=next((r for r in group if r["q_recovery_percent"] is not None
                    and r["q_recovery_percent"]>=90),None)
        prev = None
        if first is not None:
            ix = group.index(first)
            prev = group[ix-1] if ix > 0 else None
        after = ([r for r in group if r["recovery_step"]>first["recovery_step"]]
                 if first else [])
        recovery_times.append(dict(zip(("dataset","split","seed","exposure_epochs","p"),k),
                 first_step_ge_90=first["recovery_step"] if first else None,
                 previous_observed_step=prev["recovery_step"] if prev else None,
                 first_observed_epoch=first["actual_recovery_epoch"] if first else None,
                 previous_observed_epoch=prev["actual_recovery_epoch"] if prev else None,
                 dropped_below_later=(any(r["q_recovery_percent"] is not None
                                          and r["q_recovery_percent"]<90 for r in after) if first else None),
                 status="crossed_90" if first else "not_crossed_or_undefined"))
    write_csv(root/"processed"/"recovery_times.csv",recovery_times,
              fields=["dataset","split","seed","exposure_epochs","p",
                      "first_step_ge_90","previous_observed_step",
                      "first_observed_epoch","previous_observed_epoch",
                      "dropped_below_later","status"])
    verification.update({"status":"complete" if complete else "partial",
                         "observed_validation":sum(r["split"]=="validation" for r in rows),
                         "observed_test":sum(r["split"]=="test" for r in rows)})
    if complete:
        expected_paths=len(protocol["datasets"])*len(protocol["seeds"])*len(protocol["exposure_epochs"])*len(protocol["p_values"])
        run_groups=defaultdict(list)
        for r in rows:run_groups[r["run_id"]].append(r)
        if len(run_groups)!=expected_paths:
            raise ValueError("Some configured dataset/seed/exposure/p paths are missing.")
        for run_id,group in run_groups.items():
            valid=[r for r in group if r["split"]=="validation"]
            test=[r for r in group if r["split"]=="test"]
            steps_per_epoch=math.ceil(valid[0]["num_recovery"]/protocol["batch_size"])
            n=protocol["recovery_epochs"]*steps_per_epoch
            required=scheduled_recovery_steps(steps_per_epoch,protocol["recovery_epochs"],
                                              protocol["late_eval_interval"])
            reported=[r["recovery_step"] for r in valid]
            if len(reported)!=len(required) or set(reported)!=set(required):
                raise ValueError(f"Missing/duplicate scheduled validation evaluations: {run_id}")
            for r in valid:
                step=r["recovery_step"]
                completed_epochs, epoch_step=divmod(step,steps_per_epoch)
                expected_seen=completed_epochs*r["num_recovery"]+min(epoch_step*protocol["batch_size"],r["num_recovery"])
                if r["recovery_samples_seen"]!=expected_seen:
                    raise ValueError(f"Recorded sample count inconsistent with step: {run_id}, step={step}")
                if not math.isclose(r["actual_recovery_epoch"],expected_seen/r["num_recovery"],abs_tol=1e-10):
                    raise ValueError(f"Recorded fractional recovery epoch incorrect: {run_id}, step={step}")
                if r["exposure_samples_seen"]!=r["num_exposure"]*r["exposure_epochs"]:
                    raise ValueError(f"Recorded exposure image count inconsistent: {run_id}")
            if len(test)!=2 or {r["recovery_step"] for r in test}!={0,n}:
                raise ValueError(f"Missing official test endpoints: {run_id}")
    write_json(root/"processed"/"verification.json",verification)
    plot_results(root,summary,protocol)
    lines=["# 이진분류 가변 간격 step 회복 실험 분석 안내","",
        f"- 상태: {'전체 완료 및 예측값 검산' if complete else '부분 결과만 존재'}",
        f"- 데이터셋: {protocol['datasets']} (숫자 3과 8만 사용)",
        f"- p: {protocol['p_values']} / 노출 epoch: {protocol['exposure_epochs']} / 회복: {protocol['recovery_epochs']} epoch",
        f"- 반복 시드: {protocol['seeds']}",
        f"- 측정 간격: 회복 1 epoch은 매 optimizer step, 이후 epoch은 {protocol['late_eval_interval']} step 간격 + 매 epoch 마지막 step",
        f"- 검증 관측: {verification['observed_validation']}행 / test 관측: {verification['observed_test']}행","",
        "`processed/key_results.csv`: 노출 직후와 최종 회복 시점 요약.",
        "`processed/derived_results.csv`: 데이터셋·시드·노출 길이·예약된 recovery step의 지표 및 실제 이미지 수.",
        "평가되지 않은 회복 step은 관측값으로 채우지 않음; recovery_times는 관측 간격으로 보고.",
        "`processed/per_digit_results.csv`: 숫자 3·8 각각의 두 색 모두 정답인 비율.",
        "`figures/*_expXX_recovery.png`: 노출 epoch별 step 단위 지표 그래프.","",
        "Q=같은 원본이 두 색 모두에서 정답인 비율; M=두 색 평균 정확도; F=색 변경 시 예측 flip 비율.",
        "shortcut_gap=aligned_accuracy-conflict_accuracy, S=M*(1-abs(gap))*(1-F).",
        "R_Q=100*(Q_t-Q_0)/(Q_ref-Q_0); reference=동일 데이터셋·seed·노출 길이·split의 p=.5 최종 회복 Q.",
        "기준 격차가 0.001 이하이면 R_Q 미정의; 음수/100 초과를 절단하지 않음.",
        "p=.5 대조군 자체에는 R_Q 없음. S는 회복 백분율 아님.",
        "표본 SD(ddof=1)는 2개 이상의 seed가 있을 때만 기록. Step들이 독립적인 반복 seed를 의미하지 않음.",
        "데이터셋 간 원본 수·업데이트 수가 다르므로 같은 step 횟수로 비교하는 경우 주의.",
        "3·8 이진 실험 결과를 10-class 또는 현실 사회적 편향 제거에 바로 일반화하지 않음.",""]
    write_text(root/"analysis_brief.md","\n".join(lines))
    return rows, derived

def protocol_for(args):
    code = {n: digest(HERE/n) for n in CODE_FILES}
    profile = "smoke" if args.smoke else "full"
    if not args.smoke and (args.datasets!=["mnist","emnist_digits"] or args.seeds!=[42,43,44]
                           or args.p_values!=[.5,.8,.9,.99] or args.exposure_epochs!=[4,8,16,24,32]
                           or args.recovery_epochs!=5 or args.late_eval_interval!=5):
        profile="custom"
    protocol = {"version":2,"profile":profile,"datasets":args.datasets,"source_scope":"digits_3_and_8",
        "digits":list(DIGITS),"colors":list(COLORS),"target_mapping":{"3":0,"8":1},
        "seeds":args.seeds,"p_values":args.p_values,"exposure_epochs":args.exposure_epochs,
        "recovery_epochs":args.recovery_epochs,"batch_size":args.batch_size,
        "late_eval_interval":args.late_eval_interval,
        "evaluation_schedule":"epoch 1: every step; epochs 2+: every N local steps and epoch end",
        "eval_batch_size":args.eval_batch_size,"checkpoint_every_steps":args.checkpoint_every_steps,
        "model":MODEL_NAME,"learning_rate":LR,"optimizer":"Adam","loss":"CrossEntropyLoss",
        "optimizer_reset_after_exposure":True,"split_ratios":[.4,.4,.2],"hue_jitter_degrees":5.,
        "hue_baseline_degrees":[108.,288.],"smoke_source_limit":1000 if args.smoke else None,
        "device":args.device,"num_threads":args.num_threads,
        "torch_version":torch.__version__,"torchvision_version":torchvision.__version__,
        "numpy_version":np.__version__,"python_version":platform.python_version(),
        "platform":platform.system(),"cuda_version":torch.version.cuda,
        "device_name":torch.cuda.get_device_name() if args.device=="cuda" else "cpu",
        "precision":"float32; TF32 disabled", "deterministic_algorithms":True,
        "Q":"fraction of original images correctly classified with BOTH colors",
        "flip_rate":"fraction of source images with different 3-vs-8 predictions between two colors",
        "recovery_score":"M*(1-abs(shortcut_gap))*(1-flip_rate)",
        "q_reference":"same seed, exposure length, split, neutral p=.5 at final recovery step",
        "q_recovery_percent":"100*(Q_t-Q_0)/(Q_ref-Q_0); undefined for <=0.001 initial gap",
        "code_sha256":code}
    protocol["protocol_hash"] = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    return protocol


def setup(root, protocol):
    root.mkdir(parents=True,exist_ok=True)
    file = root / "protocol.json"
    if file.exists():
        if read_json(file)!=protocol:
            raise ValueError("Saved protocol/code/environment differs. Restore original or use new --output.")
    else:
        write_json(file,protocol)
        for name in CODE_FILES:
            atomic(root / "provenance" / "code" / name,
                   lambda stream,n=name:stream.write((HERE/n).read_bytes()))
        try:
            provenance = {"commit":git(HERE,"rev-parse","HEAD"),
                          "worktree_changes":git(HERE,"status","--short","--",*CODE_FILES)}
        except Exception as exc:
            provenance={"git":"unavailable","exception":type(exc).__name__}
        write_json(root / "provenance" / "git.json", provenance)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke",action="store_true",help="small real-data trial, not for research claims")
    parser.add_argument("--datasets",nargs="+",choices=["mnist","emnist_digits"],
                        help="MNIST and/or EMNIST Digits (full default: both)")
    parser.add_argument("--seeds",type=int,nargs="+")
    parser.add_argument("--p-values",type=float,nargs="+")
    parser.add_argument("--exposure-epochs",type=int,nargs="+")
    parser.add_argument("--recovery-epochs",type=int)
    parser.add_argument("--batch-size",type=int,default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size",type=int,default=512)
    parser.add_argument("--checkpoint-every-steps",type=int,default=25)
    parser.add_argument("--late-eval-interval",type=int,default=5,
                        help="Evaluate every N optimizer updates in recovery epochs 2+; epoch-end always included")
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--num-threads",type=int,default=min(8,os.cpu_count() or 1))
    parser.add_argument("--data-root",type=Path,default=HERE/"data")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--analyze-only",action="store_true")
    parser.add_argument("--publish",action="store_true")
    parser.add_argument("--no-keep-awake",action="store_true")
    parser.add_argument("--stop-after-steps",type=int,help="per-path global-step count for an exact-resume test")
    args=parser.parse_args(argv)
    args.datasets=args.datasets if args.datasets is not None else (["mnist"] if args.smoke else ["mnist","emnist_digits"])
    args.seeds=sorted(args.seeds if args.seeds is not None else ([42] if args.smoke else [42,43,44]))
    args.p_values=sorted(args.p_values if args.p_values is not None else ([.5,.99] if args.smoke else [.5,.8,.9,.99]))
    args.exposure_epochs=sorted(args.exposure_epochs if args.exposure_epochs is not None else ([1] if args.smoke else [4,8,16,24,32]))
    if args.recovery_epochs is None: args.recovery_epochs=2 if args.smoke else 5
    if args.device=="auto": args.device="cuda" if torch.cuda.is_available() else "cpu"
    args.output=(args.output or HERE/"experiment_results"/(
        "smoke_binary_step_v2" if args.smoke else "binary_step_v2_full")).resolve()
    if not args.analyze_only:
        if len(set(args.datasets))!=len(args.datasets):
            parser.error("Datasets must be unique")
        if len(set(args.seeds))!=len(args.seeds) or any(s<0 or s>=2**32 for s in args.seeds):
            parser.error("seeds must be unique uint32 values")
        if len(set(args.exposure_epochs))!=len(args.exposure_epochs) or any(e<1 for e in args.exposure_epochs):
            parser.error("exposure-epochs must be unique positive integers")
        if (len(set(args.p_values))!=len(args.p_values) or .5 not in args.p_values or
            any(not math.isfinite(p) or p<.5 or p>1. or round(p,2)!=p for p in args.p_values)):
            parser.error("p-values must include .5, be unique, and be in [.5,1] at 0.01 resolution")
    if (args.recovery_epochs<1 or args.batch_size<1 or args.eval_batch_size<1 or
        args.checkpoint_every_steps<1 or args.num_threads<1):
        parser.error("Epochs, batch sizes, checkpoint interval, and threads must be positive")
    if args.stop_after_steps is not None and args.stop_after_steps<1:
        parser.error("stop-after-steps must be positive")
    if args.smoke and args.publish:
        parser.error("Do not publish smoke/test data; run full experiment for verified research outputs")
    return args


def source_preview(root, name, source, ids):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(8,4))
    for digit,ax_row in zip(DIGITS,axes):
        subset=ids[source["labels"][ids]==digit][:4]
        for a,i in zip(ax_row,subset):
            a.imshow(source["images"][int(i)].numpy(),cmap="gray",vmin=0,vmax=255)
            a.set_title(str(digit));a.axis("off")
    fig.suptitle(f"{name} digits 3/8, after canonical orientation conversion")
    fig.tight_layout()
    f=root/"figures"/f"{name}_source_preview.png"
    f.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(f,dpi=140);plt.close(fig)


def main(argv=None):
    args=parse_args(argv)
    root=args.output
    root.mkdir(parents=True,exist_ok=True)
    try:
        with FileLock(str(root/".experiment.lock"),timeout=0),keep_awake(not args.no_keep_awake):
            if args.analyze_only:
                protocol=read_json(root/"protocol.json")
                done=(root/"status.json").exists() and read_json(root/"status.json").get("status")=="complete"
                if done:verify_manifest(root)
                build_report(root,protocol,complete=done)
                if done:write_manifest(root)
                if args.publish:print("Published:",publish(root))
                return 0
            configure(args.device,args.num_threads)
            protocol=protocol_for(args)
            setup(root,protocol)
            write_json(root/"status.json",{"status":"running","started_or_resumed_at":now(),
                                           "profile":protocol["profile"]})
            print(f"profile={protocol['profile']}, device={args.device}, "
                  f"paths={len(args.datasets)*len(args.seeds)*len(args.p_values)*len(args.exposure_epochs)} "
                  f"output={root}",flush=True)
            paths=[]
            try:
                sources={}; designs={}
                for base in args.datasets:
                    name=base+"_38"
                    source=prepare_source(root,base,load_source(base,args.data_root,not args.offline),
                                          1000 if args.smoke else None)
                    ids,test_ids=binary_ids(source)
                    if len(ids)<20 or len(test_ids)==0:
                        raise ValueError(f"Insufficient 3/8 digits in {base} source/test")
                    source_preview(root,name,source,ids)
                    write_json(root/"raw_sources"/name/"selection.json",{
                        "selected_train_digit38":len(ids),"selected_test_digit38":len(test_ids),
                        "source_name":base,"official_split_separation":True,
                        "train_source_ids":"stored in shared/*/design.npz",
                        "original_full_source_archives":f"raw_sources/{base}/"})
                    sources[name]=source
                    for seed in args.seeds:
                        designs[name,seed]=make_shared_design(root,source,ids,test_ids,seed,args.p_values,
                                                               args.exposure_epochs,args.recovery_epochs,name=name)
                for seed in args.seeds:
                    for name in [d+"_38" for d in args.datasets]:
                        for exp in args.exposure_epochs:
                            for p in args.p_values:
                                path=run_path(root,protocol,seed,p,exp,sources[name],designs[name,seed],args,name=name)
                                paths.append((name,seed,exp,p,path))
                                write_csv(root/"raw"/"observations.csv",all_observations(root))
                                write_json(root/"progress.json",{
                                    "completed_training_paths":len(paths),
                                    "expected_paths":len(args.datasets)*len(args.seeds)*len(args.exposure_epochs)*len(args.p_values),
                                    "last_run":f"{name}/seed{seed}/exp{exp}/{ptag(p)}","updated_at":now()})
                # Test is intentionally evaluated ONLY after all selected training paths finish.
                for name,seed,exp,p,path in paths:
                    evaluate_test_endpoints(root,protocol,path,sources[name],designs[name,seed],args)
                rows,_=build_report(root,protocol,complete=True)
                write_manifest(root)
                write_json(root/"status.json",{"status":"complete","finished_at":now(),
                    "training_paths":len(paths),"validation_observations":sum(r["split"]=="validation" for r in rows),
                    "test_observations":sum(r["split"]=="test" for r in rows),
                    "protocol_hash":protocol["protocol_hash"]})
            except BaseException as exc:
                write_json(root/"status.json",{"status": "interrupted" if isinstance(exc,(KeyboardInterrupt,PlannedStop)) else "failed",
                    "error":str(exc),"updated_at":now(),"profile":protocol["profile"]})
                if isinstance(exc,PlannedStop):
                    print(str(exc),flush=True)
                    return 130
                raise
            print(f"Complete. Open {root/'analysis_brief.md'}",flush=True)
            if args.publish:print("Published:",publish(root))
            return 0
    except Timeout:
        print("Another process is already using this output directory.",file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Rerun same command to resume from last checkpoint.",file=sys.stderr)
        raise SystemExit(130)
