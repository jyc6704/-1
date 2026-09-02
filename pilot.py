"""p=0.99, exposure 1 epoch만 실행하는 안전한 pilot."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    AllColorsValidation,
    ColoredMNIST,
    load_mnist,
    make_biased_color_ids,
    make_hue_offsets,
    make_neutral_color_ids,
    split_train_validation,
    subset_labels,
)
from model import SmallCNN
from train import evaluate_all_colors, train_one_epoch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def unique_run_dir(root: Path, name: str) -> Path:
    """기존 결과를 덮어쓰지 않고 _02, _03 ... 접미사를 붙인다."""
    candidate = root / name
    counter = 2
    while candidate.exists():
        candidate = root / f"{name}_{counter:02d}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-size", type=int, default=6000)
    parser.add_argument("--num-workers", type=int, default=0)  # Windows 안전 기본값
    parser.add_argument("--output", type=Path, default=Path("pilot_results"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    official_train, _ = load_mnist(download=False)
    train_subset, val_subset = split_train_validation(official_train, args.seed, args.val_size)
    train_labels = subset_labels(train_subset)

    # 전체 60,000 원본 index 기준 offset 하나를 biased/recovery/validation이 공유한다.
    hue_offsets = make_hue_offsets(len(official_train), args.seed)
    biased_ids = make_biased_color_ids(train_labels, args.p, args.seed)
    neutral_ids = make_neutral_color_ids(train_labels, args.seed + 10_000)
    biased_data = ColoredMNIST(train_subset, biased_ids, hue_offsets)
    validation = AllColorsValidation(val_subset, hue_offsets)

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        biased_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, generator=shuffle_generator,
    )
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
    metrics = evaluate_all_colors(model, val_loader, len(val_subset), device)

    run_dir = unique_run_dir(args.output, f"p{int(round(args.p * 100)):03d}_seed{args.seed}_exp01")
    result = {
        "seed": args.seed, "p": args.p, "exposure_epoch": 1,
        "train_loss": train_loss, "train_accuracy": train_accuracy,
        **metrics,
    }
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "metrics": result}, run_dir / "exposure_01.pt")
    torch.save({"hue_offsets": hue_offsets, "biased_color_ids": biased_ids, "neutral_color_ids": neutral_ids}, run_dir / "assignments.pt")
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 실행별 고유 폴더 안에 CSV를 두므로 이전 결과를 덮어쓰지 않는다.
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(f"device={device}, train={len(train_subset)}, validation={len(val_subset)}")
    print(f"biased actual p={(biased_ids == train_labels).float().mean().item():.6f}")
    matrix = torch.stack([(neutral_ids[train_labels == y].bincount(minlength=10).float() / (train_labels == y).sum()) for y in range(10)])
    print(f"neutral cell range={matrix.min().item():.6f}..{matrix.max().item():.6f}")
    print(f"sample shape={tuple(biased_data[0][0].shape)}, hue offset range={hue_offsets.min():.3f}..{hue_offsets.max():.3f}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
