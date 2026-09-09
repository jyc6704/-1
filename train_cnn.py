"""3·8 Spurious Colored MNIST만 이용해 CNN을 학습하고 shortcut을 평가한다."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    SpuriousColoredMNIST,
    load_digit38_mnist,
    make_biased_color_ids,
    make_hue_offsets,
    split_train_validation,
    subset_labels,
)
from model import BinarySmallCNN


def set_seed(seed: int) -> None:
    """모델 초기화와 DataLoader 순서를 재현 가능하게 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_evaluation_color_ids(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """같은 validation 원본에 적용할 aligned 색과 conflict 색을 만든다."""
    aligned = labels.clone()
    conflict = torch.where(labels == 3, torch.tensor(8), torch.tensor(3))
    return aligned, conflict


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    """biased train 데이터로 정확히 한 epoch 학습한다."""
    model.train()
    loss_sum = 0.0
    correct = total = 0
    for images, targets, *_ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size
    return loss_sum / total, correct / total


@torch.inference_mode()
def predict(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """validation 예측과 정답 target을 원래 순서대로 반환한다."""
    model.eval()
    predictions, targets_all = [], []
    for images, targets, *_ in loader:
        predictions.append(model(images.to(device)).argmax(dim=1).cpu())
        targets_all.append(targets)
    return torch.cat(predictions), torch.cat(targets_all)


def evaluate_shortcut(model, aligned_loader, conflict_loader, device) -> dict[str, float]:
    """모양은 같고 색만 다른 두 validation 조건으로 shortcut 의존성을 계산한다."""
    aligned_predictions, targets = predict(model, aligned_loader, device)
    conflict_predictions, conflict_targets = predict(model, conflict_loader, device)
    if not torch.equal(targets, conflict_targets):
        raise RuntimeError("Aligned와 conflict validation의 원본 순서가 다릅니다.")

    aligned_accuracy = float((aligned_predictions == targets).float().mean())
    conflict_accuracy = float((conflict_predictions == targets).float().mean())
    # 2색 환경에서는 색을 반대로 바꿨을 때 예측 class가 변한 이미지의 비율이다.
    flip_rate = float((aligned_predictions != conflict_predictions).float().mean())
    return {
        "aligned_accuracy": aligned_accuracy,
        "conflict_accuracy": conflict_accuracy,
        "neutral_accuracy": (aligned_accuracy + conflict_accuracy) / 2.0,
        "shortcut_gap": aligned_accuracy - conflict_accuracy,
        "flip_rate": flip_rate,
    }


def unique_output_directory(root: Path, name: str) -> Path:
    """기존 실험 결과를 덮어쓰지 않는 새 폴더를 만든다."""
    candidate = root / name
    suffix = 2
    while candidate.exists():
        candidate = root / f"{name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="숫자 3·8 spurious correlation CNN 학습")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)  # Windows 안전 기본값
    parser.add_argument("--output", type=Path, default=Path("cnn_results_38"))
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_train, _ = load_digit38_mnist(download=False)
    train_subset, validation_subset = split_train_validation(digit_train, args.seed)
    train_labels = subset_labels(train_subset)
    validation_labels = subset_labels(validation_subset)

    # train/validation 모두 같은 공식 원본 index 기반 Hue offset을 사용한다.
    hue_offsets = make_hue_offsets(len(digit_train.mnist), args.seed)
    biased_train_ids = make_biased_color_ids(train_labels, args.p, args.seed)
    aligned_ids, conflict_ids = make_evaluation_color_ids(validation_labels)
    train_dataset = SpuriousColoredMNIST(train_subset, biased_train_ids, hue_offsets)
    aligned_dataset = SpuriousColoredMNIST(validation_subset, aligned_ids, hue_offsets)
    conflict_dataset = SpuriousColoredMNIST(validation_subset, conflict_ids, hue_offsets)

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=shuffle_generator, **loader_options
    )
    aligned_loader = DataLoader(aligned_dataset, shuffle=False, **loader_options)
    conflict_loader = DataLoader(conflict_dataset, shuffle=False, **loader_options)

    model = BinarySmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    run_directory = unique_output_directory(
        args.output, f"BinarySmallCNN_p{int(round(args.p * 100)):03d}_seed{args.seed}"
    )

    rows = []
    print(f"device={device}, train={len(train_dataset)}, validation={len(validation_subset)}")
    print(f"biased actual p={(biased_train_ids == train_labels).float().mean().item():.6f}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        shortcut_metrics = evaluate_shortcut(
            model, aligned_loader, conflict_loader, device
        )
        row = {
            "epoch": epoch,
            "seed": args.seed,
            "p": args.p,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            **shortcut_metrics,
        }
        rows.append(row)
        print(
            f"epoch {epoch:02d} | train={train_accuracy:.4f} | "
            f"aligned={row['aligned_accuracy']:.4f} | "
            f"conflict={row['conflict_accuracy']:.4f} | "
            f"gap={row['shortcut_gap']:.4f} | flip={row['flip_rate']:.4f}"
        )

    with (run_directory / "results.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (run_directory / "config.json").write_text(
        json.dumps({**vars(args), "model_class": "BinarySmallCNN"},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "BinarySmallCNN",
            "config": {**vars(args), "output": str(args.output)},
            "last_metrics": rows[-1],
        },
        run_directory / "model_last.pt",
    )
    print(f"saved={run_directory.resolve()}")


if __name__ == "__main__":
    main()
