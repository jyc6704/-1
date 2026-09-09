# 기존 0~9 pilot 전체를 참고용 주석으로 보존한다.
# 현재 실행되는 3·8 binary pilot은 이 파일 하단에 있다.
# """p=0.99, exposure 1 epoch만 실행하는 안전한 pilot."""
#
# from __future__ import annotations
#
# import argparse
# import csv
# import json
# import random
# from pathlib import Path
#
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
#
# from dataset import (
#     AllColorsValidation,
#     ColoredMNIST,
#     load_mnist,
#     make_biased_color_ids,
#     make_hue_offsets,
#     make_neutral_color_ids,
#     split_train_validation,
#     subset_labels,
# )
# from model import SmallCNN
# from train import evaluate_all_colors, train_one_epoch
#
#
# def seed_everything(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.use_deterministic_algorithms(True)
#
#
# def unique_run_dir(root: Path, name: str) -> Path:
#     """기존 결과를 덮어쓰지 않고 _02, _03 ... 접미사를 붙인다."""
#     candidate = root / name
#     counter = 2
#     while candidate.exists():
#         candidate = root / f"{name}_{counter:02d}"
#         counter += 1
#     candidate.mkdir(parents=True)
#     return candidate
#
#
# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--p", type=float, default=0.99)
#     parser.add_argument("--batch-size", type=int, default=64)
#     parser.add_argument("--val-size", type=int, default=6000)
#     parser.add_argument("--num-workers", type=int, default=0)  # Windows 안전 기본값
#     parser.add_argument("--output", type=Path, default=Path("pilot_results"))
#     args = parser.parse_args()
#
#     seed_everything(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     official_train, _ = load_mnist(download=False)
#     train_subset, val_subset = split_train_validation(official_train, args.seed, args.val_size)
#     train_labels = subset_labels(train_subset)
#
#     # 전체 60,000 원본 index 기준 offset 하나를 biased/recovery/validation이 공유한다.
#     hue_offsets = make_hue_offsets(len(official_train), args.seed)
#     biased_ids = make_biased_color_ids(train_labels, args.p, args.seed)
#     neutral_ids = make_neutral_color_ids(train_labels, args.seed + 10_000)
#     biased_data = ColoredMNIST(train_subset, biased_ids, hue_offsets)
#     validation = AllColorsValidation(val_subset, hue_offsets)
#
#     shuffle_generator = torch.Generator().manual_seed(args.seed)
#     train_loader = DataLoader(
#         biased_data, batch_size=args.batch_size, shuffle=True,
#         num_workers=args.num_workers, generator=shuffle_generator,
#     )
#     val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#
#     model = SmallCNN().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
#     metrics = evaluate_all_colors(model, val_loader, len(val_subset), device)
#
#     run_dir = unique_run_dir(args.output, f"p{int(round(args.p * 100)):03d}_seed{args.seed}_exp01")
#     result = {
#         "seed": args.seed, "p": args.p, "exposure_epoch": 1,
#         "train_loss": train_loss, "train_accuracy": train_accuracy,
#         **metrics,
#     }
#     torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "metrics": result}, run_dir / "exposure_01.pt")
#     torch.save({"hue_offsets": hue_offsets, "biased_color_ids": biased_ids, "neutral_color_ids": neutral_ids}, run_dir / "assignments.pt")
#     (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
#     # 실행별 고유 폴더 안에 CSV를 두므로 이전 결과를 덮어쓰지 않는다.
#     with (run_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
#         writer = csv.DictWriter(csv_file, fieldnames=result.keys())
#         writer.writeheader()
#         writer.writerow(result)
#
#     print(f"device={device}, train={len(train_subset)}, validation={len(val_subset)}")
#     print(f"biased actual p={(biased_ids == train_labels).float().mean().item():.6f}")
#     matrix = torch.stack([(neutral_ids[train_labels == y].bincount(minlength=10).float() / (train_labels == y).sum()) for y in range(10)])
#     print(f"neutral cell range={matrix.min().item():.6f}..{matrix.max().item():.6f}")
#     print(f"sample shape={tuple(biased_data[0][0].shape)}, hue offset range={hue_offsets.min():.3f}..{hue_offsets.max():.3f}")
#     print(json.dumps(result, ensure_ascii=False, indent=2))
#     print(f"saved={run_dir}")
#
#
# if __name__ == "__main__":
#     main()
#

"""숫자 3·8 BinarySmallCNN: biased 데이터로 정확히 1 epoch 학습하는 pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    SpuriousColoredMNIST,
    load_digit38_mnist,
    make_biased_color_ids,
    make_hue_offsets,
    make_neutral_color_ids,
    split_train_validation,
    subset_labels,
)
from model import BinarySmallCNN
from train import evaluate_binary, train_one_epoch


def seed_everything(seed: int) -> None:
    """모델 초기화와 학습 순서를 고정한다. 같은 환경에서의 재현을 목표로 한다."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def unique_run_dir(root: Path, name: str) -> Path:
    """기존 결과 폴더가 있으면 새 번호를 붙여 덮어쓰기를 방지한다."""
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = root / (name if index == 1 else f"{name}_{index:02d}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="3·8 binary CNN 1 epoch pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    # 현재 3·8 데이터셋의 80:20 분할을 유지한다. 10-class의 val-size=6000과 구분.
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--download", action="store_true", help="원본 MNIST가 없을 때 다운로드")
    parser.add_argument("--output", type=Path, default=Path("pilot_results_38"))
    args = parser.parse_args()
    if not 0.5 <= args.p <= 1.0:
        parser.error("p는 0.5 이상 1 이하이어야 합니다.")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size는 양수, num-workers는 0 이상이어야 합니다.")
    if not 0 < args.val_ratio < 1:
        parser.error("val-ratio는 0보다 크고 1보다 작아야 합니다.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    digit_train, _ = load_digit38_mnist(root=str(args.data_root), download=args.download)
    train_subset, val_subset = split_train_validation(digit_train, args.seed, args.val_ratio)
    train_labels = subset_labels(train_subset)
    val_labels = subset_labels(val_subset)
    if set(train_labels.tolist()) != {3, 8} or set(val_labels.tolist()) != {3, 8}:
        parser.error("train과 validation 모두 숫자 3, 8을 포함하도록 val-ratio를 조정하세요.")

    # 원본 60,000개 index를 기준으로 δ를 만들어 필터링/분할 후에도 동일하게 참조한다.
    hue_offsets = make_hue_offsets(len(digit_train.mnist), args.seed)
    biased_ids = make_biased_color_ids(train_labels, args.p, args.seed)
    neutral_ids = make_neutral_color_ids(train_labels, args.seed + 10_000)
    train_data = SpuriousColoredMNIST(train_subset, biased_ids, hue_offsets)
    # 두 평가 조건에서 같은 원본, 같은 δ, 같은 순서를 사용하고 색만 교환한다.
    aligned_ids = val_labels.clone()
    conflict_ids = torch.where(val_labels == 3, 8, 3)
    aligned_data = SpuriousColoredMNIST(val_subset, aligned_ids, hue_offsets)
    conflict_data = SpuriousColoredMNIST(val_subset, conflict_ids, hue_offsets)
    options = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
    aligned_loader = DataLoader(aligned_data, shuffle=False, **options)
    conflict_loader = DataLoader(conflict_data, shuffle=False, **options)

    model = BinarySmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    config = {
        **vars(args), "data_root": str(args.data_root), "output": str(args.output),
        "model_class": "BinarySmallCNN", "digits": [3, 8],
        "target_mapping": {"3": 0, "8": 1}, "color_ids": [3, 8],
        "hue_jitter_degrees": 5, "learning_rate": 0.001,
        "optimizer": "Adam", "loss": "CrossEntropyLoss", "exposure_epochs": 1,
        "device": str(device), "torch_version": str(torch.__version__),
        "flip_rate_definition": "fraction of images whose predictions change between two colors",
    }
    run_dir = unique_run_dir(args.output, f"binary_p{args.p:g}_seed{args.seed}_exp01")
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"BinarySmallCNN | train={len(train_subset)}, validation={len(val_subset)}, device={device}",
          flush=True)
    # 반복문 없이 정확히 1 epoch만 학습한다. neutral은 배정 저장만 하며 회복 학습은 하지 않는다.
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
    result = {
        "seed": args.seed, "p": args.p, "exposure_epoch": 1,
        "train_loss": train_loss, "train_accuracy": train_accuracy,
        **evaluate_binary(model, aligned_loader, conflict_loader, device),
    }
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config, "metrics": result,
    }, run_dir / "exposure_01.pt")
    # 배정 tensor는 subset 순서 기준이므로 실제 원본 index도 함께 저장한다.
    torch.save({
        "train_original_indices": digit_train.original_indices[train_subset.indices],
        "validation_original_indices": digit_train.original_indices[val_subset.indices],
        "hue_offsets": hue_offsets, "biased_color_ids": biased_ids,
        "neutral_color_ids": neutral_ids,
        "validation_aligned_color_ids": aligned_ids,
        "validation_conflict_color_ids": conflict_ids,
    }, run_dir / "assignments.pt")
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(f"actual p={(biased_ids == train_labels).float().mean().item():.6f}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={run_dir.resolve()}")


if __name__ == "__main__":
    main()
