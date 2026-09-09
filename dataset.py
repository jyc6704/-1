"""숫자 3과 8만 사용하는 재현 가능한 Spurious Colored MNIST.

숫자 모양이 실제 예측 대상이고 색상은 의도적으로 강한 상관관계를 갖게 만든
허위 특징(spurious feature)이다. 이미지 파일을 저장하지 않고 __getitem__에서
RGB 이미지를 실시간 생성한다.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

TARGET_DIGITS = (3, 8)
COLOR_IDS = (3, 8)
LABEL_TO_COLOR = {3: 3, 8: 8}
OTHER_COLOR = {3: 8, 8: 3}
HUE_STEP = 36.0
HUE_JITTER = 5.0


class Digit38MNIST(Dataset):
    """공식 MNIST에서 숫자 3과 8만 남긴 데이터셋."""

    def __init__(self, mnist: MNIST):
        self.mnist = mnist
        mask = (mnist.targets == 3) | (mnist.targets == 8)
        self.original_indices = torch.where(mask)[0]
        self.targets = mnist.targets[self.original_indices].long()

    def __len__(self) -> int:
        return len(self.original_indices)

    def __getitem__(self, idx: int):
        original_idx = int(self.original_indices[idx])
        image, label = self.mnist[original_idx]
        return image, int(label), original_idx


def load_digit38_mnist(
    root: str = "./data", download: bool = False
) -> tuple[Digit38MNIST, Digit38MNIST]:
    """공식 train/test를 각각 불러온 뒤 3과 8만 필터링한다."""
    transform = ToTensor()
    train = MNIST(root=root, train=True, download=download, transform=transform)
    test = MNIST(root=root, train=False, download=download, transform=transform)
    return Digit38MNIST(train), Digit38MNIST(test)


def split_train_validation(
    dataset: Digit38MNIST, seed: int, val_ratio: float = 0.2
) -> tuple[Subset, Subset]:
    """3·8 공식 train을 재현 가능한 80% train / 20% validation으로 분할한다."""
    val_size = round(len(dataset) * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    return random_split(
        dataset, [len(dataset) - val_size, val_size], generator=generator
    )


def subset_labels(subset: Subset) -> torch.Tensor:
    indices = torch.as_tensor(subset.indices, dtype=torch.long)
    return subset.dataset.targets[indices].long()


def make_hue_offsets(
    num_official_images: int, seed: int, jitter: float = HUE_JITTER
) -> torch.Tensor:
    """공식 MNIST 원본 index별 δ~Uniform(-5°, +5°)를 한 번만 생성한다."""
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(num_official_images, generator=generator) * (2.0 * jitter) - jitter
    )


def make_biased_color_ids(
    labels: torch.Tensor, p: float = 0.99, seed: int = 42
) -> torch.Tensor:
    """P(color=3|digit=3)=P(color=8|digit=8)=p가 되도록 고정 배정한다.

    conflict 표본은 반대 색을 받는다: 3→color 8, 8→color 3.
    """
    if not 0.5 <= p <= 1.0:
        raise ValueError("2색 biased 조건의 p는 0.5 이상 1.0 이하여야 합니다.")
    if not torch.all((labels == 3) | (labels == 8)):
        raise ValueError("labels에는 숫자 3과 8만 있어야 합니다.")
    generator = torch.Generator().manual_seed(seed)
    color_ids = torch.empty_like(labels, dtype=torch.long)
    for digit in TARGET_DIGITS:
        indices = torch.where(labels == digit)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        num_aligned = round(len(indices) * p)
        color_ids[indices[:num_aligned]] = LABEL_TO_COLOR[digit]
        color_ids[indices[num_aligned:]] = OTHER_COLOR[digit]
    return color_ids


def make_neutral_color_ids(labels: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """각 숫자 안에서 color 3과 color 8을 50:50에 가깝게 고정 배정한다."""
    if not torch.all((labels == 3) | (labels == 8)):
        raise ValueError("labels에는 숫자 3과 8만 있어야 합니다.")
    generator = torch.Generator().manual_seed(seed)
    color_ids = torch.empty_like(labels, dtype=torch.long)
    colors = torch.tensor(COLOR_IDS, dtype=torch.long)
    for digit in TARGET_DIGITS:
        indices = torch.where(labels == digit)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        assigned = colors[torch.arange(len(indices)) % 2]
        assigned = assigned[torch.randperm(len(assigned), generator=generator)]
        color_ids[indices] = assigned
    return color_ids


def hue_to_rgb(hue_degrees: float) -> torch.Tensor:
    rgb = colorsys.hsv_to_rgb((hue_degrees % 360.0) / 360.0, 1.0, 1.0)
    return torch.tensor(rgb, dtype=torch.float32)


def colorize(
    image: torch.Tensor, color_id: int, hue_offset: float
) -> tuple[torch.Tensor, float]:
    """MNIST 획의 밝기(Value)는 유지하고 Hue만 부여한다."""
    if color_id not in COLOR_IDS:
        raise ValueError("color_id는 3 또는 8이어야 합니다.")
    hue = (HUE_STEP * color_id + hue_offset) % 360.0
    return image * hue_to_rgb(hue).view(3, 1, 1), hue


class SpuriousColoredMNIST(Dataset):
    """색 배정과 Hue offset이 epoch마다 바뀌지 않는 3·8 RGB 데이터셋."""

    def __init__(
        self, subset: Subset, color_ids: torch.Tensor, hue_offsets: torch.Tensor
    ):
        if len(subset) != len(color_ids):
            raise ValueError("subset과 color_ids 길이가 다릅니다.")
        self.subset = subset
        self.color_ids = color_ids.long()
        self.hue_offsets = hue_offsets.float()

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, digit, original_idx = self.subset[idx]
        color_id = int(self.color_ids[idx])
        colored, hue = colorize(image, color_id, float(self.hue_offsets[original_idx]))
        target = 0 if digit == 3 else 1  # 향후 2-class 학습용 target
        return colored, target, digit, color_id, hue


def label_color_distribution(
    labels: torch.Tensor, color_ids: torch.Tensor
) -> torch.Tensor:
    """행=(digit 3,8), 열=(color 3,8)인 2×2 조건부 비율 행렬."""
    result = torch.zeros((2, 2), dtype=torch.float32)
    for row, digit in enumerate(TARGET_DIGITS):
        class_colors = color_ids[labels == digit]
        for column, color_id in enumerate(COLOR_IDS):
            result[row, column] = (class_colors == color_id).float().mean()
    return result


def spurious_correlation(labels: torch.Tensor, color_ids: torch.Tensor) -> float:
    """digit(3/8)와 color(3/8)의 이진 Pearson/phi 상관계수."""
    digit_binary = (labels == 8).float()
    color_binary = (color_ids == 8).float()
    digit_centered = digit_binary - digit_binary.mean()
    color_centered = color_binary - color_binary.mean()
    denominator = torch.sqrt(
        digit_centered.square().sum() * color_centered.square().sum()
    )
    return float((digit_centered * color_centered).sum() / denominator)


def visualize_random_samples(
    dataset: SpuriousColoredMNIST, output_path=None, seed=42, show=False
):
    """랜덤 표본 20개의 digit, target, color_id, 실제 Hue를 표시한다."""
    import matplotlib.pyplot as plt

    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:20]
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for ax, idx in zip(axes.flat, indices.tolist()):
        image, target, digit, color_id, hue = dataset[idx]
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(
            f"digit={digit}, target={target}\ncolor={color_id}, H={hue:.1f}°",
            fontsize=8,
        )
        ax.axis("off")
    fig.tight_layout()
    _finish_figure(fig, output_path, show)


def visualize_label_color_heatmap(
    labels, color_ids, output_path=None, show=False, title="Label-color distribution"
):
    """3·8 숫자와 color 3·8의 2×2 조건부 분포를 표시한다."""
    import matplotlib.pyplot as plt

    matrix = label_color_distribution(labels, color_ids)
    fig, ax = plt.subplots(figsize=(5, 4))
    plot = ax.imshow(matrix.numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(2), ["color 3", "color 8"])
    ax.set_yticks(range(2), ["digit 3", "digit 8"])
    ax.set_title(title)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(plot, ax=ax, label="P(color | digit)")
    fig.tight_layout()
    _finish_figure(fig, output_path, show)


def visualize_same_image_both_colors(
    subset: Subset,
    hue_offsets: torch.Tensor,
    subset_index=0,
    output_path=None,
    show=False,
):
    """동일한 숫자 모양을 color 3과 color 8로 바꾸어 비교한다."""
    import matplotlib.pyplot as plt

    image, digit, original_idx = subset[subset_index]
    offset = float(hue_offsets[original_idx])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, color_id in zip(axes, COLOR_IDS):
        colored, hue = colorize(image, color_id, offset)
        ax.imshow(colored.permute(1, 2, 0))
        ax.set_title(f"digit={digit}, color={color_id}\nH={hue:.1f}°")
        ax.axis("off")
    fig.tight_layout()
    _finish_figure(fig, output_path, show)


def _finish_figure(fig, output_path, show: bool) -> None:
    import matplotlib.pyplot as plt

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def verify_dataset(seed: int = 42, p: float = 0.99, show_plots: bool = True) -> dict:
    """3·8 데이터 생성 조건을 자동 검사하고 요약 결과를 반환한다.

    이 함수는 모델을 생성하거나 학습하지 않는다. 데이터가 연구 설계대로
    만들어졌는지만 통계, tensor shape, 재현성 및 시각화로 확인한다.
    """
    output = Path("validation_outputs_38")
    digit_train, digit_test = load_digit38_mnist(download=False)
    train_subset, validation_subset = split_train_validation(digit_train, seed)
    labels = subset_labels(train_subset)

    # 공식 원본 index마다 offset 하나를 생성하여 모든 조건에서 공유한다.
    hue_offsets = make_hue_offsets(len(digit_train.mnist), seed)
    biased_ids = make_biased_color_ids(labels, p=p, seed=seed)
    neutral_ids = make_neutral_color_ids(labels, seed=seed + 10_000)
    biased_dataset = SpuriousColoredMNIST(train_subset, biased_ids, hue_offsets)

    biased_matrix = label_color_distribution(labels, biased_ids)
    neutral_matrix = label_color_distribution(labels, neutral_ids)
    actual_p = float((biased_ids == labels).float().mean())
    biased_phi = spurious_correlation(labels, biased_ids)
    neutral_phi = spurious_correlation(labels, neutral_ids)

    first_sample = biased_dataset[0]
    repeated_sample = biased_dataset[0]
    image, target, digit, color_id, hue = first_sample

    checks = {
        "only_digits_3_and_8": set(labels.tolist()) == {3, 8},
        "rgb_shape_3x28x28": tuple(image.shape) == (3, 28, 28),
        "valid_binary_target": target in (0, 1),
        "valid_color_id": color_id in COLOR_IDS,
        "hue_offset_within_5_degrees": bool(
            (hue_offsets.min() >= -HUE_JITTER) and (hue_offsets.max() <= HUE_JITTER)
        ),
        "biased_p_matches_setting": abs(actual_p - p) < 0.001,
        "neutral_is_half_half": bool(torch.all((neutral_matrix - 0.5).abs() < 0.001)),
        # 같은 index를 다시 요청해도 이미지·색·Hue가 동일해야 한다.
        "assignment_fixed_between_calls": (
            torch.equal(first_sample[0], repeated_sample[0])
            and first_sample[2:] == repeated_sample[2:]
        ),
        # 같은 seed로 다시 생성했을 때 완전히 같은 배정이어야 한다.
        "seed_reproducible": (
            torch.equal(biased_ids, make_biased_color_ids(labels, p=p, seed=seed))
            and torch.equal(hue_offsets, make_hue_offsets(len(digit_train.mnist), seed))
        ),
    }

    print("\n=== 3·8 Spurious Colored MNIST 생성 검증 ===")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print("\n--- 데이터 통계 ---")
    print(f"공식 train 중 3·8: {len(digit_train):,}개")
    print(f"train / validation: {len(train_subset):,} / {len(validation_subset):,}개")
    print(f"공식 test 중 3·8: {len(digit_test):,}개 (학습에 사용하지 않음)")
    print(
        f"train digit 3 / 8: {(labels == 3).sum().item():,} / {(labels == 8).sum().item():,}개"
    )
    print(f"설정 p / 실제 p: {p:.6f} / {actual_p:.6f}")
    print(f"biased phi correlation: {biased_phi:.6f}")
    print(f"neutral phi correlation: {neutral_phi:.6f}")
    print(f"biased distribution:\n{biased_matrix}")
    print(f"neutral distribution:\n{neutral_matrix}")
    print(
        f"sample: shape={tuple(image.shape)}, target={target}, digit={digit}, "
        f"color={color_id}, hue={hue:.3f}°"
    )
    print(f"Hue offset range: {hue_offsets.min():.6f}° .. {hue_offsets.max():.6f}°")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("데이터셋 검증 실패: " + ", ".join(failed))
    print("\n모든 데이터셋 검증을 통과했습니다.")

    visualize_random_samples(
        biased_dataset, output / "biased_samples_38.png", seed=seed, show=show_plots
    )
    visualize_label_color_heatmap(
        labels,
        biased_ids,
        output / "biased_heatmap_38.png",
        show=show_plots,
        title=f"Biased digit-color distribution (p={p})",
    )
    visualize_label_color_heatmap(
        labels,
        neutral_ids,
        output / "neutral_heatmap_38.png",
        show=show_plots,
        title="Neutral digit-color distribution (p=0.5)",
    )
    visualize_same_image_both_colors(
        train_subset,
        hue_offsets,
        output_path=output / "same_image_both_colors.png",
        show=show_plots,
    )
    print(f"시각화 저장 위치: {output.resolve()}")

    return {
        "checks": checks,
        "actual_p": actual_p,
        "biased_phi": biased_phi,
        "neutral_phi": neutral_phi,
        "biased_distribution": biased_matrix,
        "neutral_distribution": neutral_matrix,
    }


if __name__ == "__main__":
    # dataset.py를 직접 실행할 때만 검증과 화면 시각화를 수행한다.
    # 다른 코드에서 import할 때는 자동 실행되지 않는다.
    verify_dataset(show_plots=True)
