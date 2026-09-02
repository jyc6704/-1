"""재현 가능한 10색 Colored MNIST 데이터셋과 배정 함수."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


NUM_COLORS = 10
HUE_STEP = 36.0
HUE_JITTER = 5.0


def load_mnist(root: str = "./data", download: bool = False) -> tuple[MNIST, MNIST]:
    """공식 train/test를 불러온다. test는 학습에 절대 사용하지 않는다."""
    transform = ToTensor()
    train = MNIST(root=root, train=True, download=download, transform=transform)
    test = MNIST(root=root, train=False, download=download, transform=transform)
    return train, test


def split_train_validation(dataset: MNIST, seed: int, val_size: int = 6_000) -> tuple[Subset, Subset]:
    """모든 조건에서 재사용할 고정 54,000/6,000 분할."""
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [len(dataset) - val_size, val_size], generator=generator)


def subset_labels(subset: Subset) -> torch.Tensor:
    indices = torch.as_tensor(subset.indices, dtype=torch.long)
    return subset.dataset.targets[indices].long()


def make_hue_offsets(num_images: int, seed: int, jitter: float = HUE_JITTER) -> torch.Tensor:
    """원본 이미지별 δ를 한 번만 생성한다: Uniform(-5°, +5°)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(num_images, generator=generator) * (2.0 * jitter) - jitter


def make_biased_color_ids(labels: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    """클래스별 round(n*p)개를 aligned로, 나머지는 다른 9색에 균등 배정한다."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p는 0과 1 사이여야 합니다.")
    labels = labels.long()
    generator = torch.Generator().manual_seed(seed)
    color_ids = torch.empty_like(labels)
    for label in range(NUM_COLORS):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        num_aligned = round(len(indices) * p)
        color_ids[indices[:num_aligned]] = label
        conflict = indices[num_aligned:]
        if len(conflict):
            other = torch.tensor([c for c in range(NUM_COLORS) if c != label])
            choices = other[torch.arange(len(conflict)) % len(other)]
            choices = choices[torch.randperm(len(choices), generator=generator)]
            color_ids[conflict] = choices
    return color_ids


def make_neutral_color_ids(labels: torch.Tensor, seed: int) -> torch.Tensor:
    """각 label 내부에서 10색의 수 차이가 최대 1이 되도록 고정 배정한다."""
    labels = labels.long()
    generator = torch.Generator().manual_seed(seed)
    color_ids = torch.empty_like(labels)
    for label in range(NUM_COLORS):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        colors = torch.arange(len(indices)) % NUM_COLORS
        colors = colors[torch.randperm(len(colors), generator=generator)]
        color_ids[indices] = colors
    return color_ids


def hue_to_rgb(hue_degrees: float) -> torch.Tensor:
    rgb = colorsys.hsv_to_rgb((hue_degrees % 360.0) / 360.0, 1.0, 1.0)
    return torch.tensor(rgb, dtype=torch.float32)


def colorize(image: torch.Tensor, color_id: int, hue_offset: float) -> tuple[torch.Tensor, float]:
    hue = (HUE_STEP * color_id + hue_offset) % 360.0
    return image * hue_to_rgb(hue).view(3, 1, 1), hue


class ColoredMNIST(Dataset):
    """color_id와 δ가 생성 시점에 고정되는 on-the-fly RGB 데이터셋."""

    def __init__(self, subset: Subset, color_ids: torch.Tensor, hue_offsets: torch.Tensor):
        if len(subset) != len(color_ids):
            raise ValueError("subset과 color_ids 길이가 다릅니다.")
        self.subset = subset
        self.color_ids = color_ids.long()
        self.hue_offsets = hue_offsets.float()

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label = self.subset[idx]
        original_idx = int(self.subset.indices[idx])
        color_id = int(self.color_ids[idx])
        colored, hue = colorize(image, color_id, float(self.hue_offsets[original_idx]))
        return colored, int(label), color_id, hue


class AllColorsValidation(Dataset):
    """각 validation 원본을 10색 모두로 펼쳐 sampling noise 없이 평가한다."""

    def __init__(self, subset: Subset, hue_offsets: torch.Tensor):
        self.subset = subset
        self.hue_offsets = hue_offsets.float()

    def __len__(self) -> int:
        return len(self.subset) * NUM_COLORS

    def __getitem__(self, idx: int):
        source_idx, color_id = divmod(idx, NUM_COLORS)
        image, label = self.subset[source_idx]
        original_idx = int(self.subset.indices[source_idx])
        colored, _ = colorize(image, color_id, float(self.hue_offsets[original_idx]))
        return colored, int(label), color_id, source_idx
