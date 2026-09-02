"""데이터 통제 조건을 검사하고 확인용 그림을 파일로 저장한다."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # VS Code 터미널/CI에서도 창 없이 저장 가능
import matplotlib.pyplot as plt
import torch

from dataset import ColoredMNIST, load_mnist, make_biased_color_ids, make_hue_offsets, make_neutral_color_ids, split_train_validation, subset_labels


def main() -> None:
    seed, p = 42, 0.99
    output = Path("validation_outputs")
    output.mkdir(exist_ok=True)
    mnist, _ = load_mnist(download=False)
    train_subset, _ = split_train_validation(mnist, seed)
    labels = subset_labels(train_subset)
    offsets = make_hue_offsets(len(mnist), seed)
    biased_ids = make_biased_color_ids(labels, p, seed)
    neutral_ids = make_neutral_color_ids(labels, seed + 10_000)
    biased = ColoredMNIST(train_subset, biased_ids, offsets)
    neutral = ColoredMNIST(train_subset, neutral_ids, offsets)

    matrix = torch.stack([
        neutral_ids[labels == y].bincount(minlength=10).float() / (labels == y).sum()
        for y in range(10)
    ])
    biased_p = (biased_ids == labels).float().mean().item()
    neutral_p = (neutral_ids == labels).float().mean().item()
    sample, _, _, _ = biased[0]
    assert sample.shape == (3, 28, 28)
    assert offsets.min() >= -5 and offsets.max() <= 5
    assert abs(biased_p - p) < 0.001
    assert torch.all((matrix - 0.1).abs() < 0.001)

    generator = torch.Generator().manual_seed(seed)
    random_indices = torch.randperm(len(biased), generator=generator)[:20]
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for ax, idx in zip(axes.flat, random_indices.tolist()):
        image, label, color_id, hue = biased[idx]
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f"label={label}, color={color_id}\nH={hue:.1f}°", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "biased_samples.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    plot = ax.imshow(matrix.numpy(), vmin=0, vmax=0.2)
    ax.set(xlabel="Color ID", ylabel="Digit Label", title="Neutral label-color distribution")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    for y in range(10):
        for color_id in range(10):
            ax.text(color_id, y, f"{matrix[y, color_id]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(plot, ax=ax, label="Ratio")
    fig.tight_layout()
    fig.savefig(output / "neutral_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"total official train={len(mnist)}, pilot train={len(train_subset)}")
    print(f"biased actual p={biased_p:.6f}")
    print(f"neutral label==color ratio={neutral_p:.6f}")
    print(f"neutral matrix range={matrix.min().item():.6f}..{matrix.max().item():.6f}")
    print(f"tensor shape={tuple(sample.shape)}")
    print(f"hue offset range={offsets.min().item():.6f}..{offsets.max().item():.6f}")
    print(f"figures={output.resolve()}")


if __name__ == "__main__":
    main()
