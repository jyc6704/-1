"""3·8 Spurious Colored MNIST의 상관관계와 시각화를 검증한다."""

from pathlib import Path

import torch

from dataset import (
    SpuriousColoredMNIST,
    label_color_distribution,
    load_digit38_mnist,
    make_biased_color_ids,
    make_hue_offsets,
    make_neutral_color_ids,
    split_train_validation,
    spurious_correlation,
    subset_labels,
    visualize_label_color_heatmap,
    visualize_random_samples,
    visualize_same_image_both_colors,
)


def main() -> None:
    seed, p = 42, 0.99
    output = Path("validation_outputs_38")

    digit_train, digit_test = load_digit38_mnist(download=False)
    train_subset, validation_subset = split_train_validation(digit_train, seed)
    labels = subset_labels(train_subset)

    # 공식 MNIST 원본 index 기준 offset을 biased/neutral 조건이 공유한다.
    hue_offsets = make_hue_offsets(len(digit_train.mnist), seed)
    biased_ids = make_biased_color_ids(labels, p=p, seed=seed)
    neutral_ids = make_neutral_color_ids(labels, seed=seed + 10_000)
    biased_dataset = SpuriousColoredMNIST(train_subset, biased_ids, hue_offsets)

    biased_matrix = label_color_distribution(labels, biased_ids)
    neutral_matrix = label_color_distribution(labels, neutral_ids)
    actual_p = (biased_ids == labels).float().mean().item()
    biased_corr = spurious_correlation(labels, biased_ids)
    neutral_corr = spurious_correlation(labels, neutral_ids)
    sample = biased_dataset[0][0]

    assert set(labels.tolist()) == {3, 8}
    assert sample.shape == (3, 28, 28)
    assert hue_offsets.min() >= -5 and hue_offsets.max() <= 5
    assert abs(actual_p - p) < 0.001
    assert torch.all((neutral_matrix - 0.5).abs() < 0.001)

    print("=== Digit 3/8 Spurious Colored MNIST ===")
    print(f"official train 3/8={len(digit_train)}, official test 3/8={len(digit_test)}")
    print(f"train={len(train_subset)}, validation={len(validation_subset)}")
    print(f"digit counts: 3={(labels == 3).sum().item()}, 8={(labels == 8).sum().item()}")
    print(f"biased actual p={actual_p:.6f}")
    print(f"biased correlation(phi)={biased_corr:.6f}")
    print(f"neutral correlation(phi)={neutral_corr:.6f}")
    print(f"biased matrix:\n{biased_matrix}")
    print(f"neutral matrix:\n{neutral_matrix}")
    print(f"RGB tensor shape={tuple(sample.shape)}")
    print(f"Hue offset range={hue_offsets.min():.6f}..{hue_offsets.max():.6f}")

    # PNG 저장과 화면 표시를 함께 수행한다. 창을 닫으면 다음 창이 열린다.
    visualize_random_samples(
        biased_dataset, output / "biased_samples_38.png", seed=seed, show=True
    )
    visualize_label_color_heatmap(
        labels, biased_ids, output / "biased_heatmap_38.png", show=True,
        title=f"Biased digit-color distribution (p={p})",
    )
    visualize_label_color_heatmap(
        labels, neutral_ids, output / "neutral_heatmap_38.png", show=True,
        title="Neutral digit-color distribution (p=0.5)",
    )
    visualize_same_image_both_colors(
        train_subset, hue_offsets,
        output_path=output / "same_image_both_colors.png", show=True,
    )
    print(f"figures={output.resolve()}")


if __name__ == "__main__":
    main()
