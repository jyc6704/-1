import random
import colorsys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# ==========================================
# 기본 설정
# ==========================================

P = 0.99
SEED = 42
BATCH_SIZE = 64


# ==========================================
# 1. HSV 기준 10개의 색상 생성
#
# 숫자 0 -> Hue   0°
# 숫자 1 -> Hue  36°
# 숫자 2 -> Hue  72°
# ...
# 숫자 9 -> Hue 324°
#
# Saturation = 1
# Value = 1
# ==========================================

def make_rgb_colors():
    colors = []

    for color_id in range(10):
        hue = color_id / 10.0

        r, g, b = colorsys.hsv_to_rgb(
            hue,
            1.0,
            1.0
        )

        colors.append(
            torch.tensor(
                [r, g, b],
                dtype=torch.float32
            )
        )

    return torch.stack(colors)


RGB_COLORS = make_rgb_colors()


# ==========================================
# 2. 각 MNIST 이미지에 color_id 할당
#
# p = 0.99일 경우
#
# 99%:
#   label과 color_id가 같음
#
# 1%:
#   다른 9개 색 중 하나를 사용
#
# 각 숫자 클래스별로 p가 약 0.99가 되도록 설정
# ==========================================

def make_color_ids(labels, p=0.99, seed=42):

    # labels는 MNIST에서 이미 Tensor이므로
    # torch.tensor(labels)로 다시 만들 필요 없음
    labels = labels.long()

    generator = torch.Generator()
    generator.manual_seed(seed)

    color_ids = torch.empty(
        len(labels),
        dtype=torch.long
    )

    for label in range(10):

        # 현재 label에 해당하는 이미지의 index
        indices = torch.where(
            labels == label
        )[0]

        # index 순서를 랜덤하게 섞음
        permutation = torch.randperm(
            len(indices),
            generator=generator
        )

        indices = indices[permutation]

        # aligned 데이터 개수
        num_aligned = round(
            len(indices) * p
        )

        aligned_indices = indices[:num_aligned]
        conflict_indices = indices[num_aligned:]


        # ----------------------------------
        # Aligned 데이터
        #
        # label과 color_id를 동일하게 설정
        # ----------------------------------

        color_ids[aligned_indices] = label


        # ----------------------------------
        # Conflict 데이터
        #
        # 자기 색을 제외한 나머지 9개 색 사용
        # ----------------------------------

        other_colors = [
            color
            for color in range(10)
            if color != label
        ]

        conflict_colors = []

        # 다른 9개의 색이 최대한 균등하게
        # 나타나도록 배정
        for i in range(len(conflict_indices)):
            conflict_colors.append(
                other_colors[i % 9]
            )

        # Conflict 색상 순서도 랜덤화
        random_generator = random.Random(
            seed + label
        )

        random_generator.shuffle(
            conflict_colors
        )

        if len(conflict_indices) > 0:

            color_ids[conflict_indices] = torch.tensor(
                conflict_colors,
                dtype=torch.long
            )

    return color_ids


# ==========================================
# 3. Colored MNIST Dataset
# ==========================================

class ColoredMNIST(Dataset):

    def __init__(
        self,
        mnist_dataset,
        color_ids
    ):

        self.mnist = mnist_dataset
        self.color_ids = color_ids

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):

        # 원본 MNIST
        #
        # image shape:
        # [1, 28, 28]
        #
        # pixel 값:
        # 0 ~ 1
        image, label = self.mnist[idx]

        color_id = int(
            self.color_ids[idx]
        )

        # color_id에 대응되는 RGB 값
        #
        # [3] -> [3, 1, 1]
        rgb = RGB_COLORS[color_id].view(
            3, 1, 1
        )

        # MNIST 원래 밝기는 유지하면서
        # 색상만 적용
        #
        # [1, 28, 28]
        # ×
        # [3, 1, 1]
        #
        # ->
        #
        # [3, 28, 28]

        colored_image = image * rgb

        return (
            colored_image,
            label,
            color_id
        )


# ==========================================
# 4. MNIST 다운로드
# ==========================================

mnist_train = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)


# ==========================================
# 5. p = 0.99 색상 배치 생성
# ==========================================

train_color_ids = make_color_ids(
    mnist_train.targets,
    p=P,
    seed=SEED
)


# ==========================================
# 6. color_id 저장
#
# 나중에 다시 실행하더라도
# 동일한 색상 배치를 사용할 수 있도록 저장
# ==========================================

torch.save(
    train_color_ids,
    "train_color_ids_p099_seed42.pt"
)


# ==========================================
# 7. Colored MNIST Dataset 생성
# ==========================================

colored_train_dataset = ColoredMNIST(
    mnist_train,
    train_color_ids
)


# ==========================================
# 8. DataLoader 생성
# ==========================================

train_loader = DataLoader(
    colored_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# 9. 전체 실제 p 확인
# ==========================================

labels = mnist_train.targets

aligned_mask = (
    train_color_ids == labels
)

aligned_count = aligned_mask.sum().item()

conflict_count = (
    ~aligned_mask
).sum().item()

actual_p = aligned_mask.float().mean().item()


print("======================================")
print("전체 데이터 확인")
print("======================================")

print(
    f"전체 데이터 개수 : {len(labels)}"
)

print(
    f"설정한 p          : {P}"
)

print(
    f"실제 p            : {actual_p:.6f}"
)

print(
    f"Aligned 개수      : {aligned_count}"
)

print(
    f"Conflict 개수     : {conflict_count}"
)


# ==========================================
# 10. 클래스별 p 확인
# ==========================================

print("\n======================================")
print("클래스별 결과")
print("======================================")

for label in range(10):

    mask = (
        labels == label
    )

    total = mask.sum().item()

    class_aligned = (
        train_color_ids[mask] == label
    ).sum().item()

    class_conflict = (
        train_color_ids[mask] != label
    ).sum().item()

    ratio = (
        class_aligned / total
    )

    print(
        f"숫자 {label}: "
        f"Aligned {class_aligned}, "
        f"Conflict {class_conflict}, "
        f"전체 {total}, "
        f"p = {ratio:.4f}"
    )


# ==========================================
# 11. Conflict 데이터가 실제 존재하는지 확인
# ==========================================

conflict_indices = torch.where(
    train_color_ids != labels
)[0]


print("\n======================================")
print("Conflict 데이터 확인")
print("======================================")

print(
    f"Conflict 데이터 개수: "
    f"{len(conflict_indices)}"
)


# ==========================================
# 12. Conflict 데이터 예시 10개 출력
# ==========================================

print("\nConflict 예시 10개")

for index_tensor in conflict_indices[:10]:

    idx = index_tensor.item()

    label = labels[idx].item()

    color_id = train_color_ids[idx].item()

    hue = color_id * 36

    print(
        f"index={idx:5d}, "
        f"label={label}, "
        f"color_id={color_id}, "
        f"Hue={hue}°"
    )


# ==========================================
# 13. DataLoader 작동 확인
# ==========================================

images, labels_batch, colors_batch = next(
    iter(train_loader)
)


print("\n======================================")
print("DataLoader 확인")
print("======================================")

print(
    "Batch shape:",
    images.shape
)

print(
    "\n앞의 10개 label:"
)

print(
    labels_batch[:10]
)

print(
    "\n앞의 10개 color_id:"
)

print(
    colors_batch[:10]
)


# ==========================================
# 14. RGB 색상 자체 확인
# ==========================================

print("\n======================================")
print("사용되는 10가지 색상")
print("======================================")

for color_id in range(10):

    hue = color_id * 36

    r, g, b = RGB_COLORS[
        color_id
    ].tolist()

    print(
        f"color_id={color_id}, "
        f"Hue={hue:3d}°, "
        f"RGB=({r:.3f}, {g:.3f}, {b:.3f})"
    )
    import matplotlib.pyplot as plt


# ==========================================
# 15. Colored MNIST 이미지 시각화
# ==========================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 5)
)

for i, ax in enumerate(axes.flat):

    image, label, color_id = colored_train_dataset[i]

    # PyTorch:
    # [3, 28, 28]
    #
    # matplotlib:
    # [28, 28, 3]
    image = image.permute(1, 2, 0)

    ax.imshow(image)

    ax.set_title(
        f"Label: {label}\n"
        f"Color: {color_id} "
        f"(H={color_id * 36}°)"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()
# ==========================================
# 16. Conflict 이미지 시각화
# ==========================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(12, 5)
)

for ax, index_tensor in zip(
    axes.flat,
    conflict_indices[:10]
):

    idx = index_tensor.item()

    image, label, color_id = colored_train_dataset[idx]

    image = image.permute(1, 2, 0)

    ax.imshow(image)

    ax.set_title(
        f"Label: {label}\n"
        f"Color: {color_id} "
        f"(H={color_id * 36}°)"
    )

    ax.axis("off")

plt.tight_layout()
plt.show()