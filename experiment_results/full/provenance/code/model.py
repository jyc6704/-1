"""Downloads 모델과 동일한 구조의 10-class 및 2-class CNN."""

import torch
from torch import nn


class BinarySmallCNN(nn.Module):
    """입력 [3, 28, 28]을 숫자 3(target 0) 또는 8(target 1)로 분류한다."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 추출한 1568개 특징을 두 클래스의 logit으로 바로 변환한다.
        self.classifier = nn.Linear(32 * 7 * 7, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(torch.flatten(features, 1))


class SmallCNN(nn.Module):
    """최종 0~9 분류용 모델. 현재 3·8 학습에서는 BinarySmallCNN을 사용한다."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))
