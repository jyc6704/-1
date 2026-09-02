"""학습 1 epoch와 10색 validation 평가."""

from __future__ import annotations

import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0
    for images, labels, *_ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size
    return loss_sum / total, correct / total


@torch.inference_mode()
def evaluate_all_colors(model, loader, num_sources: int, device) -> dict[str, float]:
    """Aligned, 다른 9색 평균 Conflict, 10색 평균 Neutral 및 Flip Rate를 계산한다.

    Flip Rate는 한 이미지의 10색 예측 45쌍 중 서로 다른 예측인 쌍의 비율을
    모든 이미지에 평균한 값이다. 단순 '하나라도 변화'보다 변화 정도를 보존한다.
    """
    model.eval()
    predictions = torch.empty((num_sources, 10), dtype=torch.long)
    labels_by_source = torch.empty(num_sources, dtype=torch.long)
    for images, labels, color_ids, source_ids in loader:
        pred = model(images.to(device)).argmax(1).cpu()
        predictions[source_ids, color_ids] = pred
        labels_by_source[source_ids] = labels

    color_grid = torch.arange(10).view(1, 10)
    labels_grid = labels_by_source.view(-1, 1)
    correct = predictions.eq(labels_grid)
    aligned_mask = color_grid.eq(labels_grid)
    aligned = correct[aligned_mask].float().mean().item()
    conflict = correct[~aligned_mask].float().mean().item()
    neutral = correct.float().mean().item()

    # 10색에서 가능한 모든 unordered pair(10 choose 2=45)의 불일치율.
    pairwise_different = predictions.unsqueeze(2).ne(predictions.unsqueeze(1))
    upper_triangle = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    flip_rate = pairwise_different[:, upper_triangle].float().mean().item()
    return {
        "aligned_accuracy": aligned,
        "conflict_accuracy": conflict,
        "neutral_accuracy": neutral,
        "shortcut_gap": aligned - conflict,
        "flip_rate": flip_rate,
    }
