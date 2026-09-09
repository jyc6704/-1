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


# 기존 0~9 전용 평가: 최종 본실험 참고용으로 보존하며 현재 실행하지 않는다.
# @torch.inference_mode()
# def evaluate_all_colors(model, loader, num_sources: int, device) -> dict[str, float]:
#     """Aligned, 다른 9색 평균 Conflict, 10색 평균 Neutral 및 Flip Rate를 계산한다.
#
#     Flip Rate는 한 이미지의 10색 예측 45쌍 중 서로 다른 예측인 쌍의 비율을
#     모든 이미지에 평균한 값이다. 단순 '하나라도 변화'보다 변화 정도를 보존한다.
#     """
#     model.eval()
#     predictions = torch.empty((num_sources, 10), dtype=torch.long)
#     labels_by_source = torch.empty(num_sources, dtype=torch.long)
#     for images, labels, color_ids, source_ids in loader:
#         pred = model(images.to(device)).argmax(1).cpu()
#         predictions[source_ids, color_ids] = pred
#         labels_by_source[source_ids] = labels
#
#     color_grid = torch.arange(10).view(1, 10)
#     labels_grid = labels_by_source.view(-1, 1)
#     correct = predictions.eq(labels_grid)
#     aligned_mask = color_grid.eq(labels_grid)
#     aligned = correct[aligned_mask].float().mean().item()
#     conflict = correct[~aligned_mask].float().mean().item()
#     neutral = correct.float().mean().item()
#
#     # 10색에서 가능한 모든 unordered pair(10 choose 2=45)의 불일치율.
#     pairwise_different = predictions.unsqueeze(2).ne(predictions.unsqueeze(1))
#     upper_triangle = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
#     flip_rate = pairwise_different[:, upper_triangle].float().mean().item()
#     return {
#         "aligned_accuracy": aligned,
#         "conflict_accuracy": conflict,
#         "neutral_accuracy": neutral,
#         "shortcut_gap": aligned - conflict,
#         "flip_rate": flip_rate,
#     }
#

# ---------------------------------------------------------------------------
# Binary (3, 8): 같은 원본과 같은 δ에서 두 색을 적용한 결과를 비교한다.
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_binary(model, aligned_loader, conflict_loader, device) -> dict[str, float]:
    """Aligned/Conflict와 2색 평균 Neutral, 색 변경 Flip Rate를 계산한다.

    두 loader는 동일 subset 객체, 동일 순서(shuffle=False)를 사용해야 한다.
    Flip Rate = 색만 반대로 바꿨을 때 예측이 달라진 원본 이미지의 비율.
    """
    from torch.utils.data import SequentialSampler

    if aligned_loader.dataset.subset is not conflict_loader.dataset.subset:
        raise ValueError("두 평가 조건은 동일한 validation subset을 사용해야 합니다.")
    if not all(isinstance(loader.sampler, SequentialSampler)
               for loader in (aligned_loader, conflict_loader)):
        raise ValueError("평가 DataLoader는 shuffle=False여야 합니다.")
    if aligned_loader.batch_size != conflict_loader.batch_size:
        raise ValueError("두 평가 조건의 batch_size가 같아야 합니다.")
    if aligned_loader.drop_last or conflict_loader.drop_last:
        raise ValueError("평가에서는 마지막 batch도 포함해야 합니다.")
    model.eval()
    aligned_correct = conflict_correct = flips = total = 0
    for aligned_batch, conflict_batch in zip(aligned_loader, conflict_loader):
        images_a, targets_a, digits_a, colors_a, _ = aligned_batch
        images_c, targets_c, digits_c, colors_c, _ = conflict_batch
        if not torch.equal(targets_a, targets_c) or not torch.equal(digits_a, digits_c):
            raise ValueError("두 평가 조건의 원본 순서가 다릅니다.")
        if not torch.equal(colors_a, digits_a) or not torch.all(colors_c != digits_c):
            raise ValueError("aligned/conflict 색상 배정이 올바르지 않습니다.")
        pred_a = model(images_a.to(device)).argmax(1).cpu()
        pred_c = model(images_c.to(device)).argmax(1).cpu()
        aligned_correct += (pred_a == targets_a).sum().item()
        conflict_correct += (pred_c == targets_c).sum().item()
        flips += (pred_a != pred_c).sum().item()
        total += len(targets_a)
    if total == 0:
        raise ValueError("validation 데이터가 비어 있습니다.")
    aligned = aligned_correct / total
    conflict = conflict_correct / total
    return {
        "aligned_accuracy": aligned,
        "conflict_accuracy": conflict,
        "neutral_accuracy": (aligned + conflict) / 2,
        "shortcut_gap": aligned - conflict,
        "flip_rate": flips / total,
    }
