"""Shortcut 의존성과 회복 여부를 계산하는 순수 함수."""

from __future__ import annotations


def shortcut_gap(aligned_accuracy: float, conflict_accuracy: float) -> float:
    return aligned_accuracy - conflict_accuracy


def is_recovered(
    gap: float,
    neutral_accuracy: float,
    reference_accuracy: float,
    gap_threshold: float = 0.05,
    reference_ratio: float = 0.95,
) -> bool:
    """Gap≤5%p이면서 neutral 정확도가 reference의 95% 이상인지 판정한다."""
    return gap <= gap_threshold and neutral_accuracy >= reference_ratio * reference_accuracy


def recovery_time(rows: list[dict], reference_accuracy: float, **thresholds) -> int | None:
    """최초 회복 epoch. 끝까지 미회복이면 censored를 뜻하는 None을 반환한다."""
    for row in sorted(rows, key=lambda item: item["recovery_epoch"]):
        if is_recovered(row["shortcut_gap"], row["neutral_accuracy"], reference_accuracy, **thresholds):
            return int(row["recovery_epoch"])
    return None
