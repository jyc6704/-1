"""pilot 결과로 색상 강건성의 회복 지표를 계산한다 (추가 학습 없음).

M = 전체 색상 평균 정확도 (보조 지표)
Q = 모든 평가 색상에서 정답인 원본 비율 (주 지표)
R_Q = 100 * (Q_t - Q_0) / (Q_ref - Q_0)

현재 binary pilot은 Q = neutral_accuracy - flip_rate / 2로 계산할 수 있다.
다중 클래스에서는 이미지별 predictions/targets 또는 직접 측정한
all_colors_accuracy가 필요하다. 이진 공식을 다중 클래스에 적용하지 않는다.
Achille 등의 기준 모델 대비 성능 격차를 회복률로 확장한 것이며,
R_Q 자체는 해당 논문의 공식이 아니다. M과 Q의 가중합은 계산하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TOLERANCE = 1e-6
DEFAULT_ROOT = Path(__file__).resolve().parent / "pilot_results_38"


def probability(value, name: str) -> float:
    """모든 입력 정확도/비율의 단위는 0~1이다."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: bool이 아닌 0~1 수치가 필요합니다.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name}: 유한한 0~1 값이 필요합니다. 입력={value!r}")
    return number


def color_scores_from_predictions(predictions: list, targets: list) -> dict:
    """N x K 정수 예측과 N개 정답에서 M/Q를 직접 계산한다.

    같은 행은 동일 원본, 각 열은 모든 원본에 공통인 고정 색상이다.
    예측과 정답은 같은 라벨 체계(예: 0~9)를 사용해야 한다.
    """
    if not isinstance(predictions, list) or not isinstance(targets, list):
        raise ValueError("predictions와 targets는 list여야 합니다.")
    if not targets or len(predictions) != len(targets):
        raise ValueError("비어 있지 않은 N x K predictions와 N개 targets가 필요합니다.")
    if not isinstance(predictions[0], list) or len(predictions[0]) < 2:
        raise ValueError("원본별로 최소 2개 색상의 예측이 필요합니다.")
    num_colors = len(predictions[0])
    correct_count = all_correct_count = 0
    for row, target in zip(predictions, targets):
        if not isinstance(row, list) or len(row) != num_colors:
            raise ValueError("모든 원본은 동일한 개수/순서의 색상 예측을 가져야 합니다.")
        if type(target) is not int or any(type(pred) is not int for pred in row):
            raise ValueError("predictions와 targets에는 정수 클래스 라벨만 허용됩니다.")
        correct = sum(pred == target for pred in row)
        correct_count += correct
        all_correct_count += correct == num_colors
    return {
        "mean_color_accuracy": correct_count / (len(targets) * num_colors),
        "all_colors_accuracy": all_correct_count / len(targets),
        "num_sources": len(targets),
        "num_colors": num_colors,
    }


def _check_close(actual: float, expected: float, name: str) -> None:
    if abs(actual - expected) > TOLERANCE:
        raise ValueError(f"{name}가 다른 지표와 일치하지 않습니다: {actual} != {expected}")


def calculate_scores(row: dict, num_classes: int | None, num_colors: int | None) -> dict:
    """기존 binary 요약 결과 또는 일반적인 다색 평가 결과를 읽는다."""
    direct = None
    if "predictions" in row or "targets" in row:
        direct = color_scores_from_predictions(row.get("predictions"), row.get("targets"))
        if num_colors is not None and num_colors != direct["num_colors"]:
            raise ValueError("색상 수 설정과 predictions 열 개수가 다릅니다.")
        num_colors = direct["num_colors"]

    mean_values = [probability(row[key], key) for key in
                   ("mean_color_accuracy", "neutral_accuracy") if key in row]
    if direct is not None:
        mean_values.append(direct["mean_color_accuracy"])
    if not mean_values:
        raise ValueError("neutral_accuracy, mean_color_accuracy 또는 predictions가 필요합니다.")
    mean = mean_values[0]
    for other in mean_values[1:]:
        _check_close(mean, other, "전체 색상 평균 정확도")

    if direct is not None:
        all_colors = direct["all_colors_accuracy"]
        method = "per_source_predictions"
        if "all_colors_accuracy" in row:
            _check_close(probability(row["all_colors_accuracy"], "all_colors_accuracy"),
                         all_colors, "모든 색상 정답률")
    elif "all_colors_accuracy" in row:
        all_colors = probability(row["all_colors_accuracy"], "all_colors_accuracy")
        method = "measured_all_colors_accuracy"
    elif num_classes == 2 and num_colors == 2:
        flip = probability(row["flip_rate"], "flip_rate")
        all_colors = mean - flip / 2
        method = "binary_paired_identity"
    else:
        raise ValueError(
            "요약 수치에서 Q=M-F/2를 쓰려면 2클래스·2색 평가임이 확인되어야 합니다. "
            "config.json 또는 --num-classes 2 --num-colors 2를 지정하세요. "
            "다중 클래스는 predictions/targets 또는 직접 측정한 all_colors_accuracy가 필요합니다."
        )

    # Q <= M 및 K색의 결합 확률 하한 Q >= K*M-(K-1)을 검증한다.
    lower = max(0.0, num_colors * mean - (num_colors - 1)) if num_colors else 0.0
    if not lower - TOLERANCE <= all_colors <= mean + TOLERANCE:
        raise ValueError("M/Q 또는 flip_rate가 실현 가능한 동일 원본 평가 결과가 아닙니다.")
    all_colors = min(mean, max(lower, all_colors))  # 허용 오차 안의 반올림만 보정

    if num_classes == 2 and num_colors == 2:
        if "flip_rate" in row:
            flip = probability(row["flip_rate"], "flip_rate")
            _check_close(all_colors, mean - flip / 2, "binary Q=M-F/2")
        if "aligned_accuracy" in row and "conflict_accuracy" in row:
            aligned = probability(row["aligned_accuracy"], "aligned_accuracy")
            conflict = probability(row["conflict_accuracy"], "conflict_accuracy")
            _check_close(mean, (aligned + conflict) / 2, "binary neutral_accuracy")
            if all_colors > min(aligned, conflict) + TOLERANCE:
                raise ValueError("Q는 aligned/conflict 정확도보다 클 수 없습니다.")
    return {
        "mean_color_accuracy": mean,
        "all_colors_accuracy": all_colors,
        "num_classes": num_classes,
        "num_colors": num_colors,
        "calculation_method": method,
        **({"num_sources": direct["num_sources"]} if direct is not None else {}),
    }


def calculate_recovery(current: dict, initial: dict | None = None,
                       reference: dict | None = None, min_initial_gap: float = 0.001) -> dict:
    """회복률은 %, 잔여 격차는 %p. 미정의 값은 None, 비율은 clipping하지 않는다."""
    if not math.isfinite(min_initial_gap) or min_initial_gap < 0:
        raise ValueError("min_initial_gap은 유한한 0 이상의 값이어야 합니다.")
    result = {
        "q_recovery_percent": None,
        "q_gap_to_reference_pp": None,
        "mean_gap_to_reference_pp": None,
        "q_initial_gap_pp": None,
        "recovery_status": "missing_initial_and_reference",
    }
    if reference is not None:
        result["q_gap_to_reference_pp"] = 100 * (
            reference["all_colors_accuracy"] - current["all_colors_accuracy"])
        result["mean_gap_to_reference_pp"] = 100 * (
            reference["mean_color_accuracy"] - current["mean_color_accuracy"])
        result["recovery_status"] = "missing_initial"
    if initial is not None and reference is None:
        result["recovery_status"] = "missing_reference"
    if initial is not None and reference is not None:
        gap = reference["all_colors_accuracy"] - initial["all_colors_accuracy"]
        result["q_initial_gap_pp"] = 100 * gap
        if gap <= min_initial_gap:
            result["recovery_status"] = "initial_gap_too_small_or_nonpositive"
        else:
            result["q_recovery_percent"] = 100 * (
                current["all_colors_accuracy"] - initial["all_colors_accuracy"]) / gap
            result["recovery_status"] = "available"
    return result


def resolve_input(path: Path) -> Path:
    if path.is_dir():
        for name in ("results.csv", "result.json"):
            candidate = path / name
            if candidate.is_file():
                return candidate.resolve()
        raise ValueError(f"결과 폴더에 results.csv/result.json이 없습니다: {path}")
    if not path.is_file():
        raise ValueError(f"입력 파일이 없습니다: {path}")
    return path.resolve()


def load_results(path: Path, args) -> dict:
    path = resolve_input(path)
    config_path = path.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
    if not isinstance(config, dict):
        raise ValueError(f"config.json은 객체여야 합니다: {config_path}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload if isinstance(payload, list) else [payload]
    else:
        raise ValueError("입력 형식은 .csv 또는 .json이어야 합니다.")
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("입력에는 하나 이상의 결과 객체가 필요합니다.")
    scores = []
    for index, row in enumerate(rows, 1):
        dimensions = []
        for key, config_key, override in (("num_classes", "digits", args.num_classes),
                                          ("num_colors", "color_ids", args.num_colors)):
            candidates = []
            if key in row:
                candidates.append(int(row[key]))
            if key in config:
                candidates.append(int(config[key]))
            if config_key in config:
                candidates.append(len(config[config_key]))
            if override is not None:
                candidates.append(override)
            if candidates and (min(candidates) < 2 or len(set(candidates)) != 1):
                raise ValueError(f"{path}, 행 {index}: {key} 설정이 충돌하거나 2 미만입니다.")
            dimensions.append(candidates[0] if candidates else None)
        try:
            scores.append(calculate_scores(row, *dimensions))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}, 행 {index}: {error}") from error
    return {"path": path, "rows": rows, "scores": scores, "config": config}


def check_comparable(current: dict, other: dict) -> None:
    # 현재 pilot의 seed는 split 및 hue offset도 바꾸므로 비교 시 고정해야 한다.
    for key in ("digits", "color_ids", "target_mapping", "model_class", "val_ratio",
                "hue_jitter_degrees", "seed"):
        if key in current["config"] and key in other["config"]:
            if current["config"][key] != other["config"][key]:
                raise ValueError(f"비교 설정 불일치: {key} ({current['path']} / {other['path']})")
    for score in current["scores"]:
        for key in ("num_classes", "num_colors"):
            a, b = score[key], other["scores"][-1][key]
            if a is not None and b is not None and a != b:
                raise ValueError(f"비교 대상의 {key}가 다릅니다.")


def baseline_info(data: dict | None) -> dict | None:
    if data is None:
        return None
    row = data["rows"][-1]
    return {"source": str(data["path"]), "selected_row": len(data["rows"]),
            "exposure_epoch": row.get("exposure_epoch"),
            "recovery_epoch": row.get("recovery_epoch"), **data["scores"][-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="pilot 결과의 M/Q 및 기준 모델 대비 회복률 계산")
    parser.add_argument("input", type=Path, nargs="?",
                        help="실행 폴더, results.csv 또는 result.json. 생략하면 최신 pilot 결과")
    parser.add_argument("--initial", type=Path, help="회복 시작 전 결과. 여러 행이면 마지막 행 사용")
    parser.add_argument("--reference", type=Path, help="처음부터 중립 학습한 기준 결과. 마지막 행 사용")
    parser.add_argument("--num-classes", type=int, help="config가 없을 때 클래스 수 명시")
    parser.add_argument("--num-colors", type=int, help="config가 없을 때 평가 색상 수 명시")
    parser.add_argument("--min-initial-gap", type=float, default=0.001,
                        help="이 값 이하의 초기 Q 격차는 회복률 미정의 (0~1 단위, 기본 0.001)")
    parser.add_argument("--output", type=Path, help="새 출력 폴더. 기본: 입력 폴더/recovery_metrics[_02...]")
    args = parser.parse_args()
    try:
        if args.input is None:
            candidates = list(DEFAULT_ROOT.glob("*/results.csv"))
            if not candidates:
                raise ValueError("pilot 결과가 없습니다. 먼저 pilot.py를 실행하거나 입력 경로를 지정하세요.")
            args.input = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        current = load_results(args.input, args)
        initial = load_results(args.initial, args) if args.initial else None
        reference = load_results(args.reference, args) if args.reference else None
        for other in (initial, reference):
            if other is not None:
                check_comparable(current, other)
        if reference is not None:
            # 알려진 편향 학습 결과를 중립 reference로 잘못 사용하는 것을 차단한다.
            ref_p = reference["config"].get("p", reference["rows"][-1].get("p"))
            colors = reference["scores"][-1]["num_colors"]
            if ref_p is not None and colors is not None:
                _check_close(probability(ref_p, "reference p"), 1 / colors, "중립 reference의 p=1/K")
        records = []
        for index, (row, score) in enumerate(zip(current["rows"], current["scores"]), 1):
            stage = row.get("phase") or ("recovery" if "recovery_epoch" in row else
                                         "exposure" if "exposure_epoch" in row else "unspecified")
            record = {
                "source_row": index, "phase": stage,
                **{key: row[key] for key in ("seed", "p", "exposure_epoch", "recovery_epoch") if key in row},
                **score,
                **calculate_recovery(score, initial["scores"][-1] if initial else None,
                                     reference["scores"][-1] if reference else None,
                                     args.min_initial_gap),
            }
            records.append(record)
        notes = [
            "M/Q를 분리해 보고하며 가중합은 계산하지 않습니다.",
            "M/Q는 0~1, q_recovery_percent는 %, *_pp는 %p 단위입니다.",
            "null/CSV 빈칸은 미계산 또는 미정의입니다. 회복률을 0~100으로 제한하지 않습니다.",
            "기준 모델은 동일 평가 원본·색상·모델 조건에서 처음부터 중립 데이터로 학습해야 합니다.",
            "요약 파일만으로 동일 평가 원본이나 기준 모델의 학습 이력을 완전히 검증할 수 없습니다.",
        ]
        if initial is None or reference is None:
            notes.append("회복률 계산에는 --initial과 --reference가 모두 필요합니다.")
        if any(record["phase"] == "exposure" for record in records):
            notes.append("exposure 행은 편향 학습 결과입니다. 수치 변화가 중립 회복 학습의 증거는 아닙니다.")
        report = {"source": str(current["path"]), "initial": baseline_info(initial),
                  "reference": baseline_info(reference), "min_initial_gap": args.min_initial_gap,
                  "notes": notes, "results": records}
        if args.output is not None:
            output = args.output
            output.mkdir(parents=True, exist_ok=False)
        else:
            root = current["path"].parent
            suffix = 1
            while True:
                output = root / ("recovery_metrics" if suffix == 1 else f"recovery_metrics_{suffix:02d}")
                try:
                    output.mkdir()
                    break
                except FileExistsError:
                    suffix += 1
        (output / "recovery_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        fields = list(dict.fromkeys(key for row in records for key in row))
        with (output / "recovery_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        print(f"input={current['path']}")
        for row in records:
            recovery = row["q_recovery_percent"]
            rq = f"{recovery:.4f}%" if recovery is not None else f"N/A ({row['recovery_status']})"
            print(f"row={row['source_row']} phase={row['phase']} "
                  f"M={100 * row['mean_color_accuracy']:.4f}% "
                  f"Q={100 * row['all_colors_accuracy']:.4f}% R_Q={rq}")
        for note in notes:
            print(note)
        print(f"saved={output.resolve()}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
