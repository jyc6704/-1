# -1
과제연구1

## 현재 실험: 3·8 binary pilot

`python pilot.py`로 BinarySmallCNN을 p=0.99 biased 데이터에서 학습합니다.
pilot.py의 활성 코드에 있는 `exposure_epoch = 1`을 원하는 횟수로 바꾸면
같은 모델을 그 횟수만큼 연속 학습합니다. 매 epoch가 끝나면 학습 및 validation 지표를 출력합니다.
기존 0~9 pilot은 pilot.py 상단에 주석으로 보존되어 있습니다.

- 데이터: 공식 MNIST train의 숫자 3·8, train/validation 80:20 (seed=42).
- target: 3→0, 8→1. 색상: color 3(108°), color 8(288°), 이미지별 고정 δ ±5°.
- 모델: Conv 3→16→32, 특징 1568→2. Adam(lr=0.001), batch=64, CrossEntropyLoss.
- validation: 동일 원본에 aligned/conflict 두 색을 적용합니다.
- Neutral Accuracy는 두 색 정확도의 평균, Flip Rate는 색 변경 시 예측이 바뀐 이미지 비율입니다.
- 공식 test는 최종 평가용으로 보존합니다. neutral recovery는 수행하지 않습니다.

```powershell
python -m pip install -r requirements.txt
python pilot.py
# 코드 수정 없이 5 epoch 학습
python pilot.py --exposure-epoch 5
# MNIST 원본이 없는 경우
python pilot.py --download
```

출력은 pilot_results_38의 새 실행 폴더에 저장됩니다.
exposure_01.pt, exposure_02.pt 등은 epoch별 모델/optimizer/설정,
assignments.pt는 전체 epoch에서 공유하는 원본 index/색 배정/δ를 저장합니다.
results.csv는 epoch마다 결과를 한 행씩 기록하고,
result.json은 가장 최근 완료된 epoch의 결과를 저장합니다.
기존 Digit38CNN(1568→64→2) checkpoint는 새 모델과 호환되지 않습니다.

현재 구조: dataset.py(데이터·시각화), model.py(CNN), train.py(학습·평가),
pilot.py(실험 실행). 데이터 검증과 시각화는 `python dataset.py`로 실행합니다.

## 회복 지표 계산

`recovery_metrics.py`는 이미 저장한 결과를 읽어 다음 두 지표를 **분리해서** 계산합니다.
학습을 실행하거나 기존 pilot 결과를 수정하지 않습니다. Python 표준 라이브러리만 사용합니다.

- `mean_color_accuracy` (M): 전체 평가 색상 평균 정확도. 보조 지표입니다.
- `all_colors_accuracy` (Q): 모든 평가 색상에서 정답인 원본의 비율. 주 지표입니다.
- `q_recovery_percent`: `100 * (Q_t - Q_0) / (Q_ref - Q_0)`.
- `q_gap_to_reference_pp`: `100 * (Q_ref - Q_t)`, 기준 모델과의 잔여 격차(%p).
- `mean_gap_to_reference_pp`: `100 * (M_ref - M_t)`, 평균 정확도의 잔여 격차(%p).

M/Q 저장 단위는 0~1, 회복률은 %, 잔여 격차는 %p입니다. 가중합은 만들지 않습니다.
회복률은 기준 모델 대비 초기 성능 격차를 얼마나 해소했는지를 뜻하며,
Achille 등의 기준 모델 대비 성능 손실 관점을 정규화한 확장 지표입니다.
해당 논문이 이 회복률 공식을 직접 제시한 것은 아닙니다.

```powershell
# pilot_results_38에서 가장 최근 수정된 results.csv를 자동 선택
python recovery_metrics.py

# 분석할 실행을 직접 지정하는 것을 권장 (results.csv의 모든 epoch를 계산)
python recovery_metrics.py pilot_results_38/binary_p0.99_seed42_exp01_04

# CSV 또는 JSON 파일도 직접 지정 가능
python recovery_metrics.py pilot_results_38/binary_p0.99_seed42_exp01_04/result.json
```

입력 폴더 아래 새 `recovery_metrics` 폴더에 JSON/CSV를 저장합니다.
이미 존재하면 `_02`, `_03` 등을 붙여 보존합니다. `--output 새폴더`로 저장 위치를
지정할 수도 있습니다. 명시한 출력 폴더가 이미 존재하면 덮어쓰지 않고 오류를 냅니다.

현재 pilot은 **편향 exposure 학습**만 수행합니다. 따라서 단독 결과로는 M/Q만
계산하며, 회복률은 `null`(CSV 빈칸), 상태는 `missing_initial_and_reference`입니다.
여러 exposure epoch를 자동으로 recovery epoch로 취급하지 않습니다.

중립 회복 학습 결과와 중립 기준 모델 결과를 실제로 확보한 뒤에는 아래처럼 실행합니다.
아래 세 경로는 향후 실험 결과 경로의 예시이며, 현재 존재하는 결과가 아닙니다.

```powershell
python recovery_metrics.py recovery_run/results.csv --initial biased_run/result.json --reference neutral_reference/result.json
```

- `--initial`: 중립 회복 학습 **직전** 모델의 결과. 첫 노출 epoch가 아니라 실제 전환 시점을 지정합니다.
- `--reference`: **처음부터 중립 데이터로 학습한** 기준 모델의 결과.
- 두 인자가 여러 행의 CSV/JSON을 가리키면 마지막 행을 선택하며 선택한 행과 epoch를 출력 JSON에 기록합니다.
- 회복 결과에 `recovery_epoch` 열을 넣으면 `phase=recovery`로 기록합니다.
- reference만 지정하면 잔여 격차만 계산하고 회복률은 비워 둡니다.
- `Q_ref - Q_0 <= 0.001`(0.1%p)이면 비율이 불안정하거나 정의에 맞지 않아 회복률을
  비워 둡니다. 허용값은 `--min-initial-gap`으로 지정할 수 있습니다. 이 값은 통계적 유의성 기준이 아닙니다.
- 음수 회복률과 100% 초과 회복률은 그대로 보존합니다.
- 평가 원본·색상 집합·모델 조건을 맞추고 같은 회복 학습량에서 비교해야 합니다.
  현재 pilot의 seed는 데이터 분할과 hue offset도 바꾸므로 config에 seed가 다르면 비교를 거부합니다.
  요약 파일만으로 평가 원본의 일치나 학습 이력을 완전히 검증할 수는 없습니다.
- reference에 편향 확률 `p`가 있으면 중립 조건 `p=1/K`인지 검사합니다.

### 현재 2클래스와 향후 10클래스

현재처럼 2클래스·2색의 동일 원본 쌍이면 `Q = neutral_accuracy - flip_rate / 2`로
기존 결과만으로 Q를 정확히 복원할 수 있습니다. 실행 폴더의 `config.json`으로
클래스·색상 수를 확인합니다. config 없이 파일만 옮겼다면 다음처럼 명시하세요.

```powershell
python recovery_metrics.py copied_results.csv --num-classes 2 --num-colors 2
```

**10클래스에서는 이 공식을 쓰지 않습니다.** 기존 다중 클래스의 평균 정확도와
flip rate만으로 모든 색상 정답률을 복원할 수 없으므로 아래 중 하나가 필요합니다.

1. 평가 단계에서 직접 구한 `mean_color_accuracy`(또는 `neutral_accuracy`)와
   `all_colors_accuracy`, `num_classes`, `num_colors`를 CSV/JSON에 저장합니다.
2. JSON에 `predictions`(원본 N개 × 색상 K개 정수 예측), `targets`(N개 정답),
   `num_classes`, `num_colors`를 저장합니다. 파일이 각 행의 정답 수와 모든 색상 정답 여부를 직접 계산합니다.

예를 들어 아래는 **형식을 설명하기 위한 가상 3클래스·3색 데이터**입니다.

```json
{
  "num_classes": 3,
  "num_colors": 3,
  "recovery_epoch": 1,
  "targets": [0, 1],
  "predictions": [[0, 0, 0], [1, 2, 0]]
}
```

동일 행은 동일 원본이며, 모든 행에서 색상 열의 순서는 같아야 합니다.
3·8 실험의 target 0/1처럼 예측과 정답의 라벨 체계도 같아야 합니다.
`color_scores_from_predictions(predictions, targets)` 함수를 평가 코드에서 직접 import해도 됩니다.
