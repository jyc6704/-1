# 콜로퀴움 이후: 3·8 이진분류 step 단위 편향 회복 실험

새 실행 파일: `binary_step_experiment.py` (기존 `experiment.py`, `recovery_experiment.py` 변경 없음)

## 1. 무엇이 바뀌었나

최근 발표에서 정리한 세 가지 변경 사항을 반영했다.

| 항목 | 이번 실험 | 비고 |
|---|---|---|
| 분류 작업 | 숫자 3과 8의 **이진분류** | MNIST와 EMNIST Digits 각각 독립 실행 가능 |
| 노출 단계 | **1, 2, 5, 10 epoch** | 동일한 seed에서는 초기 모델·분할·색 배정·각 epoch 내 순서를 재사용 |
| 회복 단계 | 중립 색상으로 **10 epoch** 연속 학습 | 매 회복 epoch마다 원본별 색 배정 재생성, 같은 CNN 가중치를 이어서 학습 |
| 관측 빈도 | **매 optimizer.step() 직후** | 노출 직후 상태는 `recovery_step=0`으로 별도 평가 |
| 편향 일치 확률 | `p=0.5, 0.9, 0.99` | p=0.5는 중립 대조군이자 회복률 기준 모델 |
| 반복 실험 | seed 42, 43, 44 | seed 평균 ± 표본표준편차 (1개 seed는 SD 공란) |
| 데이터 분할 | 원본 train의 40%/40%/20% | 노출 / 회복 / validation. 공식 test는 학습하지 않음 |
| 신경망 | 기존 `BinarySmallCNN` | Adam, lr=0.001, batch=64, CrossEntropyLoss |

**같은 데이터셋, seed, 노출 epoch 수에서** 편향 강도 p만 달리한 실험들을 비교한다. 각 조건은 새 CNN을 만들지만 동일한 초기 가중치에서 시작한다. 편향 학습 후 CNN 가중치는 유지하면서 **Adam의 누적 상태만 초기화**한다.

### 핵심 지표

같은 손글씨 원본을 색상 3, 색상 8 두 가지 버전으로 평가한다.

- `aligned_accuracy = A`: 숫자에 대응되는 색으로 평가한 정확도
- `conflict_accuracy = C`: 반대 색으로 평가한 정확도
- `mean_color_accuracy = M = (A+C)/2`: 두 색에서의 평균 정확도 (`neutral_accuracy`도 동일)
- `shortcut_gap = G = A-C`
- `flip_rate = F`: 색만 바꾸었을 때 예측한 숫자가 달라지는 원본의 비율
- `all_colors_accuracy = Q`: **두 색 모두**에서 숫자를 맞힌 원본의 비율. 이진분류에서는 `Q = M-F/2`가 성립한다.
- `recovery_score = S = M × (1-|G|) × (1-F)`: 기존 합성 지표. **백분율이 아님.**
- **회복률** `R_Q(step) = 100 × (Q_step-Q_0) / (Q_ref-Q_0)`.
  - `Q_0`: 해당 p 모델의 노출 종료 직후 Q
  - `Q_ref`: **같은 데이터셋·seed·노출 epoch 수·평가 split**에서 p=0.5로 노출/회복 학습한 대조군의 최종 Q
  - `Q_ref-Q_0 <= 0.001`이면 회복률이 불안정하거나 방향이 반대여서 **미정의**(CSV 빈칸, 사유 별도 열)
  - p=0.5 자체는 대조군이므로 R_Q 없음. 음수 및 100% 초과의 R_Q는 삭제하지 않는다.

`R_Q`는 연구에서 정의한 **대조군 기준 상대 회복률**이다. 어떤 모델의 일반적인 분류 정확도가 100%라는 뜻이 아니고, 실제 사회적 편향이 그만큼 제거되었다는 의미도 아니다.

## 2. 실행

기존 저장소의 루트(`experiment.py`, `model.py`, `experiment_data.py` 등이 있는 폴더)에 세 파일을 추가한다. Python 3.10 이상을 이용한다.

```powershell
python -m pip install -r requirements-experiment.txt
# 새 코드가 옵션을 제대로 인식하는지 확인
python binary_step_experiment.py --help
# 실제 원본을 사용한 소규모 검증 (정식 결과 아님)
python binary_step_experiment.py --smoke
# 정식: MNIST + EMNIST Digits × 3 seed × 3 p × 4 exposure length
python binary_step_experiment.py
```

정식 기본값은 **72개 학습 경로**다(데이터셋 2 × seed 3 × p 3 × 노출 epoch 수 4). 각 경로는 노출 학습 + 회복 10 epoch이며, **회복의 모든 optimizer 업데이트 이후 validation 평가**를 수행하므로 이전 `experiment.py`보다 훨씬 오래 걸린다. 특히 EMNIST는 원본 수가 많아서 부담이 크다. CPU에서 정식 전체 실험을 바로 실행하기보다 먼저 smoke 테스트를 권장한다. 첫 다운로드에는 인터넷이 필요하다.

독립된 폴더에 데이터셋별로 실험하려면 다음과 같이 실행한다.

```powershell
# MNIST만 정식 수행 (새 폴더에 저장; 이후 다른 설정을 섞으면 안 됨)
python binary_step_experiment.py --datasets mnist --output experiment_results/binary_step_mnist

# EMNIST Digits만 정식 수행
python binary_step_experiment.py --datasets emnist_digits --output experiment_results/binary_step_emnist

# 작은 조건 한 개를 미리 실행하고 싶다면 p=0.5 대조군을 반드시 포함
python binary_step_experiment.py --datasets mnist --seeds 42 --p-values 0.5 0.99 --exposure-epochs 1 --recovery-epochs 2 --output experiment_results/binary_step_pilot

# 데이터셋이 이미 다운로드되어 있으면
python binary_step_experiment.py --offline
```

**중단 후 같은 명령을 다시 실행하면 저장된 체크포인트에서 재개**한다. `--checkpoint-every-steps 25`가 기본값이며, 매 epoch 종료와 노출→회복 전환 시에도 체크포인트를 기록한다. 따라서 정상 종료 이전 강제 종료에서는 마지막 체크포인트 이후의 학습·평가 최대 24 step을 재실행할 수 있다. 저장된 결과를 중복으로 기록하지는 않는다. 이 옵션을 바꾼 재실행은 프로토콜 불일치로 거부하므로 **같은 값을 유지**한다.

```powershell
# 학습 없이 예측값 검산 후 CSV·그래프 재생성
python binary_step_experiment.py --analyze-only

# 완료된 정식 실험의 결과만 GitHub에 커밋하고 push (기존 experiment.py와 동일)
python binary_step_experiment.py --publish
```

`--analyze-only`는 기본 `experiment_results/binary_step_full`에 동작한다. 별도 결과 폴더를 사용했다면 해당 `--output`도 지정해야 한다. `--publish`는 완전히 끝나고 검산된 결과만 다루며, 로컬 Git 인증/remote 환경이 필요하다. 용량과 파일 수가 상당히 커질 수 있으므로 업로드 전 저장 공간과 Git 리포지토리 크기를 확인한다. `--smoke` 결과는 정식 논문 결과로 사용하지 않으며 publish도 허용하지 않는다.

## 3. 저장되는 데이터

정식 기본 출력 루트는 `experiment_results/binary_step_full/`이다. 기존 결과 폴더나 구형 Python 파일은 건드리지 않는다.

```text
binary_step_full/
├── protocol.json                        # 모든 실험 설정·코드/환경 SHA256
├── provenance/code/                     # 실험 당시 원본 Python 코드 보관
├── raw_sources/mnist/                   # 기존처럼 공식 MNIST 원본 이미지/라벨 및 중복 검토
├── raw_sources/emnist_digits/           # EMNIST Digits 공식 원본 이미지/라벨 및 중복 검토
├── raw_sources/{dataset}_38/selection.json
├── shared/{dataset}_38/seed42/
│   ├── design.npz                       # 원본 ID/분할/HSV jitter/색상 배정/학습 순서
│   ├── color_counts.json
│   └── initial_model.pt
├── runs/{dataset}_38/seed42/exp01/p099/
│   ├── config.json
│   ├── latest.json                       # 재개용 검증된 마지막 checkpoint 포인터
│   ├── checkpoints/step_000XXXX.pt      # 가중치·Adam·난수·진행 위치
│   ├── raw/predictions/
│   │   ├── validation_block_*.npz       # 여러 consecutive step의 예측을 묶어 저장
│   │   └── test_step*.npz               # 최초/최종 공식 test의 예측
│   ├── metrics.json                     # 매 recovery step 원시 지표
│   ├── test_metrics.json
│   ├── train_log.csv                    # epoch별 학습 loss/정확도
│   └── status.json
├── raw/observations.csv                 # 모든 데이터셋·조건·평가 시점의 관측값
├── processed/
│   ├── derived_results.csv              # 매 step의 R_Q/원시 지표/참조 기준
│   ├── validation_summary.csv           # seed 평균·표본 SD·유효 n
│   ├── test_summary.csv
│   ├── key_results.csv                  # 회복 시작 시점 및 최종 시점 요약
│   ├── per_digit_results.csv            # 숫자 3,8 각각의 Q
│   ├── recovery_times.csv               # R_Q 90% 최초 도달 step 구간
│   └── verification.json                # 저장 예측값→지표 독립 재계산 결과
├── figures/                             # p 비교 및 exposure length 비교 그래프, PNG/SVG
├── analysis_brief.md                    # 초록 작성/해석 주의용 안내
├── file_manifest.json                   # 저장된 파일 크기·SHA256
├── progress.json
└── status.json                          # 완료: status="complete"
```

CSV는 기존 코드와 같이 UTF-8 BOM으로 작성하여 Excel에서 읽을 수 있다. 이미지 원본은 기존 로더에 맞춘 공식 데이터 파일로 저장하며, 색칠된 모든 이미지를 반복적으로 저장하지 않고 원본·색상 배정을 통해 재구성한다.

**원시 예측값은 매 step 보관**하되, 파일 폭증을 줄이기 위해 기본 25 step씩 묶는다. `raw/observations.csv`의 `predictions_file`, `predictions_index`를 통해 해당 step의 예측 배열을 찾을 수 있다. `test`는 전체 학습이 끝난 뒤 각 경로의 `step=0`과 최종 시점에서만 평가한다.

## 4. 주의사항

1. 완전한 이진분류에서는 flip_rate와 shortcut_gap이 자주 비슷하게 움직일 수 있으나 **두 지표가 항상 같은 것은 아니다.** 부호가 음수일 때나 양쪽 색에서 모두 잘못 예측하는 경우를 구분할 수 있다.
2. 노출 epoch 수를 바꾸면 **누적 노출 학습량 자체가 바뀌므로** 회복 step 수 외에도 그 차이를 해석해야 한다.
3. MNIST와 EMNIST Digits는 원본 이미지 수, 필기체 분포, 회복 1 epoch당 step 수가 서로 다르다. 동일한 step 수로 비교한 그래프와 동일한 epoch 수로 비교한 그래프의 해석은 다르다.
4. 회복률은 p=0.5 대조군의 **최종 Q**에 의존한다. 대조군 성능이 충분히 좋아지지 않거나 `Q_ref-Q_0`가 작으면 R_Q는 정의하지 않는다.
5. 테스트용 합성 이미지로 전체 파이프라인과 중단·재개를 확인했지만, **실제 MNIST/EMNIST를 이용한 72경로 본실험 결과는 여기서 생성하지 않았다.**
