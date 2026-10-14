# 3·8 이진분류 편향 회복 실험 — 2차 변경안

`binary_step_experiment_v2.py`는 기존 [`jyc6704/-1`](https://github.com/jyc6704/-1) 프로젝트의 실험 구조를 바탕으로 작성한 **별도 실행 파일**입니다. 기존 `experiment.py`, `recovery_experiment.py`, `binary_step_experiment.py`는 수정하지 않습니다. 이 파일은 프로젝트 루트의 기존 `experiment_data.py`, `experiment_io.py`, `model.py`를 사용합니다.

## 1. 최종 확정 조건

| 항목 | 정식 실행의 기본값 |
|---|---|
| 데이터셋 | MNIST, EMNIST Digits (각각 숫자 3·8만 사용) |
| 노출 강도 p | 0.50, 0.80, 0.90, 0.99 |
| 노출 길이 | 4, 8, 16, 24, 32 epoch |
| 회복 길이 | 5 epoch |
| 반복 seed | 42, 43, 44 |
| batch size | 64 |
| 원본 분할 | 노출 40%, 회복 40%, validation 20% (숫자별 층화 분할) |
| 노출 데이터 색상 | 같은 숫자 원본은 노출 epoch 사이에서 색상을 고정 |
| 회복 데이터 색상 | 원본별 2색 균형 배정, epoch마다 다시 추출, 조건 사이에 공유 |
| 모델/최적화 | `BinarySmallCNN`, Adam, LR 0.001, CrossEntropyLoss |
| 노출→회복 전환 | CNN 가중치 유지, Adam 상태 초기화 |
| 평가 | **회복 0시점 포함; 첫 회복 epoch 매 step; 이후 epoch는 5 step마다 + 각 epoch의 끝** |
| 검증 | 동일 validation 원본에 색상 2개를 각각 적용해 예측 비교 |

전체 정식 기본값은 2 데이터셋 × 3 seed × 5 노출 길이 × 4 p = **120개 경로**입니다. 각 경로에서 학습은 5 recovery epoch 모두 연속 수행합니다. 평가 간격만 2 epoch부터 달라집니다.

**1 optimizer step = 최대 64장의 학습 이미지를 처리하고 가중치를 1회 업데이트하는 과정**입니다. 마지막 batch는 64장보다 적을 수 있습니다.

1 recovery epoch당 회복 업데이트 수는 `ceil(num_recovery/64)`로 결정됩니다. 예를 들어 MNIST의 회복 데이터가 4,792장이면 75 step이며, 5 epoch는 375 step입니다. 첫 회복 epoch는 0,1,2,…,75에서 측정하고, 다음 epoch는 80,85,…,150에서 측정합니다. 1 epoch의 업데이트 수가 5의 배수가 아니어도 마지막 step을 반드시 측정합니다.

## 2. Windows PowerShell에서 실행

1. ZIP에서 `binary_step_experiment_v2.py`와 `test_binary_step_experiment_v2.py`를 꺼내 기존 GitHub 저장소 폴더 (`model.py`, `experiment_data.py` 등이 있는 폴더)에 넣습니다. **기존 파일을 대체하지 않습니다.**
2. 파일 탐색기에서 프로젝트 폴더 주소창에 `powershell`을 입력하여 PowerShell을 엽니다.
3. 아래 명령을 순서대로 실행합니다.

```powershell
python -m pip install -r requirements-experiment.txt
python -m unittest test_binary_step_experiment_v2 -v
python binary_step_experiment_v2.py --smoke
```

`--smoke`는 실제 원본 데이터의 일부만 사용하는 사전 작동 검증입니다. 실제 정식 결과로 사용하지 않습니다. `--smoke`의 노출 길이·회복 길이는 정식 실험보다 작습니다.

정식 **MNIST만** 먼저 실행:

```powershell
python binary_step_experiment_v2.py --datasets mnist --output experiment_results/binary_step_v2_mnist
```

정식 **MNIST와 EMNIST 모두** 실행:

```powershell
python binary_step_experiment_v2.py
```

기본 저장 폴더는 `experiment_results/binary_step_v2_full/`입니다. 이전 실험(`experiment_results/full/`, `experiment_results/binary_step_full/`)에는 저장하지 않습니다.

시간·용량이 부담된다면 먼저 MNIST만 진행해 실제 속도를 확인하세요. **전부 실행할 경우 120개 경로에 수만 번의 전체 validation 추론이 필요해 CPU에서는 상당히 오래 걸릴 수 있습니다.** 중간 종료되더라도 같은 명령과 같은 환경으로 재개할 수 있습니다.

## 3. 저장 결과

- `protocol.json`: 실험 조건, 환경, 코드 파일 SHA-256 및 평가 간격.
- `provenance/code/`: 실행 시 사용된 소스 코드 사본.
- `raw_sources/`: 공식 원본 데이터, 원본 ID, 중복 감사 및 선택 내역.
- `shared/`: 공통 데이터 분할, 색상·hue 배정, 원본 학습 순서, 초기 모델.
- `runs/<dataset>_38/seed*/exp*/p*/`: 경로별 모델 체크포인트, 상태, step별 예측 청크, 학습 로그.
- `raw/observations.csv`: 평가를 **실제로 실시한** step의 Q/M/G/F/S 및 `recovery_samples_seen`, `actual_recovery_epoch`.
- `processed/derived_results.csv`: seed별 R_Q 및 동일 시점의 중립 대조군 Q 차이.
- `processed/validation_summary.csv`, `processed/test_summary.csv`: seed 평균, 표본 표준편차.
- `processed/key_results.csv`: step 0 및 종료 시점 핵심값.
- `processed/recovery_times.csv`: R_Q ≥ 90이 처음 관측된 시점 및 **직전 측정 시점**(미측정 step을 임의로 채우지 않음).
- `figures/`: p 및 exposure 길이에 따른 지표 비교 그래프(PNG/SVG).
- `processed/verification.json`, `status.json`, `file_manifest.json`: 관측 누락·원시 예측 검산·완료 상태·해시.

지표: `Q`=동일 원본의 두 색 모두에서 정답인 비율, `M`=두 색 평균 정확도, `G`=정렬 색 정확도−충돌 색 정확도, `F`=색 변화로 예측이 달라진 비율, `S=M(1−|G|)(1−F)`. 이진분류이므로 `Q=M−F/2`가 성립합니다. `R_Q = 100×(Q_t−Q_0)/(Q_ref−Q_0)`이며, `Q_ref`는 같은 데이터셋·seed·노출 길이에서 **p=0.5 대조군의 최종 Q**입니다. 이는 절대 정확도도 아니고 S 점수 자체도 아닙니다.

## 4. 중단 및 재개

같은 설정의 같은 명령을 다시 실행하면 저장된 체크포인트부터 재개합니다. 환경, 코드, 실험 조건이 달라졌으면 서로 다른 결과로 간주하여 같은 폴더에 이어 붙이지 않도록 설계했습니다.

```powershell
# 분석값과 그래프만 다시 만들기 (학습 없음)
python binary_step_experiment_v2.py --analyze-only
```

`--analyze-only`는 기본 결과 폴더의 데이터를 분석합니다. MNIST만 다른 폴더에 돌렸으면 `--output experiment_results/binary_step_v2_mnist`를 함께 입력하세요.

## 5. 검증 범위와 한계

- 코드 문법 검사, 기본값·평가 간격·지표 계산 단위 테스트를 수행했습니다.
- 합성 이미지로 실제 훈련/관측/체크포인트 저장/최종 test·CSV·그래프 생성 및 중단 후 재개를 실행해 지표 일치를 확인했습니다.
- 이는 **합성 데이터 기반 동작 검증**이지 실제 MNIST·EMNIST 전체 120경로 실험 결과가 아닙니다.
- 5 epoch 안에 회복 기준을 충족하지 못했다고 회복 불가능이라고 결론 내려선 안 됩니다.
- 노출 epoch에 따른 초기 편향(G, F, Q) 차이도 같이 확인해야 하며, seed 간 SD는 신뢰구간과 동일하지 않습니다.
