# 10-class 회복 실험 실행 및 결과 안내

`experiment.py`가 데이터 준비 → 동일 조건의 짝지은 학습 → 전체 색상 평가 → 지표 검산 →
가공 데이터·도표·분석 자료 저장을 수행한다. 학습 코드는 보조 모듈로 나누었지만 실행은 한 명령이다.
기존 `pilot.py`, `recovery_experiment.py`와 그 결과는 그대로 보존한다.

## 1. 설치와 실행

저장소 폴더에서 실행한다. Python 3.10 이상, PyTorch와 호환되는 torchvision이 필요하다.

```powershell
python -m pip install -r requirements-experiment.txt
python experiment.py
```

첫 실행은 없는 데이터를 자동 다운로드한다. EMNIST는 약 562MB의 공식 통합 ZIP을 내려받아
공식 MD5를 확인한 뒤, 필요한 Digits의 네 IDX 파일만 스트리밍으로 추출한다. 나머지 문자
데이터는 풀지 않는다. 이 프로그램이 새로 받은 ZIP은 추출 성공 후 제거하고, 기존에 있던
ZIP은 보존한다. 네트워크 연결과 디스크 여유를 확보한다. 데이터·결과·Git 이력까지 포함해
최소 2GB의 여유 공간을 권장한다. 기존 `data/MNIST`가 있으면 그대로 사용한다.

이미 CUDA용 PyTorch가 설치되어 있다면 해당 환경에서 실행한다. 설치된 PyTorch가 CPU 전용이면
GPU 장착 여부와 관계없이 CPU로 실행된다. 출력의 `device=...`와 `protocol.json`에서 확인한다.
장치를 명시하려면 `--device cuda` 또는 `--device cpu`를 붙인다. CUDA 요청이 불가능하면 중단한다.
실행 중에는 장치·라이브러리 버전을 바꾸지 않는다.

Windows에서는 프로그램이 실행되는 동안 유휴 절전을 막고 종료 시 원래 상태로 되돌린다.
전원을 연결하고 덮개는 열어둔다. 덮개 닫기, 수동 절전, Windows 재부팅까지 막지는 못한다.
다른 운영체제에서는 운영체제의 절전 설정을 확인한다. 별도 백그라운드 작업은 만들지 않는다.

완료 후 아래 파일을 먼저 연다.

```text
experiment_results/full/analysis_brief.md
experiment_results/full/processed/key_results.csv
experiment_results/full/figures/mnist_recovery.png
experiment_results/full/figures/emnist_digits_recovery.png
```

`analysis_brief.md`는 실제 측정값과 해석 주의점을 정리한 초록 작성용 자료다.
연구 주장을 미리 만들어 내거나 최종 제출 초록을 자동 확정하지 않는다.

## 2. 기본 실험표

| 항목 | 고정 설정 |
|---|---|
| 데이터셋 | MNIST, EMNIST Digits를 별도로 학습·평가 |
| 클래스 | 숫자 0~9, 색상 10개 |
| 편향 강도 p | 0.10, 0.70, 0.80, 0.90, 0.99 |
| seed | 42, 43, 44 |
| 노출 | 2 epoch, 같은 원본의 색 배정은 두 epoch 동안 고정 |
| 회복 | 10 epoch, 숫자별 중립 색 배정을 epoch마다 다시 생성 |
| 데이터 분할 | 중복 정리한 공식 train을 원본 단위·숫자별 층화 40/40/20 노출/회복/validation |
| 공식 test | 학습·validation과 분리; 전체 선택된 학습 완료 후 시작·최종 모델만 평가 |
| 모델 | 기존 SmallCNN: Conv 3→16→32, MaxPool 두 번, Linear 1568→10 |
| 최적화 | Adam, lr=0.001, batch=64, CrossEntropyLoss, 모든 가중치 학습 |
| 전환 | 노출 종료 시 가중치를 유지하고 Adam 상태만 초기화; 중립 대조군도 동일 |
| 색 | HSV의 hue 간격 36°, S=V=1; 원본별 hue ±5° 고정, 회색조 픽셀×RGB |
| 평가 | 같은 원본에 10색을 모두 적용; 원본별 10개 예측 저장 |

총 30개 경로에 중립 대조군 6개가 포함된다. 추가 대조군 학습을 중복 실행하지 않는다.
validation은 경로당 14시점, 총 420행이고 test는 경로당 2시점, 총 60행이다.
seed 요약은 validation 140행, test 20행이다.

실행 순서는 각 seed의 양 데이터셋 p=.1/.99를 먼저 확보하고, .9, .7/.8 순서다.
순서가 결과에 영향을 주지 않도록 모든 분할·색·학습 순서를 별도 난수 일정으로 고정한다.
중간에 종료돼도 이미 완료한 경로와 평가 자료가 남는다. 일부 조건만 끝난 상태를 전체 완료로
표현하지 않는다. 본 실행의 완료 기준은 `status.json`의 `status="complete"`다.

## 3. 1 epoch, 2 epoch, 1/2 epoch의 의미

**1 epoch**는 해당 단계에 배정된 원본을 한 번씩 학습하는 것이다. **노출 2 epoch**는
노출 집합을 두 번 학습한다는 뜻이다. 회복 집합은 노출 집합과 다른 원본으로 구성된다.
노출이 끝난 순간을 **회복 t=0**으로 잡는다. 노출 2 epoch와 회복 2 epoch는 다른 단계다.

**1/2 epoch = 0.5 epoch**는 첫 회복 epoch의 약 절반을 학습한 시점이다. 별도의 모델을
처음부터 학습하지 않고 같은 모델을 잠시 평가한 뒤 이어간다. 첫 회복 epoch에서
0.25, 0.5, 0.75, 1을 관측하고 이후에는 2, 3, …, 10에서 평가한다.
평가 자체에는 역전파나 파라미터 변경이 없고 난수 상태도 복구한다.

평가는 `ceil(비율 × 해당 epoch의 batch 수)` batch 직후 수행한다. 따라서 마지막 작은 batch와
반올림 때문에 정확히 25/50/75%의 원본 수와 다를 수 있다. CSV의 `actual_recovery_epoch`,
`recovery_step`, `recovery_sources_seen`이 실제 학습량이다. 그래프 x축은 예약한 평가 시점이다.
MNIST와 EMNIST는 원본 수가 달라 같은 2 epoch라도 업데이트 수가 다르다.

## 4. 조건 통제와 데이터 누수 확인

- 같은 dataset·seed에서는 모든 p가 원본 분할, 초기 가중치, 학습 순서, 회복 색 배정,
  원본별 hue, 평가 원본을 공유한다. 파일의 해시로 공유 여부를 검산한다.
- 숫자→색상 1:1 대응 π는 seed마다 무작위로 결정하고 p 사이에는 고정한다.
  같은 seed에서는 두 데이터셋에도 같은 π를 적용한다.
- 중립 p=.1에서는 각 숫자 내부 10색의 수 차이가 최대 1개다. p>.1에서는 대응 색을
  `round(n*p)`개, 나머지를 9색에 균등 배정한다. 실제 수는 `color_counts.json`에 남긴다.
- 원본 uint8 픽셀의 SHA256으로 정확히 같은 이미지를 확인한다. 공식 test는 보존하고,
  test와 일치하는 train 원본을 제외한다. 나머지 train 내부 중복은 첫 원본만 남긴 후 분할한다.
  제거 ID·충돌 라벨·실제 수는 `raw_sources/*/selection.npz`, `audit.json`에 남긴다.
- EMNIST의 로더 원본 축을 전치해 방향을 맞추고, 원본 보관 파일은 로더가 읽은 방향을 유지한다.
  `figures/*_source_preview.png`로 실제 숫자 방향을 확인할 수 있다.
- 근접 중복·필기자 중복은 인증하지 않는다. 두 데이터셋의 교집합 비율은 이 실행에서 측정하지
  않는다. 별도 실험이므로 중복이 적다거나 독립적인 외부 검증이라고 주장하지 않는다.
- 공식 test는 학습 조건 선정에 사용하지 않는다. validation 성능을 보고 epoch를 선택하거나
  조기 종료하지 않고 사전에 정한 10 epoch까지 실행한다.

## 5. 지표와 통계

각 원본 i, 색 c에 대해 정답 여부를 z(i,c)로 두고 모든 확률은 0~1로 저장한다.

| 열 | 정의 |
|---|---|
| mean_color_accuracy (M) | 전체 원본×10색의 평균 정확도 |
| all_colors_accuracy (Q) | 한 원본을 10색 모두에서 맞힌 원본의 비율 |
| aligned_accuracy (A) | 숫자에 대응하는 색 π(y)에서의 정확도 |
| conflict_accuracy (C) | 나머지 9색의 평균 정확도 |
| shortcut_gap (G) | A−C; M=(A+9C)/10 |
| flip_rate (F) | 한 원본의 서로 다른 색 쌍 45개에서 예측이 다른 비율을 원본 전체에 평균 |
| recovery_score (S) | M×(1−abs(G))×(1−F); 연구 자체의 보조 합성 지표 |
| q_recovery_percent (R_Q) | 100×(Q_t−Q_0)/(Q_ref−Q_0) |
| q_gap_same_epoch_pp | 100×(같은 시점 중립 대조군 Q−해당 조건 Q), 단위 %p |

Q는 예측 10개에서 직접 구한다. 이진 분류의 `Q=M−F/2` 식을 10-class에 사용하지 않는다.
Q_ref는 같은 dataset·seed·평가 split의 **중립 대조군 회복 10 epoch 종료 Q**로 고정한다.
test에는 test의 reference, validation에는 validation의 reference를 사용한다.

R_Q는 seed별로 비율을 구한 후 평균한다. 먼저 Q를 평균한 뒤 비율을 구하지 않는다.
Q_ref−Q_0가 0.001 이하이면 R_Q를 비워 두고 이유를 기록한다. 중립 대조군에는 R_Q를 부여하지
않는다. 음수나 100 초과를 잘라내지 않는다. R_Q 100%는 초기 reference 격차를 메운 것이며
정확도 100%를 의미하지 않는다. S도 회복 백분율이 아니다.

요약은 seed 평균과 표본 SD(ddof=1)다. 한 seed의 SD는 빈칸으로 둔다. 각 지표별 유효 seed 수를
`*_n`에 보존한다. n=3의 SD는 신뢰구간이 아니다. 색·epoch를 독립 seed로 세지 않는다.
90% 회복 시간은 첫 관측 시점과 직전 관측 시점을 저장하며, 관측 사이 정확한 도달 시간을
알 수 없다고 표기한다. 이후 다시 90% 아래로 떨어졌는지도 저장한다.

## 6. 저장 결과

모든 경로는 `experiment_results/full/` 기준이다. CSV는 UTF-8 BOM으로 저장해 Excel에서도
한글을 읽을 수 있다. NPZ는 `numpy.load`로 여는 압축 배열 파일이다.

| 경로 | 내용 |
|---|---|
| protocol.json | 모든 조건, 라이브러리·장치, 코드 해시 |
| provenance/code/ | 실행 당시 코드 사본 |
| provenance/git.json | 시작 commit과 코드 변경 상태 |
| raw_sources/{dataset}/train_*.npz, test_*.npz | 다운로드한 전체 원본 픽셀·라벨·공식 split 내 ID |
| raw_sources/{dataset}/manifest.json | 원본 파일별 해시·개수·방향 설명 |
| raw_sources/{dataset}/audit.json, selection.npz | 중복 검토와 선택·제외 원본 ID |
| shared/{dataset}/seed*/design.npz | 분할, 대응 π, hue, 모든 epoch의 색과 원본 순서 |
| shared/{dataset}/seed*/initial_model.pt | 공유 초기 모델 |
| runs/{dataset}/seed*/p*/config.json | 경로별 조건·공유 설계 해시 |
| runs/.../raw/predictions/*.npz | 원본×10색의 개별 예측·정답·원본 ID·π |
| runs/.../checkpoints/*.pt | 모델·Adam·난수·batch 위치·누적 학습 기록 |
| runs/.../latest.json | 원자적으로 확정한 최신 checkpoint와 해시 |
| runs/.../train_log.csv | epoch별 학습 loss·정확도·원본 수·업데이트 수 |
| raw/observations.csv | 모든 seed·조건·시점의 관측값, 실제 학습량, 평가 시간 |
| processed/derived_results.csv | seed별 R_Q·reference·동일 시점 격차·미정의 이유 |
| processed/validation_summary.csv, test_summary.csv | 조건별 평균·표본 SD·유효 n |
| processed/key_results.csv | t=0,1,2,10의 핵심 요약; test는 0,10 |
| processed/per_digit_results.csv | 숫자별 Q |
| processed/recovery_times.csv | 90% 회복의 관측 구간 |
| processed/verification.json | 예측→지표 재계산, 공유 설계·원본 해시 및 완성도 검산 |
| figures/ | Q·R_Q·동시점 격차·F·S·M의 전체/첫 epoch 곡선, PNG/SVG |
| analysis_brief.md | 핵심 수치·해석 제한·초록 작성에 필요한 파일 안내 |
| file_manifest.json | 완료 산출물의 파일별 SHA256·크기 |
| status.json, progress.json | 전체 완료·중단·실패 상태와 학습 진행 수 |

원본 픽셀은 20,000장 이하씩 나눠 저장하므로 원본 shard 하나가 비압축 기준 약 16MB 이하이다.
색칠한 이미지 전체를 반복 저장하는 대신 원본과 색 배정을 저장해 정확히 재구성할 수 있다.
GitHub 업로드는 파일당 90MiB 미만인지 확인한다. 전체 결과의 총 용량은 데이터 압축률과
checkpoint 수에 따라 달라진다. `.gitignore`는 정식 결과의 `.pt`까지 추적하도록 예외를 두었다.

## 7. 재실행·복구·분석만 실행

```powershell
# 기본 결과 폴더에서 자동 재개
python experiment.py
# 이미 받은 원본만 사용
python experiment.py --offline
# 모든 결과가 완료된 뒤 학습 없이 검산·표·도표 재생성
python experiment.py --analyze-only
# 별도 반복 실험
python experiment.py --output experiment_results/replicate_02
```

각 epoch 종료와 첫 회복 epoch의 1/4 평가 시점에 checkpoint를 원자적으로 저장한다.
중단되면 마지막 확정된 checkpoint 이후의 작업만 다시 수행한다. 평가 후에도 같은 Adam 상태와
난수 상태로 이어간다. 한 폴더를 두 프로세스가 동시에 사용하면 잠금으로 막는다.
완료된 경로는 재실행해도 다시 학습하지 않는다. 재개 시 코드·설계·환경이 다르면 섞지 않고
중단한다. 기존 실행의 환경과 코드를 복원하거나 새 `--output` 폴더를 사용한다.

`--analyze-only`는 `--output`의 protocol을 읽는다. 완료된 결과라면 파일 해시를 먼저 검사하고
보고서를 재생성한다. 아직 데이터 준비도 끝나지 않은 폴더는 분석할 수 없다. 손상된 파일을
임의로 정상 처리하거나 빠진 조건을 채운 것처럼 표시하지 않는다.

CUDA와 CPU 사이, 라이브러리 버전 사이의 수치적 일치를 보장하지 않는다. 재개는 같은 환경에서
한다. Colab에 옮길 경우 저장소와 출력 폴더가 세션 종료 후에도 보존되게 별도로 백업하고,
장치가 바뀐 실행은 새 output으로 구분한다.

## 8. GitHub에 결과까지 업로드

```powershell
# 실험부터 업로드까지 한 명령
python experiment.py --publish
# 이미 끝난 결과를 검산·재정리하고 업로드
python experiment.py --analyze-only --publish
```

`--publish`는 완료·검산된 **지정 결과 폴더**의 파일만 커밋한 뒤 현재 branch를 origin에 push한다.
Git 설치·인증·커밋 사용자 설정이 필요하다. 다른 변경이 이미 stage에 있으면 섞어 커밋하지 않고
중단한다. 강제 push를 하지 않으며 원격 충돌이나 인증 실패가 나면 로컬 결과는 그대로 남는다.
원격 변경을 먼저 정상적으로 통합한 뒤 같은 명령으로 재시도할 수 있다.
`--publish`를 생략하면 네트워크 push는 하지 않고 로컬 저장소에 모든 결과를 저장한다.

## 9. 작은 사전 확인과 개발 검증

```powershell
# 두 데이터셋을 실제로 로드하되 각 train/test 최대 1,000원본만 사용하는 작은 확인
python experiment.py --smoke
# 이미 MNIST만 있는 환경에서 다운로드 없이 확인
python experiment.py --smoke --datasets mnist --offline
# 네트워크 없는 자동 검증; Git 명령도 설치되어 있어야 함
python -m unittest test_experiment test_recovery_metrics test_recovery_experiment
```

smoke는 seed42, p=.1/.99, 노출2+회복2이며 기본 출력은 `experiment_results/smoke/`다.
전체 원본 보관은 유지하지만 학습·평가 부분집합만 축소한다. 정식 30경로 결과와 혼동하지 않도록
profile과 도표에 smoke를 표시하며 기본 smoke/test 폴더는 Git에서 제외한다.
기본값을 일부 바꾼 실행은 custom으로 표시한다. smoke 실행 뒤 기본 명령을 실행하면 별도
`full/` 폴더에서 정식 실험이 시작된다.

자동 검증은 수식의 알려진 정답, 45색쌍 F, 공유 설계·색 균형, 분할, 평가의 학습 상태 보존,
중간 회복 batch 중단/재개 후 연속 실행과 최종 가중치 일치, 완료 실행의 재학습 방지,
보고서 검산, 로컬 Git 원격으로 결과만 커밋·업로드하는 동작을 검사한다.
작은 검증이나 합성 데이터 테스트의 결과를 정식 연구 결과로 쓰지 않는다.

## 10. 초록 해석에 남는 제한

MNIST와 EMNIST의 비교에는 데이터 규모 외에도 분포·필기자·원본 특성·업데이트 수 차이가 있다.
따라서 성능 차이를 데이터 수만의 인과 효과로 표현하지 않는다. 초기 편향 p만을 주요 조작변인으로
해석하고, 모델 구조·노출 기간·다른 크기의 부분집합 실험은 초록 전 필수 범위에 추가하지 않는다.
색상 편향 회복 결과를 실제 사회적 편향 제거로 일반화하지 않는다.
최종 제출 문서는 실제 완료 수치와 제출 가이드를 대조하여 별도로 작성한다.
