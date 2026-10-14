# Colab Pro+ — MNIST/EMNIST 3·8 편향 회복 전체 실험

## 시작 (추가 비용 없음, 기존 Colab Pro+ 사용)

1. `SSHS_ColoredMNIST_Colab_ProPlus.ipynb`를 [Google Colab](https://colab.research.google.com/)에서 **파일 → 노트북 업로드**로 연다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택한다. 실제 GPU 종류와 가용성은 Colab에 따라 다르다.
3. `RUN_MODE = "smoke"`로 바꾸고 노트북의 셀을 순서대로 실행하여 백업/복원을 점검한다.
4. 본실험은 `RUN_MODE = "full"`로 변경하고 다시 셀을 실행한다. 기본값은 full이다.
5. 첫 실행에서 Google Drive 연결 권한을 직접 승인한다. 연결이 완료된 후 학습 셀을 실행한다.
6. 셀에 `Starting experiment`가 출력된 뒤 **셀 자체가 실행 중인 상태**라면 브라우저를 닫거나 노트북을 꺼도 된다. Pro+ 백그라운드 실행 여부는 남은 컴퓨팅 유닛과 런타임 상태에 달려 있다.

## 기본 실험 설정

- 데이터셋: MNIST 3·8, EMNIST Digits 3·8 (서로 별도 학습)
- p: `0.5, 0.8, 0.9, 0.99`
- biased exposure: `4, 8, 16, 24, 32` epoch
- neutral recovery: `5` epoch
- seed: `42, 43, 44`
- batch size: 64, 분할: exposure/recovery/validation = 40/40/20
- 관측: 회복 시작 전 `step=0`, 첫 회복 epoch는 모든 optimizer step, 이후 epoch는 5 step마다 + 반드시 각 epoch 마지막 step.
- 지표 Q, M, G, F, S와 Q 기반 상대 회복률 R_Q, validation·최종 test 평가, 기존 CSV/그래프/원본 데이터 기록 모두 유지.

## 백업은 실제로 어떻게 작동하나?

**기본 10분마다**, 그리고 실험 실행이 정상 종료하거나 오류와 함께 중단될 때 **한 번 더** Colab 로컬 결과와 다운로드 데이터 캐시를 Google Drive로 복사한다. **백업 자체는 노트북 브라우저가 아니라 서버의 실행 중인 Python 프로세스가 수행**한다.

- 연산: `/content/SSHS_ColoredMNIST_Workspace/` (Colab의 빠른 로컬 디스크)
- 영구 복원 저장소: `내 드라이브/SSHS_ColoredMNIST_ProPlus/full/` (smoke는 `smoke/`)
- 마지막 백업 상태: `runner_status.json`와 `latest.json`
- 읽기 쉬운 결과: `view_results/` (분석 자료, 핵심 CSV, 그래프 PNG)
- 복원용 원본: `blobs/`, `snapshots/`. **이 폴더의 파일을 직접 수정·삭제하지 말 것.**

Drive 백업은 SHA256으로 각 파일을 검사하고 바뀐 파일만 새 블롭으로 저장한 후, 파일 전부 업로드가 끝났을 때에만 새 스냅샷을 확정한다. 새 스냅샷이 불완전하면 이전 정상 스냅샷으로 복원할 수 있다. 참고로 Drive의 서버 측 동기화 지연·저장공간 부족·쓰기 실패까지 완전히 방지할 수는 없다.

**브라우저를 닫아도 Colab 런타임이 살아 있고 학습 셀이 실행 중이라면 백업은 이어진다.** Colab 런타임이 종료되거나 코드 실행이 중지되면 **학습과 백업이 모두 멈춘다.** 런타임이 갑자기 종료될 경우 직전 10분 내 업데이트(그리고 마지막 로컬 체크포인트 이후 업데이트)는 손실될 수 있다. 재접속 후 다시 셀들을 실행하면 마지막 정상 스냅샷에서 복구한다. 24시간 연속 실행은 보장되지 않는다.

## 중간 종료 뒤 재시작

1. 새 Colab 런타임에 연결한다. 가능한 한 **이전과 같은 GPU 유형**, 동일한 PyTorch 버전을 사용해야 한다.
2. **똑같은 `RUN_MODE`, DRIVE_FOLDER**로 셀을 순서대로 재실행한다.
3. 실행 스크립트가 가장 최근 정상 스냅샷을 로컬 디스크에 자동 복구한 뒤 체크포인트에서 재개한다. 이미 완료된 조건은 다시 학습하지 않는다.
4. GPU/PyTorch/설정 또는 연구 코드 해시가 이전과 다르면 기존 연구의 재현성을 지키기 위해 실행이 거부될 수 있다. 오류가 발생했다면 임의로 강제 재개하지 말고 이전 환경을 복원하거나 별도 출력 폴더에서 새 실험을 시작한다.

## 저장량과 비용

추가 대여료는 없지만 **이미 구독한 Colab Pro+의 컴퓨팅 유닛**을 사용한다. Drive 저장 공간도 별도로 필요하다. MNIST·EMNIST 원본 보관, 캐시, 수많은 raw predictions 및 체크포인트를 모두 보존하므로 실제 용량은 수 GB 이상이 될 수 있다. Drive 사용량을 먼저 확인한다.

모델 체크포인트는 **각 실험 조건의 최신본과 노출 종료 시점**만 남겨 저장 공간을 절약한다. 이 두 시점이면 학습 중단 후 재개와 공식 test의 최초·최종 모델 평가가 가능하다. 중간 step의 Q·F 지표와 예측 raw 파일은 유지한다.

## 파일 설명

- `SSHS_ColoredMNIST_Colab_ProPlus.ipynb`: Colab에서 직접 실행하는 노트북. 단독으로 실행 가능. 포함된 Colab용 코드가 런타임에 자동 설치됨.
- `binary_step_experiment_colab.py`: 기존 `binary_step_experiment_v2.py`에 체크포인트 보관 정책만 추가한 Colab용 실험 코드.
- `colab_backup_runner.py`: 주기적 Drive 백업·오류 감지·스냅샷 복원 실행기.
- 기존 GitHub 저장소의 `model.py`, `experiment_data.py`, `experiment_io.py`는 **고정 커밋** `c4c1042bda3726efd11dacf8f234184f206cf9f4`에서 내려받고 Git blob SHA를 검사함. 저장소 원본은 수정하지 않음.

## 공식 안내

- Colab FAQ: https://research.google.com/colaboratory/faq.html?hl=ko
- Pro+는 컴퓨팅 유닛이 충분할 때 최대 24시간 연속 코드 실행 지원, **실행 시간 보장 아님**
- Drive에는 직접 수많은 작은 파일을 읽고 쓰면 I/O 제한이 발생할 수 있어, 로컬 디스크에서 실험하고 Drive에는 주기적 스냅샷 방식으로 백업함
