# -1
과제연구1

## 현재 실험: 3·8 binary pilot

`python pilot.py`로 BinarySmallCNN을 p=0.99 biased 데이터에서 정확히 1 epoch 학습합니다.
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
# MNIST 원본이 없는 경우
python pilot.py --download
```

출력은 pilot_results_38의 새 실행 폴더에 저장됩니다.
exposure_01.pt는 모델/optimizer/설정, assignments.pt는 원본 index/색 배정/δ,
results.csv와 result.json은 평가 결과를 포함합니다.
기존 Digit38CNN(1568→64→2) checkpoint는 새 모델과 호환되지 않습니다.

현재 구조: dataset.py(데이터·시각화), model.py(CNN), train.py(학습·평가),
pilot.py(실험 실행), validate_dataset.py(데이터 검증).
