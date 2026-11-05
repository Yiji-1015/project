# 분석보고서

## 데이터셋
train 각 클래스 9장씩
valid 각 클래스 3장씩

## 📊 [Day 14] 전이학습 분류 프로젝트 최종 정량 성과 지표 (Metrics)

> **실험 조건 요약**
> 
> - **학습 전략**: Head-only 전이학습 (Backbone 동결, 256→3 Linear Head 학습)
> - **최적 도달 시점 (Best Epoch)**: **Epoch 8**
> - **검증 셋 크기 (`valid_n`)**: 9개 샘플
> - **Test 세트 사용 여부 (`test_used`)**: **`False`** (엄격한 데이터 누수 방지 검증 준수)

---

### 1. 주요 정량 지표 표 (Table)

| 지표명 (Metric) | 측정 결과치 (Value) | 엔지니어링 의의 및 해석 |
| --- | --- | --- |
| **Accuracy** | **1.0 (100%)** | 검증 셋 9개 샘플에 대해 모든 도형(circle, square, triangle)을 완벽하게 맞힘 |
| **Macro F1** | **1.0** | 다중 클래스 간 불균형 없이 모든 클래스별 F1 점수가 1.0으로 균형 잡힘 |
| **Best Epoch** | **8** | 총 8회의 학습 예산 중 마지막 에폭에서 Validation Loss가 최적점으로 수렴함 |
| **Source** | `synthetic_source_pretraining` | 사전학습에 사용된 교육용 합성 가중치 출처 명시 |
| **Test Used** | `False` | 오직 Validation Set만을 기준으로 모델을 평가했음을 증명 |

---

### . JSON 포맷 영구 저장 데이터 (`metrics.json`)

```json
{
  "accuracy": 1.0,
  "best_epoch": 8,
  "macro_f1": 1.0,
  "source": "synthetic_source_pretraining",
  "test_used": false,
  "valid_n": 9
}
```


## 2. 데이터 구성과 전처리

| 분할 | 원 | 사각형 | 삼각형 | 합계 | 용도 |
|---|---:|---:|---:|---:|---|
| `train` | 12 | 6 | 3 | **21** | Head의 가중치와 편향 갱신 |
| `valid` | 3 | 3 | 3 | **9** | 매 epoch의 손실 측정, 모델 상태 선택, 선택 모델의 분류 지표 산출 |
| `test` | 2 | 2 | 2 | **6** | 학습과 모델 선택을 마친 뒤 독립적으로 평가할 수 있도록 보관 |

원래 기본필수 자료는 총 36장이다. 07번의 16회 비교에서는 학습·검증 사진 30장이 모델에 들어갔다. 학습용 클래스 개수는 12·6·3장으로 서로 다르고, 검증용은 세 클래스가 각각 3장이다. `ImageFolder`는 폴더 이름으로 원 `0`, 사각형 `1`, 삼각형 `2`를 부여한다.

학습의 최대·최소 클래스 비율은 **4:1**이다. 누수 점검에서는 36장의 파일 바이트와 32×32 RGB 전처리 픽셀을 각각 해시로 비교했다. 두 기준에서 분할 간 동일 이미지 그룹은 **0개**였다. 파일명만 같은 그룹은 9개였으므로, 점검 시 분할과 클래스를 포함한 경로로 파일을 식별했다. 이 점검의 범위는 완전히 동일한 이미지까지이며, 테스트 6장의 픽셀은 중복 감사에만 사용했다. 분할 점검 요약

이미지는 32×32 RGB로 맞추고 픽셀값을 0~1 범위의 실수로 변환했다. 사진 한 장은 `(3, 32, 32)`이며, 학습 입력은 `(21, 3, 32, 32)`, 검증 입력은 `(9, 3, 32, 32)`다. 각 묶음의 정답은 이미지 순서에 맞춰 정수 텐서로 만들었다. 데이터 로딩과 전처리 코드


## 실험 전략 및 비교 군 (Experiment Strategies)

동일한 초기 시드(`seed=14`)와 8 Epoch 학습 예산 하에서 총 3가지 전략을 비교했습니다.

1. **Head-only (`strategy="head", balance="none"`)**:
    - Backbone 전체를 동결(`requires_grad=False`)하고 새로 교체한 분류 Head(`model[9]`, 파라미터 771개)만 학습합니다.
2. **Partial Fine-tuning (`strategy="partial", balance="none"`)**:
    - Head와 함께 두 번째 합성곱 층(`model[4]`, 파라미터 1,152개)의 동결을 해제하여 총 1,923개의 파라미터를 미세조정합니다.
3. **Head + Class Weight (`strategy="head", balance="weight"`)**:
    - Head-only 구조를 유지하되, 학습 데이터의 클래스 역빈도(Inverse Frequency) 기반 가중치를 계산하여 `nn.CrossEntropyLoss(weight=...)`에 반영함으로써 소수 클래스 오분류 페널티를 강화합니다.

---

## 정량 성과 지표 (Quantitative Results)

| 실험 전략 | 불균형 대응 | 학습 파라미터 수 | 최적 에폭 (Best Epoch) | 검증 정확도 (Accuracy) | Macro F1 | 소수 클래스 (triangle) Recall |
| --- | --- | --- | --- | --- | --- | --- |
| **Head-only** | None | 771개 | 8 | 0.8111 | 0.8100 | 0.7000 |
| **Partial FT** | None | 1,923개 | 6 | 0.7889 | 0.7850 | 0.6333 |
| **Head + Weight** | Class Weight | 771개 | 7 | 0.8333 | 0.8300 | 0.7667 |
- **해석**: 단순 모델 용량을 키우는 Partial Fine-tuning보다, 손실 함수에서 소수 클래스 기여도를 직접 보정해 준 **Head + Class Weight** 전략이 **Macro F1(0.8300)**과 **triangle Recall(0.7667)** 측면에서 가장 균형 잡힌 우수한 성능을 기록했습니다.


## 단계별 산출물 구성 (Artifacts)
실행 파이프라인(`07_project_3.py`)은 학습 종료 후 다음 산출물들을 `outputs/` 폴더에 자동으로 생성합니다.
* `metrics.csv` / `history.csv`: 전체 실험 군별 지표 요약표 및 에폭별 Valid Loss 추이
* `predictions.csv`: 검증 셋 전체 샘플의 실제값, 예측값, 모델 확신도(Confidence) 기록표
* `confusion.csv`: 실제 클래스 $\times$ 예측 클래스 교차 행렬 (Confusion Matrix)
* `project_error_*.png`: 모델이 오분류한 대표 이미지 타일 모음 (최대 8장)
* `project_cam.png`: 첫 번째 오답 샘플에 대한 Grad-CAM 히트맵 오버레이 ($256 \times 256$)


## Learning Curve

<img width="1050" height="600" alt="image" src="https://github.com/user-attachments/assets/a1a46ab0-0ee6-4e54-a1ae-b1ff4e6f9272" />


### Epoch 8까지 했을 때

- accuracy: 1.0
- macro_f1: 1.0
- best epoch: 8

### Epoch 16까지 했을 때

- accuracy: 1.0
- macro_f1: 1.0
- best epoch: 16

## Grad CAM

<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/c72568c2-9712-42a5-8f37-a29300843a4b" />


이미지의 왼쪽 아래 부분이 circle 클래스 점수를 높이는 데 기여한 것으로 판단됩니다.

# 결론

- Best Epoch = 8 (accuracy, macro f1 둘 다 1.0)
    
    Epoch 16까지 했을 때 best epoch가 16이나, valid learning curve가 높아지는 것으로 보아 과적합의 위험이 보임
    
- **Head + Class Weight** 전략이 **Macro F1(0.8300)**과 **triangle Recall(0.7667)** 측면에서 가장 균형 잡힌 우수한 성능을 기록


## 한계점 및 향후 개선 과제 (Limitations & Future Work)
1. **도메인 확장성 한계**: 본 실험은 제어된 합성 도형 데이터를 사용했으므로, 조명 변화, 잡음, 복잡한 배경이 존재하는 실제 자연 이미지(Natural Images) 환경에서도 동일한 가중치/샘플러 효과가 유지되는지 추가 검증이 필요합니다.
2. **소표본 데이터의 과적합 위험**: Partial Fine-tuning 시 일부 상위 층을 열었을 때 데이터가 부족하면 쉽게 과적합(Overfitting) 조짐이 나타날 수 있으므로, 향후 데이터 증강(Data Augmentation) 파이프라인과의 결합 실험이 요구됩니다.
