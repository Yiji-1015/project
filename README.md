# day5_ml · 실행 전 요청 실패 예측의 실험 구조

## 실행
이 `day5_ml` 폴더를 VS Code로 열고 검수 버전의 가상환경을 선택합니다.
역할 폴더에서 이미 만든 환경을 선택해도 되고, 독립 환경이라면 `python -m venv .venv` 후 `.venv`의 Python으로 `-m pip install -r requirements-lock.txt`를 실행합니다.
아래 python은 선택한 가상환경의 Python입니다.
```text
python main.py
python test_cases.py
```
`main.py`는 소스 파일 기준으로 `data/requests.csv`를 읽고 `outputs`를 생성합니다. 다른 현재 작업 폴더에서도 절대 경로로 main.py를 실행할 수 있습니다.
학생 소스의 TODO는 상위 실습 README/함수 docstring에 맞춰 완성합니다.

## 모듈별 책임
| 파일 | 책임 |
|---|---|
| main.py | 원본 읽기 → 입력 선택 → 두 단계 분할 → train 학습 → valid·새 요청 검증 → 결과 연결 |
| preprocessing.py | 수치/범주 처리 경로와 ColumnTransformer/Pipeline 생성 |
| checks.py | 필수 열·금지 열·행 ID·전처리 행렬·학습 상태 확인 |
| provided_io.py | 제공 보조: CSV 읽기, JSON, 직접 생성 모델 저장/복원, 결과 폴더 안전 교체 |
| restore_check.py | 명시적 신뢰 후 별도 프로세스에서 예측·행렬·학습 상태 복원 대조 |
| test_cases.py | 정상·경계·실패 검증. test 부분의 모델 평가는 하지 않음 |

## 생성 결과
`split_manifest.csv`에는 request_id와 split만 저장합니다. 내부 순서는 `split_order.json`에 별도로 남깁니다.
`feature_names.json`은 8개 출력 특성명, `validation_predictions.csv`는 request_id·y_true·y_pred입니다.
`validation_features.csv`는 valid의 20×8 변환 행렬, `validation_metrics.json`은 정확도와 실패 발견 건수입니다.
`preprocessing_state.json`, `run_metadata.json`, `protocol_log.json`, `environment.json`은 상태·원본 해시·분할·호출 범위·환경을 기록합니다.
`new_predictions.csv`, `restore_result.json`, `pipeline.joblib`은 새 요청과 자체 생성 모델의 저장/복원 결과입니다.
모델은 정답 y 없이 새 요청을 예측하며, test는 ID·건수만 확인하고 모델 메서드에 전달하지 않습니다.

기대: train60/valid20/test20, 중앙값[1315,3], 출력8열, valid0.70·실패발견0/6입니다.
입력 파일을 바꾸면 대표값과 특성 수가 달라질 수 있습니다. 기본 결과 검증은 본문 고정 데이터 기준입니다.
전체 결측 train 열은 기본 계약에서 오류로 탐지합니다. 실습 04 심화의 0 보존 정책을 자동 적용하지 않습니다.

## 저장 파일 복원
직접 생성한 파일임을 확인한 후에만 실행합니다.
```text
python restore_check.py --trust-local-model
```
임의의 외부 joblib/pickle을 열지 않습니다. 저장 때와 실행 패키지 버전이 같아야 합니다.
Dummy의 예측은 항상0이므로 예측 일치만으로 복원을 판정하지 않고 변환 행렬·특성명·학습 상태도 대조합니다.

## 결과 폴더와 원본 보존
정상 재실행 시 교재가 생성한 표식이 있는 `outputs`를 교체합니다. 임의 사용자 폴더를 출력 위치로 지정하지 마세요.
원본 입력 검증/학습에 실패하면 기존 결과를 새 결과로 교체하지 않습니다. 비교용 expected_outputs는 코드에서 읽지 않습니다.
실행 환경 잠금 사본은 `outputs/requirements-lock.txt`에 기록합니다. 소스의 잠금 파일과 비교해 제출 환경을 확인하세요.

## 작성자가 채울 내용
아래는 그대로 제출하지 말고 `REPORT.md`, `TEST_LOG.md`와 함께 본인의 근거를 작성하세요.
문제·예측 시점: ______
입력/제외 열의 이유: ______
분할 선택 이유·train-only 확인: ______
valid 결과와 기준선 한계: ______
재실행 환경·개인 기여: ______
Git 이력: ______

실습 08에서 폴더를 새로 만들지 않고 이 프로젝트를 검증합니다. 분할/평가 정책은 다음 모델을 연결해도 유지합니다.
