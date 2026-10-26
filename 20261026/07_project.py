"""실습 07~08 응용: 한 파일에서 회귀·분류 각각 세 후보를 비교합니다."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 이 표는 전날의 분할 소속과 각 분할 내부의 행 순서(position)를 함께 보존합니다.
split = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = split.loc[split["split"] == "train"].sort_values("position")["request_id"]
valid_ids = split.loc[split["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
features = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train = train[features]
X_valid = valid[features]

print(train)
#             input_chars  allowed_tools task_type priority  elapsed_ms  retry_count status  failed
# request_id                                                                                       
# R009             1412.0            2.0        qa   normal        2349            2  error       1
# R026              895.0            1.0   summary     high         138            0     ok       0
# R074              290.0            5.0  research   normal         885            0     ok       0
# R067             1641.0            1.0  research      low         694            0     ok       0
# R005                NaN            5.0        qa     high        1979            1  error       1
# R093              311.0            1.0  research      low        1545            0     ok       0
# R053             1629.0            3.0        qa     high        1316            2  error       1
# R018                NaN            5.0  research     high        1942            0     ok       0
# R082              908.0            2.0        qa      low        1135            0     ok       0

# test의 ID는 분할표에만 남겨 둡니다. test 입력·정답을 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

# 제공: 전날 만든 전처리입니다. 숫자 2열은 중앙값→표준화, 문자 2열은 최빈값→원핫입니다.
num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
pre = ColumnTransformer([("num", num, features[:2]), ("cat", cat, features[2:])])

# 제공 K: PPT 실습06의 동일 원본·분할·standard 탐색 결과(분류3, 회귀15)를 재사용합니다.
regression_rows = []
classification_rows = []
predictions = []
for name, estimator in [("Dummy", DummyRegressor()), ("Linear", LinearRegression()),
                        ("KNN15", KNeighborsRegressor(n_neighbors=15))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07r: 회귀 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 elapsed_ms입니다.
    model.fit(X_train, train["elapsed_ms"])
    prediction = model.predict(X_valid)
    regression_rows.append({"model": name, "scale": "standard",
        "MAE_ms": mean_absolute_error(valid["elapsed_ms"], prediction),
        "RMSE_ms": root_mean_squared_error(valid["elapsed_ms"], prediction)})
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "regression", "model": name,
        "actual": valid["elapsed_ms"].to_numpy(), "predicted": prediction}))

for name, estimator in [("Dummy", DummyClassifier(strategy="most_frequent")),
                        ("Logistic", LogisticRegression(C=1.0, max_iter=1000)),
                        ("KNN3", KNeighborsClassifier(n_neighbors=3))]:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A07c: 분류 모델을 학습하고 valid 예측을 구하세요.
    # 힌트: 정답 열은 failed입니다.
    model.fit(X_train, train["failed"])
    prediction = model.predict(X_valid)
    tn, fp, fn, tp = confusion_matrix(valid["failed"], prediction, labels=[0, 1]).ravel()
    classification_rows.append({"model": name, "scale": "standard",
        "F1": f1_score(valid["failed"], prediction, zero_division=0),
        "Precision": precision_score(valid["failed"], prediction, zero_division=0),
        "Recall": recall_score(valid["failed"], prediction, zero_division=0), "FP": fp, "FN": fn})
    predictions.append(pd.DataFrame({"request_id": valid.index, "task": "classification", "model": name,
        "actual": valid["failed"].to_numpy(), "predicted": prediction}))

regression = pd.DataFrame(regression_rows)
classification = pd.DataFrame(classification_rows)
out = HERE / "outputs"
out.mkdir(exist_ok=True)

regression.to_csv(out / "regression_comparison.csv", index=False, encoding="utf-8-sig")
classification.to_csv(out / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False, encoding="utf-8-sig")
print("회귀 / valid 20건")
print(regression.round(3).to_string(index=False))
print("분류 / valid 20건")
print(classification.round(3).to_string(index=False))
print("저장: outputs의 비교표 2개와 validation_predictions.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")

# 회귀 / valid 20건
#  model    scale  MAE_ms  RMSE_ms
#  Dummy standard 541.260  658.745
# Linear standard 551.797  639.996
#  KNN15 standard 530.063  641.788
# 분류 / valid 20건
#    model    scale    F1  Precision  Recall  FP  FN
#    Dummy standard 0.000      0.000   0.000   0   6
# Logistic standard 0.333      0.333   0.333   4   4
#     KNN3 standard 0.545      0.600   0.500   2   3