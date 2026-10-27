"""실습 07 심화: 전날 모델과 트리·앙상블의 분류·회귀 실험표를 생성합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.

import json
import platform
import importlib.metadata as metadata
import joblib
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.inspection import permutation_importance
from preprocessing import build_model
from evaluate import classification_metrics, regression_metrics

# 제공: 후보 순서는 CV 동률일 때도 사용합니다. test·valid를 보고 후보를 더 늘리지 않습니다.
classifiers = [("Dummy", DummyClassifier(strategy="most_frequent"), False),
               ("Logistic", LogisticRegression(C=1, max_iter=1000), True),
               ("KNN3", KNeighborsClassifier(n_neighbors=3), True),
               ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42), False),
               ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1), False),
               ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1), False)]
regressors = [("Dummy", DummyRegressor(strategy="mean"), False),
              ("Linear", LinearRegression(), True),
              ("KNN15", KNeighborsRegressor(n_neighbors=15), True),
              ("Tree", DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42), False),
              ("Forest", RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                  max_features=1.0, random_state=42, n_jobs=1), False),
              ("XGB", XGBRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="reg:squarederror", eval_metric="rmse",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1), False)]
class_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
reg_folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
class_rows, reg_rows, cv_rows, prediction_rows = [], [], [], []
fitted_class, fitted_reg = {}, {}
for name, estimator, scale in classifiers:
    model = build_model(estimator, scale=scale)
    # 제공: CV에 전처리 전 X와 Pipeline을 전달합니다. fold 내부에서 전처리를 학습합니다.
    scores = cross_val_score(model, X_train, y_train, cv=class_folds, scoring="f1", n_jobs=1)
    # TODO C07a: 분류 후보를 train 전체로 학습하고 valid를 예측하세요.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    # TODO 끝
    probability = model.predict_proba(X_valid)[:, list(model.classes_).index(1)]
    class_rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                       "scale": "standard" if scale else "none",
                       "train_F1": classification_metrics(y_train, model.predict(X_train))["F1"],
                       **classification_metrics(y_valid, prediction)})
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"task": "classification", "model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "metric": "F1", "score": value})
    for request_id, actual, predicted, prob in zip(X_valid.index, y_valid, prediction, probability):
        prediction_rows.append({"request_id": request_id, "task": "classification", "model": name,
                                "actual": float(actual), "prediction": float(predicted), "probability_failed": float(prob)})
    fitted_class[name] = model
for name, estimator, scale in regressors:
    model = build_model(estimator, scale=scale)
    # 제공: 회귀는 fold별 RMSE를 평균합니다. 분류 F1과 합쳐 순위를 만들지 않습니다.
    scores = -cross_val_score(model, X_train, train["elapsed_ms"], cv=reg_folds,
                              scoring="neg_root_mean_squared_error", n_jobs=1)
    # TODO C07b: 회귀 후보를 실행시간 정답으로 학습하고 valid를 예측하세요.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    # TODO 끝
    reg_rows.append({"model": name, "CV_RMSE_ms": scores.mean(), "CV_std_ms": scores.std(),
                     "scale": "standard" if scale else "none", **regression_metrics(valid["elapsed_ms"], prediction)})
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"task": "regression", "model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "metric": "RMSE_ms", "score": value})
    for request_id, actual, predicted in zip(X_valid.index, valid["elapsed_ms"], prediction):
        prediction_rows.append({"request_id": request_id, "task": "regression", "model": name,
                                "actual": float(actual), "prediction": float(predicted), "probability_failed": None})
    fitted_reg[name] = model

# 제공: 모델 선택은 train 내부 CV로만 합니다. valid 결과는 선택된 조건의 별도 진단으로 보고합니다.
classification = pd.DataFrame(class_rows)
regression = pd.DataFrame(reg_rows)
chosen_class = classification.loc[classification["CV_F1"].idxmax(), "model"]
chosen_reg = regression.loc[regression["CV_RMSE_ms"].idxmin(), "model"]
classification.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
regression.to_csv(OUT / "regression_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(prediction_rows).to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")

# 제공: 중요도는 선택 모델과 별개로 Forest를 진단합니다. 이름에 모델과 계산 방법을 남깁니다.
forest = fitted_class["Forest"]
mdi = pd.DataFrame({"feature": forest["pre"].get_feature_names_out(),
                    "importance": forest["model"].feature_importances_})
permutation = permutation_importance(forest, X_valid, y_valid, scoring="f1", n_repeats=10, random_state=42, n_jobs=1)
permutation_table = pd.DataFrame({"feature": FEATURES, "importance": permutation.importances_mean,
                                  "repeat_std": permutation.importances_std})
# MDI의 학습8열과 순열의 원본4열은 의미·단위가 다릅니다. method와 data를 함께 저장합니다.
importance = pd.concat([mdi.assign(model="Forest", method="MDI", data="train60", repeat_std=float("nan")),
                        permutation_table.assign(model="Forest", method="permutation_F1", data="valid20")], ignore_index=True)
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")

# 제공: 각 fold의 학습/평가 ID 기록. 보관한 test ID는 여기에 들어가지 않습니다.
fold_rows = []
for task, folds in [("classification", class_folds), ("regression", reg_folds)]:
    for number, (fit_positions, eval_positions) in enumerate(folds, start=1):
        for role, positions in [("fit", fit_positions), ("evaluate", eval_positions)]:
            for position in positions:
                fold_rows.append({"task": task, "fold": number, "role": role, "request_id": X_train.index[position]})
pd.DataFrame(fold_rows).to_csv(OUT / "fold_manifest.csv", index=False, encoding="utf-8-sig")
selection = {"classification": chosen_class, "regression": chosen_reg,
             "rule": "classification: max train-CV mean F1; regression: min train-CV mean RMSE; ties: listed order",
             "classification_candidates": [x[0] for x in classifiers], "regression_candidates": [x[0] for x in regressors],
             "features": FEATURES, "seed": 42, "folds": 5, "test_evaluated": False,
             "model_parameters": {name: {key: value for key, value in estimator.get_params().items()
                                       if key in ["strategy", "C", "max_iter", "n_neighbors", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "objective", "eval_metric", "tree_method", "device", "random_state", "n_jobs"]}
                                  for name, estimator, _ in classifiers},
             "regression_model_parameters": {name: {key: value for key, value in estimator.get_params().items()
                                                  if key in ["strategy", "n_neighbors", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "objective", "eval_metric", "tree_method", "device", "random_state", "n_jobs"]}
                                             for name, estimator, _ in regressors}}
(OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
environment = {"python": platform.python_version(), **{name: metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]}}
(OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
# 직접 생성한 파일만 같은 환경에서 restore.py로 복원합니다. 외부 joblib/pickle 파일은 읽지 않습니다.
joblib.dump({"classification": fitted_class[chosen_class], "regression": fitted_reg[chosen_reg]}, OUT / "selected_models.joblib")
print("분류 비교")
print(classification.round(4).to_string(index=False))
print("회귀 비교")
print(regression.round(3).to_string(index=False))
print("CV로 선택한 분류:", chosen_class, "/ 회귀:", chosen_reg)
print("중요도: Forest 진단용 / test20은 학습·변환·예측·평가하지 않았습니다.")
print("저장 완료: outputs/의 비교표·CV·중요도·선택·예측·환경·모델")
