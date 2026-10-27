"""제공 모듈: 앞 시간의 평가지표를 두 함수로 재사용합니다. TODO는 없습니다."""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, mean_absolute_error, root_mean_squared_error, r2_score)

def classification_metrics(actual, prediction):
    """실패1을 양성으로 고정하고, valid의 점수와 오류 건수를 반환합니다."""
    tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(actual, prediction),
            "Precision": precision_score(actual, prediction, zero_division=0),
            "Recall": recall_score(actual, prediction, zero_division=0),
            "F1": f1_score(actual, prediction, zero_division=0),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

def regression_metrics(actual, prediction):
    """실행시간 정답의 MAE·RMSE(ms)와 단위 없는 R²를 반환합니다."""
    return {"MAE_ms": mean_absolute_error(actual, prediction),
            "RMSE_ms": root_mean_squared_error(actual, prediction), "R2": r2_score(actual, prediction)}
 