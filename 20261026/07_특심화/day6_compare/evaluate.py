"""지표·학습·검증 담당. 비교표의 점수는 모두 validation 결과입니다."""
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error,root_mean_squared_error,r2_score,
    accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
from preprocessing import build_pipeline
from checks import before_model,check_finite
from course_io import snapshot

def regression_metrics(y,pred):
    """실습02 재사용. MAE_ms, RMSE_ms, R2 dict를 계산한다. 원시 값을 반올림하지 않는다."""
    # TODO D6-07-H1: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-07-H1: 구현 후 다시 실행하세요.")

def classification_metrics(y,pred):
    """실습03 재사용. labels=[0,1]. Accuracy,Precision,Recall,F1,TN,FP,FN,TP 반환.
    실패=1, zero_division=0. 오탐은 FP, 실제 실패를 놓친 것은 FN이다.
    """
    # TODO D6-07-H2: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-07-H2: 구현 후 다시 실행하세요.")

def fit_candidates(candidates,X_train,y_train,X_valid,y_valid,task):
    """{이름:estimator}별로 독립 Pipeline을 학습한다.
    반환 (comparison DataFrame, long prediction DataFrame, fitted_models dict, trace list).
    task는 regression/classification. train-only fit, valid predict, 입력·통계 검사를 포함한다.
    make_records()는 제공 코드다. 학생은 핵심 학습 흐름을 구현한다.
    """
    # TODO D6-07-H3: 바로 위 함수 설명의 입력·반환 계약을 완성하세요.
    raise NotImplementedError("TODO D6-07-H3: 구현 후 다시 실행하세요.")

def make_records(task,name,ids,y,pred,model,X):
    """행 순서를 유지하는 제공 코드. 회귀 잔차는 실제-예측입니다."""
    probability=None
    if task=='classification':
        positive=list(model['model'].classes_).index(1)
        probability=model.predict_proba(X)[:,positive]
    records=[]
    for i,request_id in enumerate(ids):
        row={'request_id':request_id,'split':'valid','task':task,'model':name,
             'y_true':float(y.iloc[i]),'y_pred':float(pred[i]),
             'residual_ms':float(y.iloc[i]-pred[i]) if task=='regression' else None,
             'probability_failed':float(probability[i]) if probability is not None else None}
        records.append(row)
    return records

def select_models(regression,classification):
    """제공: 사전 지표로 정렬. 완전 동점이면 먼저 등록한 단순 후보를 우선합니다."""
    r=regression.sort_values('RMSE_ms',ascending=True,kind='stable').iloc[0]
    c=classification.sort_values('F1',ascending=False,kind='stable').iloc[0]
    return {'regression':{'model':r['model'],'metric':'RMSE_ms','value':float(r['RMSE_ms'])},
            'classification':{'model':c['model'],'metric':'F1','value':float(c['F1'])}}
