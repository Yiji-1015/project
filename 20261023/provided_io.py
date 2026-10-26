"""제공 코드: 파일 경로·입출력·모델 저장·결과 폴더 교체.
데이터 원본과 expected_outputs는 덮어쓰거나 정답으로 읽지 않습니다.
"""
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import math
import os
import shutil
import sys
import uuid
import numpy as np
import pandas as pd
import joblib
from checks import FEATURES,NUM,CAT,validate_features

PACKAGES=["numpy","pandas","scikit-learn","scipy","joblib","threadpoolctl","python-dateutil","pytz","tzdata","six"]

def versions():
    return {"python":sys.version.split()[0], **{k:md.version(k) for k in PACKAGES}}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    """결과 파일에는 비표준 NaN/Infinity 토큰을 허용하지 않습니다."""
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def read_raw(path):
    """학습 원본을 읽고 행별 계약만 검사합니다. 통계는 학습하지 않습니다."""
    frame=pd.read_csv(path,encoding="utf-8",dtype={"request_id":str})
    required={"request_id","failed",*FEATURES}
    if required-set(frame.columns): raise ValueError(f"원본 누락 열: {sorted(required-set(frame.columns))}")
    if frame.empty: raise ValueError("원본이 비어 있습니다.")
    ids=frame["request_id"]
    if ids.isna().any() or ids.str.strip().eq("").any() or not ids.is_unique:
        raise ValueError("원본 요청 ID 결측/중복")
    target=pd.to_numeric(frame["failed"],errors="raise")
    if target.isna().any() or not target.isin([0,1]).all():
        raise ValueError("정답은 관측된 0 또는 1이어야 합니다.")
    frame=frame.set_index("request_id")
    # 사전에 정한 자료형/공백 규칙만 적용합니다. 평균·최빈값은 여기서 계산하지 않습니다.
    selected=validate_features(frame[FEATURES],training=False)
    for col in FEATURES: frame[col]=selected[col]
    frame["failed"]=target.to_numpy(dtype=int)
    if "status" in frame:
        if not frame["status"].isin(["ok","error"]).all(): raise ValueError("알 수 없는 status")
        if not ((frame["status"]=="error").astype(int)==frame["failed"]).all():
            raise ValueError("status와 failed 생성식 불일치")
    return frame

def read_new(path):
    frame=pd.read_csv(path,encoding="utf-8",dtype={"request_id":str})
    if "request_id" not in frame or frame["request_id"].isna().any():
        raise ValueError("새 요청에 유효한 request_id가 필요합니다.")
    return frame.set_index("request_id")

def persist_and_verify(model,folder,X_valid):
    """직접 생성한 Pipeline만 저장 직후 복원합니다.
    folder/pipeline.joblib에 dump 후 load. 예측뿐 아니라 변환 행렬·특성명도 대조합니다.
    반환: restored_equal(bool), matrix_equal(bool), model_sha256(str) 사전.
    외부에서 받은 joblib 파일은 이 함수에 넣지 않습니다.
    """
    # 제공: 앞 단원 재사용 또는 입출력 보조 코드입니다.
    folder=Path(folder)
    file=folder/"pipeline.joblib"
    joblib.dump(model,file)
    # 방금 이 프로세스가 생성한 파일만 읽습니다. 임의 파일 복원은 별도 신뢰 승인이 필요합니다.
    restored=joblib.load(file)
    a=model.predict(X_valid); b=restored.predict(X_valid)
    matrix_a=model.named_steps["pre"].transform(X_valid)
    matrix_b=restored.named_steps["pre"].transform(X_valid)
    same=bool(np.array_equal(a,b))
    matrices=bool(np.allclose(matrix_a,matrix_b))
    names_same=np.array_equal(model.named_steps["pre"].get_feature_names_out(),
                              restored.named_steps["pre"].get_feature_names_out())
    if not (same and matrices and names_same): raise AssertionError("저장/복원 결과 불일치")
    return {"restored_equal":same,"matrix_equal":matrices,"model_sha256":sha256(file)}

def publish(stage,output):
    """관리 표식이 있는 outputs만 교체합니다. 교체 실패 시 기존 결과를 복원합니다.
    이 함수는 제공된 파일 관리 보조 코드로, 학습자가 구현할 전처리 범위 밖입니다.
    """
    stage=Path(stage); output=Path(output)
    marker=".day5_generated"
    if output.exists() and not output.is_dir(): raise ValueError("출력 경로가 폴더가 아닙니다.")
    if output.exists() and any(output.iterdir()) and not (output/marker).is_file():
        raise ValueError("관리 표식이 없는 비어 있지 않은 폴더를 덮어쓸 수 없습니다.")
    (stage/marker).write_text("Day5 generated outputs\n",encoding="utf-8")
    backup=output.with_name(".day5-backup-"+uuid.uuid4().hex)
    moved=False
    try:
        if output.exists(): os.replace(output,backup); moved=True
        os.replace(stage,output)
    except BaseException:
        if moved and not output.exists(): os.replace(backup,output)
        raise
    else:
        if moved: shutil.rmtree(backup)
