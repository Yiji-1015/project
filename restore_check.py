"""신뢰를 확인한 자체 생성 Pipeline을 별도 프로세스에서 복원합니다.
joblib/pickle은 임의 코드 실행이 가능하므로 외부의 알 수 없는 파일을 읽지 마세요.
SHA256은 우발적 변조 탐지일 뿐 출처/안전성 인증이 아닙니다.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from provided_io import sha256,versions
from checks import FEATURES,snapshot,validate_features

ROOT=Path(__file__).resolve().parent

def verify_restore(output_dir, input_path, trusted=False):
    if not trusted:
        raise ValueError("직접 생성한 파일임을 확인한 뒤 --trust-local-model을 지정하세요.")
    output=Path(output_dir)
    meta=json.loads((output/"run_metadata.json").read_text(encoding="utf-8"))
    if sha256(input_path)!=meta["input_sha256"]: raise ValueError("입력 파일 해시 불일치")
    saved_env=json.loads((output/"environment.json").read_text(encoding="utf-8"))
    current=versions()
    for key in ["python","numpy","scipy","pandas","scikit-learn","joblib"]:
        if saved_env[key]!=current[key]: raise ValueError(f"복원 환경 버전 불일치: {key}")
    manifest=json.loads((output/"artifact_manifest.json").read_text(encoding="utf-8"))
    for name,wanted in manifest.items():
        if Path(name).name!=name or sha256(output/name)!=wanted: raise ValueError(f"결과 파일 해시 불일치: {name}")
    raw=pd.read_csv(input_path,encoding="utf-8",dtype={"request_id":str}).set_index("request_id")
    order=json.loads((output/"split_order.json").read_text(encoding="utf-8"))
    # 복원 검증도 valid 행만 사용합니다. test의 값은 모델에 전달하지 않습니다.
    X_valid=validate_features(raw.loc[order["valid"],FEATURES])
    model=joblib.load(output/"pipeline.joblib")
    predictions=model.predict(X_valid)
    expected=pd.read_csv(output/"validation_predictions.csv",dtype={"request_id":str})
    if not expected["request_id"].tolist()==X_valid.index.tolist(): raise AssertionError("예측 ID 순서 불일치")
    if not np.array_equal(expected["y_pred"].to_numpy(),predictions): raise AssertionError("복원 예측 불일치")
    original_matrix=pd.read_csv(output/"validation_features.csv",index_col="request_id")
    transformed=model.named_steps["pre"].transform(X_valid)
    names=model.named_steps["pre"].get_feature_names_out().tolist()
    if not np.allclose(original_matrix.to_numpy(),transformed): raise AssertionError("복원 전처리 행렬 불일치")
    if list(original_matrix.columns)!=names: raise AssertionError("복원 특성명 불일치")
    saved_state=json.loads((output/"preprocessing_state.json").read_text(encoding="utf-8"))
    if snapshot(model.named_steps["pre"])!=saved_state: raise AssertionError("복원 학습 상태 불일치")
    return {"fresh_process_restore":True,"predictions_equal":True,"matrix_equal":True,
            "state_equal":True,"rows":len(predictions),"features":len(names),"test_model_calls":0}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs")
    parser.add_argument("--input",type=Path,default=ROOT/"data"/"requests.csv")
    parser.add_argument("--trust-local-model",action="store_true")
    args=parser.parse_args()
    print(json.dumps(verify_restore(args.output,args.input,args.trust_local_model),ensure_ascii=False,indent=2))
