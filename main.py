"""Day5: 실행 전 요청 실패 예측을 위한 분할·전처리·기준선 프로젝트.
main.py를 import해도 학습하거나 결과를 쓰지 않습니다.
"""
from pathlib import Path
import argparse
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from checks import FEATURES,NUM,CAT,validate_features,check_splits,snapshot,assert_train_state,check_ready
from preprocessing import make_model
from provided_io import read_raw,read_new,write_json,versions,sha256,persist_and_verify,publish

ROOT=Path(__file__).resolve().parent

def split_xy(X,y,seed=42):
    """제공: 실습 03에서 완성한 두 단계 층화 분할을 재사용합니다."""
    Xdev,Xtest,ydev,ytest=train_test_split(X,y,test_size=.20,random_state=seed,stratify=y)
    Xtr,Xva,ytr,yva=train_test_split(Xdev,ydev,test_size=.25,random_state=seed,stratify=ydev)
    return {"train":(Xtr,ytr),"valid":(Xva,yva),"test":(Xtest,ytest)}

def train_and_validate(model,X_train,y_train,X_valid):
    """model을 train만으로 학습하고 학습 직후 상태, train/valid 변환, valid 예측 반환.
    반환: (state_before, train_ready, valid_ready, predictions).
    모델의 fit은 한 번이며 X_valid에는 transform/predict만 사용합니다.
    """
    # TODO D5-07-B2: 위 함수 설명의 입력·반환·검사 조건을 구현하세요.
    raise NotImplementedError("D5-07-B2를 완성하세요.")

def predict_new(model,frame):
    """입력 계약 검사 → 학습된 pre 변환 → predict, 정답 y 없이 수행합니다.
    반환: (정규화한 입력, 변환 행렬, 예측 배열). 새 범주를 배우려고 재학습하지 않습니다.
    """
    # 제공: 앞 단원 재사용 또는 입출력 보조 코드입니다.
    checked=validate_features(frame,training=False)
    ready=model.named_steps["pre"].transform(checked)
    predictions=model.predict(checked)
    return checked,ready,predictions

def run_project(input_path=None,output_dir=None,new_path=None):
    """한 번의 실행으로 결과를 재생성합니다. test에는 모델 메서드를 호출하지 않습니다."""
    source=Path(input_path) if input_path is not None else ROOT/"data"/"requests.csv"
    output=Path(output_dir) if output_dir is not None else ROOT/"outputs"
    source=source.resolve(); output=output.resolve()
    if output==source.parent or output in source.parents or output==ROOT:
        raise ValueError("소스/데이터 폴더는 출력 경로로 사용할 수 없습니다.")
    original=read_raw(source)
    X=original[FEATURES].copy(); y=original["failed"].copy()
    parts=split_xy(X,y,42); check_splits(parts,X.index)
    Xtr,ytr=parts["train"]; Xva,yva=parts["valid"]
    Xtr=validate_features(Xtr,training=True); Xva=validate_features(Xva)
    model=make_model()
    before,train_ready,valid_ready,pred=train_and_validate(model,Xtr,ytr,Xva)
    names=model.named_steps["pre"].get_feature_names_out().tolist()
    check_ready(train_ready,valid_ready,names)
    if len(pred)!=len(yva): raise AssertionError("검증 예측 수 불일치")
    probe=Xva.copy(); probe["input_chars"]=1_000_000.
    model.named_steps["pre"].transform(probe)
    after=snapshot(model.named_steps["pre"])
    assert_train_state(before,after,Xtr)
    metrics={"correct":int((pred==yva.to_numpy()).sum()),"total":len(yva),
        "accuracy":float((pred==yva.to_numpy()).mean()),
        "found_failed":int(((pred==1)&(yva.to_numpy()==1)).sum()),
        "total_failed":int((yva==1).sum())}
    new_source=Path(new_path) if new_path is not None else ROOT/"data"/"new_requests.csv"
    new,ready_new,pred_new=predict_new(model,read_new(new_source))
    assert_train_state(before,snapshot(model.named_steps["pre"]),Xtr)
    # 실제 호출 경로와 같은 범위만 기록합니다. 이는 접근 통제/보안 시스템의 대체물이 아닙니다.
    events=[{"operation":"fit","split":"train","rows":len(Xtr),"ids":Xtr.index.tolist()},
        {"operation":"transform","split":"train","rows":len(Xtr)},
        {"operation":"transform","split":"valid","rows":len(Xva)},
        {"operation":"predict","split":"valid","rows":len(Xva),"ids":Xva.index.tolist()},
        {"operation":"transform","split":"valid_probe","rows":len(probe)},
        {"operation":"predict","split":"new","rows":len(new)}]
    # test 내용/정답은 파일로 출력하지 않습니다. 분할 확인용 ID·건수만 남깁니다.
    manifest=pd.DataFrame([{"request_id":rid,"split":name}
        for name,(xx,_) in parts.items() for rid in xx.index]).sort_values("request_id")
    predictions=pd.DataFrame({"request_id":Xva.index,"y_true":yva.to_numpy(),"y_pred":pred})
    feature_frame=pd.DataFrame(valid_ready,columns=names,index=Xva.index)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".day5-stage-",dir=output.parent) as directory:
        stage=Path(directory)
        manifest.to_csv(stage/"split_manifest.csv",index=False,lineterminator="\n")
        feature_frame.to_csv(stage/"validation_features.csv",index=True,lineterminator="\n")
        predictions.to_csv(stage/"validation_predictions.csv",index=False,lineterminator="\n")
        pd.DataFrame({"request_id":new.index,"y_pred":pred_new}).to_csv(stage/"new_predictions.csv",index=False,lineterminator="\n")
        write_json(stage/"feature_names.json",names)
        write_json(stage/"validation_metrics.json",metrics)
        write_json(stage/"preprocessing_state.json",before)
        write_json(stage/"protocol_log.json",events)
        restoration=persist_and_verify(model,stage,Xva)
        write_json(stage/"restore_result.json",restoration)
        env=versions()
        write_json(stage/"environment.json",env)
        write_json(stage/"split_order.json",{k:xx.index.tolist() for k,(xx,_) in parts.items()})
        meta={"input_sha256":sha256(source),"seed_first":42,"seed_second":42,
            "test_size_first":0.20,"test_size_second":0.25,"source_rows":len(X),
            "split_sizes":{k:len(xx) for k,(xx,_) in parts.items()},
            "train_class_counts":{str(k):int(v) for k,v in ytr.value_counts().sort_index().items()},
            "valid_class_counts":{str(k):int(v) for k,v in yva.value_counts().sort_index().items()},
            "feature_input":FEATURES,"feature_count":len(names),"fit_ids":Xtr.index.tolist(),
            "test_model_calls":sum(e["split"]=="test" for e in events),"synthetic":True}
        write_json(stage/"run_metadata.json",meta)
        # 학습 전 소스와 결과 원본 파일이 그대로인지 확인합니다.
        hashes={p.name:sha256(p) for p in stage.iterdir() if p.is_file()}
        write_json(stage/"artifact_manifest.json",hashes)
        publish(stage,output)
    # 전처리/출력 검증을 모두 통과한 뒤에만 실행 환경 잠금을 갱신합니다.
    lock="\n".join(f"{k}=={v}" for k,v in env.items() if k!="python")+"\n"
    (output/"requirements-lock.txt").write_text(lock,encoding="utf-8")
    # lock까지 포함한 파일 무결성 목록을 다시 작성합니다.
    hashes={p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!="artifact_manifest.json"}
    write_json(output/"artifact_manifest.json",hashes)
    return {"split_sizes":meta["split_sizes"],"train_shape":list(train_ready.shape),
            "valid_shape":list(valid_ready.shape),"numeric_medians":before["numeric_medians"],
            "validation":metrics,"new_predictions":pred_new.tolist(),
            "state_unchanged":before==after,"restored_equal":restoration["restored_equal"],
            "test_model_calls":meta["test_model_calls"]}

if __name__=="__main__":
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--new-input",type=Path)
    args=parser.parse_args()
    print(json.dumps(run_project(args.input,args.output,args.new_input),ensure_ascii=False,indent=2,allow_nan=False))
