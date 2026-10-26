"""입력·분할·학습 상태 검사. test 모델 평가는 수행하지 않습니다."""
import numpy as np
import pandas as pd

NUM=["input_chars","allowed_tools"]
CAT=["task_type","priority"]
FEATURES=NUM+CAT
FORBIDDEN={"request_id","failed","status","elapsed_ms","retry_count"}

def validate_features(frame, training=False, extra="error"):
    """입력 DataFrame을 계약에 맞게 검증하고 정규화한 복사본을 반환.
    1) 비어 있는 입력/중복 열/중복 index/누락 열/금지 열을 거부합니다.
    2) extra='drop'이면 일반 추가 열만 제외, extra='error'이면 오류입니다.
    3) NUM은 숫자/결측, 관측값은 유한·0 이상·정수 값이어야 합니다.
    4) CAT은 문자열/결측만 허용하고 strip/lower, 빈칸/None→np.nan을 적용합니다.
    5) training=True일 때 전체 결측 열은 명확한 오류. 추론의 결측은 허용합니다.
    """
    # 제공: 앞 단원 재사용 또는 입출력 보조 코드입니다.
    if extra not in {"error","drop"}:
        raise ValueError("extra 정책은 error 또는 drop")
    if not isinstance(frame,pd.DataFrame) or frame.empty:
        raise ValueError("입력 DataFrame이 비어 있습니다.")
    if not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError("중복 열 또는 중복 요청 ID")
    missing=set(FEATURES)-set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    forbidden=FORBIDDEN & set(frame.columns)
    if forbidden:
        raise ValueError(f"forbidden columns: {sorted(forbidden)}")
    additional=set(frame.columns)-set(FEATURES)
    if additional and extra=="error":
        raise ValueError(f"extra columns: {sorted(additional)}")
    out=frame[FEATURES].copy()
    for col in NUM:
        values=pd.to_numeric(out[col],errors="raise").astype(float)
        observed=values.dropna().to_numpy()
        if not np.isfinite(observed).all() or (observed<0).any() or (observed%1!=0).any():
            raise ValueError(f"{col}: 유한한 0 이상의 정수 관측값 또는 결측이어야 합니다.")
        out[col]=values
    for col in CAT:
        def norm(value):
            if pd.isna(value): return np.nan
            if not isinstance(value,str): raise ValueError(f"{col}: 문자열 또는 결측이어야 합니다.")
            return value.strip().lower() or np.nan
        out[col]=out[col].map(norm).astype(object)
    empty=out.columns[out.isna().all()].tolist()
    if training and empty:
        raise ValueError(f"전체 결측 train 열: {empty}; 정책을 정한 후 다시 진행하세요.")
    return out

def check_splits(parts, all_ids):
    """세 부분 이름·내부 ID 고유성·X/y 순서·상호 독립·전체 포함을 검증.
    어떤 자료도 fit/transform/predict하지 않습니다. 실패하면 ValueError, 성공하면 True.
    """
    # 제공: 앞 단원 재사용 또는 입출력 보조 코드입니다.
    if set(parts)!={"train","valid","test"}:
        raise ValueError("train/valid/test가 모두 필요합니다.")
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError("원본 요청 ID 중복")
    seen=set()
    for name,(xx,yy) in parts.items():
        if xx.empty or not xx.index.is_unique or not yy.index.is_unique:
            raise ValueError(f"{name}: 빈 부분 또는 중복 ID")
        if not xx.index.equals(yy.index):
            raise ValueError(f"{name}: X/y index 불일치")
        ids=set(xx.index)
        if seen & ids:
            raise ValueError("분할 간 요청 ID 중복")
        seen |= ids
    if seen!=set(all_ids):
        raise ValueError("분할 ID 합집합과 원본 불일치")
    return True

def snapshot(pre):
    """제공: fitted 학습 상태를 JSON 직렬화 가능한 독립 값으로 복사합니다."""
    num=pre.named_transformers_["num"]
    cat=pre.named_transformers_["cat"]
    return {"numeric_medians":num.named_steps["imputer"].statistics_.tolist(),
            "scaler_mean":num.named_steps["scaler"].mean_.tolist(),
            "scaler_scale":num.named_steps["scaler"].scale_.tolist(),
            "scaler_seen":int(num.named_steps["scaler"].n_samples_seen_),
            "categorical_modes":cat.named_steps["imputer"].statistics_.tolist(),
            "categories":[c.tolist() for c in cat.named_steps["encoder"].categories_],
            "feature_names":pre.get_feature_names_out().tolist()}

def assert_train_state(before,after,X_train):
    """상태가 변하지 않았고 NUM 중앙값이 train 값인지 확인, 실패하면 AssertionError.
    scaler가 본 행 수 역시 train 행 수와 같아야 합니다. 통계 출처와 불변을 함께 확인합니다.
    """
    # 제공: 앞 단원 재사용 또는 입출력 보조 코드입니다.
    if before != after:
        raise AssertionError("valid/새 입력 처리 후 학습 상태가 변했습니다.")
    if not np.allclose(before["numeric_medians"],X_train[NUM].median().to_numpy()):
        raise AssertionError("대표값의 출처가 train이 아닙니다.")
    if before["scaler_seen"] != len(X_train):
        raise AssertionError("scaler가 학습한 행 수가 train과 다릅니다.")
    return True

def check_ready(train_ready,valid_ready,names):
    """제공: 행렬의 유한성, 열 수, 특성명 고유성과 금지 열 유입을 확인합니다."""
    if train_ready.ndim!=2 or valid_ready.ndim!=2:
        raise ValueError("전처리 출력은 2차원이어야 합니다.")
    if train_ready.shape[1]!=valid_ready.shape[1] or valid_ready.shape[1]!=len(names):
        raise ValueError("전처리 출력 열 수가 일치하지 않습니다.")
    if not np.isfinite(train_ready).all() or not np.isfinite(valid_ready).all():
        raise ValueError("전처리 출력에 NaN 또는 inf")
    if len(names)!=len(set(names)):
        raise ValueError("특성명 중복")
    for name in names:
        base=name.split("__",1)[-1]
        if any(base==bad or base.startswith(bad+"_") for bad in FORBIDDEN):
            raise ValueError(f"금지 특성명: {name}")
    return True
