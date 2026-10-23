"""열별 전처리 구성. 함수는 매번 새 미학습 객체를 반환합니다."""
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from checks import NUM, CAT

def make_numeric():
    """제공: 중앙값으로 대체한 후 train 기준으로 표준화합니다."""
    return Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])

def make_categorical():
    """제공: object/np.nan 입력을 최빈값으로 채우고 train 범주로 인코딩합니다."""
    return Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
        ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

def make_preprocessor():
    """num 경로에는 NUM, cat 경로에는 CAT을 연결한 ColumnTransformer 반환.
    입력으로 주지 않은 열은 버리지만, 입력 계약 감사 자체를 대체하지는 않습니다.
    """
    # TODO D5-07-B1: 위 함수 설명의 입력·반환·검사 조건을 구현하세요.
    return ColumnTransformer([("num", make_numeric(), NUM), ("cat", make_categorical(), CAT)])

    # raise NotImplementedError("D5-07-B1를 완성하세요.")

def make_model():
    """제공: 오늘은 입력 패턴을 사용하지 않는 기준선만 연결합니다."""
    return Pipeline([("pre",make_preprocessor()),
                     ("model",DummyClassifier(strategy="most_frequent"))])
