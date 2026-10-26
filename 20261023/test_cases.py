"""제공: 프로젝트의 정상·경계·실패 테스트. test split의 모델 성능은 계산하지 않습니다."""
from pathlib import Path
import unittest
import numpy as np
from main import split_xy,train_and_validate
from preprocessing import make_model
from checks import FEATURES,validate_features,check_splits,snapshot,assert_train_state
from provided_io import read_raw,read_new

ROOT=Path(__file__).resolve().parent

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=read_raw(ROOT/"data"/"requests.csv")
        cls.X=raw[FEATURES];cls.y=raw["failed"]
        cls.parts=split_xy(cls.X,cls.y)

    def test_disjoint_aligned_splits(self):
        self.assertTrue(check_splits(self.parts,self.X.index))
        self.assertEqual([len(self.parts[n][0]) for n in ["train","valid","test"]],[60,20,20])

    def test_same_request_in_two_parts_is_rejected(self):
        bad=dict(self.parts);bad["valid"]=self.parts["train"]
        with self.assertRaises(ValueError):check_splits(bad,self.X.index)

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):validate_features(self.X.drop(columns="priority"))

    def test_future_information_is_rejected(self):
        with self.assertRaises(ValueError):validate_features(self.X.assign(status="ok"),extra="drop")

    def test_wholly_missing_train_column_is_rejected(self):
        with self.assertRaises(ValueError):validate_features(self.parts["train"][0].assign(priority=np.nan),training=True)

    def test_valid_and_unknown_have_same_schema(self):
        tr,ytr=self.parts["train"];va,_=self.parts["valid"]
        model=make_model();before,a,b,pred=train_and_validate(model,tr,ytr,va)
        new=validate_features(read_new(ROOT/"data"/"new_requests.csv"))
        converted=model.named_steps["pre"].transform(new)
        self.assertEqual(a.shape,(60,8));self.assertEqual(b.shape,(20,8));self.assertEqual(converted.shape,(2,8))
        self.assertTrue(np.isfinite(b).all());self.assertTrue(np.isfinite(converted).all())
        self.assertTrue(assert_train_state(before,snapshot(model.named_steps["pre"]),tr))

    def test_numeric_outlier_probe_does_not_refit(self):
        tr,ytr=self.parts["train"];va,_=self.parts["valid"]
        model=make_model();before,_,_,_=train_and_validate(model,tr,ytr,va)
        model.named_steps["pre"].transform(va.assign(input_chars=1_000_000.))
        self.assertEqual(before,snapshot(model.named_steps["pre"]))

    def test_infinite_numeric_is_rejected(self):
        with self.assertRaises(ValueError):validate_features(self.X.assign(input_chars=np.inf))

if __name__=="__main__":unittest.main(verbosity=2)
