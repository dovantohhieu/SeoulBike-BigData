"""Kiểm tra Spark khi có đủ môi trường: dữ liệu hỏng phải bị từ chối, đặc trưng khớp."""
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import RAW, FEATURES


@unittest.skipUnless(importlib.util.find_spec("pyspark") and shutil.which("java"), "Cần PySpark và Java")
class SparkDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.spark_pipeline import session
        cls.spark=session("ContractTests")
        cls.temp=tempfile.TemporaryDirectory()
        cls.raw=pd.read_csv(RAW,encoding="latin1")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        cls.temp.cleanup()

    def test_spark_rejects_missing_duplicate_and_invalid_values(self):
        from src.spark_data import read_seoul
        cases={"missing":self.raw.drop(index=100),
               "duplicate":pd.concat([self.raw,self.raw.iloc[[0]]],ignore_index=True)}
        for label,column,value in [("negative_solar","Solar Radiation (MJ/m2)",-1),
                                   ("infinite_dew","Dew point temperature(°C)",float("inf")),
                                   ("bad_humidity","Humidity(%)",101)]:
            changed=self.raw.copy();changed.loc[0,column]=value;cases[label]=changed
        for label,frame in cases.items():
            with self.subTest(case=label):
                p=Path(self.temp.name)/f"{label}.csv"
                frame.to_csv(p,index=False,encoding="latin1")
                with self.assertRaises(ValueError):read_seoul(self.spark,p)

    def test_spark_features_equal_pandas_and_reload_model(self):
        from src.spark_pipeline import spark_features
        from src.data import load_data,feature_frame
        from src.predict import load_model,predict_frame
        from src.common import ROOT
        sf=spark_features(self.spark).orderBy("timestamp").toPandas().set_index("timestamp")
        pf=feature_frame(load_data()).dropna()
        self.assertTrue(sf.index.equals(pf.index))
        np.testing.assert_array_equal(sf[FEATURES],pf[FEATURES])
        model=load_model("SparkRandomForest")
        saved=pd.read_csv(ROOT/"results/predictions_spark.csv",parse_dates=["timestamp"]).set_index("timestamp")
        pred=predict_frame(model,pf.loc[saved.index],"SparkRandomForest")
        np.testing.assert_allclose(pred,saved["prediction"],rtol=0,atol=1e-8)


if __name__=="__main__":unittest.main()
