"""Kiểm thử sản phẩm đã có model/kết quả. Chạy sau --all, không gọi trong bootstrap."""
import unittest
import importlib.util
import shutil
import numpy as np
import pandas as pd
from src.common import ROOT
from src.data import load_data
from src.predict import metadata,load_model,forecast

class SavedModelTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("pyspark") and shutil.which("java"),"Cần PySpark và Java")
    def test_dashboard_runs_real_spark_etl_button(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=120).run()
        at.sidebar.radio[0].set_value("Xử lý Big Data trực tiếp").run()
        at.button[0].click().run(timeout=180)
        self.assertEqual(len(at.exception),0)
        self.assertEqual(len(at.error),0)
        self.assertTrue(any("Spark thật" in s.value for s in at.success))

    def test_replay_saved_model_matches_exported_predictions(self):
        history=load_data();reg=metadata();name=reg["selected_model"]
        model=load_model(name)
        pred=pd.read_csv(ROOT/"results/predictions.csv",parse_dates=["timestamp"]).set_index("timestamp")
        for timestamp in ["2018-10-01 08:00","2018-11-04 08:00","2018-11-30 23:00"]:
            ts,value,_=forecast(history.loc[history.index<pd.Timestamp(timestamp)],model,name)
            self.assertEqual(ts,pd.Timestamp(timestamp))
            self.assertAlmostEqual(value,pred.loc[ts,name],places=8)
        ts,value,_=forecast(history,model,name)
        self.assertEqual(ts,pd.Timestamp("2018-12-01 00:00"))
        self.assertTrue(np.isfinite(value) and value>=0)

    def test_metric_csv_recomputes_from_predictions(self):
        from src.metrics import regression_metrics
        pred=pd.read_csv(ROOT/"results/predictions.csv")
        scores=pd.read_csv(ROOT/"results/test_metrics.csv")
        for r in scores.itertuples():
            metric=regression_metrics(pred["count"],pred[r.model])
            for name in ["MAE","RMSE","MAPE_positive_pct"]:
                self.assertAlmostEqual(metric[name],getattr(r,name),places=8)

    def test_dashboard_pages_and_forecast(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=45).run()
        self.assertEqual(len(at.exception),0)
        for page in ["Xử lý Big Data trực tiếp","Thử dự báo","Đánh giá mô hình","Spark và hiệu năng","Hướng dẫn demo"]:
            at.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(at.exception),0,page)
            if page=="Thử dự báo":
                at.button[0].click().run()
                self.assertEqual(len(at.exception),0)
                self.assertTrue(len(at.success)>0)
                at.radio[0].set_value("Giờ kế tiếp sau bộ dữ liệu").run()
                at.button[0].click().run()
                self.assertEqual(len(at.exception),0)
                self.assertIn("01/12/2018",at.success[0].value)

if __name__=="__main__":unittest.main()
