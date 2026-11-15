import unittest
import numpy as np
import pandas as pd
from io import StringIO
from src.data import load_data, feature_frame, next_features, validate_hourly
from src.common import FEATURES, FOLDS, TEST_START
from src.metrics import regression_metrics

class ForecastContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df=load_data()

    def test_raw_grid_and_zero_semantics(self):
        self.assertEqual(len(self.df),8760)
        self.assertEqual(int((self.df["count"]==0).sum()),295)
        self.assertTrue((self.df.loc[self.df["count"]==0,"functioning"]=="No").all())

    def test_future_target_and_weather_cannot_change_feature(self):
        pos=500
        old=feature_frame(self.df)
        altered=self.df.copy()
        altered.iloc[pos:,altered.columns.get_indexer(["count","temperature","humidity","wind","rain","snow"])]=9999
        new=feature_frame(altered)
        np.testing.assert_allclose(old.iloc[pos],new.iloc[pos])

    def test_rolling_excludes_current_hour(self):
        x=feature_frame(self.df)
        self.assertAlmostEqual(x.iloc[500]["mean_24"],self.df["count"].iloc[476:500].mean())
        self.assertEqual(x.iloc[500]["lag_168"],self.df["count"].iloc[332])

    def test_next_features_matches_backtest(self):
        pos=1000
        expected=feature_frame(self.df).iloc[[pos]]
        actual=next_features(self.df.iloc[:pos])
        pd.testing.assert_frame_equal(actual,expected)

    def test_reject_missing_hour(self):
        with self.assertRaisesRegex(ValueError,"liên tục"):
            validate_hourly(self.df.drop(self.df.index[100]))

    def test_reject_duplicate_hour(self):
        with self.assertRaisesRegex(ValueError,"trùng"):
            validate_hourly(pd.concat([self.df.iloc[:2],self.df.iloc[[1]]]))

    def test_need_one_week_history(self):
        with self.assertRaisesRegex(ValueError,"168"):
            next_features(self.df.iloc[:167])

    def test_metrics_keep_zero_in_mae_rmse(self):
        m=regression_metrics([0,10],[5,8])
        self.assertEqual(m["MAE"],3.5)
        self.assertAlmostEqual(m["RMSE"],np.sqrt(14.5))
        self.assertEqual(m["MAPE_positive_pct"],20.0)
        self.assertEqual(m["MAPE_n"],1)
        self.assertEqual(m["MAPE_coverage_pct"],50.0)
        self.assertIsNone(regression_metrics([0],[2])["MAPE_positive_pct"])

    def test_cv_is_before_holdout(self):
        for start,end in FOLDS:
            self.assertLess(pd.Timestamp(start),pd.Timestamp(end))
            self.assertLessEqual(pd.Timestamp(end),pd.Timestamp(TEST_START))
            self.assertTrue((self.df.index[self.df.index<start] < pd.Timestamp(start)).all())

    def test_feature_count(self):
        self.assertEqual(len(FEATURES),21)
        self.assertEqual(len(feature_frame(self.df).dropna()),8592)

    def test_reject_empty_and_invalid_unused_raw_column(self):
        with self.assertRaisesRegex(ValueError,"rỗng"):
            validate_hourly(self.df.iloc[:0])
        raw=pd.read_csv("data/raw/SeoulBikeData.csv",encoding="latin1").head(200)
        for value in [float("inf"),-1]:
            modified=raw.copy()
            modified.loc[0,"Solar Radiation (MJ/m2)"]=value
            with self.assertRaises(ValueError):
                load_data(StringIO(modified.to_csv(index=False)))

if __name__ == "__main__":
    unittest.main()
