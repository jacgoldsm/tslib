from __future__ import annotations

from tslib.pyspark_api import SparkOpts
from pyspark.sql import SparkSession, DataFrame
import pandas as pd
import numpy as np

from pandas.testing import assert_frame_equal

spark = SparkSession.builder.getOrCreate()

dates = [
    "2000-01-01",
    "2000-01-01",
    "2000-01-01",
    "2000-02-01",
    "2000-02-01",
    "2000-02-01",
    "2000-04-01",
    "2000-04-01",
    "2000-05-01",
    "2000-05-01",
    "2000-09-01",
    "2000-09-01",
    "2000-09-01",
]

nums = [
    1,
    1,
    1,
    2,
    2,
    2,
    4,
    4,
    5,
    5,
    9,
    9,
    9,
]

panel = spark.createDataFrame(pd.DataFrame(
    {
        "id": [1, 2, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 3],
        "date": [pd.to_datetime(date) for date in dates],
        "sex": ["m", "f", "f", "m", "f", "f", "m", "f", "m", "f", "m", "f", "f"],
        "credit_score": [
            750,
            820,
            640,
            760,
            810,
            670,
            740,
            840,
            745,
            620,
            780,
            800,
            630,
        ],
    }
))

panel_args = SparkOpts(
    ts_column="date",
    panel_column="id",
    start="1999-12-01",
    end="2000-10-01",
    freq="1m",
)

panel_ts = panel.ts(panel_args)
filled = panel_ts.tsfill()
panel_with_lags = panel_ts.with_lag("lag_credit", column="credit_score", back=2)
panel_with_leads = panel_ts.with_lead("lead_credit", column="credit_score", forward=2)

panel_with_diffs = panel_ts.with_difference(
    "credit_change", column="credit_score", back=2
)

panel_num = panel.toPandas()
panel_num["date"] = nums
panel_args_num = panel_args.assign(freq=1, start=0, end=10)
panel_num = spark.createDataFrame(panel_num)
panel_num_ts = panel_num.ts(panel_args_num)

filled_num = panel_num_ts.tsfill()
panel_with_lags_num = panel_num_ts.with_lag("lag_credit", column="credit_score", back=2)
panel_with_leads_num = panel_num_ts.with_lead(
    "lead_credit", column="credit_score", forward=2
)

panel_with_diffs_num = panel_num_ts.with_difference(
    "credit_change", column="credit_score", back=2
)

complete_dates = [
    pd.to_datetime(date)
    for date in [
        "1999-12-01",
        "2000-01-01",
        "2000-02-01",
        "2000-03-01",
        "2000-04-01",
        "2000-05-01",
        "2000-06-01",
        "2000-07-01",
        "2000-08-01",
        "2000-09-01",
        "2000-10-01",
    ]
]

complete_nums = list(range(11))

scores = [
    None,
    750,
    760,
    None,
    740,
    745,
    None,
    None,
    None,
    780,
    None,
    None,
    820,
    810,
    None,
    840,
    None,
    None,
    None,
    None,
    800,
    None,
    None,
    640,
    670,
    None,
    None,
    620,
    None,
    None,
    None,
    630,
    None,
]

expected_filled = pd.DataFrame(
    {
        "id": [1] * 11 + [2] * 11 + [3] * 11,
        "date": complete_dates * 3,
        "sex": [None, "m", "m", None, "m", "m", None, None, None, "m", None]
        + [None, "f", "f", None, "f", None, None, None, None, "f", None]
        + [None, "f", "f", None, None, "f", None, None, None, "f", None],
        "credit_score": scores,
    }
)

expected_filled_num = expected_filled.copy()
expected_filled_num["date"] = np.array(complete_nums * 3)

expected_panel_with_lags = panel.toPandas()
expected_panel_with_lags["lag_credit"] = [
    None,
    None,
    None,
    None,
    None,
    None,
    760,
    810,
    None,
    None,
    None,
    None,
    None,
]
expected_panel_with_lags_num = expected_panel_with_lags.copy()
expected_panel_with_lags_num["date"] = np.array(nums)

expected_panel_with_leads = panel.toPandas()
expected_panel_with_leads["lead_credit"] = [
    None,
    None,
    None,
    740,
    840,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]
expected_panel_with_leads_num = expected_panel_with_leads.copy()
expected_panel_with_leads_num["date"] = np.array(nums)
expected_panel_with_diff = panel.toPandas()
expected_panel_with_diff["credit_change"] = [
    None,
    None,
    None,
    None,
    None,
    None,
    -20,
    30,
    None,
    None,
    None,
    None,
    None,
]
expected_panel_with_diff_num = expected_panel_with_diff.copy()
expected_panel_with_diff_num["date"] = np.array(nums)

def compares_equal(
    df1: DataFrame | pd.DataFrame,
    df2: DataFrame | pd.DataFrame,
    check_dtype: bool = False,
) -> bool:
    if isinstance(df1, DataFrame):
        df1 = df1.toPandas()
    if isinstance(df2, DataFrame):
        df2 = df2.toPandas()
    df1 = df1.sort_values(["id","date"]).reset_index(drop=True).sort_index(axis=1)
    df2 = df2.sort_values(["id","date"]).reset_index(drop=True).sort_index(axis=1)
    try:
        assert_frame_equal(df1, df2, check_dtype=check_dtype, check_index_type=False)
        return True
    except AssertionError:
        return False

def test_panel():
    assert compares_equal(filled, expected_filled)
    assert compares_equal(filled_num, expected_filled_num)
    assert compares_equal(panel_with_lags, expected_panel_with_lags)
    assert compares_equal(panel_with_lags_num, expected_panel_with_lags_num)
    assert compares_equal(panel_with_leads, expected_panel_with_leads)
    assert compares_equal(panel_with_leads_num, expected_panel_with_leads_num)
    assert compares_equal(panel_with_diffs, expected_panel_with_diff)
    assert compares_equal(panel_with_diffs_num, expected_panel_with_diff_num)
