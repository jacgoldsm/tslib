from __future__ import annotations

from tslib.pandas_api import PandasOpts
from tslib.pyspark_api import SparkOpts
import pandas as pd
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.dataframe import DataFrame
spark = SparkSession.builder.getOrCreate()
from pandas.testing import assert_frame_equal

cookies = pd.DataFrame(
    {
        "year": [2000, 2001, 2002, 2003, 2008],
        "favorite": [
            "Chocolate Chip",
            "Chocolate Chip",
            "Oatmeal Raisin",
            "Sugar",
            "M&M",
        ],
        "n": [10, 20, 15, 12, 40],
    }
)
cookies_ps = spark.createDataFrame(cookies)
cookies_args = PandasOpts(ts_column="year", freq=1, start=1999)
cookies_ts = cookies.ts(cookies_args)
ps_cookies_ts = cookies_ps.ts(cookies_args)
cookies_full = cookies_ts.tsfill()
ps_cookies_full = ps_cookies_ts.tsfill()
cookies_with_lag = cookies_ts.with_lag("previous_favorite", column="favorite")
ps_cookies_with_lag = ps_cookies_ts.with_lag("previous_favorite", column="favorite")
cookies_with_lead = cookies_ts.with_lead("next_favorite", column="favorite")
ps_cookies_with_lead = ps_cookies_ts.with_lead("next_favorite", column="favorite")
cookies_with_diff = cookies_ts.with_difference("change_in_panelists", column="n")
ps_cookies_with_diff = ps_cookies_ts.with_difference("change_in_panelists", column="n")


cookies_date = pd.DataFrame(
    {
        "year": (
            pd.to_datetime(i)
            for i in [
                "2000-01-01",
                "2001-01-01",
                "2002-01-01",
                "2003-01-01",
                "2008-01-01",
            ]
        ),
        "favorite": [
            "Chocolate Chip",
            "Chocolate Chip",
            "Oatmeal Raisin",
            "Sugar",
            "M&M",
        ],
        "n": [10, 20, 15, 12, 40],
    }
)

ps_cookies_date = spark.createDataFrame(cookies_date)
cookies_date_args = PandasOpts(ts_column="year", freq="Y", start="1999-01-01")
cookies_date_ts = cookies_date.ts(cookies_date_args)
ps_cookies_date_ts = ps_cookies_date.ts(cookies_date_args)
cookies_date_full = cookies_date_ts.tsfill()
ps_cookies_date_full = ps_cookies_date_ts.tsfill()
cookies_date_with_lag = cookies_date_ts.with_lag("previous_favorite", column="favorite")
ps_cookies_date_with_lag = ps_cookies_date_ts.with_lag("previous_favorite", column="favorite")
cookies_date_with_lead = cookies_date_ts.with_lead("next_favorite", column="favorite")
ps_cookies_date_with_lead = ps_cookies_date_ts.with_lead("next_favorite", column="favorite")
cookies_date_with_diff = cookies_date_ts.with_difference(
    "change_in_panelists", column="n"
)
ps_cookies_date_with_diff = ps_cookies_date_ts.with_difference(
    "change_in_panelists", column="n"
)

expected_filled = pd.DataFrame(
    {
        "year": [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008],
        "favorite": [
            None,
            "Chocolate Chip",
            "Chocolate Chip",
            "Oatmeal Raisin",
            "Sugar",
            None,
            None,
            None,
            None,
            "M&M",
        ],
        "n": [None, 10, 20, 15, 12, None, None, None, None, 40],
    }
)
ps_expected_filled = spark.createDataFrame(expected_filled)
expected_filled_date = expected_filled.copy()
expected_filled_date.year = [pd.to_datetime(f"{i}-01-01") for i in expected_filled.year]
ps_expected_filled_date = spark.createDataFrame(expected_filled_date)

expected_lag = cookies.copy()
expected_lag["previous_favorite"] = [
    None,
    "Chocolate Chip",
    "Chocolate Chip",
    "Oatmeal Raisin",
    None,
]
ps_expected_lag = spark.createDataFrame(expected_lag)

expected_lag_date = cookies_date.copy()
expected_lag_date["previous_favorite"] = [
    None,
    "Chocolate Chip",
    "Chocolate Chip",
    "Oatmeal Raisin",
    None,
]
ps_expected_lag_date = spark.createDataFrame(expected_lag_date)

expected_lead = cookies.copy()
expected_lead["next_favorite"] = [
    "Chocolate Chip",
    "Oatmeal Raisin",
    "Sugar",
    None,
    None,
]
ps_expected_lead = spark.createDataFrame(expected_lead)

expected_lead_date = cookies_date.copy()
expected_lead_date["next_favorite"] = [
    "Chocolate Chip",
    "Oatmeal Raisin",
    "Sugar",
    None,
    None,
]
ps_expected_lead_date = spark.createDataFrame(expected_lead_date)

expected_diff = cookies.copy()
expected_diff["change_in_panelists"] = [None, 10, -5, -3, None]
ps_expected_diff = spark.createDataFrame(expected_diff)

expected_diff_date = cookies_date.copy()
expected_diff_date["change_in_panelists"] = [None, 10, -5, -3, None]
ps_expected_diff_date = spark.createDataFrame(expected_diff_date)



def compares_equal(
    df1: pd.DataFrame | DataFrame,
    df2: pd.DataFrame | DataFrame,
    check_dtype: bool = False,
) -> bool:
    if isinstance(df1, DataFrame):
        df1 = df1.toPandas().sort_values("year")
        df2 = df2.toPandas().sort_values("year")
        df1.index = np.arange(len(df1.index))
        df2.index = np.arange(len(df2.index))
    try:
        assert_frame_equal(df1, df2, check_dtype=check_dtype, check_index_type=False)
        return True
    except AssertionError:
        return False


def test_cookies():
    assert compares_equal(cookies_date_full, expected_filled_date)
    assert compares_equal(cookies_date_with_lag, expected_lag_date)
    assert compares_equal(cookies_date_with_lead, expected_lead_date)
    assert compares_equal(cookies_date_with_diff, expected_diff_date)

    assert compares_equal(cookies_full, expected_filled)
    assert compares_equal(cookies_with_lag, expected_lag)
    assert compares_equal(cookies_with_lead, expected_lead)
    assert compares_equal(cookies_with_diff, expected_diff)



def test_cookies_ps():
    assert compares_equal(ps_cookies_date_full, ps_expected_filled_date)
    assert compares_equal(ps_cookies_date_with_lag, ps_expected_lag_date)
    assert compares_equal(ps_cookies_date_with_lead, ps_expected_lead_date)
    assert compares_equal(ps_cookies_date_with_diff, ps_expected_diff_date)

    assert compares_equal(ps_cookies_full, ps_expected_filled)
    assert compares_equal(ps_cookies_with_lag, ps_expected_lag)
    assert compares_equal(ps_cookies_with_lead, ps_expected_lead)
    assert compares_equal(ps_cookies_with_diff, ps_expected_diff)
