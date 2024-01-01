from __future__ import annotations

from tslib.pyspark_api import SparkOpts
from pyspark.sql import SparkSession, DataFrame

import pandas as pd
from pandas.testing import assert_frame_equal

spark = SparkSession.builder.getOrCreate()


cookies = spark.createDataFrame(pd.DataFrame(
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
))
cookies_args = SparkOpts(ts_column="year", freq=1, start=1999)
cookies_ts = cookies.ts(cookies_args)
cookies_full = cookies_ts.tsfill()
cookies_with_lag = cookies_ts.with_lag("previous_favorite", column="favorite")
cookies_with_lead = cookies_ts.with_lead("next_favorite", column="favorite")
cookies_with_diff = cookies_ts.with_difference("change_in_panelists", column="n")


cookies_date = spark.createDataFrame(pd.DataFrame(
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
))

cookies_date_args = SparkOpts(ts_column="year", freq="Y", start="1999-01-01")
cookies_date_ts = cookies_date.ts(cookies_date_args)
cookies_date_full = cookies_date_ts.tsfill()
cookies_date_with_lag = cookies_date_ts.with_lag("previous_favorite", column="favorite")
cookies_date_with_lead = cookies_date_ts.with_lead("next_favorite", column="favorite")
cookies_date_with_diff = cookies_date_ts.with_difference(
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
expected_filled_date = expected_filled.copy()
expected_filled_date.year = [pd.to_datetime(f"{i}-01-01") for i in expected_filled.year]

expected_lag = cookies.toPandas()
expected_lag["previous_favorite"] = [
    None,
    "Chocolate Chip",
    "Chocolate Chip",
    "Oatmeal Raisin",
    None,
]

expected_lag_date = cookies_date.toPandas()
expected_lag_date["previous_favorite"] = [
    None,
    "Chocolate Chip",
    "Chocolate Chip",
    "Oatmeal Raisin",
    None,
]
expected_lead = cookies.toPandas()
expected_lead["next_favorite"] = [
    "Chocolate Chip",
    "Oatmeal Raisin",
    "Sugar",
    None,
    None,
]

expected_lead_date = cookies_date.toPandas()
expected_lead_date["next_favorite"] = [
    "Chocolate Chip",
    "Oatmeal Raisin",
    "Sugar",
    None,
    None,
]

expected_diff = cookies.toPandas()
expected_diff["change_in_panelists"] = [None, 10, -5, -3, None]

expected_diff_date = cookies_date.toPandas()
expected_diff_date["change_in_panelists"] = [None, 10, -5, -3, None]


def compares_equal(
    df1: DataFrame | pd.DataFrame,
    df2: DataFrame | pd.DataFrame,
    check_dtype: bool = False,
) -> bool:
    if isinstance(df1, DataFrame):
        df1 = df1.toPandas().sort_values("year").reset_index(drop=True)
    if isinstance(df2, DataFrame):
        df2 = df2.toPandas().sort_values("year").reset_index(drop=True)
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
