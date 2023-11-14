from tslib.pandas_api import TimeOpts
import pandas as pd

from pandas.testing import assert_frame_equal

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

panel = pd.DataFrame(
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
)

panel_args = TimeOpts(
    ts_column="date",
    panel_column=panel["id"],
    start="1999-12-01",
    end="2000-10-01",
    freq="1m",
)

filled = panel.ts.tsfill(panel_args)
# panel_with_lags = panel.ts.with_lag(panel_args,column="credit_score",name="lag_credit",back=2)
# panel_with_leads = panel.ts.with_lead(panel_args,column="credit_score",name="lag_credit",forward=2)
# panel_with_diffs = panel.ts.with_difference(panel_args,column="credit_score",name="credit_change",back=2)

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


def compares_equal(
    df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = False
) -> bool:
    try:
        assert_frame_equal(df1, df2, check_dtype=check_dtype)
        return True
    except AssertionError:
        return False


def test_panel():
    assert compares_equal(filled, expected_filled)
