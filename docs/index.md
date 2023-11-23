# tslib
A correct and consistent API for dealing with leads, lags, differences, and filling in gaps in time-series and panel data. Available in Pandas (and Pyspark, coming soon.)

In Pandas, importing `tslib` grants access to the `.ts` accessor, allowing for idiomatic creation of lags, leads, and differences with time series and panel data. 

## Installation
`tslib` is not currently on PyPI. Install from GitHub with:

`pip install tslib@git+https://github.com/jacgoldsm/tslib`

## Getting Started—Time-Series Data
```python
import pandas as pd
from tslib.pandas_api import TimeOpts

# Define our Data Frame. `tslib` works with dates stored as numbers or as Pandas dates.
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
# Define our time series arguments. 
# Set the time-series column, frequency, and start of the time-series.
cookies_args = TimeOpts(ts_column="year", freq=1, start=1999)
# define our time-series DataFrame
cookies_ts = cookies.ts(cookies_args)
# create a DataFrame with all the gaps in the time-series filled in
print(cookies_ts.tsfill())
# create a DataFrame with the lagged value of `favorite`. 
# Note that lag respects gaps in the data,
# so year 2008 has no lag since there's no value for 2007
print(cookies_ts.with_lag("previous_favorite",column="favorite"))
# the same, but with a lead
print(cookies_ts.with_lead("next_favorite",column="favorite"))
# the same, but with differencing
print(cookies_ts.with_difference("change_in_panelists",column="n"))
```

## Getting Started: Panel Data
```python
import pandas as pd
from tslib.pandas_api import TimeOpts

# Define our Data Frame. `tslib` works with dates stored as numbers or as Pandas dates.
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
print(panel)

# define our Time-Series options
panel_args = TimeOpts(
    ts_column="date",
    panel_column=panel["id"],
    start="1999-12-01",
    end="2000-10-01",
    freq="1m",
)

# set up our time-series DataFrame
panel_ts = panel.ts(panel_args)

# fill in our complete panel
print(panel_ts.tsfill())
# create our lagged data with lag of 2, with gaps preserved
print(panel_ts.with_lag("lag_credit",column="credit_score",back=2))
# the same, but with a lead
print(panel_ts.with_lead("lag_credit",column="credit_score",forward=2))
# the same, but with differencing
print(panel_ts.with_difference(name="credit_change",column="credit_score",back=2))
```

## Contributing

Before making a pull request, run the tests to ensure that none of them are broken. You can do 
this with the following code:
```bash
python3 -m pip install pytest
python3 -m pytest
```