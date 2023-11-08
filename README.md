# tslib
A correct and consistent API for dealing with leads, lags, differences, and filling in gaps in time-series and panel data. Available in Pandas and (and Pyspark, coming soon.)

In Pandas, importing `tslib` grants access to the `.ts` accessor, allowing for idiomatic creation of lags, leads, and differences with time series data. `tslib` also grants access to the `.xt` accessor for the same methods applied to panel data. 

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
# create a DataFrame with all the gaps in the time-series filled in
print(cookies.ts.tsfill(cookies_args))
# create a DataFrame with the lagged value of `favorite`. 
# Note that lag respects gaps in the data,
# so year 2008 has no lag since there's no value for 2007
print(cookies.ts.with_lag(cookies_args, "favorite", name="previous_favorite"))
# the same, but with a lead
print(cookies.ts.with_lead(cookies_args, "favorite", name="next_favorite"))
# the same, but with differencing
print(cookies.ts.with_difference(cookies_args, "n", name="change_in_panelists"))
```

## Contributing

Before making a pull request, run the tests to ensure that none of them are broken. You can do 
this with the following code:
```bash
python3 -m pip install pytest
python3 -m pytest
```

