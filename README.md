# tslib
A correct and consistent API for dealing with leads, lags, differences, and filling in gaps in time-series and panel data. Available in Pandas and PySpark.

In Pandas and PySpark, importing `tslib` grants access to the `.ts` accessor, allowing for idiomatic creation of lags, leads, and differences with time series and panel data. 

See complete documentation [here](https://jacgoldsm.github.io/tslib/).

## Installation
`tslib` is not currently on PyPI. Install from GitHub with:

`pip install tslib@git+https://github.com/jacgoldsm/tslib`

## Note
`tslib` works with dates stored as numbers or as dates/timestamps, but it will almost always be more efficient
with integer dates, especially with PySpark.

## Getting Started—Time-Series Data
```python
import pandas as pd
from tslib.pandas_api import PandasOpts

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
print(cookies)
    year       favorite   n
0  2000  Chocolate Chip  10
1  2001  Chocolate Chip  20
2  2002  Oatmeal Raisin  15
3  2003           Sugar  12
4  2008             M&M  40
# Define our time series arguments. 
# Set the time-series column, frequency, and start of the time-series.
cookies_args = PandasOpts(ts_column="year", freq=1, start=1999)
# define our time-series DataFrame
cookies_ts = cookies.ts(cookies_args)
# create a DataFrame with all the gaps in the time-series filled in
print(cookies_ts.tsfill())
    year       favorite     n
0  1999             NaN  <NA>
1  2000  Chocolate Chip    10
2  2001  Chocolate Chip    20
3  2002  Oatmeal Raisin    15
4  2003           Sugar    12
5  2004             NaN  <NA>
6  2005             NaN  <NA>
7  2006             NaN  <NA>
8  2007             NaN  <NA>
9  2008             M&M    40
# create a DataFrame with the lagged value of `favorite`. 
# Note that lag respects gaps in the data,
# so year 2008 has no lag since there's no value for 2007
print(cookies_ts.with_lag("previous_favorite",column="favorite"))
    year       favorite   n previous_favorite
0  2000  Chocolate Chip  10               NaN
1  2001  Chocolate Chip  20    Chocolate Chip
2  2002  Oatmeal Raisin  15    Chocolate Chip
3  2003           Sugar  12    Oatmeal Raisin
4  2008             M&M  40               NaN
# the same, but with a lead
print(cookies_ts.with_lead("next_favorite",column="favorite"))
   year        favorite   n   next_favorite
0  2000  Chocolate Chip  10  Chocolate Chip
1  2001  Chocolate Chip  20  Oatmeal Raisin
2  2002  Oatmeal Raisin  15           Sugar
3  2003           Sugar  12             NaN
4  2008             M&M  40             NaN
# the same, but with differencing
print(cookies_ts.with_difference("change_in_panelists",column="n"))
    year       favorite   n  change_in_panelists
0  2000  Chocolate Chip  10                  NaN
1  2001  Chocolate Chip  20                 10.0
2  2002  Oatmeal Raisin  15                 -5.0
3  2003           Sugar  12                 -3.0
4  2008             M&M  40                  NaN
```

## Getting Started: Panel Data
```python
import pandas as pd
from tslib.pandas_api import PandasOpts

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
    id       date sex  credit_score
0    1 2000-01-01   m           750
1    2 2000-01-01   f           820
2    3 2000-01-01   f           640
3    1 2000-02-01   m           760
4    2 2000-02-01   f           810
5    3 2000-02-01   f           670
6    1 2000-04-01   m           740
7    2 2000-04-01   f           840
8    1 2000-05-01   m           745
9    3 2000-05-01   f           620
10   1 2000-09-01   m           780
11   2 2000-09-01   f           800
12   3 2000-09-01   f           630
# define our Time-Series options
panel_args = PandasOpts(
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
    id       date  sex  credit_score
0    1 1999-12-01  NaN          <NA>
1    1 2000-01-01    m           750
2    1 2000-02-01    m           760
3    1 2000-03-01  NaN          <NA>
4    1 2000-04-01    m           740
5    1 2000-05-01    m           745
6    1 2000-06-01  NaN          <NA>
7    1 2000-07-01  NaN          <NA>
8    1 2000-08-01  NaN          <NA>
9    1 2000-09-01    m           780
10   1 2000-10-01  NaN          <NA>
11   2 1999-12-01  NaN          <NA>
12   2 2000-01-01    f           820
13   2 2000-02-01    f           810
14   2 2000-03-01  NaN          <NA>
15   2 2000-04-01    f           840
16   2 2000-05-01  NaN          <NA>
17   2 2000-06-01  NaN          <NA>
18   2 2000-07-01  NaN          <NA>
19   2 2000-08-01  NaN          <NA>
20   2 2000-09-01    f           800
21   2 2000-10-01  NaN          <NA>
22   3 1999-12-01  NaN          <NA>
23   3 2000-01-01    f           640
24   3 2000-02-01    f           670
25   3 2000-03-01  NaN          <NA>
26   3 2000-04-01  NaN          <NA>
27   3 2000-05-01    f           620
28   3 2000-06-01  NaN          <NA>
29   3 2000-07-01  NaN          <NA>
30   3 2000-08-01  NaN          <NA>
31   3 2000-09-01    f           630
32   3 2000-10-01  NaN          <NA>
# create our lagged data with lag of 2, with gaps preserved
print(panel_ts.with_lag("lag_credit",column="credit_score",back=2))
    id       date sex  credit_score  lag_credit
0    1 2000-01-01   m           750        <NA>
1    2 2000-01-01   f           820        <NA>
2    3 2000-01-01   f           640        <NA>
3    1 2000-02-01   m           760        <NA>
4    2 2000-02-01   f           810        <NA>
5    3 2000-02-01   f           670        <NA>
6    1 2000-04-01   m           740         760
7    2 2000-04-01   f           840         810
8    1 2000-05-01   m           745        <NA>
9    3 2000-05-01   f           620        <NA>
10   1 2000-09-01   m           780        <NA>
11   2 2000-09-01   f           800        <NA>
12   3 2000-09-01   f           630        <NA>
# the same, but with a lead
print(panel_ts.with_lead("lag_credit",column="credit_score",forward=2))
    id       date sex  credit_score  lead_credit
0    1 2000-01-01   m           750         <NA>
1    2 2000-01-01   f           820         <NA>
2    3 2000-01-01   f           640         <NA>
3    1 2000-02-01   m           760          740
4    2 2000-02-01   f           810          840
5    3 2000-02-01   f           670         <NA>
6    1 2000-04-01   m           740         <NA>
7    2 2000-04-01   f           840         <NA>
8    1 2000-05-01   m           745         <NA>
9    3 2000-05-01   f           620         <NA>
10   1 2000-09-01   m           780         <NA>
11   2 2000-09-01   f           800         <NA>
12   3 2000-09-01   f           630         <NA>
# the same, but with differencing
print(panel_ts.with_difference(name="credit_change",column="credit_score",back=2))
    id       date sex  credit_score  credit_change
0    1 2000-01-01   m           750           <NA>
1    2 2000-01-01   f           820           <NA>
2    3 2000-01-01   f           640           <NA>
3    1 2000-02-01   m           760           <NA>
4    2 2000-02-01   f           810           <NA>
5    3 2000-02-01   f           670           <NA>
6    1 2000-04-01   m           740            -20
7    2 2000-04-01   f           840             30
8    1 2000-05-01   m           745           <NA>
9    3 2000-05-01   f           620           <NA>
10   1 2000-09-01   m           780           <NA>
11   2 2000-09-01   f           800           <NA>
12   3 2000-09-01   f           630           <NA>
```

## Contributing

Before making a pull request, run the tests to ensure that none of them are broken. You can do 
this with the following code:
```bash
python3 -m pip install pytest
python3 -m pytest
```

## Building the Docs

To rebuild the docs, run the following

```bash
python3 -m pip install mkdocs mkdocstrings[python] mkdocs-material
mkdocs build
```

See what it looks like on a local server with `mkdocs serve`.
Once it looks right, deploy with `mkdocs gh-deploy`.

Once it's deployed, delete the site/ repository so it doesn't pollute the main branch.