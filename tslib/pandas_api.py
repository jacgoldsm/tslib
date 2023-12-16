from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Iterable, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

from tslib.pandas_utils import (
    _is_acceptable_type,
    _is_date,
    _is_numeric,
    _is_numeric_scalar,
    TimeDict,
    _range,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass
class PandasOpts:
    ts_column: str | pd.Series
    panel_column: str | pd.Series | None = None
    freq: int | float | str | pd.offsets.DateOffset | None = None
    start: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None
    end: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None

    def assign(self, inplace=False, **kwargs) -> PandasOpts | None:
        import copy

        if not inplace:
            out = copy.copy(self)
        else:
            out = None
        for kwarg in kwargs.keys():
            if kwarg not in ("ts_column", "panel_column", "freq", "start", "end"):
                raise ValueError(f"Invalid option: {kwarg}")
            if inplace:
                setattr(self, kwarg, kwargs[kwarg])
            else:
                setattr(out, kwarg, kwargs[kwarg])
        return out

    @staticmethod
    def _extract_opt(opts: PandasOpts | dict, opt: str):
        if isinstance(opts, PandasOpts):
            return getattr(opts, opt)
        else:
            return opts[opt]

    @staticmethod
    def _extract_opts(opts: PandasOpts | dict):
        if isinstance(opts, PandasOpts):
            return (opts.ts_column, opts.panel_column, opts.freq, opts.start, opts.end)
        else:
            out = PandasOpts(
                opts["ts_column"],
                opts.get("panel_column", None),
                opts.get("freq", None),
                opts.get("start", None),
                opts.get("end", None),
            )
            return (out.ts_column, out.panel_column, out.freq, out.start, out.end)


@pd.api.extensions.register_dataframe_accessor("ts")
class TSAccessor:
    def __init__(self, obj: pd.DataFrame):
        self._obj = obj

    def __call__(self, opts: PandasOpts | dict) -> TSAccessor:
        self._opts = opts
        return self

    def tsset(
        self,
        ts_args: PandasOpts | dict,
    ) -> TimeDict:
        data = self._obj.copy()
        ts_column, panel_column, freq, start, end = PandasOpts._extract_opts(ts_args)
        out_dict = TimeDict()
        if isinstance(ts_column, (str, bytes)):
            ts_column_name = ts_column
            ts_column = self._obj[ts_column]
            if len(ts_column) < 2:
                raise ValueError("A time series needs at least two records.")
        else:
            ts_column_name: str = (
                str(ts_column.name) if ts_column.name is not None else "time_series"
            )
        out_dict.ts_column_name = ts_column_name
        if not _is_acceptable_type(ts_column):
            raise Exception(f"{ts_column.dtype} not an acceptable time-series type")
        if len(ts_column.index) != ts_column.nunique() and panel_column is None:
            raise Exception("Time-series column cannot contain duplicates")
        if _is_numeric(ts_column) and not _is_numeric_scalar(freq):
            raise Exception(
                "For a numeric time-series variable" "Frequency must be a number."
            )
        data.reset_index(drop=True)
        is_date = _is_date(ts_column)
        if is_date:
            out_dict.freq = pd.tseries.frequencies.to_offset(freq)  # type: ignore
            assert not isinstance(start, (int, float))
            assert not isinstance(end, (int, float))
        else:
            assert isinstance(freq, (int, float))
            out_dict.freq = freq

        if start is not None:
            if is_date:
                start = pd.to_datetime(start)
        else:
            start = ts_column.min()

        if end is not None:
            if is_date:
                end = pd.to_datetime(end)
        else:
            end = ts_column.max()

        if is_date:
            complete_time_series = pd.period_range(
                start=start, end=end, freq=out_dict.freq
            ).to_timestamp()  # type: ignore
        else:
            # note that `end` should be inclusive whereas range is top-exclusive
            complete_time_series = np.arange(start, end + freq, out_dict.freq)
        if set(ts_column[(ts_column >= start) & (ts_column <= end)]) - set(
            complete_time_series
        ):
            raise Exception("Time series structure doesn't match frequency specified.")

        if panel_column is not None:
            if isinstance(panel_column, (str, bytes)):
                panel_column_name = panel_column
                panel_column = self._obj[panel_column]
            else:
                panel_column_name: str | None = (
                    str(panel_column.name) if panel_column.name is not None else "panel"
                )
        else:
            panel_column_name = None

        out_dict.complete_time_series = complete_time_series
        out_dict.data = data
        out_dict.start, out_dict.end = start, end
        out_dict.is_date = is_date
        out_dict.panel_column_name = panel_column_name
        out_dict.data.sort_values(ts_column_name, inplace=True)
        return out_dict

    def tsfill(
        self,
        *,
        fill_value: Any = None,
        method: str | None = None,
        sentinel: str | None = None,
        keep_index: bool = False,
        avoid_float_casts: bool = True,
        opts_replacement: PandasOpts | dict | None = None,
    ) -> pd.DataFrame:
        """
        Fill in holes in time-series or panel data

        Args:
            fill_value: Value to fill in NAs passed to `pandas.DataFrame.reindex`
            method: Method of filling in NAs passed to `pandas.DataFrame.reindex`.
            sentinel: If not None, the name of a column indicating if a row was
                present in the original data (True) or was filled in by `tsfill`
                (False)
            keep_index: If True, the index of the data returned will be the index of
                the original data with null values for filled-in observations
            avoid_float_casts: Use Pandas nullable dtypes to avoid casting integer columns
                to float when NAs are filled in. Has no effect for `pyspark.pandas` DataFrames.
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing PandasOpts arguments from `ts()`


        Returns:
            Pandas DataFrame

        Examples:
            >>> from tslib.pandas_api import PandasOpts
            >>> import pandas as pd
            >>> cookies = pd.DataFrame(
            ...   {
            ...        "year": [2000, 2001, 2002, 2003, 2008],
            ...        "favorite": [
            ...         "Chocolate Chip",
            ...         "Chocolate Chip",
            ...         "Oatmeal Raisin",
            ...         "Sugar",
            ...         "M&M",
            ...    ],
            ...    "n": [10, 20, 15, 12, 40],
            ...   }
            ... )
            >>> cookies
               year        favorite   n
            0  2000  Chocolate Chip  10
            1  2001  Chocolate Chip  20
            2  2002  Oatmeal Raisin  15
            3  2003           Sugar  12
            4  2008             M&M  40
            >>> cookies_args = PandasOpts(ts_column="year", freq=1, start=1999)
            >>> cookies_ts = cookies.ts(cookies_args)
            >>> cookies_full = cookies_ts.tsfill()
            >>> cookies_full
               year        favorite     n
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

        """
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        is_panel = PandasOpts._extract_opt(ts_args, "panel_column") is not None
        ts = self.tsset(ts_args=ts_args)
        assert ts.data is not None
        out = ts.data.copy()
        if sentinel is not None:
            sentinel = str(sentinel)
            out[sentinel] = True
        if keep_index:
            out["__index__"] = out.index
        if avoid_float_casts:
            # prevent ints from being turned into floats because of NAs
            out = out.convert_dtypes(
                convert_string=False, infer_objects=False, convert_floating=False  # type: ignore
            )

        if not is_panel:
            out.set_index(ts.ts_column_name, drop=True, inplace=True)
            out.index.name = None
            out = out.reindex(
                ts.complete_time_series, method=method, fill_value=fill_value
            )
            out.index.name = "__temporary_index__"
            out.reset_index(
                drop=False, inplace=True
            )  # this will create a column called "__temporary_index__"
            out.rename(columns={"__temporary_index__": ts.ts_column_name}, inplace=True)
            out.index.name = None

        else:
            out.set_index(ts.ts_column_name, drop=False, inplace=True)
            out.index.name = None
            out_grouped = out.groupby(ts.panel_column_name)
            new_groups: list[None | pd.DataFrame] = [
                None for _ in range(len(out_grouped))
            ]
            for i, (key, _) in zip(range(len(out_grouped)), out_grouped.groups.items()):
                subset = out.loc[out[ts.panel_column_name] == key]
                if ts.engine == pd:
                    new_groups[i] = subset.reindex(
                        ts.complete_time_series, method=method, fill_value=fill_value
                    )
                else:
                    ts.complete_time_series.name = None
                    new_groups[i] = subset.reindex(
                        ts.complete_time_series, fill_value=fill_value
                    )
                new_groups[i][ts.ts_column_name] = ts.complete_time_series
                new_groups[i][ts.panel_column_name] = key
            out = pd.concat(new_groups, axis=0)  # type: ignore
        if sentinel is not None:
            out.fillna({sentinel: False}, inplace=True)
        if keep_index:
            out.set_index("__index__", drop=True, inplace=True)
            out.index.name = None
        else:
            out.reset_index(drop=True, inplace=True)
        return out

    def with_lag(
        self,
        col_name: str,
        column: str | pd.Series,
        back: int = 1,
        *,
        opts_replacement: PandasOpts | dict | None = None,
    ) -> pd.DataFrame:
        """
        Add a lag column to a Pandas DataFrame

        Args:
            col_name: What to name the lag column
            column: Column to take the lag of
            back: How many records to go back. Negative values are "leads"
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing PandasOpts arguments from `ts()`

        Returns:
            Pandas DataFrame

        Examples:
            >>> from tslib.pandas_api import PandasOpts
            >>> import pandas as pd
            >>> cookies = pd.DataFrame(
            ...   {
            ...        "year": [2000, 2001, 2002, 2003, 2008],
            ...        "favorite": [
            ...         "Chocolate Chip",
            ...         "Chocolate Chip",
            ...         "Oatmeal Raisin",
            ...         "Sugar",
            ...         "M&M",
            ...    ],
            ...    "n": [10, 20, 15, 12, 40],
            ...   }
            ... )
            >>> cookies
               year        favorite   n
            0  2000  Chocolate Chip  10
            1  2001  Chocolate Chip  20
            2  2002  Oatmeal Raisin  15
            3  2003           Sugar  12
            4  2008             M&M  40
            >>> cookies_args = PandasOpts(ts_column="year", freq=1, start=1999)
            >>> cookies_ts = cookies.ts(cookies_args)
            >>> cookies_lag = cookies_ts.with_lag("previous_favorite", column="favorite")
            >>> cookies_lag
               year        favorite   n previous_favorite
            0  2000  Chocolate Chip  10               NaN
            1  2001  Chocolate Chip  20    Chocolate Chip
            2  2002  Oatmeal Raisin  15    Chocolate Chip
            3  2003           Sugar  12    Oatmeal Raisin
            4  2008             M&M  40               NaN

        """
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        ts = self.tsset(ts_args=ts_args)
        assert ts.data is not None
        assert ts.freq is not None
        if isinstance(column, str):
            column_string: str = column
            column = ts.data[column]
        else:
            column_string: str = column.name

        if ts.panel_column_name is None:
            if ts.is_date:
                # pessimisation: we can't reliably predict the next element of the series
                # if the column is a date, so we have to use the "complete" time series.
                # Thus, tsfill the data and then use the "shift"-based lag method.
                # Then filter to the original series.
                out = ts.data.ts(ts_args).tsfill(
                    sentinel="__sentinel__", keep_index=True
                )
                out[col_name] = out[column_string].shift(back)
                out = out[out["__sentinel__"]]
                out.drop(["__sentinel__"], inplace=True, axis=1)
            else:
                assert ts.ts_column_name is not None
                lagged_col = ts.data[ts.ts_column_name] + (ts.freq * back)  # type: ignore
                new = pd.DataFrame(
                    {ts.ts_column_name: lagged_col, col_name: ts.data[column_string]}
                )
                out = ts.data.merge(new, on=ts.ts_column_name, how="left")
        else:
            if ts.is_date:
                out = ts.data.ts(ts_args).tsfill(
                    sentinel="__sentinel__", keep_index=True
                )
                out[col_name] = out.groupby(ts.panel_column_name)[column_string].shift(
                    back
                )
                if ts.engine == pd:
                    out.drop(["__sentinel__"], inplace=True, axis=1)
                else:
                    out = out.drop(["__sentinel__"], axis=1)
            else:
                assert ts.ts_column_name is not None
                lagged_col = ts.data[ts.ts_column_name] + (ts.freq * back)  # type: ignore
                new = pd.DataFrame(
                    {
                        ts.ts_column_name: lagged_col,
                        col_name: ts.data[column_string],
                        ts.panel_column_name: ts.data[ts.panel_column_name],
                    }
                )
                out = ts.data.merge(
                    new, on=[ts.ts_column_name, ts.panel_column_name], how="left"
                )

        out.sort_index(inplace=True)
        return out

    def with_lead(
        self,
        col_name: str,
        column: pd.Series | str,
        forward: int | None = 1,
        opts_replacement: PandasOpts | dict | None = None,
    ) -> pd.DataFrame:
        """
        Add a lead column to a Pandas DataFrame

        Args:
            col_name: What to name the lead column
            column: Column to take the lead of
            forward: How many records to go forward. Negative values are "lags"
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing PandasOpts arguments from `ts()`

        Returns:
            Pandas DataFrame or pandas-on-Spark DataFrame

        Examples:
            >>> from tslib.pandas_api import PandasOpts
            >>> import pandas as pd
            >>> cookies = pd.DataFrame(
            ...   {
            ...        "year": [2000, 2001, 2002, 2003, 2008],
            ...        "favorite": [
            ...         "Chocolate Chip",
            ...         "Chocolate Chip",
            ...         "Oatmeal Raisin",
            ...         "Sugar",
            ...         "M&M",
            ...    ],
            ...    "n": [10, 20, 15, 12, 40],
            ...   }
            ... )
            >>> cookies
               year        favorite   n
            0  2000  Chocolate Chip  10
            1  2001  Chocolate Chip  20
            2  2002  Oatmeal Raisin  15
            3  2003           Sugar  12
            4  2008             M&M  40
            >>> cookies_args = PandasOpts(ts_column="year", freq=1, start=1999)
            >>> cookies_ts = cookies.ts(cookies_args)
            >>> cookies_lead = cookies_ts.with_lead("next_favorite", column="favorite")
            >>> cookies_lead
               year        favorite   n   next_favorite
            0  2000  Chocolate Chip  10  Chocolate Chip
            1  2001  Chocolate Chip  20  Oatmeal Raisin
            2  2002  Oatmeal Raisin  15           Sugar
            3  2003           Sugar  12             NaN
            4  2008             M&M  40             NaN
        """
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        return self.with_lag(
            col_name, column, back=-1 * forward, opts_replacement=ts_args
        )

    def with_difference(
        self,
        col_name: str,
        column: str | pd.Series,
        back: int | None = 1,
        *,
        opts_replacement: PandasOpts | dict | None = None,
    ) -> pd.DataFrame:
        """
        Add a difference column to a Pandas DataFrame

        Args:
            col_name: What to name the difference column
            column: Column to take the difference of
            back: How many records to go back to compute difference.
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing PandasOpts arguments from `ts()`

        Returns:
            Pandas DataFrame or pandas-on-Spark DataFrame

        Examples:
            >>> from tslib.pandas_api import PandasOpts
            >>> import pandas as pd
            >>> cookies = pd.DataFrame(
            ...   {
            ...        "year": [2000, 2001, 2002, 2003, 2008],
            ...        "favorite": [
            ...         "Chocolate Chip",
            ...         "Chocolate Chip",
            ...         "Oatmeal Raisin",
            ...         "Sugar",
            ...         "M&M",
            ...    ],
            ...    "n": [10, 20, 15, 12, 40],
            ...   }
            ... )
            >>> cookies
               year        favorite   n
            0  2000  Chocolate Chip  10
            1  2001  Chocolate Chip  20
            2  2002  Oatmeal Raisin  15
            3  2003           Sugar  12
            4  2008             M&M  40
            >>> cookies_args = PandasOpts(ts_column="year", freq=1, start=1999)
            >>> cookies_ts = cookies.ts(cookies_args)
            >>> cookies_diff = cookies_ts.with_difference("change_in_panelists", column="n")
            >>> cookies_diff
               year        favorite   n  change_in_panelists
            0  2000  Chocolate Chip  10                  NaN
            1  2001  Chocolate Chip  20                 10.0
            2  2002  Oatmeal Raisin  15                 -5.0
            3  2003           Sugar  12                 -3.0
            4  2008             M&M  40                  NaN
        """
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        lag = self.with_lag(
            "__difference_dummy__", column, back, opts_replacement=ts_args
        )["__difference_dummy__"]
        curr = self._obj[column]
        out = self._obj.copy()
        out[col_name] = curr - lag
        return out
