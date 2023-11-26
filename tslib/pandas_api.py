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

try:
    from pyspark.pandas.extensions import (
        register_dataframe_accessor as pyspark_register_dataframe_accessor,
    )
except ImportError:
    pass

if TYPE_CHECKING:
    import pyspark.pandas as ps
    import numpy as np


@dataclass
class TimeOpts:
    ts_column: str | pd.Series | ps.Series
    panel_column: str | pd.Series | ps.Series | None = None
    freq: int | float | str | pd.offsets.DateOffset | None = None
    start: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None
    end: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None

    def assign(self, inplace=False, **kwargs) -> TimeOpts | None:
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
    def _extract_opt(opts: TimeOpts | dict, opt: str):
        if isinstance(opts, TimeOpts):
            return getattr(opts, opt)
        else:
            return opts[opt]

    @staticmethod
    def _extract_opts(opts: TimeOpts | dict):
        if isinstance(opts, TimeOpts):
            return (opts.ts_column, opts.panel_column, opts.freq, opts.start, opts.end)
        else:
            out = TimeOpts(
                opts["ts_column"],
                opts.get("panel_column", None),
                opts.get("freq", None),
                opts.get("start", None),
                opts.get("end", None),
            )
            return (out.ts_column, out.panel_column, out.freq, out.start, out.end)


def _maybe_pyspark(name: str):
    def deco(cls: type):
        try:
            return pyspark_register_dataframe_accessor(name)(cls)
        except Exception:
            return cls
    return deco
        

# _maybe_pyspark("ts")(TSAccessor)
# _maybe_pyspark("ts") returns
# def deco(cls):
#   try:
#     return pyspark_register_dataframe_accessor("ts")
#   except Exception:
#     return cls
#
# so deco(TSAccessor) either returns
# pyspark_register_dataframe_accessor("ts")(TSAccessor)
# or TSAccessor


@_maybe_pyspark("ts")
@pd.api.extensions.register_dataframe_accessor("ts")
class TSAccessor:
    def __init__(self, obj: pd.DataFrame | ps.DataFrame):
        try:
            import pyspark.pandas as ps
            ps.set_option('compute.ops_on_diff_frames', True)
            if isinstance(obj, ps.DataFrame):
                self._engine = ps
            else:
                self._engine = pd
        except ImportError:
            self._engine = pd

        self._obj = obj

    def __call__(self, opts: TimeOpts | dict) -> TSAccessor:
        self._opts = opts
        return self

    def tsset(
        self,
        ts_args: TimeOpts | dict,
    ) -> TimeDict:
        data = self._obj.copy()
        ts_column, panel_column, freq, start, end = TimeOpts._extract_opts(ts_args)
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
            # we don't have a way of making the complete time series in Spark right now,
            # so we're limited to Pandas-sized time series. That's probably fine though not ideal:
            # we can still have a time series for every minute since 1950 without issue
            complete_time_series = pd.period_range(
                start=start, end=end, freq=out_dict.freq
            ).to_timestamp()  # type: ignore
            if self._engine != pd:
                complete_time_series = self._engine.from_pandas(complete_time_series)
        else:
            # note that `end` should be inclusive whereas range is top-exclusive
            complete_time_series = _range(self._engine)(
                start, end + freq, out_dict.freq
            )
        if self._engine == pd:
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
        out_dict.engine = self._engine
        return out_dict

    def tsfill(
        self,
        *,
        fill_value: Any = None,
        method: str | None = None,
        sentinel: str | None = None,
        keep_index: bool = False,
        avoid_float_casts: bool = True,
        opts_replacement: TimeOpts | dict | None = None,
    ) -> pd.DataFrame | ps.DataFrame:
        """
        Fill in holes in time-series or panel data

        Args:
            fill_value: Value to fill in NAs passed to `pandas.DataFrame.reindex`
            method: Method of filling in NAs passed to `pandas.DataFrame.reindex`.
                Not available on PySpark.
            sentinel: If not None, the name of a column indicating if a row was
                present in the original data (True) or was filled in by `tsfill`
                (False)
            keep_index: If True, the index of the data returned will be the index of
                the original data with null values for filled-in observations
            avoid_float_casts: Use Pandas nullable dtypes to avoid casting integer columns
                to float when NAs are filled in. Has no effect for `pyspark.pandas` DataFrames.
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing TimeOpts arguments from `ts()`


        Returns:
            Pandas DataFrame or pandas-on-Spark DataFrame

        Examples:
            >>> from tslib.pandas_api import TimeOpts
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
            >>> cookies_args = TimeOpts(ts_column="year", freq=1, start=1999)
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
        is_panel = TimeOpts._extract_opt(ts_args, "panel_column") is not None
        ts = self.tsset(ts_args=ts_args)
        if method is not None and ts.engine != pd:
            raise Exception("`method` argument not available for PySpark DataFrames")
        assert ts.data is not None
        out = ts.data.copy()
        if sentinel is not None:
            sentinel = str(sentinel)
            out[sentinel] = True
        if keep_index:
            out["__index__"] = out.index
        if avoid_float_casts and ts.engine == pd:
            # prevent ints from being turned into floats because of NAs
            out = out.convert_dtypes(
                convert_string=False, infer_objects=False, convert_floating=False  # type: ignore
            )

        if not is_panel:
            out.set_index(ts.ts_column_name,drop=True,inplace=True)
            out.index.name = None
            if ts.engine == pd:
                out = out.reindex(
                    ts.complete_time_series, method=method, fill_value=fill_value
                )
            else:
                ts.complete_time_series.name = None
                out = out.reindex(
                    ts.complete_time_series, fill_value=fill_value
                )
            out.index.name = "__temporary_index__"
            out.reset_index(drop=False,inplace=True) # this will create a column called "__temporary_index__"
            out.rename(columns={"__temporary_index__":ts.ts_column_name},inplace=True)
            out.index.name = None

        else:
            out.set_index(ts.ts_column_name,drop=False,inplace=True)
            out.index.name = None
            out_grouped = out.groupby(ts.panel_column_name)
            new_groups: list[None | pd.DataFrame | ps.DataFrame] = [
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
            out = ts.engine.concat(new_groups, axis=0)  # type: ignore
        if sentinel is not None:
            out.fillna({sentinel:False},inplace=True)
        if keep_index:
            out.set_index("__index__",drop=True,inplace=True)
            out.index.name = None
        else:
            out.reset_index(drop=True,inplace=True)
        return out

    def with_lag(
        self,
        col_name: str,
        column: str | pd.Series,
        back: int = 1,
        *,
        opts_replacement: TimeOpts | dict | None = None,
    ) -> pd.DataFrame | ps.DataFrame:
        """
        Add a lag column to a Pandas DataFrame

        Args:
            col_name: What to name the lag column
            column: Column to take the lag of
            back: How many records to go back. Negative values are "leads"
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing TimeOpts arguments from `ts()`

        Returns:
            Pandas DataFrame or pandas-on-Spark DataFrame

        Examples:
            >>> from tslib.pandas_api import TimeOpts
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
            >>> cookies_args = TimeOpts(ts_column="year", freq=1, start=1999)
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
                if ts.engine == pd:
                    out.drop(["__sentinel__"], inplace=True, axis=1)
                else:
                    out = out.drop(["__sentinel__"], axis=1)
            else:
                assert ts.ts_column_name is not None
                lagged_col = ts.data[ts.ts_column_name] + (ts.freq * back)  # type: ignore
                if ts.engine == pd:
                    new = ts.engine.DataFrame(
                        {ts.ts_column_name: lagged_col, col_name: ts.data[column_string]}
                    )
                else: # Why does pyspark.pandas require us to do this???
                    new = ts.engine.DataFrame(index=ts.data.index)
                    new[ts.ts_column_name] = lagged_col
                    new[col_name] = ts.data[column_string]
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
                if ts.engine == pd:
                    new = ts.engine.DataFrame(
                        {
                            ts.ts_column_name: lagged_col,
                            col_name: ts.data[column_string],
                            ts.panel_column_name: ts.data[ts.panel_column_name],
                        }
                    )
                else: 
                    new = ts.engine.DataFrame(index=ts.data.index)
                    new[ts.ts_column_name] = lagged_col
                    new[col_name] = ts.data[column_string]
                    new[ts.panel_column_name] = ts.data[ts.panel_column_name]
                out = ts.data.merge(
                    new, on=[ts.ts_column_name, ts.panel_column_name], how="left"
                )

        out.sort_index(inplace=True)
        return out

    def with_lead(
        self,
        col_name: str,
        column: pd.Series | ps.Series | str,
        forward: int | None = 1,
        opts_replacement: TimeOpts | dict | None = None,
    ) -> pd.DataFrame | ps.DataFrame:
        """
        Add a lead column to a Pandas DataFrame

        Args:
            col_name: What to name the lead column
            column: Column to take the lead of
            forward: How many records to go forward. Negative values are "lags"
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing TimeOpts arguments from `ts()`

        Returns:
            Pandas DataFrame or pandas-on-Spark DataFrame

        Examples:
            >>> from tslib.pandas_api import TimeOpts
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
            >>> cookies_args = TimeOpts(ts_column="year", freq=1, start=1999)
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
        opts_replacement: TimeOpts | dict | None = None,
    ) -> pd.DataFrame | ps.DataFrame:
        """
        Add a difference column to a Pandas DataFrame

        Args:
            col_name: What to name the difference column
            column: Column to take the difference of
            back: How many records to go back to compute difference.
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing TimeOpts arguments from `ts()`

        Returns:
            Pandas DataFrame or pandas-on-Spark DataFrame

        Examples:
            >>> from tslib.pandas_api import TimeOpts
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
            >>> cookies_args = TimeOpts(ts_column="year", freq=1, start=1999)
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

