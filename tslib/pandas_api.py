from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Iterable
from dataclasses import dataclass
from datetime import datetime

from tslib.utils import (
    _is_acceptable_type,
    _is_date,
    _is_numeric,
    _is_numeric_scalar,
    TimeDict,
)


@dataclass
class TimeOpts:
    ts_column: str | pd.Series
    panel_column: str | pd.Series | None = None
    freq: int | float | str | pd.offsets.DateOffset | None = None
    start: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None
    end: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None

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


@pd.api.extensions.register_dataframe_accessor("ts")
class TSAccessor:
    def __init__(self, obj: pd.DataFrame):
        self._obj = obj

    def __call__(self, opts: TimeOpts | dict):
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
        data.index = np.arange(len(data.index))  # type: ignore
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
            # note that `end` should be inclusive whereas np.arange is top-exclusive
            complete_time_series = np.arange(
                start=start, stop=end + freq, step=out_dict.freq
            )
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
        method: str | None = None,
        opts_replacement: TimeOpts | dict | None = None,
        fill_value: Any = None,
        sentinel: str | None = None,
        avoid_float_casts: bool = True,
    ) -> pd.DataFrame:
        """
        Fill in holes in time-series or panel data

        Parameters
        ----------
        ts_args: TimeOpts or compatible dict
            Arguments for the time-series structure of the data.
        method: str or None, default None
            Fill method passed to `pandas.DataFrame.reindex`
        fill_value: Any, default None
            Value to fill in NAs passed to `pandas.DataFrame.reindex`
        sentinel: str, default None
            If not None, the name of a column indicating if a row was
            present in the original data (True) or was filled in by `tsfill`
            (False)
        avoid_float_casts: bool, default True
            Use Pandas nullable dtypes to avoid casting integer columns
            to float when NAs are filled in.
        """
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        is_panel = TimeOpts._extract_opt(ts_args, "panel_column") is not None
        if not is_panel:
            ts = self.tsset(ts_args=ts_args)
        else:
            ts = self.tsset(ts_args=ts_args)
        out = ts.data
        assert out is not None
        if sentinel is not None:
            sentinel = str(sentinel)
            out[sentinel] = True
        if avoid_float_casts:
            # prevent ints from being turned into floats because of NAs
            out = out.convert_dtypes(
                convert_string=False, infer_objects=False, convert_floating=False  # type: ignore
            )

        assert ts.ts_column_name is not None
        if not is_panel:
            assert isinstance(ts.complete_time_series, Iterable)
            out.index = out[ts.ts_column_name]  # type: ignore
            out = out.reindex(
                ts.complete_time_series, method=method, fill_value=fill_value
            )
            out[ts.ts_column_name] = ts.complete_time_series

        else:
            assert ts.panel_column_name is not None
            out.index = out[ts.ts_column_name]  # type: ignore
            out_grouped = out.groupby(ts.panel_column_name)
            new_groups: list[None | pd.DataFrame] = [
                None for _ in range(len(out_grouped))
            ]
            for i, (key, _) in zip(range(len(out_grouped)), out_grouped.groups.items()):
                subset = out.loc[out[ts.panel_column_name] == key]
                new_groups[i] = subset.reindex(
                    ts.complete_time_series, method=method, fill_value=fill_value
                )
                new_groups[i][ts.ts_column_name] = ts.complete_time_series
                new_groups[i][ts.panel_column_name] = key
            out = pd.concat(new_groups, axis=0)  # type: ignore
        if sentinel is not None:
            out[sentinel] = out[sentinel].fillna(False)
        out.index = np.arange(len(out.index))
        return out

    def with_lag(
        self,
        col_name: str,
        column: str | pd.Series,
        back: int = 1,
        *,
        opts_replacement: TimeOpts | dict | None = None,
    ) -> pd.DataFrame:
        """
        Add a lag column to a Pandas DataFrame

        Parameters
        ----------
        col_name: str, default None
            What to name the lag column
        column: str or pandas.Series
            Column to take the lag of
        back: int, default 1
            How many records to go back. Negative values are "leads"
        opts_replacement: TimeOpts or compatible dict or None
            Replace Arguments for the time-series structure of the data.
            Defaults to the existing TimeOpts arguments from `ts()`
        """
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        ts = self.tsset(ts_args=ts_args)
        assert ts.data is not None
        assert ts.freq is not None
        if isinstance(column,str):
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
                out = ts.data.ts(ts_args).tsfill(sentinel="__sentinel__")
                out[col_name] = out[column_string].shift(back)
                out = out[out["__sentinel__"]]
                out.drop("__sentinel__", inplace=True, axis=1)
            else:
                assert ts.ts_column_name is not None
                lagged_col = ts.data[ts.ts_column_name] + (ts.freq * back)  # type: ignore
                new = pd.DataFrame({ts.ts_column_name: lagged_col, col_name: ts.data[column_string]})
                out = ts.data.merge(new, on=ts.ts_column_name, how="left")
        out.index = np.arange(len(out.index))
        return out

    def with_lead(
        self,
        col_name: str,
        column: pd.Series | str,
        forward=1,
        opts_replacement: TimeOpts | dict | None = None,
    ) -> pd.DataFrame:
        """
        Add a lead column to a Pandas DataFrame

        Parameters
        ----------
        col_name: str, default None
            What to name the lead column
        column: pandas.Series
            Column to take the lead of
        forward: int, default 1
            How many records to go forward. Negative values are "lags"
        opts_replacement: TimeOpts or compatible dict or None
            Replace Arguments for the time-series structure of the data.
            Defaults to the existing TimeOpts arguments from `ts()`
        """
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        return self.with_lag(col_name, column, back=-1 * forward, opts_replacement=ts_args)

    def with_difference(
        self,
        col_name: str,
        column: str | pd.Series,
        back=1,
        *,
        opts_replacement: TimeOpts | dict | None = None,
    ) -> pd.DataFrame:
        """
        Add a difference column to a Pandas DataFrame

        Parameters
        ----------
        col_name: str, default None
            What to name the difference column
        column: pandas.Series
            Column to take the difference of
        back: int, default 1
            How many records to go back to compute difference.
        opts_replacement: TimeOpts or compatible dict or None
            Replace Arguments for the time-series structure of the data.
            Defaults to the existing TimeOpts arguments from `ts()`
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
