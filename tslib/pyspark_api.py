from __future__ import annotations

from dataclasses import dataclass
from pyspark.sql import SparkSession,DataFrame
from datetime import datetime
import pandas as pd
from typing import Any
import numpy as np
from tslib.pandas_api import PandasOpts
from tslib.pyspark_api import _is_acceptable_type,_is_numeric,_is_date, _is_numeric_scalar
import pyspark
import pyspark.sql.functions as F

class SparkOpts(PandasOpts):
    ts_column: str
    panel_column: str | None = None
    freq: int | float | str | pd.offsets.DateOffset | None = None
    start: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None
    end: int | float | str | pd.Timestamp | datetime | np.datetime64 | None = None

@dataclass
class _TimeDict:
    data: pd.DataFrame | None = None
    ts_column_name: str | None = None
    panel_column_name: str | None = None
    complete_time_series: pd.Series | None = None
    freq: pd.offsets.DateOffset | int | float | None = None
    start: pd.Timestamp | np.datetime64 | datetime | None = None
    end: pd.Timestamp | np.datetime64 | datetime | None = None
    is_date: bool | None = None

class TSAccessor:
    def __init__(self, obj: pyspark.sql.dataframe.DataFrame):
        self._obj = obj

    def __call__(self, opts):
        self._opts = opts
        return self

    def tsset(self, ts_args: SparkOpts | dict):
        out_dict = _TimeDict()
        ts_column, panel_column, freq, start, end = SparkOpts._extract_opts(ts_args)
        if not _is_acceptable_type(self._obj,ts_column):
            raise Exception(f"{ts_column.dtype} not an acceptable time-series type")
        if _is_numeric(self._obj,ts_column) and not _is_numeric_scalar(freq):
            raise Exception(
                "For a numeric time-series variable" "Frequency must be a number."
            )
        
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
            end = self._obj.agg(F.max(ts_column).alias("__min__")).toPandas()["__min__"][0]

        if end is not None:
            if is_date:
                end = pd.to_datetime(end)
        else:
            end = self._obj.agg(F.max(ts_column).alias("__max__")).toPandas()["__max__"][0]

        if is_date:
            complete_time_series = pd.period_range(
                start=start, end=end, freq=out_dict.freq
            ).to_timestamp()  # type: ignore
        else:
            # note that `end` should be inclusive whereas range is top-exclusive
            complete_time_series = np.arange(
                start, end + freq, out_dict.freq
            )
        is_date = _is_date(self._obj,ts_column)
        out_dict.complete_time_series = complete_time_series
        out_dict.data = self._obj
        out_dict.start, out_dict.end = start, end
        out_dict.is_date = is_date
        out_dict.panel_column_name = panel_column
        return out_dict

    def tsfill(
        self,
        *,
        fill_value: Any = None,
        method: str | None = None,
        sentinel: str | None = None,
        keep_index: bool = False,
        avoid_float_casts: bool = True,
        opts_replacement: SparkOpts | dict | None = None,
    ) -> pyspark.sql.dataframe.DataFrame:
        """
        Fill in holes in time-series or panel data

        Args:
            fill_value: Value to fill in NAs passed to `pandas.DataFrame.reindex`
            method: Method of filling in NAs passed to `pandas.DataFrame.reindex`.
            sentinel: If not None, the name of a column indicating if a row was
                present in the original data (True) or was filled in by `tsfill`
                (False)
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing TimeOpts arguments from `ts()`


        Returns:
            Pyspark DataFrame

        Examples:
            >>> from tslib.pandas_api import TimeOpts
            >>> import pandas as pd
            >>> from pyspark.sql import SparkSession
            >>> spark = SparkSession.builder.getOrCreate()
            >>> cookies = spark.createDataFrame(pd.DataFrame(
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
            ... ))
            >>> cookies.toPandas()
               year        favorite   n
            0  2000  Chocolate Chip  10
            1  2001  Chocolate Chip  20
            2  2002  Oatmeal Raisin  15
            3  2003           Sugar  12
            4  2008             M&M  40
            >>> cookies_args = PysparkOpts(ts_column="year", freq=1, start=1999)
            >>> cookies_ts = cookies.ts(cookies_args)
            >>> cookies_full = cookies_ts.tsfill()
            >>> cookies_full.toPandas()
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
        spark = SparkSession.builder.getOrCreate()
        if opts_replacement is not None:
            ts_args = opts_replacement
        else:
            ts_args = self._opts
        is_panel = SparkOpts._extract_opt(ts_args, "panel_column") is not None
        ts = self.tsset(ts_args=ts_args)
        assert ts.data is not None
        full = spark.createDataFrame(pd.DataFrame({ts.ts_column_name:ts.complete_time_series}))
        if not is_panel:
            out = full.join(ts.data, ts.ts_column_name, "fullouter")
        else:
            ids = ts.data.select(ts.panel_column_name).distinct()
            full = full.crossJoin(ids)
            out = full.join(ts.data, [ts.ts_column_name,ts.panel_column_name], "fullouter")
        return out
       

class _ts:
    def __get__(self, obj,objtype=None):
        return TSAccessor(obj)
DataFrame.ts = _ts()



