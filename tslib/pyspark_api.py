from __future__ import annotations

from dataclasses import dataclass
from pyspark.sql import SparkSession, DataFrame
from datetime import datetime
import pandas as pd
from typing import Any
import numpy as np
from tslib.pandas_api import PandasOpts
from tslib.pyspark_utils import (
    _is_acceptable_type,
    _is_numeric,
    _is_date,
    _is_numeric_scalar,
    offsets_since_epoch,
)
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window


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
        if not _is_acceptable_type(self._obj, ts_column):
            raise Exception(f"{ts_column.dtype} not an acceptable time-series type")
        if _is_numeric(self._obj, ts_column) and not _is_numeric_scalar(freq):
            raise Exception(
                "For a numeric time-series variable" "Frequency must be a number."
            )
        is_date = _is_date(self._obj,ts_column)
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
            start = self._obj.agg(F.min(ts_column).alias("__min__")).toPandas()[
                "__min__"
            ][0]

        if end is not None:
            if is_date:
                end = pd.to_datetime(end)
        else:
            end = self._obj.agg(F.max(ts_column).alias("__max__")).toPandas()[
                "__max__"
            ][0]

        if is_date:
            complete_time_series = pd.period_range(
                start=start, end=end, freq=out_dict.freq
            ).to_timestamp()  # type: ignore
        else:
            # note that `end` should be inclusive whereas range is top-exclusive
            complete_time_series = np.arange(start, end + freq, out_dict.freq)

        data = self._obj
        if is_date:
            data = data.withColumn('__offsets_since_epoch__', offsets_since_epoch(freq)(ts_column))

        is_date = _is_date(self._obj, ts_column)
        out_dict.complete_time_series = complete_time_series
        out_dict.data = data
        out_dict.start, out_dict.end = start, end
        out_dict.is_date = is_date
        out_dict.panel_column_name = panel_column
        out_dict.ts_column_name = ts_column
        return out_dict

    def tsfill(
        self,
        *,
        fill_value: Any = None,
        sentinel: str | None = None,
        opts_replacement: SparkOpts | dict | None = None,
    ) -> pyspark.sql.dataframe.DataFrame:
        """
        Fill in holes in time-series or panel data

        Args:
            fill_value: Value to fill in NAs generated by `tsfill`
            sentinel: If not None, the name of a column indicating if a row was
                present in the original data (True) or was filled in by `tsfill`
                (False)
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing TimeOpts arguments from `ts()`


        Returns:
            Pyspark DataFrame

        Note:
            `tsfill` may shuffle the order of columns and rows in the data.

        Examples:
            >>> from tslib.pyspark_api import SparkOpts
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
            >>> cookies_args = SparkOpts(ts_column="year", freq=1, start=1999)
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
        full = spark.createDataFrame(
            pd.DataFrame({ts.ts_column_name: ts.complete_time_series})
        )
        dta = ts.data
        if sentinel is None and fill_value is not None:
            sentinel = "__sentinel_dropped__"
        if sentinel is not None:
            dta = dta.withColumn(sentinel, F.lit(True))
        if not is_panel:
            out = full.join(dta, ts.ts_column_name, "left")
        else:
            ids = ts.data.select(ts.panel_column_name).distinct()
            full = full.crossJoin(ids)
            out = full.join(dta, [ts.ts_column_name, ts.panel_column_name], "left")
        if sentinel is not None:
            out = out.withColumn(sentinel, F.coalesce(F.col(sentinel), F.lit(False)))
            if fill_value is not None:
                # we can't use fillna here since it doesn't admit a predicate.
                # instead, use `sentinel` as predicate and fill in only rows for which sentinel=TRUE.
                xt_cols = [ts.ts_column_name,ts.panel_column_name] if is_panel else [ts.ts_column_name]
                other_cols = [col for col in out.columns if col not in xt_cols]
                exprs = [F.expr(f"CASE WHEN {sentinel} THEN {fill_value} ELSE {col} END") for col in other_cols]
                out = out.select(xt_cols, exprs)
        return out.drop("__sentinel_dropped__", "__offsets_since_epoch__")

    def with_lag(
        self,
        col_name: str,
        column: str,
        back: int = 1,
        *,
        opts_replacement: SparkOpts | dict | None = None,
    ) -> pyspark.sql.dataframe.DataFrame:
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

        Note:
            `with_lag` may shuffle the order of columns and rows in the data.

        Examples:
            >>> from tslib.pyspark_api import SparkOpts
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
            >>> cookies_args = SparkOpts(ts_column="year", freq=1, start=1999)
            >>> cookies_ts = cookies.ts(cookies_args)
            >>> cookies_lag = cookies_ts.with_lag("previous_favorite", column="favorite")
            >>> cookies_lag.toPandas()
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
        is_panel = SparkOpts._extract_opt(ts_args, "panel_column") is not None
        ts = self.tsset(ts_args=ts_args)
        assert ts.data is not None
        if not ts.is_date:
            # in this case, we just order by ts.ts_column_name
            # and use ts.freq as the rangeSpec
            dta = ts.data
            order_column = ts.ts_column_name
            offset = ts.freq
            range_spec = (-1 * back * offset, -1 * back * offset)
        else:
            # in this case, we order by the numeric value of ts.ts_column
            # and use ts.freq.n as the rangeSpec
            dta = ts.data
            order_column = "__offsets_since_epoch__"
            offset = ts.freq.n
            range_spec = (-1 * back * offset, -1 * back * offset)

        if not is_panel:
            return dta.withColumn(
                    col_name,
                    F.max(column).over(
                        Window.orderBy(order_column).rangeBetween(
                            *range_spec
                        )
                    ),
            ).drop("__offsets_since_epoch__")
        else:
            return dta.withColumn(
                    col_name,
                    F.max(column).over(
                        Window.partitionBy(ts.panel_column_name)
                        .orderBy(order_column)
                        .rangeBetween(*range_spec)
                    ),
            ).drop("__offsets_since_epoch__")
    
            
    def with_lead(
            self,
            col_name: str,
            column: str,
            forward: int | None = 1,
            opts_replacement: SparkOpts | dict | None = None,
        ) -> pyspark.sql.dataframe.DataFrame:
            """
            Add a lead column to a PySpark DataFrame

            Args:
                col_name: What to name the lead column
                column: Column to take the lead of
                forward: How many records to go forward. Negative values are "lags"
                opts_replacement: Replace Arguments for the time-series structure of the data.
                    Defaults to the existing SparkOpts arguments from `ts()`

            Returns:
                PySpark DataFrame

            Note:
                `with_lag` may shuffle the order of columns and rows in the data.

            Examples:
                >>> from tslib.pyspark_api import SparkOpts
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
                >>> cookies_args = SparkOpts(ts_column="year", freq=1, start=1999)
                >>> cookies_ts = cookies.ts(cookies_args)
                >>> cookies_lead = cookies_ts.with_lead("next_favorite", column="favorite")
                >>> cookies_lead.toPandas()
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
            ).drop("__offsets_since_epoch__")

    def with_difference(
        self,
        col_name: str,
        column: str,
        back: int | None = 1,
        *,
        opts_replacement: SparkOpts | dict | None = None,
    ) -> pyspark.sql.dataframe.DataFrame:
        """
        Add a difference column to a Pandas DataFrame

        Args:
            col_name: What to name the difference column
            column: Column to take the difference of
            back: How many records to go back to compute difference.
            opts_replacement: Replace Arguments for the time-series structure of the data.
                Defaults to the existing PandasOpts arguments from `ts()`

        Returns:
            PySpark DataFrame

        Note:
            `with_lag` may shuffle the order of columns and rows in the data.

        Examples:
            >>> from tslib.pyspark_api import SparkOpts
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
            >>> cookies_args = SparkOpts(ts_column="year", freq=1, start=1999)
            >>> cookies_ts = cookies.ts(cookies_args)
            >>> cookies_diff = cookies_ts.with_difference("change_in_panelists", column="n")
            >>> cookies_diff.toPandas()
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
        )
        out = lag.withColumn(col_name, F.col(column) - F.col("__difference_dummy__")).drop("__difference_dummy__")
        return out.drop("__offsets_since_epoch__")



class _ts:
    def __get__(self, obj, objtype=None):
        return TSAccessor(obj)


DataFrame.ts = _ts()
