from __future__ import annotations

import pyspark
from pyspark.sql.types import IntegerType
import re
from numbers import Number
import pandas as pd
from pyspark.sql.functions import pandas_udf

def _is_numeric(df: pyspark.sql.dataframe.DataFrame, col: str):
    return (
        re.match(
            r"decimal|double|float|integer|long|short|bigint|tinyint|smallint",
            dict(df.dtypes)[col],
        )
        is not None
    )


def _is_numeric_scalar(num):
    return isinstance(num, Number)


def _is_acceptable_type(df: pyspark.sql.dataframe.DataFrame, col: str):
    return _is_numeric(df, col) or _is_date(df, col)


def _is_date(df: pyspark.sql.dataframe.DataFrame, col: str):
    return re.match(r"date|timestamp", dict(df.dtypes)[col]) is not None

def offsets_since_epoch(o: pd.DateOffset):
    """We have to curry the DateOffset into the UDF bc all the arguments have to be pd.Series"""
    @pandas_udf(IntegerType())
    def _inner(s: pd.Series) -> pd.Series:
        """Converts a date column to an integer column based on the interval of the data.
        For example, if the data are monthly, the output should be months since the epoch,
        if the data are minutely then it should be the minutes since epoch, etc."""
        nonlocal o
        if isinstance(o,str):
            o = pd.tseries.frequencies.to_offset(o)
        idx = pd.Index(s)
        int_idx = idx.to_period(o).astype(int) // o.n
        return pd.Series(int_idx)
    return _inner
