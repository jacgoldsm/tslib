from __future__ import annotations

import pyspark
import re
from numbers import Number


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
