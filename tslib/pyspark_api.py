from __future__ import annotations

from dataclasses import dataclass
import pyspark.sql.functions as F
import pandas as pd
from datetime import datetime
from typing import TYPE_CHECKING, Any
from pyspark.sql import SparkSession
from pyspark.sql.column import Column
import pyspark
from pyspark.pandas.extensions import register_dataframe_accessor


from tslib.pandas_api import TSAccessor as PandasAccessor, TimeOpts, TimeDict

from tslib.pyspark_utils import (
    _is_acceptable_type,
    _is_date,
    _is_numeric,
    _is_numeric_scalar,
    TimeDict,
)

if TYPE_CHECKING:
    import numpy as np

spark = SparkSession.builder.getOrCreate()


@register_dataframe_accessor
class TSAccessor(PandasAccessor):
    def __init__(self, obj: pyspark.sql.dataframe.DataFrame | pyspark.pandas.DataFrame):
        if isinstance(obj, pyspark.sql.dataframe.DataFrame):
            self._obj = obj.pandas_api()
        else:
            self._obj = obj
