from __future__ import annotations

import pandas as pd
import numpy as np
from warnings import warn
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
    is_scalar,
)
from numbers import Number
from typing import Any, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TimeDict:
    data: pd.DataFrame | None = None
    ts_column_name: str | None = None
    panel_column_name: str | None = None
    original_index: pd.Series | pd.Index | None = None
    complete_time_series: pd.Series | None = None
    freq: pd.offsets.DateOffset | int | float | None = None
    start: pd.Timestamp | np.datetime64 | datetime | None = None
    end: pd.Timestamp | np.datetime64 | datetime | None = None
    is_date: bool | None = None


def _is_numeric(ser: pd.Series) -> bool:
    if is_bool_dtype(ser):
        warn(
            "Parsing boolean data type as numeric time-series index"
            "This is probably not what you want."
        )
    if is_object_dtype(ser):
        if all(isinstance(obj, Number) for obj in ser):
            return True
    return is_numeric_dtype(ser)


def _is_date(ser: pd.Series) -> bool:
    if is_object_dtype(ser):
        if all(isinstance(obj, datetime) for obj in ser):
            return True
    return is_datetime64_any_dtype(ser)


def _is_acceptable_type(ser: pd.Series) -> bool:
    return _is_date(ser) or _is_numeric(ser)


def _is_numeric_scalar(val: Any) -> bool:
    return isinstance(val, Number) or (is_scalar(val) and is_numeric_dtype(val))

def offsets_since_epoch(s: pd.Series, o: pd.DateOffset) -> np.ndarray:
    """Converts a date column to an integer column based on the interval of the data.
    For example, if the data are monthly, the output should be months since the epoch,
    if the data are minutely then it should be the minutes since epoch, etc."""
    if isinstance(o,str):
        o = pd.tseries.frequencies.to_offset(o)
    idx = pd.Index(s)
    int_idx = idx.to_period(o).astype(int) // o.n
    return np.array(int_idx)
