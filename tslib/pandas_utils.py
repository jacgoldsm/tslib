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
    from types import ModuleType


def _range(engine):
    if engine == pd:
        return np.arange
    else:
        return engine.range

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
    engine: ModuleType | None = None


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
