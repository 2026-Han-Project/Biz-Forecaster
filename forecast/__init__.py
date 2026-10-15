"""매입장부 + 마감 기록 한 줄로 돌아가는 빵집 수요예측, 권장 생산량 엔진.

    from forecast import Config, load_inputs, fit, next_open_days
    model = fit(load_inputs({...}), Config())
    plan = model.plan(next_open_days(model, 1))
"""

from forecast.config import Config, WEEKDAY_NAMES
from forecast.inputs import Inputs, InputError, load_inputs, FILE_NAMES
from forecast.engine import (
    ForecastModel, fit, next_open_days, daily_usage, reverse_recipe,
    restore_demand, parse_result, cumulative_share, service_level,
)

__all__ = [
    "Config", "WEEKDAY_NAMES", "Inputs", "InputError", "load_inputs", "FILE_NAMES",
    "ForecastModel", "fit", "next_open_days", "daily_usage", "reverse_recipe",
    "restore_demand", "parse_result", "cumulative_share", "service_level",
]
