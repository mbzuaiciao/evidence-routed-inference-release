"""Return-data validation and frequency conventions."""

from eri.data.frequency import validate_periods_per_year
from eri.data.returns import validate_returns

__all__ = ["validate_periods_per_year", "validate_returns"]
