"""
Includes the functionality related to demand monitoring and forecasting
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Demand:
    """Keeps track of the capacity of a firm"""

    initial: float
    constrained: float
    current: float
    history: List = field(default_factory=list)

    def volatility(self) -> float:
        """The demand volatility"""
        return self.initial * 0.03


class ForecastAlgorithm:
    """An algorithm for demand forecasting"""

    def __init__(self, history: list) -> None:
        self.history = history

    def add_to_history(self, value):
        """Adds a value to the history"""
        self.history.append(value)


class PreviousValue(ForecastAlgorithm):
    """Forecast based on the last value"""

    @property
    def forecast(self):
        """Provides the forecast of the model"""
        if len(self.history) > 0:
            return np.floor(self.history[-1])
        else:
            raise RuntimeError("No history provided")


class Average10(ForecastAlgorithm):
    """Forecast based on the last value"""

    @property
    def forecast(self):
        """Provides the forecast of the model"""
        if len(self.history) > 0:
            return np.floor(np.mean(self.history[-10:]))
        else:
            raise RuntimeError("No history provided")


class ExponentialSmooting(ForecastAlgorithm):
    """Forecast based on the last value"""

    @property
    def forecast(self, alpha=0.05):
        """Provides the forecast of the model using only the last 10 values"""
        if len(self.history) > 0:
            # Use only the last 10 values from the history (or all if fewer than 10)
            last_values = self.history[-10:]
            
            smoothed_value = last_values[0]  # Start with the first value in the sliced history
            for value in last_values[1:]:
                smoothed_value = alpha * value + (1 - alpha) * smoothed_value
            return np.floor(smoothed_value)
        else:
            raise RuntimeError("No history provided")
