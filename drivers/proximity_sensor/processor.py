"""Processing and calibration for proximity sensor frames."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable


DEFAULT_PROXIMITY_CHANNELS = ("jjj1_1", "jjj2_1", "jjj1_2", "jjj2_2")


class ProximitySensorProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.proximity_channels = tuple(config.get("proximity_channels", DEFAULT_PROXIMITY_CHANNELS))
        self.calibration = config.get("calibration", {})
        self.min_value = float(config.get("distance_min_value", 1e-10))

    def process(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        proximity = {name: float(parsed.get(name, 0.0)) for name in self.proximity_channels}
        distance = {
            name: self.calculate_distance(value, self.calibration.get(name, {}))
            for name, value in proximity.items()
        }

        return {
            "raw": dict(parsed),
            "proximity": proximity,
            "distance": distance,
            "force_torque": {
                "sensor1": self._force_torque(parsed, suffix="1"),
                "sensor2": self._force_torque(parsed, suffix="2"),
            },
            "invalid": {
                "jjj_invalid_1": parsed.get("jjj_invalid_1"),
                "jjj_invalid_2": parsed.get("jjj_invalid_2"),
            },
        }

    def calculate_distance(self, capacitance_value: float, coeff: Dict[str, Any]) -> float:
        a = float(coeff.get("a", 0.0))
        b = float(coeff.get("b", 0.0))
        c = float(coeff.get("c", 0.0))
        safe_value = max(abs(float(capacitance_value)), self.min_value)
        log_value = math.log10(safe_value)
        return a * log_value * log_value + b * log_value + c

    @staticmethod
    def _force_torque(parsed: Dict[str, Any], suffix: str) -> Dict[str, float]:
        return {
            "fx": float(parsed.get(f"fx{suffix}", 0.0)),
            "fy": float(parsed.get(f"fy{suffix}", 0.0)),
            "fz": float(parsed.get(f"fz{suffix}", 0.0)),
            "mx": float(parsed.get(f"mx{suffix}", 0.0)),
            "my": float(parsed.get(f"my{suffix}", 0.0)),
        }
