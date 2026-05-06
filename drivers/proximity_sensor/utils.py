from __future__ import annotations

from typing import Any, Dict

from .serial_sensor import ProximitySensor


def create_proximity_sensor(config: Dict[str, Any]) -> ProximitySensor:
    return ProximitySensor(config)


def initialize_proximity_sensor(sensor: ProximitySensor, init_analyzer: bool = False) -> bool:
    print("=== Proximity sensor initialization ===")
    if not sensor.connect():
        print("Proximity sensor serial connection failed")
        return False
    print("Proximity sensor serial connection OK")
    if init_analyzer and not sensor.init_analyzer():
        print("Proximity analyzer init command failed")
        return False
    return True
