"""Serial driver for the proximity sensor."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import serial

from .base import ProximitySensorBase
from .processor import ProximitySensorProcessor
from .protocol import ProximitySensorProtocol


class ProximitySensor(ProximitySensorBase):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        serial_cfg = config["serial"]
        frame_cfg = config["frame"]

        self.serial_cfg = serial_cfg
        self.commands = config.get("commands", {})
        self.ser: Optional[serial.Serial] = None
        self.protocol = ProximitySensorProtocol(
            frame_head=frame_cfg["head"],
            frame_tail=frame_cfg["tail"],
            frame_size=int(frame_cfg["size"]),
            fields=list(frame_cfg["fields"]),
        )
        self.processor = ProximitySensorProcessor(config.get("processing", {}))

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(
                port=self.serial_cfg["port"],
                baudrate=int(self.serial_cfg.get("baud", self.serial_cfg.get("baudrate", 115200))),
                timeout=float(self.serial_cfg.get("timeout", 0.03)),
            )
            self.connected = True
            return True
        except Exception:
            self.connected = False
            return False

    def disconnect(self) -> bool:
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.connected = False
            return True
        except Exception:
            return False

    def init_analyzer(self) -> bool:
        init_cfg = self.commands.get("init", {})
        frames = init_cfg.get("frames", [])
        delay_s = float(init_cfg.get("delay_ms", 50)) / 1000.0
        if not frames:
            return True

        for idx, frame in enumerate(frames):
            if not self._write_hex(frame):
                return False
            if idx < len(frames) - 1:
                time.sleep(delay_s)
        return True

    def zero(self) -> bool:
        frame = self.commands.get("zero")
        return self._write_hex(frame) if frame else True

    def read_frames(self) -> List[Dict[str, Any]]:
        if not self.connected or not self.ser:
            return []

        chunk = self.ser.read(self.ser.in_waiting or 1)
        if not chunk:
            return []

        parsed_list = self.protocol.feed(chunk)
        results: List[Dict[str, Any]] = []
        for parsed in parsed_list:
            result = self.processor.process(parsed)
            result["timestamp"] = datetime.now().isoformat(timespec="milliseconds")
            for key, value in parsed.items():
                result.setdefault(key, value)
            results.append(result)
        return results

    def read_frame(self) -> Optional[Dict[str, Any]]:
        frames = self.read_frames()
        return frames[-1] if frames else None

    def _write_hex(self, frame_hex: Any) -> bool:
        if not self.connected or not self.ser or not self.ser.is_open:
            return False
        if isinstance(frame_hex, bytes):
            payload = frame_hex
        else:
            payload = bytes.fromhex(str(frame_hex))
        self.ser.write(payload)
        return True
