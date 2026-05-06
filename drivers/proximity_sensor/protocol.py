"""Frame parser for the proximity sensor serial protocol."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def _to_bytes(values: Iterable[int]) -> bytes:
    return bytes(int(v) & 0xFF for v in values)


class ProximitySensorProtocol:
    def __init__(
        self,
        frame_head: Iterable[int],
        frame_tail: int,
        frame_size: int,
        fields: List[str],
    ):
        self.frame_head = _to_bytes(frame_head)
        self.frame_tail = int(frame_tail) & 0xFF
        self.frame_size = int(frame_size)
        self.fields = list(fields)
        self.buffer = bytearray()

        payload_size = self.frame_size - len(self.frame_head) - 1
        if payload_size < len(self.fields) * 2:
            raise ValueError("Frame payload is smaller than the configured field list")

    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        if chunk:
            self.buffer.extend(chunk)

        frames: List[Dict[str, Any]] = []
        while len(self.buffer) >= self.frame_size:
            start = self.buffer.find(self.frame_head)
            if start < 0:
                keep = max(len(self.frame_head) - 1, 0)
                self.buffer = self.buffer[-keep:] if keep else bytearray()
                break

            if start > 0:
                del self.buffer[:start]

            if len(self.buffer) < self.frame_size:
                break

            raw_frame = bytes(self.buffer[: self.frame_size])
            parsed = self._parse_one(raw_frame)
            if parsed is None:
                del self.buffer[: len(self.frame_head)]
                continue

            del self.buffer[: self.frame_size]
            frames.append(parsed)

        return frames

    def _parse_one(self, frame: bytes) -> Optional[Dict[str, Any]]:
        if not frame.startswith(self.frame_head):
            return None
        if frame[-1] != self.frame_tail:
            return None

        payload = frame[len(self.frame_head) : -1]
        parsed: Dict[str, Any] = {}
        for idx, name in enumerate(self.fields):
            offset = idx * 2
            parsed[name] = int.from_bytes(payload[offset : offset + 2], "little", signed=True)
        return parsed
