"""Base interface for the proximity sensor driver."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ProximitySensorBase(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        pass

    @abstractmethod
    def init_analyzer(self) -> bool:
        pass

    @abstractmethod
    def zero(self) -> bool:
        pass

    @abstractmethod
    def read_frame(self) -> Optional[Dict[str, Any]]:
        pass

    def read_frames(self) -> List[Dict[str, Any]]:
        frame = self.read_frame()
        return [frame] if frame is not None else []
