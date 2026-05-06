from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


PROXIMITY_CHANNELS = ("jjj1_1", "jjj2_1", "jjj1_2", "jjj2_2")


@dataclass
class ProximitySampleRecord:
    timestamp: str
    jjj1_1: Any
    jjj2_1: Any
    jjj1_2: Any
    jjj2_2: Any
    pitch: str
    roll: str


class ProximityDatasetWriter:
    """CSV writer for proximity sensor acquisition samples."""

    def __init__(self, path: str, mode: str = "a", write_header: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, mode=mode, newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if write_header:
            self._writer.writerow(self.header())
            self._file.flush()

    @classmethod
    def create_new(cls, output_dir: str, prefix: str = "proximity_collection") -> "ProximityDatasetWriter":
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(output_dir) / f"{prefix}_{now}.csv"
        return cls(str(path), mode="w", write_header=True)

    @classmethod
    def open_existing(cls, path: str) -> "ProximityDatasetWriter":
        csv_path = Path(path)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        return cls(str(csv_path), mode="a", write_header=write_header)

    @staticmethod
    def header() -> List[str]:
        return ["timestamp", *PROXIMITY_CHANNELS, "pitch", "roll"]

    @staticmethod
    def read_existing_pitch_roll_pairs(path: str) -> Set[Tuple[float, float]]:
        existing: Set[Tuple[float, float]] = set()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    existing.add((round(float(row[-2]), 1), round(float(row[-1]), 1)))
                except Exception:
                    continue
        return existing

    def close(self) -> None:
        try:
            self._file.flush()
            self._file.close()
        except Exception:
            pass

    def save_frame(self, frame: Dict[str, Any], pitch: str, roll: str) -> ProximitySampleRecord:
        proximity = frame.get("proximity", {})
        record = ProximitySampleRecord(
            timestamp=str(frame.get("timestamp", datetime.now().isoformat(timespec="milliseconds"))),
            jjj1_1=proximity.get("jjj1_1"),
            jjj2_1=proximity.get("jjj2_1"),
            jjj1_2=proximity.get("jjj1_2"),
            jjj2_2=proximity.get("jjj2_2"),
            pitch=pitch,
            roll=roll,
        )
        self._writer.writerow([
            record.timestamp,
            record.jjj1_1,
            record.jjj2_1,
            record.jjj1_2,
            record.jjj2_2,
            record.pitch,
            record.roll,
        ])
        self._file.flush()
        return record
