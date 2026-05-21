from __future__ import annotations

import csv
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
import yaml
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from drivers.proximity_sensor.utils import create_proximity_sensor, initialize_proximity_sensor


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "default.yaml")
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

PROXIMITY_CHANNELS = ("jjj1_1", "jjj2_1", "jjj1_2", "jjj2_2")
DISTANCE_LABELS = {
    "jjj1_1": "ZS距离(mm)",
    "jjj2_1": "YS距离(mm)",
    "jjj1_2": "ZX距离(mm)",
    "jjj2_2": "YX距离(mm)",
}


def load_cfg() -> Dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def frange(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    v = float(start)
    eps = abs(step) * 1e-6
    while v <= stop + eps:
        vals.append(round(v, 6))
        v += step
    return vals


class AutoMode(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class AutoState(str, Enum):
    IDLE = "IDLE"
    INIT = "INIT"
    MOVING_TO_ANGLE = "MOVING_TO_ANGLE"
    APPROACHING_AND_RECORDING = "APPROACHING_AND_RECORDING"
    RETRACTING = "RETRACTING"
    DONE = "DONE"
    ESTOP = "ESTOP"
    ERROR = "ERROR"


class AutoStopRequested(Exception):
    pass


DYNAMIC_PROFILE_ALL = "all_recommended"
DYNAMIC_AXIS_HOLD = "hold"
DYNAMIC_AXIS_LINEAR = "linear"
DYNAMIC_AXIS_SINE = "sine"
DYNAMIC_PROFILE_ORDER = (
    "pitch_sweep",
    "roll_sweep",
    "diag_same",
    "diag_opposite",
    "pitch_sine",
    "roll_sine",
    "sine_phase0",
    "sine_phase90",
    "small_fast_sine",
    "large_slow_sine",
    "pitch_sweep_roll_sine",
)
DYNAMIC_PROFILE_UI_OPTIONS = (
    (DYNAMIC_PROFILE_ALL, "全部推荐动态组"),
    ("pitch_sweep", "Pitch单轴扫描"),
    ("roll_sweep", "Roll单轴扫描"),
    ("diag_same", "同向对角扫描"),
    ("diag_opposite", "反向对角扫描"),
    ("pitch_sine", "Pitch正弦摆动"),
    ("roll_sine", "Roll正弦摆动"),
    ("sine_phase0", "Pitch/Roll同相正弦"),
    ("sine_phase90", "Pitch/Roll 90°相位正弦"),
    ("small_fast_sine", "小幅高频相位正弦"),
    ("large_slow_sine", "大幅低频相位正弦"),
    ("pitch_sweep_roll_sine", "Pitch扫描+Roll正弦"),
)


@dataclass(frozen=True)
class AutoAcquisitionParams:
    mode: AutoMode
    speed_mm_s: float
    output_dir: str
    dynamic_profile_key: str = DYNAMIC_PROFILE_ALL
    pitch_min: float = -3.0
    pitch_max: float = 3.0
    pitch_step: float = 0.1
    roll_min: float = -3.0
    roll_max: float = 3.0
    roll_step: float = 0.1
    approach_target_mm: float = 15.0
    retract_mm: float = 14.0
    retract_speed_mm_s: float = 0.5
    start_distance_mm: float = 8.0
    stop_distance_mm: float = 1.0
    angle_timeout_s: float = 20.0
    torque_timeout_s: float = 60.0
    dynamic_duration_s: float = 8.0
    dynamic_pitch_start_deg: float = -3.0
    dynamic_pitch_end_deg: float = 3.0
    dynamic_roll_amplitude_deg: float = 2.0
    dynamic_roll_period_s: float = 8.0


@dataclass(frozen=True)
class DynamicTrajectorySpec:
    key: str
    label: str
    pitch_mode: str
    roll_mode: str
    pitch_start: float
    roll_start: float
    pitch_end: float = 0.0
    roll_end: float = 0.0
    pitch_center: float = 0.0
    roll_center: float = 0.0
    pitch_amplitude: float = 0.0
    roll_amplitude: float = 0.0
    pitch_phase_rad: float = 0.0
    roll_phase_rad: float = 0.0
    duration_s: float = 8.0
    period_s: float = 8.0


class AutoAcquisitionWorker(QtCore.QObject):
    sig_log = QtCore.pyqtSignal(str)
    sig_status = QtCore.pyqtSignal(str)
    sig_state = QtCore.pyqtSignal(str)
    sig_progress = QtCore.pyqtSignal(int, int)
    sig_file = QtCore.pyqtSignal(str)
    sig_finished = QtCore.pyqtSignal()

    def __init__(
        self,
        sensor,
        sensor_lock: threading.Lock,
        hardware_lock: threading.RLock,
        motion,
        torque,
        params: AutoAcquisitionParams,
    ):
        super().__init__()
        self.sensor = sensor
        self.sensor_lock = sensor_lock
        self.hardware_lock = hardware_lock
        self.motion = motion
        self.torque = torque
        self.params = params
        self._stop_requested = False
        self._state = AutoState.IDLE
        self._frame_queue: List[Dict] = []
        self._queue_mutex = QtCore.QMutex()
        self._wake = QtCore.QWaitCondition()
        self._csv_file = None
        self._writer: Optional[csv.writer] = None
        self._csv_path = ""
        self._recording = False
        self._record_t0: Optional[float] = None
        self._active_pitch = 0.0
        self._active_roll = 0.0
        self._current_index = 0
        self._total_points = 0
        self._dynamic_started_at: Optional[float] = None
        self._last_roll_cmd_ts = 0.0
        self._active_dynamic_spec: Optional[DynamicTrajectorySpec] = None

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self._set_state(AutoState.INIT)
            self._initialize_hardware()
            if self.params.mode == AutoMode.STATIC:
                self._open_csv()
                self._run_static_mode()
            else:
                self._run_dynamic_mode()
            if not self._stop_requested:
                self._finalize_hardware()
                self._set_state(AutoState.DONE)
                self.sig_status.emit("自动采集完成")
        except AutoStopRequested:
            self._set_state(AutoState.ESTOP)
            self.sig_log.emit("自动采集已急停")
            self.sig_status.emit("自动采集已急停")
            self._safe_stop_all()
        except Exception as exc:
            self._set_state(AutoState.ERROR)
            self.sig_log.emit(f"自动采集失败：{exc}")
            self.sig_status.emit("自动采集失败")
            self._safe_stop_all()
        finally:
            self._close_csv()
            self.sig_finished.emit()

    @QtCore.pyqtSlot(dict)
    def on_frame(self, frame: Dict):
        self._queue_mutex.lock()
        try:
            self._frame_queue.append(frame)
            if len(self._frame_queue) > 200:
                self._frame_queue = self._frame_queue[-200:]
            self._wake.wakeOne()
        finally:
            self._queue_mutex.unlock()

    @QtCore.pyqtSlot()
    def request_stop(self):
        self._stop_requested = True
        self._set_state(AutoState.ESTOP)
        self._safe_stop_all()
        self._queue_mutex.lock()
        try:
            self._wake.wakeAll()
        finally:
            self._queue_mutex.unlock()

    def _set_state(self, state: AutoState):
        self._state = state
        self.sig_state.emit(state.value)
        self.sig_status.emit(f"自动采集：{state.value}")

    def _open_csv(self, profile_key: Optional[str] = None):
        self._close_csv()
        os.makedirs(self.params.output_dir, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_name = "static" if self.params.mode == AutoMode.STATIC else "dynamic"
        speed_tag = f"{self.params.speed_mm_s:.1f}".replace(".", "p")
        profile_part = f"_{profile_key}" if profile_key else ""
        self._csv_path = os.path.join(
            self.params.output_dir,
            f"auto_{mode_name}{profile_part}_v{speed_tag}_{now}.csv",
        )
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow([
            "timestamp",
            "z_position_mm",
            "pitch_deg",
            "roll_deg",
            "c1",
            "c2",
            "c3",
            "c4",
            "d1_mm",
            "d2_mm",
            "d3_mm",
            "d4_mm",
        ])
        self._csv_file.flush()
        self.sig_file.emit(self._csv_path)
        self.sig_log.emit(f"自动采集CSV已创建：{self._csv_path}")

    def _close_csv(self):
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._writer = None

    def _initialize_hardware(self):
        from drivers.motioncard.utils import full_axis_initialization

        with self.sensor_lock:
            if not self.sensor.zero():
                raise RuntimeError("接近觉清零失败")
        self.sig_log.emit("接近觉已清零")

        for axis in (0, 1):
            self._check_stop()
            self.sig_log.emit(f"轴 {axis} 初始化...")
            if not full_axis_initialization(self.motion, axis):
                raise RuntimeError(f"轴 {axis} 初始化失败")
            self.sig_log.emit(f"轴 {axis} 回原点...")
            self._home_motion_axis(axis, timeout_s=60.0)
        self._move_axes_to(0.0, 0.0)

        self.sig_log.emit("力控电机回原点...")
        with self.hardware_lock:
            self.torque.home(0)
        self._wait_torque_done(timeout_s=120.0)
        with self.hardware_lock:
            self.torque.trigger_command(25)
        self._sleep_ms(200)
        self.sig_log.emit("力控电机力值已清零")

    def _finalize_hardware(self):
        from drivers.motioncard.utils import full_axis_initialization

        self.sig_log.emit("采集完成，开始收尾回零...")
        self._home_torque_and_zero_force()

        for axis in (0, 1, 2, 3):
            self._check_stop()
            self.sig_log.emit(f"收尾：轴 {axis} 初始化...")
            if not full_axis_initialization(self.motion, axis):
                raise RuntimeError(f"收尾：轴 {axis} 初始化失败")
            self.sig_log.emit(f"收尾：轴 {axis} 回原点...")
            self._home_motion_axis(axis, timeout_s=60.0)
            self.sig_log.emit(f"收尾：轴 {axis} 回原点完成")

        self.sig_log.emit("采集完成收尾：力控电机和四轴均已回原点")

    def _reset_between_dynamic_groups(self):
        from drivers.motioncard.utils import full_axis_initialization

        self.sig_log.emit("动态组采集完成，开始组间复位...")
        self._home_torque_and_zero_force(log_prefix="组间复位")
        for axis in (0, 1, 2, 3):
            self._check_stop()
            self.sig_log.emit(f"组间复位：轴 {axis} 初始化...")
            if not full_axis_initialization(self.motion, axis):
                raise RuntimeError(f"组间复位：轴 {axis} 初始化失败")
            self.sig_log.emit(f"组间复位：轴 {axis} 回原点...")
            self._home_motion_axis(axis, timeout_s=60.0)
            self.sig_log.emit(f"组间复位：轴 {axis} 回原点完成")
        self.sig_log.emit("组间复位完成，准备下一组动态采集")

    def _home_torque_and_zero_force(self, log_prefix: str = "收尾"):
        self.sig_log.emit(f"{log_prefix}：力控电机回原点...")
        with self.hardware_lock:
            self.torque.home(0)
        self._wait_torque_done(timeout_s=120.0)
        self.sig_log.emit(f"{log_prefix}：力控电机力值清零...")
        with self.hardware_lock:
            self.torque.trigger_command(25)
        self._sleep_ms(200)
        self.sig_log.emit(f"{log_prefix}：力控电机回原点并力值清零完成")

    def _home_motion_axis(self, axis: int, timeout_s: float):
        with self.hardware_lock:
            self.motion.home(axis)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._check_stop()
            self._check_motion_io(axes=(axis,), fail_on_limit=False)
            with self.hardware_lock:
                done = self.motion.is_home_done(axis)
            if done:
                self._wait_motion_axis_idle(axis, timeout_s=3.0)
                self._set_axis_position_zero(axis)
                return
            self._sleep_ms(20)
        with self.hardware_lock:
            self.motion.stop(axis, mode=1)
        raise TimeoutError(f"轴 {axis} 回原点超时")

    def _wait_motion_axis_idle(self, axis: int, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        stable = 0
        while time.monotonic() < deadline:
            self._check_stop()
            with self.hardware_lock:
                done = self.motion.is_done(axis)
            stable = stable + 1 if done else 0
            if stable >= 3:
                return
            self._sleep_ms(50)
        self.sig_log.emit(f"轴 {axis} 回零后等待空闲超时，继续尝试清零位置")

    def _set_axis_position_zero(self, axis: int):
        last_error = None
        for attempt in range(1, 6):
            self._check_stop()
            try:
                with self.hardware_lock:
                    self.motion.set_position(axis, 0.0)
                return
            except Exception as exc:
                last_error = exc
                self.sig_log.emit(f"轴 {axis} 位置寄存器清零重试 {attempt}/5：{exc}")
                self._sleep_ms(150)
        raise RuntimeError(f"轴 {axis} 位置寄存器清零失败：{last_error}")

    def _run_static_mode(self):
        pitch_vals = frange(self.params.pitch_min, self.params.pitch_max, self.params.pitch_step)
        roll_vals = frange(self.params.roll_min, self.params.roll_max, self.params.roll_step)
        points = [(p, r) for p in pitch_vals for r in roll_vals]
        self._total_points = len(points)

        for idx, (pitch, roll) in enumerate(points, start=1):
            self._check_stop()
            self._current_index = idx
            self.sig_progress.emit(idx, self._total_points)
            self._active_pitch = pitch
            self._active_roll = roll
            self.sig_log.emit(f"静态点 {idx}/{self._total_points}: pitch={pitch:.1f}, roll={roll:.1f}")
            self._set_state(AutoState.MOVING_TO_ANGLE)
            self._move_axes_to(pitch, roll)
            self._approach_record_and_retract(target_pitch=pitch, target_roll=roll)

    def _run_dynamic_mode(self):
        specs = self._selected_dynamic_specs()
        self._total_points = len(specs)

        for idx, spec in enumerate(specs, start=1):
            self._check_stop()
            self._current_index = idx
            self.sig_progress.emit(idx, self._total_points)
            self._active_dynamic_spec = spec
            self._open_csv(profile_key=spec.key)
            self.sig_log.emit(f"动态组 {idx}/{self._total_points}: {spec.label}")
            self._move_axes_to(spec.pitch_start, spec.roll_start)
            self._set_state(AutoState.APPROACHING_AND_RECORDING)
            self._dynamic_started_at = None
            self._last_roll_cmd_ts = 0.0
            self.sig_log.emit("动态模式：等待4路距离全部小于 8 mm 后启动Pitch/Roll轨迹")
            self._approach_record_and_retract(
                target_pitch=spec.pitch_start,
                target_roll=spec.roll_start,
                dynamic=True,
            )
            self._close_csv()
            if idx < self._total_points:
                self._reset_between_dynamic_groups()
        self._active_dynamic_spec = None

    def _move_axes_to(self, pitch: float, roll: float):
        self._set_state(AutoState.MOVING_TO_ANGLE)
        deadline = time.monotonic() + self.params.angle_timeout_s
        while time.monotonic() < deadline:
            self._check_stop()
            self._check_motion_io()
            with self.hardware_lock:
                pos0 = self.motion.get_position(0)
                pos1 = self.motion.get_position(1)
                done0 = self.motion.is_done(0)
                done1 = self.motion.is_done(1)
            axis0_ready = abs(pos0 - pitch) <= 0.01
            axis1_ready = abs(pos1 - roll) <= 0.01
            if axis0_ready and axis1_ready:
                return
            if done0 and not axis0_ready:
                self._move_motion_axis_abs(0, pitch)
            if done1 and not axis1_ready:
                self._move_motion_axis_abs(1, roll)
            self._sleep_ms(20)
        raise TimeoutError("Pitch/Roll 轴到位超时")

    def _start_dynamic_trajectory(self):
        spec = self._require_dynamic_spec()
        pitch_speed = self._dynamic_axis_speed(spec, "pitch")
        roll_speed = self._dynamic_axis_speed(spec, "roll")
        with self.hardware_lock:
            self.motion.set_profile(0, 0.0, max(pitch_speed, 0.05), 0.1, 0.1, 0.0)
            self.motion.set_profile(1, 0.0, max(roll_speed, 0.05), 0.1, 0.1, 0.0)
        self._start_dynamic_axis(0, spec, "pitch")
        self._start_dynamic_axis(1, spec, "roll")

    def _move_motion_axis_abs(self, axis: int, target: float, eps: float = 0.002):
        with self.hardware_lock:
            current = self.motion.get_position(axis)
            if abs(current - target) <= eps:
                return
            if not self.motion.is_done(axis):
                return
            try:
                self.motion.move_abs(axis, target)
            except Exception as exc:
                raise RuntimeError(
                    f"轴 {axis} 绝对运动到 {target:.3f}° 失败，当前位置 {current:.3f}°：{exc}"
                ) from exc

    def _command_dynamic_axis(self, axis: int, target: float):
        with self.hardware_lock:
            if not self.motion.is_done(axis):
                return
        self._move_motion_axis_abs(axis, target, eps=0.01)

    def _start_dynamic_axis(self, axis: int, spec: DynamicTrajectorySpec, axis_name: str):
        mode = spec.pitch_mode if axis_name == "pitch" else spec.roll_mode
        if mode == DYNAMIC_AXIS_LINEAR:
            target = spec.pitch_end if axis_name == "pitch" else spec.roll_end
        else:
            target = self._dynamic_axis_target(spec, axis_name, 0.0)
        self._command_dynamic_axis(axis, target)

    def _approach_record_and_retract(self, target_pitch: float, target_roll: float, dynamic: bool = False):
        self._recording = False
        self._record_t0 = None
        self._clear_frame_queue()
        self._set_state(AutoState.APPROACHING_AND_RECORDING)
        with self.hardware_lock:
            self.torque.set_profile(0, 0.0, self.params.speed_mm_s, 50.0, 50.0, 0.0)
            self.torque.set_band(0, 0.05)
            self.torque.move_abs(0, self.params.approach_target_mm)

        while True:
            self._check_stop()
            self._check_motion_io()
            if dynamic and self._recording:
                self._update_dynamic_motion()
            frame = self._wait_next_frame(timeout_ms=500)
            if frame is None:
                self._check_stop()
                raise TimeoutError("接近觉传感器无新数据")
            distances = self._frame_distances(frame)
            frame_ts = float(frame.get("_host_monotonic", time.monotonic()))

            if (not self._recording) and all(d < self.params.start_distance_mm for d in distances):
                self._recording = True
                self._record_t0 = frame_ts
                if dynamic:
                    self._dynamic_started_at = frame_ts
                    self._last_roll_cmd_ts = 0.0
                    self._start_dynamic_trajectory()
                    self.sig_log.emit("触发记录：4路距离全部小于 8 mm，开始记录并启动动态轨迹")
                else:
                    self.sig_log.emit("触发记录：4路距离全部小于 8 mm")

            if any(d < self.params.stop_distance_mm for d in distances):
                self.sig_log.emit("停止记录：至少一路距离小于 1 mm")
                with self.hardware_lock:
                    self.torque.stop(0)
                break

            if self._recording:
                self._write_aligned_frame(frame, target_pitch, target_roll)

            with self.hardware_lock:
                torque_done = self.torque.is_done(0)
            if torque_done:
                self.sig_log.emit("力控电机已到达下压目标，结束本轮记录")
                break

        self._recording = False
        if dynamic:
            for axis in (0, 1):
                try:
                    with self.hardware_lock:
                        self.motion.stop(axis, mode=1)
                except Exception:
                    pass
        self._retract_torque()

    def _update_dynamic_motion(self):
        if self._dynamic_started_at is None:
            return
        spec = self._require_dynamic_spec()
        elapsed = time.monotonic() - self._dynamic_started_at
        now = time.monotonic()
        if elapsed <= spec.duration_s and now - self._last_roll_cmd_ts >= 0.05:
            if spec.pitch_mode == DYNAMIC_AXIS_SINE:
                pitch_target = self._dynamic_axis_target(spec, "pitch", elapsed)
                self._command_dynamic_axis(0, pitch_target)
            if spec.roll_mode == DYNAMIC_AXIS_SINE:
                roll_target = self._dynamic_axis_target(spec, "roll", elapsed)
                self._command_dynamic_axis(1, roll_target)
            self._last_roll_cmd_ts = now

    def _selected_dynamic_specs(self) -> List[DynamicTrajectorySpec]:
        key = self.params.dynamic_profile_key
        keys = DYNAMIC_PROFILE_ORDER if key == DYNAMIC_PROFILE_ALL else (key,)
        return [self._make_dynamic_spec(item) for item in keys]

    def _make_dynamic_spec(self, key: str) -> DynamicTrajectorySpec:
        p = self.params
        pitch_min = p.pitch_min
        pitch_max = p.pitch_max
        roll_min = p.roll_min
        roll_max = p.roll_max
        pitch_amp = min(2.0, max(abs(pitch_min), abs(pitch_max)))
        roll_amp = min(2.0, max(abs(roll_min), abs(roll_max)))

        specs = {
            "pitch_sweep": DynamicTrajectorySpec(
                key="pitch_sweep",
                label="Pitch单轴扫描",
                pitch_mode=DYNAMIC_AXIS_LINEAR,
                roll_mode=DYNAMIC_AXIS_HOLD,
                pitch_start=pitch_min,
                pitch_end=pitch_max,
                roll_start=0.0,
                roll_end=0.0,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "roll_sweep": DynamicTrajectorySpec(
                key="roll_sweep",
                label="Roll单轴扫描",
                pitch_mode=DYNAMIC_AXIS_HOLD,
                roll_mode=DYNAMIC_AXIS_LINEAR,
                pitch_start=0.0,
                pitch_end=0.0,
                roll_start=roll_min,
                roll_end=roll_max,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "diag_same": DynamicTrajectorySpec(
                key="diag_same",
                label="同向对角扫描",
                pitch_mode=DYNAMIC_AXIS_LINEAR,
                roll_mode=DYNAMIC_AXIS_LINEAR,
                pitch_start=pitch_min,
                pitch_end=pitch_max,
                roll_start=roll_min,
                roll_end=roll_max,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "diag_opposite": DynamicTrajectorySpec(
                key="diag_opposite",
                label="反向对角扫描",
                pitch_mode=DYNAMIC_AXIS_LINEAR,
                roll_mode=DYNAMIC_AXIS_LINEAR,
                pitch_start=pitch_min,
                pitch_end=pitch_max,
                roll_start=roll_max,
                roll_end=roll_min,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "pitch_sine": DynamicTrajectorySpec(
                key="pitch_sine",
                label="Pitch正弦摆动",
                pitch_mode=DYNAMIC_AXIS_SINE,
                roll_mode=DYNAMIC_AXIS_HOLD,
                pitch_start=0.0,
                roll_start=0.0,
                pitch_amplitude=pitch_amp,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "roll_sine": DynamicTrajectorySpec(
                key="roll_sine",
                label="Roll正弦摆动",
                pitch_mode=DYNAMIC_AXIS_HOLD,
                roll_mode=DYNAMIC_AXIS_SINE,
                pitch_start=0.0,
                roll_start=0.0,
                roll_amplitude=roll_amp,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "sine_phase0": DynamicTrajectorySpec(
                key="sine_phase0",
                label="Pitch/Roll同相正弦",
                pitch_mode=DYNAMIC_AXIS_SINE,
                roll_mode=DYNAMIC_AXIS_SINE,
                pitch_start=0.0,
                roll_start=0.0,
                pitch_amplitude=pitch_amp,
                roll_amplitude=roll_amp,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "sine_phase90": DynamicTrajectorySpec(
                key="sine_phase90",
                label="Pitch/Roll 90°相位正弦",
                pitch_mode=DYNAMIC_AXIS_SINE,
                roll_mode=DYNAMIC_AXIS_SINE,
                pitch_start=0.0,
                roll_start=roll_amp,
                pitch_amplitude=pitch_amp,
                roll_amplitude=roll_amp,
                roll_phase_rad=math.pi / 2.0,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
            "small_fast_sine": DynamicTrajectorySpec(
                key="small_fast_sine",
                label="小幅高频相位正弦",
                pitch_mode=DYNAMIC_AXIS_SINE,
                roll_mode=DYNAMIC_AXIS_SINE,
                pitch_start=0.0,
                roll_start=1.0,
                pitch_amplitude=1.0,
                roll_amplitude=1.0,
                roll_phase_rad=math.pi / 2.0,
                duration_s=max(4.0, p.dynamic_duration_s),
                period_s=4.0,
            ),
            "large_slow_sine": DynamicTrajectorySpec(
                key="large_slow_sine",
                label="大幅低频相位正弦",
                pitch_mode=DYNAMIC_AXIS_SINE,
                roll_mode=DYNAMIC_AXIS_SINE,
                pitch_start=0.0,
                roll_start=min(3.0, max(abs(roll_min), abs(roll_max))),
                pitch_amplitude=min(3.0, max(abs(pitch_min), abs(pitch_max))),
                roll_amplitude=min(3.0, max(abs(roll_min), abs(roll_max))),
                roll_phase_rad=math.pi / 2.0,
                duration_s=max(12.0, p.dynamic_duration_s),
                period_s=12.0,
            ),
            "pitch_sweep_roll_sine": DynamicTrajectorySpec(
                key="pitch_sweep_roll_sine",
                label="Pitch扫描+Roll正弦",
                pitch_mode=DYNAMIC_AXIS_LINEAR,
                roll_mode=DYNAMIC_AXIS_SINE,
                pitch_start=pitch_min,
                pitch_end=pitch_max,
                roll_start=0.0,
                roll_amplitude=roll_amp,
                duration_s=p.dynamic_duration_s,
                period_s=p.dynamic_roll_period_s,
            ),
        }
        try:
            return specs[key]
        except KeyError as exc:
            raise ValueError(f"未知动态轨迹组：{key}") from exc

    def _require_dynamic_spec(self) -> DynamicTrajectorySpec:
        if self._active_dynamic_spec is None:
            raise RuntimeError("动态轨迹未设置")
        return self._active_dynamic_spec

    def _dynamic_axis_speed(self, spec: DynamicTrajectorySpec, axis_name: str) -> float:
        mode = spec.pitch_mode if axis_name == "pitch" else spec.roll_mode
        if mode == DYNAMIC_AXIS_HOLD:
            return 0.05
        if mode == DYNAMIC_AXIS_LINEAR:
            start = spec.pitch_start if axis_name == "pitch" else spec.roll_start
            end = spec.pitch_end if axis_name == "pitch" else spec.roll_end
            return abs(end - start) / max(spec.duration_s, 1e-6)
        amplitude = spec.pitch_amplitude if axis_name == "pitch" else spec.roll_amplitude
        return 4.0 * abs(amplitude) / max(spec.period_s, 1e-6)

    def _dynamic_axis_target(self, spec: DynamicTrajectorySpec, axis_name: str, elapsed_s: float) -> float:
        mode = spec.pitch_mode if axis_name == "pitch" else spec.roll_mode
        if axis_name == "pitch":
            start = spec.pitch_start
            end = spec.pitch_end
            center = spec.pitch_center
            amplitude = spec.pitch_amplitude
            phase = spec.pitch_phase_rad
        else:
            start = spec.roll_start
            end = spec.roll_end
            center = spec.roll_center
            amplitude = spec.roll_amplitude
            phase = spec.roll_phase_rad

        if mode == DYNAMIC_AXIS_HOLD:
            return start
        if mode == DYNAMIC_AXIS_LINEAR:
            progress = min(max(elapsed_s / max(spec.duration_s, 1e-6), 0.0), 1.0)
            return start + (end - start) * progress
        if mode == DYNAMIC_AXIS_SINE:
            angle = 2.0 * math.pi * elapsed_s / max(spec.period_s, 1e-6) + phase
            return center + amplitude * math.sin(angle)
        raise RuntimeError(f"未知动态轴模式：{mode}")

    def _retract_torque(self):
        self._set_state(AutoState.RETRACTING)
        with self.hardware_lock:
            current = self.torque.get_position(0)
            self.torque.set_profile(0, 0.0, self.params.retract_speed_mm_s, 50.0, 50.0, 0.0)
            self.torque.move_abs(0, current - self.params.retract_mm)
        self._wait_torque_done(timeout_s=self.params.torque_timeout_s)

    def _write_aligned_frame(self, frame: Dict, target_pitch: float, target_roll: float):
        if not self._writer or self._record_t0 is None:
            return
        timestamp = str(frame.get("_host_timestamp", datetime.now().isoformat(timespec="milliseconds")))
        proximity = frame.get("proximity", {})
        distance = frame.get("distance", {})
        with self.hardware_lock:
            torque_status = self.torque.read_status()
            pitch = self.motion.get_position(0)
            roll = self.motion.get_position(1)
        row = [
            timestamp,
            f"{float(torque_status['position']):.6f}",
            f"{float(pitch):.6f}",
            f"{float(roll):.6f}",
            *[proximity.get(name, "") for name in PROXIMITY_CHANNELS],
            *[distance.get(name, "") for name in PROXIMITY_CHANNELS],
        ]
        self._writer.writerow(row)
        self._csv_file.flush()

    def _wait_next_frame(self, timeout_ms: int) -> Optional[Dict]:
        self._queue_mutex.lock()
        try:
            if not self._frame_queue:
                self._wake.wait(self._queue_mutex, timeout_ms)
            if not self._frame_queue:
                return None
            return self._frame_queue.pop(0)
        finally:
            self._queue_mutex.unlock()

    def _clear_frame_queue(self):
        self._queue_mutex.lock()
        try:
            self._frame_queue.clear()
        finally:
            self._queue_mutex.unlock()

    def _frame_distances(self, frame: Dict) -> List[float]:
        distance = frame.get("distance", {})
        return [float(distance.get(name, float("inf"))) for name in PROXIMITY_CHANNELS]

    def _wait_torque_done(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        stable = 0
        while time.monotonic() < deadline:
            self._check_stop()
            with self.hardware_lock:
                moving = not self.torque.is_done(0)
                vel = abs(self.torque.get_velocity(0))
            stable = stable + 1 if (not moving) or vel < 0.01 else 0
            if stable >= 3:
                return
            self._sleep_ms(20)
        raise TimeoutError("力控电机到位超时")

    def _check_motion_io(self, axes=(0, 1), fail_on_limit: bool = True):
        for axis in axes:
            with self.hardware_lock:
                io = self.motion.read_axis_io(axis)
            if io.get("alm"):
                raise RuntimeError(f"轴 {axis} 伺服报警")
            if io.get("emg"):
                raise RuntimeError(f"轴 {axis} 急停触发")
            if fail_on_limit and (io.get("pel") or io.get("nel")):
                raise RuntimeError(f"轴 {axis} 触发限位")

    def _safe_stop_all(self):
        try:
            with self.hardware_lock:
                self.torque.stop(0)
        except Exception:
            pass
        for axis in (0, 1):
            try:
                with self.hardware_lock:
                    self.motion.stop(axis, mode=1)
            except Exception:
                pass

    def _check_stop(self):
        if self._stop_requested:
            raise AutoStopRequested()

    def _sleep_ms(self, ms: int):
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            self._check_stop()
            QtCore.QThread.msleep(min(10, max(1, int((end - time.monotonic()) * 1000))))


class ProximityAcquisitionWindow(QtWidgets.QWidget):
    sig_frame = QtCore.pyqtSignal(dict)
    sig_status = QtCore.pyqtSignal(str)
    sig_log = QtCore.pyqtSignal(str)

    def __init__(self, sensor, sensor_cfg: Dict):
        super().__init__()
        self.sensor = sensor
        self.sensor_cfg = sensor_cfg
        self.cfg = load_cfg()

        self.current_frame: Optional[Dict] = None

        self._running = True
        self._io_lock = threading.Lock()
        self._op_lock = threading.Lock()
        self._op_thread: Optional[threading.Thread] = None
        self._hardware_lock = threading.RLock()
        self.motion: Optional[object] = None
        self.torque: Optional[object] = None
        self._torque_status_error_logged = False
        self.auto_thread: Optional[QtCore.QThread] = None
        self.auto_worker: Optional[AutoAcquisitionWorker] = None
        self.auto_running = False

        self.curves = {}
        self.curve_data = {name: [] for name in PROXIMITY_CHANNELS}
        self.plot_window: Optional[pg.GraphicsLayoutWidget] = None

        self.setWindowTitle("接近觉传感器数据采集")
        self.resize(1180, 760)
        self._build_ui()

        self.sig_frame.connect(self.update_ui_with_frame)
        self.sig_status.connect(self.status_label.setText)
        self.sig_log.connect(self._append_log)

        self.torque_status_timer = QtCore.QTimer(self)
        self.torque_status_timer.timeout.connect(self.refresh_torque_status)
        self.torque_status_timer.start(300)

        self.acq_thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self.acq_thread.start()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_init = QtWidgets.QPushButton("初始化分析仪")
        self.btn_zero = QtWidgets.QPushButton("接近觉清零")
        self.btn_plot = QtWidgets.QPushButton("绘制曲线")

        self.btn_init.clicked.connect(self.init_analyzer)
        self.btn_zero.clicked.connect(self.zero_sensor)
        self.btn_plot.clicked.connect(self.show_plot_window)
        for btn in (self.btn_init, self.btn_zero, self.btn_plot):
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

        grid = QtWidgets.QGridLayout()
        self.pitch_edit = QtWidgets.QLineEdit("0")
        self.roll_edit = QtWidgets.QLineEdit("0")
        self.pitch_edit.setReadOnly(True)
        self.roll_edit.setReadOnly(True)

        self.raw_edits = {name: QtWidgets.QLineEdit("0") for name in PROXIMITY_CHANNELS}
        self.distance_edits = {name: QtWidgets.QLineEdit("0.00") for name in PROXIMITY_CHANNELS}
        for edit in list(self.raw_edits.values()) + list(self.distance_edits.values()):
            edit.setReadOnly(True)

        r = 0
        grid.addWidget(QtWidgets.QLabel("俯仰角"), r, 0)
        grid.addWidget(self.pitch_edit, r, 1)
        grid.addWidget(QtWidgets.QLabel("横滚角"), r, 2)
        grid.addWidget(self.roll_edit, r, 3)
        r += 1
        for idx, name in enumerate(PROXIMITY_CHANNELS):
            grid.addWidget(QtWidgets.QLabel(name), r, idx)
        r += 1
        for idx, name in enumerate(PROXIMITY_CHANNELS):
            grid.addWidget(self.raw_edits[name], r, idx)
        r += 1
        for idx, name in enumerate(PROXIMITY_CHANNELS):
            grid.addWidget(QtWidgets.QLabel(DISTANCE_LABELS[name]), r, idx)
        r += 1
        for idx, name in enumerate(PROXIMITY_CHANNELS):
            grid.addWidget(self.distance_edits[name], r, idx)
        root.addLayout(grid)

        control_group = QtWidgets.QGroupBox("采集硬件控制")
        control_grid = QtWidgets.QGridLayout(control_group)
        self.btn_motion_home = QtWidgets.QPushButton("运动控制卡四轴回零")
        self.btn_motion_home.clicked.connect(self.home_motion_axes)
        self.btn_torque_home = QtWidgets.QPushButton("力控电机回零")
        self.btn_torque_home.clicked.connect(self.home_torque_motor)
        self.btn_torque_force_zero = QtWidgets.QPushButton("力控电机力清零")
        self.btn_torque_force_zero.clicked.connect(self.zero_torque_force)

        self.torque_abs_pos = QtWidgets.QDoubleSpinBox()
        self.torque_abs_pos.setRange(-10000.0, 10000.0)
        self.torque_abs_pos.setDecimals(3)
        self.torque_abs_pos.setValue(0.0)
        self.torque_abs_pos.setSuffix(" mm")
        self.torque_abs_vel = QtWidgets.QDoubleSpinBox()
        self.torque_abs_vel.setRange(0.1, 1000.0)
        self.torque_abs_vel.setDecimals(3)
        self.torque_abs_vel.setValue(2.0)
        self.torque_abs_vel.setSuffix(" mm/s")
        self.torque_abs_acc = QtWidgets.QDoubleSpinBox()
        self.torque_abs_acc.setRange(0.1, 5000.0)
        self.torque_abs_acc.setDecimals(3)
        self.torque_abs_acc.setValue(50.0)
        self.torque_abs_acc.setSuffix(" mm/s²")
        self.torque_abs_dec = QtWidgets.QDoubleSpinBox()
        self.torque_abs_dec.setRange(0.1, 5000.0)
        self.torque_abs_dec.setDecimals(3)
        self.torque_abs_dec.setValue(50.0)
        self.torque_abs_dec.setSuffix(" mm/s²")
        self.torque_abs_band = QtWidgets.QDoubleSpinBox()
        self.torque_abs_band.setRange(0.001, 10.0)
        self.torque_abs_band.setDecimals(3)
        self.torque_abs_band.setValue(0.1)
        self.torque_abs_band.setSuffix(" mm")
        self.btn_torque_abs_move = QtWidgets.QPushButton("力控电机绝对位移")
        self.btn_torque_abs_move.clicked.connect(self.move_torque_abs)

        self.torque_pos_edit = QtWidgets.QLineEdit("未连接")
        self.torque_force_edit = QtWidgets.QLineEdit("未连接")
        self.torque_vel_edit = QtWidgets.QLineEdit("未连接")
        self.torque_moving_edit = QtWidgets.QLineEdit("未连接")
        for edit in (
            self.torque_pos_edit,
            self.torque_force_edit,
            self.torque_vel_edit,
            self.torque_moving_edit,
        ):
            edit.setReadOnly(True)

        row = 0
        control_grid.addWidget(self.btn_motion_home, row, 0)
        control_grid.addWidget(self.btn_torque_home, row, 1)
        control_grid.addWidget(self.btn_torque_force_zero, row, 2)
        row += 1
        control_grid.addWidget(QtWidgets.QLabel("目标位置"), row, 0)
        control_grid.addWidget(self.torque_abs_pos, row, 1)
        control_grid.addWidget(QtWidgets.QLabel("速度"), row, 2)
        control_grid.addWidget(self.torque_abs_vel, row, 3)
        control_grid.addWidget(self.btn_torque_abs_move, row, 4)
        row += 1
        control_grid.addWidget(QtWidgets.QLabel("加速度"), row, 0)
        control_grid.addWidget(self.torque_abs_acc, row, 1)
        control_grid.addWidget(QtWidgets.QLabel("减速度"), row, 2)
        control_grid.addWidget(self.torque_abs_dec, row, 3)
        control_grid.addWidget(QtWidgets.QLabel("定位带宽"), row, 4)
        control_grid.addWidget(self.torque_abs_band, row, 5)
        row += 1
        control_grid.addWidget(QtWidgets.QLabel("实时位置"), row, 0)
        control_grid.addWidget(self.torque_pos_edit, row, 1)
        control_grid.addWidget(QtWidgets.QLabel("实时力值"), row, 2)
        control_grid.addWidget(self.torque_force_edit, row, 3)
        control_grid.addWidget(QtWidgets.QLabel("实时速度"), row, 4)
        control_grid.addWidget(self.torque_vel_edit, row, 5)
        row += 1
        control_grid.addWidget(QtWidgets.QLabel("运动状态"), row, 0)
        control_grid.addWidget(self.torque_moving_edit, row, 1)
        root.addWidget(control_group)

        auto_group = QtWidgets.QGroupBox("自动化采集")
        auto_grid = QtWidgets.QGridLayout(auto_group)
        self.auto_mode_combo = QtWidgets.QComboBox()
        self.auto_mode_combo.addItem("静态网格采集", AutoMode.STATIC.value)
        self.auto_mode_combo.addItem("动态轨迹采集", AutoMode.DYNAMIC.value)
        self.auto_speed_combo = QtWidgets.QComboBox()
        for speed in (0.5, 0.7, 0.9):
            self.auto_speed_combo.addItem(f"{speed:.1f} mm/s", speed)
        self.auto_dynamic_profile_combo = QtWidgets.QComboBox()
        for key, label in DYNAMIC_PROFILE_UI_OPTIONS:
            self.auto_dynamic_profile_combo.addItem(label, key)
        self.auto_approach_target = QtWidgets.QDoubleSpinBox()
        self.auto_approach_target.setRange(0.1, 1000.0)
        self.auto_approach_target.setDecimals(3)
        self.auto_approach_target.setValue(15.0)
        self.auto_approach_target.setSuffix(" mm")
        self.auto_pitch_min = QtWidgets.QDoubleSpinBox()
        self.auto_pitch_min.setRange(-180.0, 180.0)
        self.auto_pitch_min.setDecimals(3)
        self.auto_pitch_min.setValue(-3.0)
        self.auto_pitch_min.setSuffix(" °")
        self.auto_pitch_max = QtWidgets.QDoubleSpinBox()
        self.auto_pitch_max.setRange(-180.0, 180.0)
        self.auto_pitch_max.setDecimals(3)
        self.auto_pitch_max.setValue(3.0)
        self.auto_pitch_max.setSuffix(" °")
        self.auto_roll_min = QtWidgets.QDoubleSpinBox()
        self.auto_roll_min.setRange(-180.0, 180.0)
        self.auto_roll_min.setDecimals(3)
        self.auto_roll_min.setValue(-3.0)
        self.auto_roll_min.setSuffix(" °")
        self.auto_roll_max = QtWidgets.QDoubleSpinBox()
        self.auto_roll_max.setRange(-180.0, 180.0)
        self.auto_roll_max.setDecimals(3)
        self.auto_roll_max.setValue(3.0)
        self.auto_roll_max.setSuffix(" °")
        self.auto_angle_step = QtWidgets.QDoubleSpinBox()
        self.auto_angle_step.setRange(0.001, 30.0)
        self.auto_angle_step.setDecimals(3)
        self.auto_angle_step.setValue(0.1)
        self.auto_angle_step.setSuffix(" °")
        self.auto_retract_distance = QtWidgets.QDoubleSpinBox()
        self.auto_retract_distance.setRange(0.1, 1000.0)
        self.auto_retract_distance.setDecimals(3)
        self.auto_retract_distance.setValue(14.0)
        self.auto_retract_distance.setSuffix(" mm")
        self.auto_retract_speed = QtWidgets.QDoubleSpinBox()
        self.auto_retract_speed.setRange(0.01, 1000.0)
        self.auto_retract_speed.setDecimals(3)
        self.auto_retract_speed.setValue(0.5)
        self.auto_retract_speed.setSuffix(" mm/s")
        self.btn_auto_start = QtWidgets.QPushButton("开始自动采集")
        self.btn_auto_estop = QtWidgets.QPushButton("急停")
        self.btn_auto_estop.setEnabled(False)
        self.btn_auto_estop.setStyleSheet("background:#d32f2f;color:white;font-weight:bold;")
        self.btn_auto_start.clicked.connect(self.start_auto_acquisition)
        self.btn_auto_estop.clicked.connect(self.estop_auto_acquisition)
        self.auto_state_label = QtWidgets.QLabel("IDLE")
        self.auto_progress_label = QtWidgets.QLabel("0/0")
        self.auto_file_edit = QtWidgets.QLineEdit()
        self.auto_file_edit.setReadOnly(True)

        auto_grid.addWidget(QtWidgets.QLabel("模式"), 0, 0)
        auto_grid.addWidget(self.auto_mode_combo, 0, 1)
        auto_grid.addWidget(QtWidgets.QLabel("下压速度"), 0, 2)
        auto_grid.addWidget(self.auto_speed_combo, 0, 3)
        auto_grid.addWidget(self.btn_auto_start, 0, 4)
        auto_grid.addWidget(self.btn_auto_estop, 0, 5)
        auto_grid.addWidget(QtWidgets.QLabel("下压位移"), 1, 0)
        auto_grid.addWidget(self.auto_approach_target, 1, 1)
        auto_grid.addWidget(QtWidgets.QLabel("状态"), 1, 2)
        auto_grid.addWidget(self.auto_state_label, 1, 3)
        auto_grid.addWidget(QtWidgets.QLabel("进度"), 1, 4)
        auto_grid.addWidget(self.auto_progress_label, 1, 5)
        auto_grid.addWidget(QtWidgets.QLabel("Pitch范围"), 2, 0)
        auto_grid.addWidget(self.auto_pitch_min, 2, 1)
        auto_grid.addWidget(QtWidgets.QLabel("到"), 2, 2)
        auto_grid.addWidget(self.auto_pitch_max, 2, 3)
        auto_grid.addWidget(QtWidgets.QLabel("角度步进"), 2, 4)
        auto_grid.addWidget(self.auto_angle_step, 2, 5)
        auto_grid.addWidget(QtWidgets.QLabel("Roll范围"), 3, 0)
        auto_grid.addWidget(self.auto_roll_min, 3, 1)
        auto_grid.addWidget(QtWidgets.QLabel("到"), 3, 2)
        auto_grid.addWidget(self.auto_roll_max, 3, 3)
        auto_grid.addWidget(QtWidgets.QLabel("回退距离"), 4, 0)
        auto_grid.addWidget(self.auto_retract_distance, 4, 1)
        auto_grid.addWidget(QtWidgets.QLabel("回退速度"), 4, 2)
        auto_grid.addWidget(self.auto_retract_speed, 4, 3)
        auto_grid.addWidget(QtWidgets.QLabel("动态轨迹组"), 4, 4)
        auto_grid.addWidget(self.auto_dynamic_profile_combo, 4, 5)
        auto_grid.addWidget(QtWidgets.QLabel("CSV"), 5, 0)
        auto_grid.addWidget(self.auto_file_edit, 5, 1, 1, 5)
        root.addWidget(auto_group)

        self.status_label = QtWidgets.QLabel("状态：空闲")
        root.addWidget(self.status_label)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

    def _append_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def _run_op(self, name: str, fn):
        with self._op_lock:
            if self._op_thread and self._op_thread.is_alive():
                self._append_log("当前已有运动任务在执行，请等待完成")
                return

            def runner():
                self.sig_status.emit(f"状态：{name}中")
                try:
                    fn()
                    self.sig_status.emit(f"状态：{name}完成")
                except Exception as exc:
                    self.sig_status.emit(f"状态：{name}失败")
                    self.sig_log.emit(f"{name}失败：{exc}")

            self._op_thread = threading.Thread(target=runner, daemon=True)
            self._op_thread.start()

    def _ensure_motion_connected(self):
        from drivers.motioncard.ltsmc_dll import LTSMCMotionCard

        with self._hardware_lock:
            if self.motion is not None and getattr(self.motion, "connected", False):
                return self.motion
            motion_cfg = dict(self.cfg["motioncard"])
            dll_path = motion_cfg.get("dll_path")
            if dll_path and not os.path.isabs(dll_path):
                motion_cfg["dll_path"] = os.path.join(PROJECT_ROOT, dll_path)
            self.motion = LTSMCMotionCard(motion_cfg)
            self.motion.connect()
            self.sig_log.emit("运动控制卡已连接")
            return self.motion

    def _ensure_torque_connected(self):
        from drivers.torque_motor.torque_card import TorqueMotorCard

        with self._hardware_lock:
            if self.torque is not None and getattr(self.torque, "connected", False):
                return self.torque
            self.torque = TorqueMotorCard(self.cfg.get("torque_motor", {}))
            self.torque.connect()
            self.torque.set_servo(True)
            self.sig_log.emit("力控电机已连接")
            self._torque_status_error_logged = False
            return self.torque

    def refresh_torque_status(self):
        if self.torque is None or not getattr(self.torque, "connected", False):
            self.torque_pos_edit.setText("未连接")
            self.torque_force_edit.setText("未连接")
            self.torque_vel_edit.setText("未连接")
            self.torque_moving_edit.setText("未连接")
            return

        try:
            with self._hardware_lock:
                status = self.torque.read_status()
        except Exception as exc:
            self.torque_pos_edit.setText("读取失败")
            self.torque_force_edit.setText("读取失败")
            self.torque_vel_edit.setText("读取失败")
            self.torque_moving_edit.setText("读取失败")
            if not self._torque_status_error_logged:
                self._append_log(f"力控电机实时状态读取失败：{exc}")
                self._torque_status_error_logged = True
            return

        self._torque_status_error_logged = False
        self.torque_pos_edit.setText(f"{status['position']:.3f} mm")
        self.torque_force_edit.setText(f"{status['force']:.3f} N")
        self.torque_vel_edit.setText(f"{status['velocity']:.3f} mm/s")
        self.torque_moving_edit.setText("运动中" if status.get("moving") else "停止")

    def _wait_torque_stop(self, timeout_s: float = 60.0, vel_eps: float = 0.01):
        torque = self._ensure_torque_connected()
        t0 = time.time()
        stable = 0
        while time.time() - t0 < timeout_s:
            moving = not torque.is_done(0)
            vel = abs(torque.get_velocity(0))
            stable = stable + 1 if (not moving) or vel < vel_eps else 0
            if stable >= 3:
                return
            time.sleep(0.05)
        try:
            torque.stop(0)
        except Exception:
            pass
        raise TimeoutError("力控电机等待停止超时")

    def home_motion_axes(self):
        def job():
            from drivers.motioncard.utils import full_axis_initialization, perform_homing

            card = self._ensure_motion_connected()
            for axis in (0, 1, 2, 3):
                self.sig_log.emit(f"运动控制卡轴 {axis} 初始化...")
                if not full_axis_initialization(card, axis):
                    raise RuntimeError(f"轴 {axis} 初始化失败")
                self.sig_log.emit(f"运动控制卡轴 {axis} 回零...")
                if not perform_homing(card, axis, timeout=60.0):
                    raise RuntimeError(f"轴 {axis} 回零失败")
                self.sig_log.emit(f"运动控制卡轴 {axis} 回零完成")
                time.sleep(0.2)
            self.sig_log.emit("运动控制卡四轴回零完成")

        self._run_op("运动控制卡四轴回零", job)

    def home_torque_motor(self):
        def job():
            torque = self._ensure_torque_connected()
            torque.home(0)
            self._wait_torque_stop(timeout_s=120.0)
            self.sig_log.emit(f"力控电机回零完成，当前位置 {torque.get_position(0):.3f} mm")

        self._run_op("力控电机回零", job)

    def zero_torque_force(self):
        def job():
            torque = self._ensure_torque_connected()
            try:
                torque.stop(0)
            except Exception:
                pass
            torque.trigger_command(25)
            time.sleep(0.2)
            self.sig_log.emit("力控电机力清零命令已发送")

        self._run_op("力控电机力清零", job)

    def move_torque_abs(self):
        pos = float(self.torque_abs_pos.value())
        vel = float(self.torque_abs_vel.value())
        acc = float(self.torque_abs_acc.value())
        dec = float(self.torque_abs_dec.value())
        band = float(self.torque_abs_band.value())

        def job():
            torque = self._ensure_torque_connected()
            torque.set_profile(0, 0.0, vel, acc, dec, 0.0)
            torque.set_band(0, band)
            torque.move_abs(0, pos)
            self._wait_torque_stop(timeout_s=120.0)
            self.sig_log.emit(f"力控电机绝对位移完成，当前位置 {torque.get_position(0):.3f} mm")

        self._run_op("力控电机绝对位移", job)

    def _acquire_loop(self):
        while self._running:
            try:
                with self._io_lock:
                    frames = self.sensor.read_frames() if hasattr(self.sensor, "read_frames") else []
                    if not frames:
                        frame = self.sensor.read_frame()
                        frames = [frame] if frame is not None else []
                if not frames:
                    time.sleep(0.005)
                    continue
                for frame in frames:
                    frame["_host_monotonic"] = time.monotonic()
                    frame["_host_timestamp"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    self.sig_frame.emit(frame)
            except Exception as exc:
                self.sig_status.emit(f"读取失败：{exc}")
                time.sleep(0.2)

    def update_ui_with_frame(self, frame: Dict):
        self.current_frame = frame
        proximity = frame.get("proximity", {})
        distance = frame.get("distance", {})
        for name in PROXIMITY_CHANNELS:
            raw_v = proximity.get(name, 0.0)
            dist_v = distance.get(name, 0.0)
            self.raw_edits[name].setText(f"{raw_v:.0f}")
            self.distance_edits[name].setText(f"{dist_v:.2f}")
            self.curve_data[name].append(raw_v)
            self.curve_data[name] = self.curve_data[name][-100:]
        self._refresh_curves()

    def init_analyzer(self):
        with self._io_lock:
            ok = self.sensor.init_analyzer()
        self._append_log("分析仪初始化命令已发送" if ok else "分析仪初始化失败")

    def zero_sensor(self):
        with self._io_lock:
            ok = self.sensor.zero()
        self._append_log("清零命令已发送" if ok else "清零命令发送失败")

    def show_plot_window(self):
        if self.plot_window is None:
            self.plot_window = pg.GraphicsLayoutWidget()
            self.plot_window.setWindowTitle("接近觉传感器实时曲线")
            self.plot_window.resize(800, 600)
            plot = self.plot_window.addPlot(title="接近觉实时原始值")
            plot.addLegend()
            pens = {"jjj1_1": "r", "jjj2_1": "g", "jjj1_2": "b", "jjj2_2": "y"}
            self.curves = {name: plot.plot(pen=pens[name], name=name) for name in PROXIMITY_CHANNELS}
        self.plot_window.show()

    def _refresh_curves(self):
        if not self.plot_window:
            return
        for name, curve in self.curves.items():
            curve.setData(self.curve_data[name])

    def _auto_params_from_ui(self) -> AutoAcquisitionParams:
        pitch_min = float(self.auto_pitch_min.value())
        pitch_max = float(self.auto_pitch_max.value())
        roll_min = float(self.auto_roll_min.value())
        roll_max = float(self.auto_roll_max.value())
        if pitch_min > pitch_max:
            raise ValueError("Pitch最小值不能大于最大值")
        if roll_min > roll_max:
            raise ValueError("Roll最小值不能大于最大值")
        return AutoAcquisitionParams(
            mode=AutoMode(self.auto_mode_combo.currentData()),
            speed_mm_s=float(self.auto_speed_combo.currentData()),
            output_dir=DEFAULT_LOG_DIR,
            dynamic_profile_key=str(self.auto_dynamic_profile_combo.currentData()),
            pitch_min=pitch_min,
            pitch_max=pitch_max,
            pitch_step=float(self.auto_angle_step.value()),
            roll_min=roll_min,
            roll_max=roll_max,
            roll_step=float(self.auto_angle_step.value()),
            approach_target_mm=float(self.auto_approach_target.value()),
            retract_mm=float(self.auto_retract_distance.value()),
            retract_speed_mm_s=float(self.auto_retract_speed.value()),
            start_distance_mm=8.0,
            stop_distance_mm=1.0,
        )

    def start_auto_acquisition(self):
        if self.auto_running:
            return
        if self._op_thread and self._op_thread.is_alive():
            QMessageBox.warning(self, "无法开始", "当前已有运动任务在执行。")
            return
        try:
            motion = self._ensure_motion_connected()
            torque = self._ensure_torque_connected()
        except Exception as exc:
            QMessageBox.critical(self, "硬件连接失败", str(exc))
            return

        try:
            params = self._auto_params_from_ui()
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return
        self.auto_thread = QtCore.QThread(self)
        self.auto_worker = AutoAcquisitionWorker(self.sensor, self._io_lock, self._hardware_lock, motion, torque, params)
        self.auto_worker.moveToThread(self.auto_thread)

        self.auto_thread.started.connect(self.auto_worker.run)
        self.sig_frame.connect(self.auto_worker.on_frame, QtCore.Qt.DirectConnection)
        self.auto_worker.sig_log.connect(self._append_log)
        self.auto_worker.sig_status.connect(self.status_label.setText)
        self.auto_worker.sig_state.connect(self.auto_state_label.setText)
        self.auto_worker.sig_progress.connect(self._update_auto_progress)
        self.auto_worker.sig_file.connect(self._update_auto_file)
        self.auto_worker.sig_finished.connect(self._on_auto_finished)
        self.auto_worker.sig_finished.connect(self.auto_thread.quit)
        self.auto_worker.sig_finished.connect(self.auto_worker.deleteLater)
        self.auto_thread.finished.connect(self.auto_thread.deleteLater)

        self.auto_running = True
        self.btn_auto_start.setEnabled(False)
        self.btn_auto_estop.setEnabled(True)
        self.btn_init.setEnabled(False)
        self.btn_zero.setEnabled(False)
        self.btn_motion_home.setEnabled(False)
        self.btn_torque_home.setEnabled(False)
        self.btn_torque_force_zero.setEnabled(False)
        self.btn_torque_abs_move.setEnabled(False)
        self.auto_state_label.setText("INIT")
        self.auto_thread.start()

    def estop_auto_acquisition(self):
        if self.auto_worker:
            threading.Thread(target=self.auto_worker.request_stop, daemon=True).start()
        self._append_log("自动采集急停已触发")

    def _update_auto_progress(self, current: int, total: int):
        self.auto_progress_label.setText(f"{current}/{total}")

    def _update_auto_file(self, path: str):
        self.auto_file_edit.setText(path)

    def _on_auto_finished(self):
        if self.auto_worker:
            try:
                self.sig_frame.disconnect(self.auto_worker.on_frame)
            except Exception:
                pass
        self.auto_worker = None
        self.auto_thread = None
        self.auto_running = False
        self.btn_auto_start.setEnabled(True)
        self.btn_auto_estop.setEnabled(False)
        self.btn_init.setEnabled(True)
        self.btn_zero.setEnabled(True)
        self.btn_motion_home.setEnabled(True)
        self.btn_torque_home.setEnabled(True)
        self.btn_torque_force_zero.setEnabled(True)
        self.btn_torque_abs_move.setEnabled(True)

    def closeEvent(self, event):
        self._running = False
        if self.auto_worker:
            try:
                threading.Thread(target=self.auto_worker.request_stop, daemon=True).start()
            except Exception:
                pass
        if self.auto_thread and self.auto_thread.isRunning():
            self.auto_thread.quit()
            self.auto_thread.wait(3000)
        try:
            self.torque_status_timer.stop()
        except Exception:
            pass
        try:
            if self.acq_thread.is_alive():
                self.acq_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.sensor.disconnect()
        except Exception:
            pass
        try:
            if self.torque:
                self.torque.disconnect()
        except Exception:
            pass
        try:
            if self.motion:
                self.motion.disconnect()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    cfg = load_cfg()
    sensor_cfg = cfg.get("sensor", {}).get("proximity_sensor")
    if not sensor_cfg:
        raise ValueError("配置文件缺少 sensor.proximity_sensor")

    sensor = create_proximity_sensor(sensor_cfg)
    if not initialize_proximity_sensor(sensor):
        raise RuntimeError("接近觉传感器初始化失败")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = ProximityAcquisitionWindow(sensor, sensor_cfg)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
