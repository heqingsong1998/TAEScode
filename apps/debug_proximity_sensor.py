from __future__ import annotations

import csv
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
import yaml
from PyQt5 import QtCore, QtWidgets

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from drivers.proximity_sensor.utils import create_proximity_sensor, initialize_proximity_sensor


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
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


def _format_coeff(value: float) -> str:
    return f"{float(value):.10g}"


def sync_proximity_calibration_to_yaml(coeffs: Dict[str, Dict[str, float]]) -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out: List[str] = []
    in_proximity = False
    in_processing = False
    in_calibration = False
    calibration_indent = -1
    replaced = set()

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped and not stripped.startswith("#"):
            if indent == 2 and stripped == "proximity_sensor:":
                in_proximity = True
                in_processing = False
                in_calibration = False
            elif in_proximity and indent <= 2 and stripped != "proximity_sensor:":
                in_proximity = False
                in_processing = False
                in_calibration = False

            if in_proximity and indent == 4 and stripped == "processing:":
                in_processing = True
                in_calibration = False
            elif in_processing and indent <= 4 and stripped != "processing:":
                in_processing = False
                in_calibration = False

            if in_processing and indent == 6 and stripped == "calibration:":
                in_calibration = True
                calibration_indent = indent
            elif in_calibration and indent <= calibration_indent and stripped != "calibration:":
                in_calibration = False

        if in_calibration and indent > calibration_indent and ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            if key in coeffs:
                c = coeffs[key]
                out.append(
                    f"{' ' * indent}{key}: "
                    f"{{a: {_format_coeff(c['a'])}, b: {_format_coeff(c['b'])}, c: {_format_coeff(c['c'])}}}\n"
                )
                replaced.add(key)
                continue

        out.append(line)

    missing = set(coeffs) - replaced
    if missing:
        raise RuntimeError(f"default.yaml中未找到以下标定通道：{sorted(missing)}")

    with open(CONFIG_PATH, "w", encoding="utf-8", newline="") as f:
        f.writelines(out)


class ProximityCalibrationWindow(QtWidgets.QWidget):
    sig_frame = QtCore.pyqtSignal(dict)
    sig_status = QtCore.pyqtSignal(str)
    sig_log = QtCore.pyqtSignal(str)
    sig_calib_state = QtCore.pyqtSignal(str, int)

    def __init__(self, sensor, sensor_cfg: Dict):
        super().__init__()
        self.sensor = sensor
        self.sensor_cfg = sensor_cfg
        self.cfg = load_cfg()

        self.current_frame: Optional[Dict] = None
        self._running = True
        self._io_lock = threading.Lock()
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[Dict] = None
        self._op_lock = threading.Lock()
        self._op_thread: Optional[threading.Thread] = None
        self._hardware_lock = threading.RLock()

        self.motion: Optional[object] = None
        self.torque: Optional[object] = None
        self.calib_csv_path: Optional[str] = None
        self.calib_next_label = 0

        self.curves = {}
        self.curve_data = {name: [] for name in PROXIMITY_CHANNELS}
        self.plot_window: Optional[pg.GraphicsLayoutWidget] = None
        self._torque_status_error_logged = False

        self.setWindowTitle("接近觉传感器标定")
        self.resize(1180, 760)
        self._build_ui()

        self.sig_frame.connect(self.update_ui_with_frame)
        self.sig_status.connect(self.status_label.setText)
        self.sig_log.connect(self._append_log)
        self.sig_calib_state.connect(self._update_calib_state)

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
        self.raw_edits = {name: QtWidgets.QLineEdit("0") for name in PROXIMITY_CHANNELS}
        self.distance_edits = {name: QtWidgets.QLineEdit("0.00") for name in PROXIMITY_CHANNELS}
        for edit in list(self.raw_edits.values()) + list(self.distance_edits.values()):
            edit.setReadOnly(True)

        r = 0
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

        control_group = QtWidgets.QGroupBox("标定硬件控制")
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
        control_grid.addWidget(QtWidgets.QLabel("实时力"), row, 2)
        control_grid.addWidget(self.torque_force_edit, row, 3)
        control_grid.addWidget(QtWidgets.QLabel("实时速度"), row, 4)
        control_grid.addWidget(self.torque_vel_edit, row, 5)
        row += 1
        control_grid.addWidget(QtWidgets.QLabel("运动状态"), row, 0)
        control_grid.addWidget(self.torque_moving_edit, row, 1)
        root.addWidget(control_group)

        calib_group = QtWidgets.QGroupBox("标定采样与拟合")
        calib_grid = QtWidgets.QGridLayout(calib_group)
        self.calib_duration = QtWidgets.QDoubleSpinBox()
        self.calib_duration.setRange(0.5, 60.0)
        self.calib_duration.setDecimals(1)
        self.calib_duration.setValue(5.0)
        self.calib_duration.setSuffix(" s")
        self.calib_next_label_text = QtWidgets.QLabel("0")
        self.calib_file_edit = QtWidgets.QLineEdit()
        self.calib_file_edit.setReadOnly(True)
        self.btn_calib_sample = QtWidgets.QPushButton("标定采样(5s均值)")
        self.btn_calib_sample.clicked.connect(self.sample_calibration_average)
        self.btn_fit_calib = QtWidgets.QPushButton("计算二次拟合并同步配置")
        self.btn_fit_calib.clicked.connect(self.fit_calibration_quadratic)

        calib_grid.addWidget(QtWidgets.QLabel("采样时长"), 0, 0)
        calib_grid.addWidget(self.calib_duration, 0, 1)
        calib_grid.addWidget(QtWidgets.QLabel("下一个序号标签"), 0, 2)
        calib_grid.addWidget(self.calib_next_label_text, 0, 3)
        calib_grid.addWidget(self.btn_calib_sample, 0, 4)
        calib_grid.addWidget(self.btn_fit_calib, 0, 5)
        calib_grid.addWidget(QtWidgets.QLabel("标定CSV"), 1, 0)
        calib_grid.addWidget(self.calib_file_edit, 1, 1, 1, 5)
        root.addWidget(calib_group)

        self.status_label = QtWidgets.QLabel("状态：空闲")
        root.addWidget(self.status_label)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.log, 1)

    def _append_log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def _update_calib_state(self, path: str, next_label: int):
        self.calib_file_edit.setText(path)
        self.calib_next_label_text.setText(str(next_label))

    def _run_op(self, name: str, fn):
        with self._op_lock:
            if self._op_thread and self._op_thread.is_alive():
                self._append_log("当前已有标定/运动任务在执行，请等待完成")
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
            self.sig_log.emit("力控电机回零...")
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
            try:
                self.sig_log.emit(f"力控电机力清零命令已发送，当前力 {torque.get_force():.3f} N")
            except Exception:
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
            self.sig_log.emit(
                f"力控电机绝对位移：target={pos:.3f} mm, v={vel:.3f} mm/s, "
                f"acc={acc:.3f}, dec={dec:.3f}, band={band:.3f} mm"
            )
            torque.move_abs(0, pos)
            self._wait_torque_stop(timeout_s=120.0)
            self.sig_log.emit(f"力控电机绝对位移完成，当前位置 {torque.get_position(0):.3f} mm")

        self._run_op("力控电机绝对位移", job)

    def _latest_frame_copy(self) -> Optional[Dict]:
        with self._latest_lock:
            return dict(self._latest_frame) if self._latest_frame is not None else None

    def _calib_header(self) -> List[str]:
        return (
            ["label", "start_time", "end_time", "duration_s", "frame_count"]
            + [f"{name}_mean" for name in PROXIMITY_CHANNELS]
            + [f"{name}_distance_mean" for name in PROXIMITY_CHANNELS]
        )

    def _ensure_calib_csv(self) -> str:
        if self.calib_csv_path:
            return self.calib_csv_path
        os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.calib_csv_path = os.path.join(DEFAULT_LOG_DIR, f"proximity_calibration_{now}.csv")
        with open(self.calib_csv_path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(self._calib_header())
        self.sig_calib_state.emit(self.calib_csv_path, self.calib_next_label)
        self.sig_log.emit(f"已创建标定CSV：{self.calib_csv_path}")
        return self.calib_csv_path

    def sample_calibration_average(self):
        duration_s = float(self.calib_duration.value())

        def job():
            path = self._ensure_calib_csv()
            label = self.calib_next_label
            self.sig_log.emit(f"开始标定采样：label={label}, duration={duration_s:.1f}s")
            t0 = time.time()
            start_time = datetime.now().isoformat(timespec="milliseconds")
            seen = set()
            raw_values = {name: [] for name in PROXIMITY_CHANNELS}
            distance_values = {name: [] for name in PROXIMITY_CHANNELS}

            while time.time() - t0 < duration_s:
                frame = self._latest_frame_copy()
                if frame is None:
                    time.sleep(0.01)
                    continue
                key = frame.get("timestamp") or id(frame)
                if key in seen:
                    time.sleep(0.005)
                    continue
                seen.add(key)
                proximity = frame.get("proximity", {})
                distance = frame.get("distance", {})
                for name in PROXIMITY_CHANNELS:
                    raw_values[name].append(float(proximity.get(name, np.nan)))
                    distance_values[name].append(float(distance.get(name, np.nan)))
                time.sleep(0.005)

            frame_count = len(seen)
            if frame_count == 0:
                raise RuntimeError("采样时间内没有读到有效接近觉数据")

            raw_means = [float(np.nanmean(raw_values[name])) for name in PROXIMITY_CHANNELS]
            distance_means = [float(np.nanmean(distance_values[name])) for name in PROXIMITY_CHANNELS]
            row = [
                label,
                start_time,
                datetime.now().isoformat(timespec="milliseconds"),
                duration_s,
                frame_count,
                *raw_means,
                *distance_means,
            ]

            with open(path, mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            self.calib_next_label += 1
            self.sig_calib_state.emit(path, self.calib_next_label)
            mean_text = ", ".join(f"{n}={v:.3f}" for n, v in zip(PROXIMITY_CHANNELS, raw_means))
            self.sig_log.emit(f"标定采样完成：label={label}, frames={frame_count}, {mean_text}")

        self._run_op("标定采样", job)

    def fit_calibration_quadratic(self):
        def job():
            path = self.calib_csv_path
            if not path or not os.path.exists(path):
                raise RuntimeError("没有可拟合的标定CSV，请先执行标定采样")

            with open(path, mode="r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if len(rows) < 3:
                raise RuntimeError("二次拟合至少需要3个标定样本")

            labels = np.asarray([float(row["label"]) for row in rows], dtype=float)
            coeffs: Dict[str, Dict[str, float]] = {}
            output_rows = []
            self.sig_log.emit("二次拟合结果：label = a*log10(raw)^2 + b*log10(raw) + c")

            for name in PROXIMITY_CHANNELS:
                raw = np.asarray([float(row[f"{name}_mean"]) for row in rows], dtype=float)
                x = np.log10(np.maximum(np.abs(raw), 1e-10))
                valid = np.isfinite(x) & np.isfinite(labels)
                if valid.sum() < 3 or np.unique(x[valid]).size < 3:
                    self.sig_log.emit(f"{name}: 有效样本不足，跳过")
                    continue

                a, b, c = np.polyfit(x[valid], labels[valid], deg=2)
                y_hat = a * x[valid] * x[valid] + b * x[valid] + c
                y = labels[valid]
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                coeffs[name] = {"a": float(a), "b": float(b), "c": float(c)}
                output_rows.append([name, a, b, c, r2, int(valid.sum())])
                self.sig_log.emit(
                    f"{name}: a={a:.10g}, b={b:.10g}, c={c:.10g}, R²={r2:.6f}, n={int(valid.sum())}"
                )

            if not coeffs:
                raise RuntimeError("没有得到有效拟合结果")

            fit_path = os.path.splitext(path)[0] + "_fit.csv"
            with open(fit_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "a", "b", "c", "r2", "n"])
                writer.writerows(output_rows)

            sync_proximity_calibration_to_yaml(coeffs)
            if hasattr(self.sensor, "processor"):
                self.sensor.processor.calibration.update(coeffs)
            self.cfg = load_cfg()
            self.sig_log.emit(f"拟合参数已保存：{fit_path}")
            self.sig_log.emit(f"拟合参数已同步到：{CONFIG_PATH}")

        self._run_op("二次拟合", job)

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
                    with self._latest_lock:
                        self._latest_frame = frame
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

    def closeEvent(self, event):
        self._running = False
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
    win = ProximityCalibrationWindow(sensor, sensor_cfg)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
