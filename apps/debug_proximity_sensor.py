from __future__ import annotations

import csv
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyqtgraph as pg
import yaml
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem

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
MODE_SENSOR_CALIBRATION = "传感器标定"
MODE_SURFACE_TILT_COLLECTION = "面倾斜度采集"


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


class ProximitySensorWindow(QtWidgets.QWidget):
    sig_frame = QtCore.pyqtSignal(dict)
    sig_status = QtCore.pyqtSignal(str)
    sig_log = QtCore.pyqtSignal(str)
    sig_calib_state = QtCore.pyqtSignal(str, int)

    def __init__(self, sensor, sensor_cfg: Dict):
        super().__init__()
        self.sensor = sensor
        self.sensor_cfg = sensor_cfg
        self.cfg = load_cfg()
        self.collection_cfg = sensor_cfg.get("collection", {})

        self.collecting = False
        self.data_recording_ready = False
        self.moving_forward = True
        self.sampled_once = False

        self.csv_file_path: Optional[str] = None
        self.csv_file: Optional[object] = None
        self.csv_writer = None

        self.current_frame: Optional[Dict] = None
        self.current_highlighted_row = -1
        self.sample_count = 0

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

        self.all_pairs = self.get_all_pitch_roll_pairs()
        self.remaining_pairs = set(self.all_pairs)

        self.setWindowTitle("接近觉传感器数据采集")
        self.resize(1180, 760)
        self._build_ui()
        self.update_table_with_missing_combinations(self.remaining_pairs)

        self.sig_frame.connect(self.update_ui_with_frame)
        self.sig_status.connect(self.status_label.setText)
        self.sig_log.connect(self._append_log)
        self.sig_calib_state.connect(self._update_calib_state)

        self.acq_thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self.acq_thread.start()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_init = QtWidgets.QPushButton("初始化分析仪")
        self.btn_zero = QtWidgets.QPushButton("清零")
        self.btn_create = QtWidgets.QPushButton("创建数据文件")
        self.btn_select = QtWidgets.QPushButton("选择数据文件")
        self.btn_start = QtWidgets.QPushButton("开始采集")
        self.btn_stop = QtWidgets.QPushButton("停止采集")
        self.btn_plot = QtWidgets.QPushButton("绘制曲线")
        self.btn_reset_sample = QtWidgets.QPushButton("复位样本")
        self.btn_delete_sample = QtWidgets.QPushButton("删除样本对")

        self.btn_init.clicked.connect(self.init_analyzer)
        self.btn_zero.clicked.connect(self.zero_sensor)
        self.btn_create.clicked.connect(self.create_csv_file)
        self.btn_select.clicked.connect(self.select_csv_file)
        self.btn_start.clicked.connect(self.on_start_collect)
        self.btn_stop.clicked.connect(self.on_stop_collect)
        self.btn_plot.clicked.connect(self.show_plot_window)
        self.btn_reset_sample.clicked.connect(self.on_reset_sample_clicked)
        self.btn_delete_sample.clicked.connect(self.delete_sample_pair)

        for btn in (
            self.btn_init,
            self.btn_zero,
            self.btn_create,
            self.btn_select,
            self.btn_start,
            self.btn_stop,
            self.btn_plot,
            self.btn_reset_sample,
            self.btn_delete_sample,
        ):
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

        grid = QtWidgets.QGridLayout()
        self.mode_choice = QtWidgets.QComboBox()
        self.mode_choice.addItems([MODE_SENSOR_CALIBRATION, MODE_SURFACE_TILT_COLLECTION])

        self.pitch_edit = QtWidgets.QLineEdit("0")
        self.roll_edit = QtWidgets.QLineEdit("0")
        self.pitch_edit.textChanged.connect(self.reset_sample_count)
        self.roll_edit.textChanged.connect(self.reset_sample_count)
        self.selected_file_path = QtWidgets.QLineEdit()
        self.selected_file_path.setReadOnly(True)
        self.sample_pairs = QtWidgets.QLineEdit()
        self.sample_pairs.setReadOnly(True)
        self.sample_num = QtWidgets.QLabel("0")

        self.condition_dis = QtWidgets.QCheckBox("距离条件满足")
        self.condition_dis.setEnabled(False)
        self.return_journey = QtWidgets.QCheckBox("回程")
        self.return_journey.setEnabled(False)

        self.raw_edits = {name: QtWidgets.QLineEdit("0") for name in PROXIMITY_CHANNELS}
        self.distance_edits = {name: QtWidgets.QLineEdit("0.00") for name in PROXIMITY_CHANNELS}
        for edit in list(self.raw_edits.values()) + list(self.distance_edits.values()):
            edit.setReadOnly(True)

        r = 0
        grid.addWidget(QtWidgets.QLabel("模式"), r, 0)
        grid.addWidget(self.mode_choice, r, 1)
        grid.addWidget(QtWidgets.QLabel("俯仰角"), r, 2)
        grid.addWidget(self.pitch_edit, r, 3)
        grid.addWidget(QtWidgets.QLabel("横滚角"), r, 4)
        grid.addWidget(self.roll_edit, r, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("数据文件"), r, 0)
        grid.addWidget(self.selected_file_path, r, 1, 1, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("当前样本对"), r, 0)
        grid.addWidget(self.sample_pairs, r, 1)
        grid.addWidget(QtWidgets.QLabel("样本数"), r, 2)
        grid.addWidget(self.sample_num, r, 3)
        grid.addWidget(self.condition_dis, r, 4)
        grid.addWidget(self.return_journey, r, 5)

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
        self.btn_fit_calib = QtWidgets.QPushButton("计算二次拟合")
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

        main_split = QtWidgets.QSplitter()
        self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.cellClicked.connect(self.fill_angles_from_table)
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        main_split.addWidget(self.tableWidget)
        main_split.addWidget(self.log)
        main_split.setStretchFactor(0, 1)
        main_split.setStretchFactor(1, 2)
        root.addWidget(main_split, 1)

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
            return self.torque

    def _wait_torque_stop(self, timeout_s: float = 60.0, vel_eps: float = 0.01):
        torque = self._ensure_torque_connected()
        t0 = time.time()
        stable = 0
        while time.time() - t0 < timeout_s:
            try:
                moving = not torque.is_done(0)
            except Exception:
                moving = True
            try:
                vel = abs(torque.get_velocity(0))
            except Exception:
                vel = 999.0
            if (not moving) or vel < vel_eps:
                stable += 1
            else:
                stable = 0
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
            pos = torque.get_position(0)
            self.sig_log.emit(f"力控电机回零完成，当前位置 {pos:.3f} mm")

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
                force = torque.get_force()
                self.sig_log.emit(f"力控电机力清零命令已发送，当前力 {force:.3f} N")
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
            cur = torque.get_position(0)
            self.sig_log.emit(f"力控电机绝对位移完成，当前位置 {cur:.3f} mm")

        self._run_op("力控电机绝对位移", job)

    def _latest_frame_copy(self) -> Optional[Dict]:
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            return dict(self._latest_frame)

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

                key = frame.get("timestamp")
                if key is None:
                    key = id(frame)
                if key in seen:
                    time.sleep(0.005)
                    continue
                seen.add(key)

                proximity = frame.get("proximity", {})
                distance = frame.get("distance", {})
                for name in PROXIMITY_CHANNELS:
                    try:
                        raw_values[name].append(float(proximity.get(name, np.nan)))
                    except Exception:
                        raw_values[name].append(float("nan"))
                    try:
                        distance_values[name].append(float(distance.get(name, np.nan)))
                    except Exception:
                        distance_values[name].append(float("nan"))
                time.sleep(0.005)

            frame_count = len(seen)
            if frame_count == 0:
                raise RuntimeError("5秒内没有读到有效接近觉数据")

            raw_means = [
                float(np.nanmean(raw_values[name])) if raw_values[name] else float("nan")
                for name in PROXIMITY_CHANNELS
            ]
            distance_means = [
                float(np.nanmean(distance_values[name])) if distance_values[name] else float("nan")
                for name in PROXIMITY_CHANNELS
            ]
            end_time = datetime.now().isoformat(timespec="milliseconds")
            row = [label, start_time, end_time, duration_s, frame_count] + raw_means + distance_means

            with open(path, mode="a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            self.calib_next_label += 1
            self.sig_calib_state.emit(path, self.calib_next_label)
            mean_text = ", ".join(
                f"{name}={value:.3f}" for name, value in zip(PROXIMITY_CHANNELS, raw_means)
            )
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
            output_rows = []
            self.sig_log.emit("二次拟合结果：label = a*x^2 + b*x + c")

            for name in PROXIMITY_CHANNELS:
                x = np.asarray([float(row[f"{name}_mean"]) for row in rows], dtype=float)
                valid = np.isfinite(x) & np.isfinite(labels)
                if valid.sum() < 3:
                    self.sig_log.emit(f"{name}: 有效样本不足，跳过")
                    continue
                if np.unique(x[valid]).size < 3:
                    self.sig_log.emit(f"{name}: x取值少于3个唯一值，无法二次拟合")
                    continue

                a, b, c = np.polyfit(x[valid], labels[valid], deg=2)
                y_hat = a * x[valid] * x[valid] + b * x[valid] + c
                y = labels[valid]
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                output_rows.append([name, a, b, c, r2, int(valid.sum())])
                self.sig_log.emit(
                    f"{name}: a={a:.10g}, b={b:.10g}, c={c:.10g}, R²={r2:.6f}, n={int(valid.sum())}"
                )

            if not output_rows:
                raise RuntimeError("没有得到有效拟合结果")

            fit_path = os.path.splitext(path)[0] + "_fit.csv"
            with open(fit_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["channel", "a", "b", "c", "r2", "n"])
                writer.writerows(output_rows)
            self.sig_log.emit(f"拟合参数已保存：{fit_path}")

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

        self._update_collection_gate(distance)
        self.save_data_row()
        self._refresh_curves()

    def _update_collection_gate(self, distance: Dict[str, float]):
        if self.mode_choice.currentText() != MODE_SURFACE_TILT_COLLECTION:
            return

        distances = [float(distance.get(name, 0.0)) for name in PROXIMITY_CHANNELS]
        start_min = float(self.collection_cfg.get("start_distance_min_mm", 1.0))
        start_max = float(self.collection_cfg.get("start_distance_max_mm", 7.0))
        stop_distance = float(self.collection_cfg.get("stop_distance_mm", 1.0))

        if any(d < stop_distance for d in distances) and self.data_recording_ready:
            self.status_label.setText("数据采集完成，停止记录")
            self.data_recording_ready = False
            self.sampled_once = True
            self.moving_forward = False
            self.condition_dis.setChecked(False)
            self.return_journey.setChecked(True)
            self.btn_reset_sample.setEnabled(True)
            self._remove_current_pair_if_valid()
        elif all(start_min <= d < start_max for d in distances) and self.moving_forward and not self.sampled_once:
            self.status_label.setText("距离条件满足，正在记录")
            self.data_recording_ready = True
            self.condition_dis.setChecked(True)
        elif not self.moving_forward:
            self.status_label.setText("正在回程" if all(d > stop_distance for d in distances) else "请开始回程")
            self.data_recording_ready = False
            self.condition_dis.setChecked(False)
        else:
            self.status_label.setText("距离条件未满足，暂停记录")
            self.data_recording_ready = False
            self.condition_dis.setChecked(False)

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
            self.curves = {
                name: plot.plot(pen=pens[name], name=name)
                for name in PROXIMITY_CHANNELS
            }
        self.plot_window.show()

    def _refresh_curves(self):
        if not self.plot_window:
            return
        for name, curve in self.curves.items():
            curve.setData(self.curve_data[name])

    def on_start_collect(self):
        if self.csv_writer is None:
            QMessageBox.warning(self, "未准备好", "请先创建或选择数据文件。")
            return
        self.collecting = True
        mode = self.mode_choice.currentText()
        if mode == MODE_SENSOR_CALIBRATION:
            self.status_label.setText("正在采集标定数据")
        else:
            self.status_label.setText("正在采集倾角数据，等待距离条件满足")
        self._append_log("开始采集")

    def on_stop_collect(self):
        self.collecting = False
        if self.csv_file:
            self.csv_file.flush()
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None
        self._append_log("采集已停止，数据文件已关闭")

    def create_csv_file(self):
        os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file_path = os.path.join(DEFAULT_LOG_DIR, f"proximity_{now}.csv")
        if self.csv_file:
            self.csv_file.close()
        self.csv_file = open(self.csv_file_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self._csv_header())
        self.selected_file_path.setText(self.csv_file_path)
        self.sample_num.setText("0")
        self._append_log(f"已创建数据文件：{self.csv_file_path}")

    def select_csv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "逗号分隔文件 (*.csv);;所有文件 (*)")
        if not file_path:
            return

        self.csv_file_path = file_path
        self.selected_file_path.setText(file_path)
        existing_pairs = self.get_existing_pitch_roll_from_csv(file_path)
        self.remaining_pairs = self.all_pairs - existing_pairs
        self.update_table_with_missing_combinations(self.remaining_pairs)

        if self.csv_file:
            self.csv_file.close()
        self.csv_file = open(file_path, mode="a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self._append_log(f"已选择数据文件：{file_path}")

    def _csv_header(self) -> List[str]:
        if self.mode_choice.currentText() == MODE_SENSOR_CALIBRATION:
            return [
                "timestamp",
                "jjj1_1",
                "jjj2_1",
                "jjj1_2",
                "jjj2_2",
                "distance_zs",
                "distance_ys",
                "distance_zx",
                "distance_yx",
            ]
        return ["timestamp", "jjj1_1", "jjj2_1", "jjj1_2", "jjj2_2", "pitch", "roll"]

    def save_data_row(self):
        if not self.collecting or self.csv_writer is None or self.current_frame is None:
            return
        mode = self.mode_choice.currentText()
        if mode == MODE_SURFACE_TILT_COLLECTION and (
            not self.data_recording_ready or not self.moving_forward or self.sampled_once
        ):
            return

        proximity = self.current_frame.get("proximity", {})
        distance = self.current_frame.get("distance", {})
        ts = self.current_frame.get("timestamp", datetime.now().isoformat(timespec="milliseconds"))

        if mode == MODE_SENSOR_CALIBRATION:
            row = [
                ts,
                proximity.get("jjj1_1"),
                proximity.get("jjj2_1"),
                proximity.get("jjj1_2"),
                proximity.get("jjj2_2"),
                distance.get("jjj1_1"),
                distance.get("jjj2_1"),
                distance.get("jjj1_2"),
                distance.get("jjj2_2"),
            ]
        else:
            row = [
                ts,
                proximity.get("jjj1_1"),
                proximity.get("jjj2_1"),
                proximity.get("jjj1_2"),
                proximity.get("jjj2_2"),
                self.pitch_edit.text().strip() or "0",
                self.roll_edit.text().strip() or "0",
            ]
            self.sample_count += 1
            self.sample_num.setText(str(self.sample_count))

        self.csv_writer.writerow(row)
        if self.csv_file:
            self.csv_file.flush()

    def reset_sample_count(self):
        self.sample_count = 0
        self.sample_num.setText("0")

    def on_reset_sample_clicked(self):
        if self.mode_choice.currentText() != MODE_SURFACE_TILT_COLLECTION:
            return
        pitch = self.pitch_edit.text().strip()
        roll = self.roll_edit.text().strip()
        self.sample_pairs.setText(f"({pitch}, {roll})")
        self.sampled_once = False
        self.moving_forward = True
        self.return_journey.setChecked(False)
        self.condition_dis.setChecked(False)
        self.status_label.setText("样本已复位，等待距离条件满足")
        self.zero_sensor()

    def delete_sample_pair(self):
        if self.mode_choice.currentText() != MODE_SURFACE_TILT_COLLECTION:
            return
        parsed = self._current_pair_from_text()
        if parsed is None:
            QMessageBox.warning(self, "警告", "没有可删除的有效样本对。")
            return
        self.remove_pair_from_table(*parsed)
        self.remaining_pairs.discard(parsed)
        self.sample_pairs.setText("")
        self._append_log(f"已删除样本对 {parsed}")

    def fill_angles_from_table(self, row: int, column: int):
        if self.current_highlighted_row >= 0:
            for col in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(self.current_highlighted_row, col)
                if item:
                    item.setBackground(QColor("white"))

        for col in range(self.tableWidget.columnCount()):
            item = self.tableWidget.item(row, col)
            if item:
                item.setBackground(QColor(255, 255, 150))
        self.current_highlighted_row = row

        pitch_item = self.tableWidget.item(row, 0)
        roll_item = self.tableWidget.item(row, 1)
        if pitch_item and roll_item:
            self.pitch_edit.setText(pitch_item.text())
            self.roll_edit.setText(roll_item.text())
            self.sample_pairs.setText(f"({pitch_item.text()}, {roll_item.text()})")
            self.sampled_once = False
            self.moving_forward = True

    def get_all_pitch_roll_pairs(self) -> Set[Tuple[float, float]]:
        pitch_vals = frange(
            self.collection_cfg.get("pitch_min", -3.0),
            self.collection_cfg.get("pitch_max", 3.0),
            self.collection_cfg.get("pitch_step", 1.0),
        )
        roll_vals = frange(
            self.collection_cfg.get("roll_min", -3.0),
            self.collection_cfg.get("roll_max", 3.0),
            self.collection_cfg.get("roll_step", 1.0),
        )
        return {(round(p, 1), round(r, 1)) for p in pitch_vals for r in roll_vals}

    def get_existing_pitch_roll_from_csv(self, path: str) -> Set[Tuple[float, float]]:
        existing: Set[Tuple[float, float]] = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            pitch = round(float(row[-2]), 1)
                            roll = round(float(row[-1]), 1)
                            existing.add((pitch, roll))
                        except Exception:
                            continue
        except Exception as exc:
            self._append_log(f"读取数据文件失败：{exc}")
        return existing

    def update_table_with_missing_combinations(self, missing_pairs: Set[Tuple[float, float]]):
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(len(missing_pairs))
        self.tableWidget.setHorizontalHeaderLabels(["俯仰角", "横滚角"])
        for idx, (pitch, roll) in enumerate(sorted(missing_pairs)):
            self.tableWidget.setItem(idx, 0, QTableWidgetItem(str(pitch)))
            self.tableWidget.setItem(idx, 1, QTableWidgetItem(str(roll)))

    def remove_pair_from_table(self, pitch: float, roll: float):
        for row in range(self.tableWidget.rowCount()):
            p_item = self.tableWidget.item(row, 0)
            r_item = self.tableWidget.item(row, 1)
            if not p_item or not r_item:
                continue
            if round(float(p_item.text()), 1) == pitch and round(float(r_item.text()), 1) == roll:
                self.tableWidget.removeRow(row)
                break

    def _remove_current_pair_if_valid(self):
        try:
            pitch = round(float(self.pitch_edit.text().strip()), 1)
            roll = round(float(self.roll_edit.text().strip()), 1)
        except ValueError:
            return
        self.remove_pair_from_table(pitch, roll)
        self.remaining_pairs.discard((pitch, roll))

    def _current_pair_from_text(self) -> Optional[Tuple[float, float]]:
        text = self.sample_pairs.text().strip()
        if not text:
            return None
        try:
            text = text.strip("()")
            pitch_s, roll_s = text.split(",", 1)
            return round(float(pitch_s.strip()), 1), round(float(roll_s.strip()), 1)
        except Exception:
            return None

    def closeEvent(self, event):
        self._running = False
        try:
            if self.acq_thread.is_alive():
                self.acq_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            with self._op_lock:
                op_thread = self._op_thread
            if op_thread and op_thread.is_alive():
                op_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.on_stop_collect()
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
    win = ProximitySensorWindow(sensor, sensor_cfg)
    win.show()
    app.exec_()


if __name__ == "__main__":
    main()
