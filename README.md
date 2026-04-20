# 运动采集项目（精简版）

本仓库当前包含以下设备相关代码：

1. 运动控制卡驱动（雷赛 LTSMC）
2. 六轴力传感器驱动（M8128B1）
3. 8x8 阵列触觉传感器驱动（串口帧协议）
4. 力矩电机驱动（motormaster）

## 当前目录结构

```text
.
├── apps/
│   ├── __init__.py
│   ├── debug_array_sensor_3d.py # 阵列触觉传感器3D显示
│   ├── debug_sensor.py          # 六轴传感器调试
│   ├── debug_motion.py         # LTSMC 运动控制调试
│   └── debug_torque_motor.py   # 力矩电机调试 UI
├── config/
│   └── default.yaml            # 运动卡 / M8128B1 / 力矩电机配置
├── drivers/
│   ├── __init__.py
│   ├── array_sensor/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── compensation.py
│   │   ├── processor.py
│   │   ├── protocol.py
│   │   ├── serial_sensor.py
│   │   ├── volterra_plus.py
│   │   └── utils.py
│   ├── motioncard/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ltsmc_dll.py
│   │   └── utils.py
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── m8128b1.py
│   │   └── utils.py
│   └── torque_motor/
│       ├── __init__.py
│       ├── base.py
│       ├── torque_card.py
│       └── sdk_standalone-main-V4/
├── workflows/
│   ├── acquisition/
│   │   ├── collect_dataset_ui.py
│   │   └── dataset_writer.py
│   ├── training/
│   │   ├── single_frame_dataset.py
│   │   └── train_single_frame_mlp.py
│   └── validation/
│       ├── infer_single_frame_mlp.py
│       └── validate_predict_ui.py
├── datasets/
│   └── <run_id>/...
├── LTSMC.dll
├── LTSMC.lib
└── README.md
```

## 依赖环境

- Python 3.10+
- Windows（使用 LTSMC.dll 时）
- pyserial
- pyyaml
- PyQt5（`apps/debug_torque_motor.py`）
- pyqtgraph（`apps/debug_array_sensor_3d.py`）
- motormaster（力矩电机 SDK）

## 快速运行

### 1) 运动控制卡调试

```bash
python apps/debug_motion.py
```

### 2) 力矩电机调试 UI

```bash
python apps/debug_torque_motor.py
```

### 3) M8128B1 六轴传感器

可通过 `drivers/sensors/utils.py` 创建并初始化：

```python
from drivers.sensors.utils import create_sensor, initialize_sensor

cfg = {
    "port": "COM5",
    "baudrate": 115200,
    "channels": 6,
    "rate_hz": 200,
}

sensor = create_sensor("m8128b1", cfg)
ok = initialize_sensor(sensor)
```

### 4) 阵列触觉传感器 3D 可视化

```bash
python apps/debug_array_sensor_3d.py
```

### 5) 轴0/轴1 + 力矩电机 + 阵列传感器 数据集采集 UI

```bash
python workflows/acquisition/collect_dataset_ui.py
```

输出将写入 `datasets/<run_id>/`，包含：
- `samples/sample_*.npz`（单点单文件，便于断点续采与训练读取）
- `manifest.csv` / `manifest.jsonl`（索引与标签）
- `run_meta.json`（参数快照）

### 6) 训练（20帧展开为20条单帧样本）

采集后可直接运行训练脚本（2层 MLP 回归 `theta0/theta1`）：

```bash
python -m workflows.training.train_single_frame_mlp \
  --dataset-root datasets \
  --output-dir workflows/training/artifacts
```

说明：
- 脚本会把每个 `sample_xxxxxx.npz` 的多帧数据展开为多条单帧训练样本（同一角度标签）。
- 默认按 `sample_id` 划分训练/验证集（避免同一采样点的帧泄漏到验证集）。
- 训练产物：
  - `workflows/training/artifacts/single_frame_mlp_model.npz`
  - `workflows/training/artifacts/single_frame_mlp_meta.json`

快速验证推理：

```bash
python -m workflows.validation.infer_single_frame_mlp \
  --model workflows/training/artifacts/single_frame_mlp_model.npz \
  --sample-npz datasets/<run_id>/samples/sample_000001.npz
```

## 当前 workflows 目录（采集 / 训练 / 验证）

```text
workflows/
├── acquisition/
│   ├── collect_dataset_ui.py  # 采集UI（当前可用）
│   └── dataset_writer.py      # 训练友好的样本落盘
├── training/
│   ├── single_frame_dataset.py      # 读取样本并按帧展开
│   └── train_single_frame_mlp.py    # 2层MLP训练脚本
└── validation/
    ├── infer_single_frame_mlp.py    # 推理与快速验证
    └── validate_predict_ui.py       # 验证UI
```

## 配置说明

- `config/default.yaml` 中包含：
  - `motioncard`
  - `axes`
  - `sensor.m8128b1`
  - `sensor.array_sensor`
  - `torque_motor`
