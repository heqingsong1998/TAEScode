# TAES 实验代码（接近觉传感器标定 / 数据采集）

该项目为 **TAES 论文所需实验代码**（仓库描述：*“包括接近觉传感器标定，以及数据采集”*），主要包含：

- **接近觉（proximity）传感器**：串口数据帧解析、标定参数配置、采集扫描参数（pitch/roll/距离）
- **运动控制**：雷赛 **LTSMC** 运动控制卡（`LTSMC.dll`）驱动与调试
- **力/触觉采集**：M8128B1 六轴力传感器、8×8 阵列触觉传感器（串口帧协议 + 数据处理/补偿）
- **工作流**：采集 UI → 数据落盘 → 训练（MLP）→ 推理/验证

> 说明：本仓库强依赖 **Windows + 串口设备**（尤其 LTSMC DLL）。

---

## 目录结构

```text
.
├── apps/                          # 调试与可视化入口脚本
│   ├── debug_motion.py            # LTSMC 运动控制调试
│   ├── debug_sensor.py            # M8128B1 六轴传感器调试
│   ├── debug_array_sensor_3d.py   # 阵列触觉 3D 显示
│   └── debug_torque_motor.py      # 力矩电机调试 UI
├── config/
│   └── default.yaml               # 运动卡/多轴/各类传感器/电机配置与标定参数
├── drivers/
│   ├── motioncard/                # LTSMC 控制卡（DLL）
│   ├── sensors/                   # 六轴力传感器（M8128B1）
│   ├── array_sensor/              # 8×8 阵列触觉（协议/处理/补偿/Volterra）
│   └── torque_motor/              # 力矩电机（含 SDK）
├── workflows/
│   ├── acquisition/               # 数据采集（UI + 写盘）
│   ├── training/                  # 训练
│   └── validation/                # 推理/验证
├── datasets/                      # 采集输出（按 run_id 组织）
├── logs/
├── LTSMC.dll
├── LTSMC.lib
└── README.md
```

---

## 环境与依赖

### Python
- Python **3.10+**

建议虚拟环境：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### 常用依赖（按当前代码使用情况）

- `pyserial`：串口通信（六轴/阵列/接近觉等）
- `pyyaml`：读取 YAML 配置
- `PyQt5`：UI 脚本（如 `debug_torque_motor.py`、采集/验证 UI）
- `pyqtgraph`��阵列触觉 3D 可视化

> 仓库未提供统一 requirements 文件；如果你希望我补一个 `requirements.txt` / `environment.yml`，告诉我你常用的运行入口（只跑采集？还是训练/验证也跑）。

### 系统/硬件

- **Windows**：使用 `LTSMC.dll` 时必须
- 串口设备（默认示例见 `config/default.yaml`）：
  - M8128B1：如 `COM5`
  - 阵列触觉：如 `COM3`（波特率 2,000,000）
  - 接近觉：如 `COM1`
  - 力矩电机：如 `\\.\COM17`

---

## 配置说明（config/default.yaml）

核心配置文件：`config/default.yaml`

### 1) motioncard（运动卡）
- `driver: dll`
- `dll_path: ./LTSMC.dll`
- 也预留了 `tcp` 与 `serial` 配置字段

### 2) axes（多轴参数）
- `pulse_mode`、`equiv`（等效脉冲）
- 速度规划：`profile.vmin/vmax/acc/dec/s_time`
- 回零：`homing` 与 `home_profile`（含 `home_dir/mode/source` 等）

### 3) sensor（传感器）
- `m8128b1`：串口、通道数、采样率等
- `array_sensor`：
  - 串口参数
  - 帧结构（head/size/data_size/校验等）
  - 映射（8×8 映射矩阵）
  - 处理开关：温漂/动态/滞回/标定/Volterra 等
- `proximity_sensor`：
  - 串口帧结构（head/tail/fields）
  - `processing.calibration`：接近觉标定参数（a/b/c）
  - `collection`：采集扫描参数（pitch/roll 范围与步长、距离起止等）

### 4) torque_motor（力矩电机）
- 串口端口、波特率、slave 等

---

## 快速运行（调试入口）

### 1) 运动控制卡调试（LTSMC）

```bash
python apps/debug_motion.py
```

### 2) 力矩电机调试 UI

```bash
python apps/debug_torque_motor.py
```

### 3) M8128B1 六轴传感器调试

```bash
python apps/debug_sensor.py
```

也可以直接通过工具函数创建/初始化：

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

### 4) 阵列触觉 3D 可视化

```bash
python apps/debug_array_sensor_3d.py
```

---

## 数据采集（workflows/acquisition）

### 采集 UI

```bash
python workflows/acquisition/collect_dataset_ui.py
```

输出目录：`datasets/<run_id>/`

- `samples/sample_*.npz`：单样本文件（便于断点续采/训练读取）
- `manifest.csv` / `manifest.jsonl`：样本索引与标签
- `run_meta.json`：采集参数快照

---

## 训练与验证

### 训练（单帧展开的残差 MLP，回归 theta0/theta1）

```bash
python -m workflows.training.train_single_frame_mlp \
  --dataset-root datasets \
  --output-dir workflows/training/artifacts
```

训练脚本要点：
- 会把每个 `sample_xxxxxx.npz` 的多帧数据展开为多条单帧样本（共享同一角度标签）
- 默认按 `sample_id` 划分训练/验证集，避免同一采样点的帧泄漏

产物（默认）：
- `workflows/training/artifacts/single_frame_mlp_model.npz`
- `workflows/training/artifacts/single_frame_mlp_meta.json`

### 推理/快速验证

```bash
python -m workflows.validation.infer_single_frame_mlp \
  --model workflows/training/artifacts/single_frame_mlp_model.npz \
  --sample-npz datasets/<run_id>/samples/sample_000001.npz
```

---

## FAQ

- **LTSMC 相关脚本无法运行**：确认 Windows 环境、`LTSMC.dll` 在仓库根目录且可被加载。
- **串口打不开/读数异常**：检查 `config/default.yaml` 中端口号/波特率/timeout；阵列触觉默认波特率很高（2,000,000）。
- **UI 启动失败**：安装 `PyQt5`、`pyqtgraph`。

---

## License

仓库当前未提供 LICENSE 文件。如需开源发布，建议补充 LICENSE（MIT/Apache-2.0/GPL 等）。
