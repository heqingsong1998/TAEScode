from __future__ import annotations

import argparse
from typing import Dict

import numpy as np


class SingleFrameMLP:
    def __init__(self, model_path: str):
        m = np.load(model_path, allow_pickle=False)
        self.is_two_stage = all(
            k in m for k in ("coarse_w1", "coarse_b1", "coarse_w2", "coarse_b2", "fine_w1", "fine_b1", "fine_w2", "fine_b2")
        )
        self.is_residual = all(k in m for k in ("w1", "b1", "w2", "b2", "w_skip", "b_skip", "w3", "b3"))
        if self.is_two_stage:
            self.coarse_w1 = np.asarray(m["coarse_w1"], dtype=np.float32)
            self.coarse_b1 = np.asarray(m["coarse_b1"], dtype=np.float32)
            self.coarse_w2 = np.asarray(m["coarse_w2"], dtype=np.float32)
            self.coarse_b2 = np.asarray(m["coarse_b2"], dtype=np.float32)
            self.fine_w1 = np.asarray(m["fine_w1"], dtype=np.float32)
            self.fine_b1 = np.asarray(m["fine_b1"], dtype=np.float32)
            self.fine_w2 = np.asarray(m["fine_w2"], dtype=np.float32)
            self.fine_b2 = np.asarray(m["fine_b2"], dtype=np.float32)
        elif self.is_residual:
            self.w1 = np.asarray(m["w1"], dtype=np.float32)
            self.b1 = np.asarray(m["b1"], dtype=np.float32)
            self.w2 = np.asarray(m["w2"], dtype=np.float32)
            self.b2 = np.asarray(m["b2"], dtype=np.float32)
            self.w_skip = np.asarray(m["w_skip"], dtype=np.float32)
            self.b_skip = np.asarray(m["b_skip"], dtype=np.float32)
            self.w3 = np.asarray(m["w3"], dtype=np.float32)
            self.b3 = np.asarray(m["b3"], dtype=np.float32)
        else:
            self.w1 = np.asarray(m["w1"], dtype=np.float32)
            self.b1 = np.asarray(m["b1"], dtype=np.float32)
            self.w2 = np.asarray(m["w2"], dtype=np.float32)
            self.b2 = np.asarray(m["b2"], dtype=np.float32)
        self.x_mean = np.asarray(m["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(m["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(m["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(m["y_std"], dtype=np.float32)

    @staticmethod
    def _forward(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        z1 = x @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        y = a1 @ w2 + b2
        return y

    @staticmethod
    def _forward_residual(
        x: np.ndarray,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
        w_skip: np.ndarray,
        b_skip: np.ndarray,
        w3: np.ndarray,
        b3: np.ndarray,
    ) -> np.ndarray:
        z1 = x @ w1 + b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ w2 + b2 + x @ w_skip + b_skip
        a2 = np.maximum(z2, 0.0)
        y = a2 @ w3 + b3
        return y

    def predict(self, frame_feature: np.ndarray) -> np.ndarray:
        x = np.asarray(frame_feature, dtype=np.float32).reshape(1, -1)
        x_n = (x - self.x_mean) / self.x_std
        if self.is_two_stage:
            coarse_y_n = self._forward(
                x_n, self.coarse_w1, self.coarse_b1, self.coarse_w2, self.coarse_b2
            )
            fine_x_n = np.concatenate([x_n, coarse_y_n], axis=1)
            y_n = self._forward(
                fine_x_n, self.fine_w1, self.fine_b1, self.fine_w2, self.fine_b2
            )
        elif self.is_residual:
            y_n = self._forward_residual(
                x_n,
                self.w1,
                self.b1,
                self.w2,
                self.b2,
                self.w_skip,
                self.b_skip,
                self.w3,
                self.b3,
            )
        else:
            y_n = self._forward(x_n, self.w1, self.b1, self.w2, self.b2)
        y = y_n * self.y_std + self.y_mean
        return y[0]


def build_feature_from_frame(
    frame: Dict,
    use_raw: bool = True,
    use_relative: bool = False,
    use_force: bool = False,
    use_pressure: bool = False,
    use_temp: bool = False,
) -> np.ndarray:
    blocks = []
    if use_raw:
        blocks.append(np.asarray(frame.get("raw", [0.0] * 64), dtype=np.float32).reshape(-1))
    if use_relative:
        blocks.append(np.asarray(frame.get("relative", [0.0] * 64), dtype=np.float32).reshape(-1))
    if use_force:
        f = frame.get("force", {}) or {}
        blocks.append(np.asarray([f.get("fx", 0.0), f.get("fy", 0.0), f.get("fz", 0.0)], dtype=np.float32))
    if use_pressure:
        p = frame.get("pressure", {}) or {}
        blocks.append(np.asarray([p.get("fx", 0.0), p.get("fy", 0.0), p.get("fz", 0.0)], dtype=np.float32))
    if use_temp:
        blocks.append(np.asarray([frame.get("temp1", 0.0), frame.get("temp2", 0.0)], dtype=np.float32))

    if not blocks:
        raise ValueError("至少开启一个特征源")
    return np.concatenate(blocks, axis=0)


def demo_predict_on_npz(model_path: str, sample_npz: str, frame_index: int = 0) -> None:
    model = SingleFrameMLP(model_path)
    with np.load(sample_npz, allow_pickle=False) as d:
        frame = {
            "raw": d["raw"][frame_index],
            "relative": d["relative"][frame_index],
            "force": {"fx": d["force"][frame_index][0], "fy": d["force"][frame_index][1], "fz": d["force"][frame_index][2]},
            "pressure": {
                "fx": d["pressure"][frame_index][0],
                "fy": d["pressure"][frame_index][1],
                "fz": d["pressure"][frame_index][2],
            },
            "temp1": d["temp"][frame_index][0],
            "temp2": d["temp"][frame_index][1],
        }
        feat = build_feature_from_frame(frame)
        pred = model.predict(feat)
        label = np.asarray(d["theta_cmd"], dtype=np.float32)

    print(f"预测角度: theta0={pred[0]:.4f}°, theta1={pred[1]:.4f}°")
    print(f"样本标签: theta0={label[0]:.4f}°, theta1={label[1]:.4f}°")


def main() -> None:
    parser = argparse.ArgumentParser(description="单帧残差MLP推理/快速验证（兼容历史MLP格式）")
    parser.add_argument("--model", required=True, help="single_frame_mlp_model.npz 路径")
    parser.add_argument("--sample-npz", help="可选：用于快速验证的 sample_xxxxxx.npz")
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()

    if args.sample_npz:
        demo_predict_on_npz(args.model, args.sample_npz, args.frame_index)
    else:
        print("已加载模型，可在代码中使用 SingleFrameMLP.predict(...) 做在线推理（支持残差MLP与历史MLP）。")


if __name__ == "__main__":
    main()
