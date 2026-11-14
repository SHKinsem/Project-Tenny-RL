"""NumPy forward for torque200 exports; derived from the original deployment kernel.

Uses independent 24D weights. Does not load or modify original firmware arrays.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np



ACTION_SIZE = 4
LAYER_SIZES = (24, 64, 64, 8)


def _floats(block: str) -> np.ndarray:
    cleaned = re.sub(r"//[^\n]*", " ", block)
    return np.array(
        [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?", cleaned)],
        dtype=np.float32,
    )


def load_exported(path: Path) -> dict:
    text = path.read_text()
    comment = next(
        (ln.strip() for ln in text.splitlines() if ln.startswith("// Extracted from checkpoint:")),
        "",
    )
    arrays = {}
    for name in ("state_mean", "state_std", "weights", "biases"):
        match = re.search(rf"float {name}\[\] = \{{(.*?)\}};", text, re.S)
        if not match:
            raise ValueError(f"{path}: missing {name}")
        arrays[name] = _floats(match.group(1))
    n_in, n_h1, n_h2, n_out = LAYER_SIZES
    w = arrays["weights"]
    b = arrays["biases"]
    expected_w = n_in * n_h1 + n_h1 * n_h2 + n_h2 * n_out
    expected_b = n_h1 + n_h2 + n_out
    if w.size != expected_w or b.size != expected_b:
        raise ValueError(f"{path}: weight/bias size {w.size}/{b.size}, expected {expected_w}/{expected_b}")
    if arrays["state_mean"].size != n_in or arrays["state_std"].size != n_in:
        raise ValueError(f"{path}: mean/std size {arrays['state_mean'].size}")
    i0, i1 = 0, n_h1 * n_in
    i2 = i1 + n_h2 * n_h1
    b0, b1 = 0, n_h1
    b2 = b1 + n_h2
    return {
        "comment": comment,
        "mean": arrays["state_mean"],
        "std": arrays["state_std"],
        "w1": w[i0:i1].reshape(n_h1, n_in),
        "w2": w[i1:i2].reshape(n_h2, n_h1),
        "w3": w[i2:].reshape(n_out, n_h2),
        "b1": b[b0:b1],
        "b2": b[b1:b2],
        "b3": b[b2:],
    }


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class RL_Policy:
    """对应固件 RL_Policy：history_length=1 时就是当前 24 维观测。"""

    def __init__(self, path: Path, history_len: int = 1, one_obs_len: int = 24) -> None:
        exported = load_exported(path)
        self.checkpoint_comment = exported["comment"]
        self.mean = exported["mean"]
        self.std = exported["std"]
        self.w1, self.w2, self.w3 = exported["w1"], exported["w2"], exported["w3"]
        self.b1, self.b2, self.b3 = exported["b1"], exported["b2"], exported["b3"]
        self.one_observation_length = one_obs_len
        self.history_length = history_len
        self.history_obs_length = one_obs_len * history_len
        self.input_history = np.zeros(self.history_obs_length, dtype=np.float32)
        self.actions_ = np.zeros(ACTION_SIZE, dtype=np.float32)

    def clean_history(self) -> None:
        self.input_history[:] = 0.0

    def _update_history(self, new_input: np.ndarray) -> None:
        n = self.one_observation_length
        if self.history_length > 1:
            self.input_history[:-n] = self.input_history[n:]
        self.input_history[-n:] = new_input

    def _forward(self, x: np.ndarray) -> np.ndarray:
        h1 = _silu(self.w1 @ x + self.b1)
        h2 = _silu(self.w2 @ h1 + self.b2)
        return np.tanh(self.w3 @ h2 + self.b3)

    def run(self, new_input, action_dim: int = ACTION_SIZE) -> np.ndarray:
        x = np.asarray(new_input, dtype=np.float32).reshape(-1)
        if x.size != self.one_observation_length:
            raise ValueError(f"obs dim {x.size} != {self.one_observation_length}")
        self._update_history(x)
        norm = (self.input_history - self.mean) / self.std
        out = self._forward(norm)
        self.actions_ = np.array(out[:action_dim], dtype=np.float32)
        return self.actions_.copy()
