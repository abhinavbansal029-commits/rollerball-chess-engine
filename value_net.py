"""A small NumPy value network used as a learned evaluation term.

Input: one-hot planes (piece type x colour x square) + side to move.
Output: tanh value in [-1, 1] from White's perspective.
"""
import numpy as np

from board import N, SQUARES, Board
from engine import material_eval

PLANES = 8  # 4 piece types x 2 colours
FEATURES = PLANES * N * N + 1


def features(b: Board) -> np.ndarray:
    x = np.zeros(FEATURES, dtype=np.float32)
    for i in SQUARES:
        p = b.sq[i]
        if p:
            plane = (abs(p) - 1) + (0 if p > 0 else 4)
            x[plane * N * N + i] = 1.0
    x[-1] = b.side
    return x


class ValueNet:
    def __init__(self, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.05, (FEATURES, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, 0.05, hidden).astype(np.float32)
        self.b2 = np.float32(0.0)

    def forward(self, x):
        h = np.tanh(x @ self.W1 + self.b1)
        return float(np.tanh(h @ self.W2 + self.b2)), h

    def value(self, b: Board) -> float:
        return self.forward(features(b))[0]

    def update(self, x, target, lr=0.01):
        """One SGD step on 0.5 * (v - target)^2."""
        v, h = self.forward(x)
        dv = (v - target) * (1 - v * v)
        dh = dv * self.W2 * (1 - h * h)
        self.W2 -= lr * dv * h
        self.b2 -= lr * dv
        self.W1 -= lr * np.outer(x, dh)
        self.b1 -= lr * dh
        return (v - target) ** 2

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(hidden=d["b1"].shape[0])
        net.W1, net.b1, net.W2, net.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        return net


def hybrid_eval(net: ValueNet, weight=200):
    """Handcrafted material score plus a learned positional term."""
    def evaluate(b: Board) -> int:
        return material_eval(b) + int(weight * net.value(b) * b.side)
    return evaluate
