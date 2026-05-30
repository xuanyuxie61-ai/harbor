
import numpy as np
import math
import re


class NewtonInterpolator1D:

    def __init__(self, xd: np.ndarray, yd: np.ndarray):
        self.xd = np.asarray(xd, dtype=float).flatten()
        self.yd = np.asarray(yd, dtype=float).flatten()
        if self.xd.size != self.yd.size:
            raise ValueError("xd 与 yd 长度必须一致")
        if self.xd.size < 1:
            raise ValueError("至少需要一个插值节点")
        self.nd = self.xd.size
        self.cd = self._compute_divided_differences()

    def _compute_divided_differences(self) -> np.ndarray:
        cd = self.yd.copy()
        for i in range(1, self.nd):
            for j in range(self.nd - 1, i - 1, -1):
                denom = self.xd[j] - self.xd[j - i]
                if abs(denom) < 1e-14:
                    denom = np.sign(denom) * 1e-14 if denom != 0 else 1e-14
                cd[j] = (cd[j] - cd[j - 1]) / denom
        return cd

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).flatten()
        ni = xi.size
        yi = np.full(ni, self.cd[self.nd - 1], dtype=float)
        for i in range(self.nd - 2, -1, -1):
            yi = self.cd[i] + (xi - self.xd[i]) * yi
        return yi

    def error_bound(self, xi: np.ndarray, max_derivative: float) -> np.ndarray:
        xi = np.asarray(xi, dtype=float).flatten()
        prod = np.ones_like(xi)
        for j in range(self.nd):
            prod *= np.abs(xi - self.xd[j])
        factorial = float(math.factorial(self.nd))
        return max_derivative * prod / factorial


def filename_increment(filename: str) -> str:
    if not filename:
        raise ValueError("filename_increment: 输入文件名为空")
    chars = list(filename)
    changed = 0
    for idx in range(len(chars) - 1, -1, -1):
        c = chars[idx]
        if '0' <= c <= '8':
            chars[idx] = chr(ord(c) + 1)
            return ''.join(chars)
        elif c == '9':
            chars[idx] = '0'
            changed += 1
    if changed == 0:
        return ' '
    return ''.join(chars)


def safe_inverse_sqrt(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_safe = np.where(x > eps, x, eps)
    return 1.0 / np.sqrt(x_safe)


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])
