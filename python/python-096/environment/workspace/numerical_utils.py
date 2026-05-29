"""
numerical_utils.py
==================
博士级数值工具模块

核心算法来源：
  - 800_newton_interp_1d：牛顿一维差值插值
  - 429_file_name_sequence：文件名递增序列

在电磁学波束赋形中的角色：
  1. Newton 插值用于天线口径面上相位分布的快速逼近
  2. 文件序列工具用于批量仿真输出的命名管理
"""

import numpy as np
import math
import re


class NewtonInterpolator1D:
    r"""
    一维牛顿差值插值器

    数学模型：
      给定节点 x_0, x_1, ..., x_{n-1} 及对应函数值 f_i = f(x_i)，
      定义 k 阶差商递归地：

        f[x_i] = f_i
        f[x_i, ..., x_{i+k}] = (f[x_{i+1},...,x_{i+k}] - f[x_i,...,x_{i+k-1}]) / (x_{i+k} - x_i)

      Newton 插值多项式：
        P_n(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ... + c_{n-1} \prod_{j=0}^{n-2}(x-x_j)

      其中 c_k = f[x_0, ..., x_k] 为差商系数。

    误差估计（博士级）：
      若 f \in C^n[a,b]，则存在 \xi(x) \in [a,b] 使得
        f(x) - P_n(x) = f^{(n)}(\xi(x)) / n! * \prod_{j=0}^{n-1}(x-x_j)
    """

    def __init__(self, xd: np.ndarray, yd: np.ndarray):
        """
        参数：
            xd: 插值节点，形状 (nd,)
            yd: 节点函数值，形状 (nd,)
        """
        self.xd = np.asarray(xd, dtype=float).flatten()
        self.yd = np.asarray(yd, dtype=float).flatten()
        if self.xd.size != self.yd.size:
            raise ValueError("xd 与 yd 长度必须一致")
        if self.xd.size < 1:
            raise ValueError("至少需要一个插值节点")
        self.nd = self.xd.size
        self.cd = self._compute_divided_differences()

    def _compute_divided_differences(self) -> np.ndarray:
        """计算差商系数（除差表的第一行）。"""
        cd = self.yd.copy()
        for i in range(1, self.nd):
            for j in range(self.nd - 1, i - 1, -1):
                denom = self.xd[j] - self.xd[j - i]
                if abs(denom) < 1e-14:
                    denom = np.sign(denom) * 1e-14 if denom != 0 else 1e-14
                cd[j] = (cd[j] - cd[j - 1]) / denom
        return cd

    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """
        在 xi 处求插值多项式的值。

        参数：
            xi: 目标点，形状任意
        返回：
            yi: 插值结果，展平为 (ni,)
        """
        xi = np.asarray(xi, dtype=float).flatten()
        ni = xi.size
        yi = np.full(ni, self.cd[self.nd - 1], dtype=float)
        for i in range(self.nd - 2, -1, -1):
            yi = self.cd[i] + (xi - self.xd[i]) * yi
        return yi

    def error_bound(self, xi: np.ndarray, max_derivative: float) -> np.ndarray:
        r"""
        返回误差上界估计 |f^{(n)}(\xi) / n! * \prod (x - x_j)|。

        参数：
            xi: 评估点
            max_derivative: |f^{(n)}| 在区间上的上界估计
        """
        xi = np.asarray(xi, dtype=float).flatten()
        prod = np.ones_like(xi)
        for j in range(self.nd):
            prod *= np.abs(xi - self.xd[j])
        factorial = float(math.factorial(self.nd))
        return max_derivative * prod / factorial


def filename_increment(filename: str) -> str:
    """
    将文件名中的末尾数字加 1（含进位与循环回绕）。

    来源：429_file_name_sequence
    在项目中用于批量输出波束方向图数据文件的自动命名。
    """
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
    """数值鲁棒的 1/sqrt(x) 计算，处理非正输入。"""
    x = np.asarray(x, dtype=float)
    x_safe = np.where(x > eps, x, eps)
    return 1.0 / np.sqrt(x_safe)


def rotation_matrix_z(theta: float) -> np.ndarray:
    """绕 Z 轴的旋转矩阵（用于阵列方向旋转）。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def rotation_matrix_y(theta: float) -> np.ndarray:
    """绕 Y 轴的旋转矩阵。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])
