# -*- coding: utf-8 -*-
"""
utils.py
通用工具模块

融合来源:
- 513_hello: 程序启动与日志输出
- 1261_timestamp: 时间戳功能
- 587_imshow_numeric: 数值矩阵格式化输出
- 1006_random_data: 球面均匀采样、布朗运动

功能:
- 日志记录与时间戳
- 数值矩阵格式化打印
- 球面上均匀随机采样（用于初始化拉格朗日粒子）
- 布朗运动随机位移生成
"""

import time
import numpy as np


def log_message(message: str):
    """
    打印带时间戳的日志信息。
    融合自 513_hello 与 1261_timestamp。
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {message}")


def print_numeric_matrix(mat, title="", fmt="{:12.6f}"):
    """
    格式化打印数值矩阵。
    融合自 587_imshow_numeric，去除可视化，保留数值输出。
    """
    if title:
        print(f"\n{title}")
    mat = np.atleast_2d(np.asarray(mat, dtype=float))
    for row in mat:
        line = " ".join(fmt.format(v) for v in row)
        print(line)


def uniform_on_sphere_phong(n, rng=None):
    """
    在三维单位球面上生成 n 个均匀分布的随机点（Phong 修正法）。
    融合自 1006_random_data 的 uniform_on_hemisphere_phong 与
    uniform_on_hypersphere。

    算法:
      1. 生成三维独立标准正态随机变量 X ~ N(0, I_3)
      2. 单位化  U = X / ||X||_2
      则 U 在 S^2 上服从均匀分布。

    数学基础:
      若 X 的联合概率密度为
        p(x) = (2*pi)^(-3/2) * exp(-||x||^2 / 2)
      则球坐标下径向与角度独立，角度部分在球面上均匀。
    """
    if rng is None:
        rng = np.random.default_rng()
    xyz = rng.standard_normal(size=(n, 3))
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    # 边界处理：避免除零
    norms = np.where(norms < 1e-15, 1.0, norms)
    return xyz / norms


def brownian_displacement(n_steps, dim=3, dt=1.0, D=0.5, rng=None):
    """
    生成布朗运动的位移序列。
    融合自 1006_random_data 的 brownian。

    物理模型:
      爱因斯坦关系: <dx^2> = 2 * D * dt
      其中 D 为扩散系数，dt 为时间步长。

    离散形式:
      dW ~ N(0, sqrt(2*D*dt))
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(2.0 * D * dt)
    return rng.normal(loc=0.0, scale=sigma, size=(n_steps, dim))


def direction_uniform_nd(dim, rng=None):
    """
    在 N 维空间中生成一个均匀随机方向向量。
    融合自 1006_random_data 的 direction_uniform_nd。
    """
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(size=dim)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        v[0] = 1.0
        norm = 1.0
    return v / norm


def safe_divide(a, b, default=0.0):
    """
    安全除法，处理接近零的分母。
    """
    b = np.asarray(b, dtype=float)
    result = np.empty_like(b, dtype=float)
    mask = np.abs(b) > 1e-15
    result[mask] = a / b[mask]
    result[~mask] = default
    return result


def clip_gradient(grad, max_norm=1e3):
    """
    梯度裁剪，防止数值爆炸。
    """
    gnorm = np.linalg.norm(grad)
    if gnorm > max_norm:
        return grad * (max_norm / gnorm)
    return grad
