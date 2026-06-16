"""
mesh_generator.py — 结构化正交网格生成器
=========================================

融合自:
  - 748_medit_to_fem: MEDIT 网格解析与拓扑转换
  - 204_compass_search: 坐标拉伸参数的无导数优化

物理背景:
  磁重联计算域 Omega = [-Lx, Lx] x [-Ly, Ly]，采用双曲正切拉伸
  使电流片附近 (y = 0) 网格最密，捕获扩散区尺度 delta ~ (eta*L/v_A)^{1/2}。

坐标拉伸函数:
  y_j = Ly * tanh(sigma * (2j/Ny - 1)) / tanh(sigma)

  其中 sigma 为拉伸因子，通过 compass search 优化使电流片内
  最小网格尺度满足 delta_min < d_i (离子惯性长度)。
"""

import numpy as np


class MeshGenerator:
    """二维结构化正交网格，带双曲正切拉伸。"""

    def __init__(self, nx=128, ny=64, x_range=(-6.4, 6.4),
                 y_range=(-3.2, 3.2), stretching_factor=1.2):
        self.nx = nx
        self.ny = ny
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.sigma = stretching_factor
        self.dx_uniform = (self.x_max - self.x_min) / self.nx
        self.dy_uniform = (self.y_max - self.y_min) / self.ny
        self.x_nodes = None
        self.y_nodes = None
        self.x_centers = None
        self.y_centers = None
        self.dx_min = None
        self.dy_min = None
        self.max_aspect_ratio = None

    def build_structured_mesh(self):
        """生成带拉伸的节点和中心坐标。"""
        self.x_nodes = np.linspace(self.x_min, self.x_max, self.nx + 1)
        self.x_centers = 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])
        self._apply_tanh_stretching()
        self._optimize_stretching()

    def _apply_tanh_stretching(self):
        """应用双曲正切拉伸到 y 方向。"""
        j_idx = np.arange(self.ny + 1)
        eta = 2.0 * j_idx / self.ny - 1.0
        self.y_nodes = self.y_max * np.tanh(self.sigma * eta) / np.tanh(self.sigma)
        self.y_centers = 0.5 * (self.y_nodes[:-1] + self.y_nodes[1:])

    def _optimize_stretching(self):
        """
        Compass search 无导数优化拉伸因子 sigma。
        目标函数: J(sigma) = (delta_min(sigma)/d_i - 0.5)^2 + 0.01*(sigma - 1)^2
        """
        d_i = 0.05
        target_ratio = 0.5
        sigma = self.sigma
        delta = 0.5
        best_cost = self._stretch_cost(sigma, d_i, target_ratio)
        for _ in range(50):
            improved = False
            for direction in [1.0, -1.0]:
                sigma_trial = sigma + direction * delta
                if sigma_trial < 0.1 or sigma_trial > 10.0:
                    continue
                cost = self._stretch_cost(sigma_trial, d_i, target_ratio)
                if cost < best_cost:
                    best_cost = cost
                    sigma = sigma_trial
                    improved = True
                    break
            if not improved:
                delta *= 0.5
                if delta < 1e-6:
                    break
        self.sigma = sigma
        self._apply_tanh_stretching()

    def _stretch_cost(self, sigma, d_i, target_ratio):
        """计算拉伸代价函数。"""
        dy_min = self.y_max * (1.0 - np.tanh(sigma * (1.0 - 2.0 / self.ny))) / np.tanh(sigma)
        ratio = abs(dy_min) / d_i
        return (ratio - target_ratio) ** 2 + 0.01 * (sigma - 1.0) ** 2

    def compute_metrics(self):
        """计算网格质量度量。"""
        dx_arr = np.diff(self.x_nodes)
        dy_arr = np.diff(self.y_nodes)
        self.dx_min = float(np.min(np.abs(dx_arr)))
        self.dy_min = float(np.min(np.abs(dy_arr)))
        dx_2d = np.abs(dx_arr)[:, None] * np.ones((1, self.ny))
        dy_2d = np.ones((self.nx, 1)) * np.abs(dy_arr)[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            aspect = np.where(dy_2d > 1e-30,
                              np.maximum(dx_2d / dy_2d, dy_2d / (dx_2d + 1e-30)),
                              1.0)
        self.max_aspect_ratio = float(np.max(aspect))

    def get_cell_areas(self):
        """返回二维单元面积数组 (nx, ny)。"""
        dx_arr = np.abs(np.diff(self.x_nodes))
        dy_arr = np.abs(np.diff(self.y_nodes))
        return np.outer(dx_arr, dy_arr)

    def get_inverse_metrics(self):
        """返回网格逆度量 xi_x, eta_y。"""
        dx_arr = np.abs(np.diff(self.x_nodes))
        dy_arr = np.abs(np.diff(self.y_nodes))
        return 1.0 / (dx_arr + 1e-30), 1.0 / (dy_arr + 1e-30)
