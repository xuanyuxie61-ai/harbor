# -*- coding: utf-8 -*-
"""
sinc_interpolator.py
Sinc 谱插值模块

基于种子项目 1082_sinc 的归一化 sinc 函数及其导数计算，
用于裂隙介质中浓度场的高精度谱插值。

Sinc 函数在信号处理和数值分析中具有重要地位：
    - 它是理想的低通滤波器
    - 满足 Shannon 采样定理的精确重构核
    - 在均匀网格上具有指数收敛率

核心公式：
    归一化 Sinc 函数：
        sinc(x) = sin(πx) / (πx),  x ≠ 0
        sinc(0) = 1
    
    Sinc 导数：
        sinc'(x) = [πx cos(πx) - sin(πx)] / (πx²),  x ≠ 0
        sinc'(0) = 0
    
    Sinc 插值：
        f(x) ≈ Σ f(x_k) sinc((x - x_k)/h)
    
    Whittaker-Shannon 重构：
        f(x) = Σ_{n=-∞}^{∞} f(nh) sinc((x - nh)/h)

在水文地质中的应用：
    - 浓度场的亚网格分辨率插值
    - 粒子追踪中的速度场插值
    - 突破曲线的高精度重构
"""

import numpy as np
from typing import Union


class SincInterpolator:
    """
    Sinc 谱插值器

    使用归一化 sinc 函数进行一维和二维场的高精度插值。
    适用于裂隙介质示踪剂浓度场的后处理与分析。
    """

    @staticmethod
    def sincn(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        归一化 sinc 函数

        Parameters
        ----------
        x : float or np.ndarray
            自变量

        Returns
        -------
        float or np.ndarray
            sinc(x) 的值
        """
        x = np.asarray(x, dtype=float)
        result = np.ones_like(x)
        mask = np.abs(x) > 1e-15
        if np.any(mask):
            x_mask = x[mask]
            result[mask] = np.sin(np.pi * x_mask) / (np.pi * x_mask)
        return result if result.shape != () else float(result)

    @staticmethod
    def sincn_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        sinc 函数的一阶导数

        公式：
            d/dx sinc(x) = [πx cos(πx) - sin(πx)] / (πx²)

        Parameters
        ----------
        x : float or np.ndarray
            自变量

        Returns
        -------
        float or np.ndarray
            sinc'(x) 的值
        """
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        mask = np.abs(x) > 1e-15
        if np.any(mask):
            x_m = x[mask]
            pi_x = np.pi * x_m
            result[mask] = (pi_x * np.cos(pi_x) - np.sin(pi_x)) / (pi_x * x_m)
        return result if result.shape != () else float(result)

    @staticmethod
    def sincn_second_derivative(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        sinc 函数的二阶导数

        公式：
            d²/dx² sinc(x) = [(2 - π²x²)sin(πx) - 2πx cos(πx)] / (πx³)

        Parameters
        ----------
        x : float or np.ndarray
            自变量

        Returns
        -------
        float or np.ndarray
            sinc''(x) 的值
        """
        x = np.asarray(x, dtype=float)
        result = np.full_like(x, -np.pi ** 2 / 3.0)
        mask = np.abs(x) > 1e-15
        if np.any(mask):
            x_m = x[mask]
            pi_x = np.pi * x_m
            result[mask] = ((2.0 - pi_x ** 2) * np.sin(pi_x) - 2.0 * pi_x * np.cos(pi_x)) / (pi_x * x_m ** 2)
        return result if result.shape != () else float(result)

    @staticmethod
    def interpolate_1d(x_grid: np.ndarray, f_values: np.ndarray,
                       x_query: np.ndarray, truncation: int = None) -> np.ndarray:
        """
        一维 Sinc 插值

        基于 Whittaker-Shannon 采样定理：
            f(x) ≈ Σ_{k} f(x_k) sinc((x - x_k)/h)

        Parameters
        ----------
        x_grid : np.ndarray
            均匀采样网格点 (N,)
        f_values : np.ndarray
            网格点上的函数值 (N,)
        x_query : np.ndarray
            查询点 (M,)
        truncation : int, optional
            sinc 求和的截断半宽，默认使用所有点

        Returns
        -------
        np.ndarray
            插值结果 (M,)
        """
        if len(x_grid) < 2:
            raise ValueError("x_grid 至少需要 2 个点")
        if len(x_grid) != len(f_values):
            raise ValueError("x_grid 和 f_values 长度必须相同")

        # 检查均匀性
        h = x_grid[1] - x_grid[0]
        if not np.allclose(np.diff(x_grid), h, rtol=1e-10):
            raise ValueError("x_grid 必须是均匀网格")

        N = len(x_grid)
        if truncation is None:
            truncation = N

        f_interp = np.zeros(len(x_query))

        for qi, xq in enumerate(x_query):
            # 找到最近的网格点索引
            k0 = int(round((xq - x_grid[0]) / h))
            k_min = max(0, k0 - truncation)
            k_max = min(N - 1, k0 + truncation)

            s = 0.0
            for k in range(k_min, k_max + 1):
                s += f_values[k] * SincInterpolator.sincn((xq - x_grid[k]) / h)
            f_interp[qi] = s

        return f_interp

    @staticmethod
    def interpolate_2d(x_grid: np.ndarray, y_grid: np.ndarray,
                       f_values: np.ndarray, x_query: np.ndarray,
                       y_query: np.ndarray, truncation: int = None) -> np.ndarray:
        """
        二维可分离 Sinc 插值

        公式：
            f(x, y) ≈ Σ_i Σ_j f(x_i, y_j) sinc((x-x_i)/h_x) sinc((y-y_j)/h_y)

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            均匀网格坐标
        f_values : np.ndarray
            二维场值 (ny, nx)
        x_query, y_query : np.ndarray
            查询点坐标（可广播）
        truncation : int, optional
            截断半宽

        Returns
        -------
        np.ndarray
            插值结果
        """
        if f_values.ndim != 2:
            raise ValueError("f_values 必须为二维数组")

        ny, nx = f_values.shape
        if len(x_grid) != nx or len(y_grid) != ny:
            raise ValueError("网格维度不匹配")

        hx = x_grid[1] - x_grid[0]
        hy = y_grid[1] - y_grid[0]

        if truncation is None:
            truncation = max(nx, ny)

        x_query = np.atleast_1d(x_query)
        y_query = np.atleast_1d(y_query)

        result = np.zeros((len(y_query), len(x_query)))

        for j, yq in enumerate(y_query):
            for i, xq in enumerate(x_query):
                kx0 = int(round((xq - x_grid[0]) / hx))
                ky0 = int(round((yq - y_grid[0]) / hy))

                kx_min = max(0, kx0 - truncation)
                kx_max = min(nx - 1, kx0 + truncation)
                ky_min = max(0, ky0 - truncation)
                ky_max = min(ny - 1, ky0 + truncation)

                s = 0.0
                for ky in range(ky_min, ky_max + 1):
                    sy = SincInterpolator.sincn((yq - y_grid[ky]) / hy)
                    for kx in range(kx_min, kx_max + 1):
                        sx = SincInterpolator.sincn((xq - x_grid[kx]) / hx)
                        s += f_values[ky, kx] * sx * sy
                result[j, i] = s

        return result

    @staticmethod
    def derivative_1d(x_grid: np.ndarray, f_values: np.ndarray,
                      x_query: np.ndarray) -> np.ndarray:
        """
        一维 Sinc 导数插值

        公式：
            f'(x) ≈ Σ f(x_k) * (1/h) * sinc'((x - x_k)/h)

        Parameters
        ----------
        x_grid : np.ndarray
            均匀网格
        f_values : np.ndarray
            网格函数值
        x_query : np.ndarray
            查询点

        Returns
        -------
        np.ndarray
            导数插值结果
        """
        if len(x_grid) < 2:
            raise ValueError("x_grid 至少需要 2 个点")

        h = x_grid[1] - x_grid[0]
        N = len(x_grid)

        df = np.zeros(len(x_query))
        for qi, xq in enumerate(x_query):
            s = 0.0
            for k in range(N):
                s += f_values[k] * SincInterpolator.sincn_derivative((xq - x_grid[k]) / h) / h
            df[qi] = s

        return df
