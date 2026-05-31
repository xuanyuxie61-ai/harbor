# -*- coding: utf-8 -*-
"""
error_norms.py
误差分析与范数计算模块

核心公式与物理背景
------------------
1. L¹ 范数（积分意义）
   对定义域 Ω 上的函数 f，其 L¹ 范数为
       ‖f‖_{L¹} = ∫_Ω |f(x)| dx
   离散近似：‖f‖_{L¹} ≈ Σ_i |f_i| · ΔV_i

2. L² 范数与能量范数
   ‖f‖_{L²} = √(∫_Ω |f|² dx)
   在有限差分/有限元中常作为收敛度量。

3. 相对误差
   ε_rel = ‖f - f_ref‖ / ‖f_ref‖

4. 收敛阶估计（Richardson 外推）
   若误差 e(h) ≈ C·h^p，则对两个步长 h₁, h₂：
       p ≈ log(e(h₁)/e(h₂)) / log(h₁/h₂)

融合来源
--------
- 812_norm_l1 : L1 范数计算与误差度量
"""

import numpy as np
from typing import Callable, Optional, Tuple


class ErrorNorms:
    """
    提供多种范数与误差度量工具，用于评估数值解的精度与收敛性。
    """

    @staticmethod
    def l1_norm_discrete(values: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        离散 L¹ 范数：
            ‖u‖_{L¹,d} = Σ_i |u_i| · V_i
        若 volumes 为 None，则退化为 ℓ¹ 范数 Σ|u_i|。
        """
        values = np.asarray(values)
        if volumes is None:
            return float(np.sum(np.abs(values)))
        volumes = np.asarray(volumes)
        if values.shape != volumes.shape:
            raise ValueError("values 与 volumes 形状不匹配")
        return float(np.sum(np.abs(values) * volumes))

    @staticmethod
    def l2_norm_discrete(values: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        离散 L² 范数：
            ‖u‖_{L²,d} = √( Σ_i |u_i|² · V_i )
        """
        values = np.asarray(values)
        if volumes is None:
            return float(np.sqrt(np.sum(values ** 2)))
        volumes = np.asarray(volumes)
        return float(np.sqrt(np.sum((values ** 2) * volumes)))

    @staticmethod
    def linf_norm(values: np.ndarray) -> float:
        """ℓ^∞ 范数：max |u_i|"""
        return float(np.max(np.abs(values)))

    @staticmethod
    def relative_l2_error(approx: np.ndarray, exact: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        相对 L² 误差：
            ε = ‖u_approx - u_exact‖_{L²} / ‖u_exact‖_{L²}
        若分母为 0，返回绝对 L² 误差。
        """
        diff = approx - exact
        num = ErrorNorms.l2_norm_discrete(diff, volumes)
        den = ErrorNorms.l2_norm_discrete(exact, volumes)
        if den < 1e-30:
            return num
        return num / den

    @staticmethod
    def relative_l1_error(approx: np.ndarray, exact: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """相对 L¹ 误差"""
        diff = approx - exact
        num = ErrorNorms.l1_norm_discrete(diff, volumes)
        den = ErrorNorms.l1_norm_discrete(exact, volumes)
        if den < 1e-30:
            return num
        return num / den

    @staticmethod
    def convergence_order(errors: np.ndarray, resolutions: np.ndarray) -> np.ndarray:
        """
        根据误差序列与对应分辨率序列估计收敛阶。
        返回每对相邻点的局部收敛阶 p。

        公式
        ----
        p_k = log(e_{k+1} / e_k) / log(h_{k+1} / h_k)
        """
        errors = np.asarray(errors)
        resolutions = np.asarray(resolutions)
        if len(errors) != len(resolutions):
            raise ValueError("errors 与 resolutions 长度必须相同")
        if len(errors) < 2:
            return np.array([])
        p = np.log(errors[1:] / errors[:-1]) / np.log(resolutions[1:] / resolutions[:-1])
        return p

    @staticmethod
    def richardson_extrapolation(uh: np.ndarray, uh2: np.ndarray, p: float = 2.0) -> np.ndarray:
        """
        Richardson 外推：
        已知 u_h 与 u_{h/2}，若误差展开为 u_h = u_exact + C·h^p + O(h^{p+1})，则
            u_extrap = (2^p · u_{h/2} - u_h) / (2^p - 1)
        """
        factor = 2.0 ** p
        return (factor * uh2 - uh) / (factor - 1.0)

    @staticmethod
    def residual_norm(residual: np.ndarray, volumes: Optional[np.ndarray] = None) -> float:
        """
        计算离散残差向量的 L² 范数，用于判断迭代收敛。
        """
        return ErrorNorms.l2_norm_discrete(residual, volumes)

    @staticmethod
    def compute_quality_metrics(field: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
        """
        计算标量场的数值质量指标（最大值、最小值、均值、标准差、动态范围）。
        """
        arr = np.asarray(field)
        if mask is not None:
            arr = arr[mask.astype(bool)]
        return {
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "dynamic_range_db": 20.0 * np.log10((np.max(np.abs(arr)) + 1e-30) / (np.min(np.abs(arr)) + 1e-30)),
        }
