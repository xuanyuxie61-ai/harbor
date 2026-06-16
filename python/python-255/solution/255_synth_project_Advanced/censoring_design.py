# -*- coding: utf-8 -*-
"""
censoring_design.py
======================================================================
最优光谱窗口选择 —— 截断设计框架

物理背景:
    实际观测的系外行星光谱受限于:
    (1) 仪器噪声: 某些波段 SNR 过低, 信息量为负
    (2) 恒星污染: 恒星黑子/光斑影响透射深度测量
    (3) 地球大气: 地面观测受 telluric 吸收影响
    (4) 数据截断: 饱和吸收线无法提供线性信息

    本模块采用最优截断设计 (来自 1105_ZhanzhongyuGAO_Optimal-Censoring-Design-Framework)
    的思想, 选择最优的光谱子集 (窗口) 用于反演:
        - 最小化反演参数的后验协方差
        - 最大化 Fisher 信息矩阵的行列式 (D-最优)

    对数正态分布模型 (移植自 1105 的 func_lognormal):
        观测误差 epsilon ~ LogNormal(mu, sigma)
    截断观测: 仅当 |epsilon| < c 时接受数据点。

数学公式:
    Fisher 信息矩阵:
        I(theta) = sum_i w_i (dF_i/dtheta)^T Sigma^{-1} (dF_i/dtheta)
    D-最优设计:
        max_w log det(I(theta))
        s.t. sum w_i = 1, w_i >= 0                          (1)

    A-最优设计:
        min_w trace(I(theta)^{-1})                           (2)

依赖: numpy, scipy
======================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional, List, Callable
from scipy.stats import norm, lognorm
from scipy.special import comb


class CensoringDesigner:
    """
    最优光谱窗口设计器。

    移植自 1105 项目的核心算法:
    - truncated_normal_pdf : 截断正态分布 PDF
    - convolve_pdfs        : 递归卷积
    - compute_alpha        : 经验第一类错误率
    - compute_H            : 控制限
    """

    def __init__(
        self,
        n_channels: int = 50,
        censoring_threshold: float = 5.0,
        snr_min: float = 3.0,
        resolution: int = 500,
    ):
        self.n_channels = n_channels
        self.c = censoring_threshold
        self.snr_min = snr_min
        self.resolution = resolution
        self._validate()

    def _validate(self) -> None:
        if self.n_channels < 5:
            raise ValueError(f"光谱通道数过少: {self.n_channels}")
        if self.c <= 0:
            raise ValueError(f"截断阈值必须为正: {self.c}")
        if self.snr_min <= 0:
            raise ValueError(f"最小 SNR 必须为正: {self.snr_min}")

    @staticmethod
    def truncated_normal_pdf(
        k: float, sigma: float, tau: float, resolution: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        截断正态分布 PDF (移植自 1105 func_lognormal.truncated_normal_pdf)。

        物理意义: 观测噪声在截断阈值内的概率分布。
        """
        x = np.linspace(k - 6 * sigma, tau, resolution)
        pdf = norm.pdf(x, loc=k, scale=sigma)
        cdf_max = norm.cdf(tau, loc=k, scale=sigma)
        if cdf_max < 1.0e-30:
            cdf_max = 1.0e-30
        pdf = pdf / cdf_max
        return x, pdf

    @staticmethod
    def convolve_pdfs(
        x: np.ndarray, pdf: np.ndarray, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        递归卷积 (移植自 1105 func_lognormal.convolve_pdfs)。
        计算 n 个独立同分布随机变量之和的 PDF。
        """
        result_pdf = pdf.copy()
        result_x = x.copy()
        for _ in range(n - 1):
            result_pdf = np.convolve(result_pdf, pdf, mode="full")
            result_x = np.linspace(
                result_x[0] + x[0],
                result_x[-1] + x[-1],
                len(result_pdf),
            )
            dx_new = np.diff(result_x)
            norm_factor = np.sum(result_pdf[:-1] * dx_new)
            if norm_factor > 1.0e-30:
                result_pdf = result_pdf / norm_factor
        return result_x, result_pdf

    @staticmethod
    def empirical_cdf(x: np.ndarray, pdf: np.ndarray) -> np.ndarray:
        dx = np.diff(x)
        cdf = np.zeros_like(x)
        cdf[1:] = np.cumsum(pdf[:-1] * dx)
        if cdf[-1] > 1.0e-30:
            cdf = cdf / cdf[-1]
        return cdf

    def compute_fisher_weights(
        self,
        jacobian: np.ndarray,
        noise_variance: np.ndarray,
    ) -> np.ndarray:
        """
        基于 Fisher 信息的通道权重优化 (D-最优设计)。

        输入:
            jacobian : (n_channels, n_params) Jacobian 矩阵
            noise_variance : 各通道噪声方差
        输出:
            weights : 最优通道权重 (和为 1)
        """
        n_ch, n_p = jacobian.shape
        if n_ch < n_p:
            return np.ones(n_ch) / n_ch

        fisher_full = np.zeros((n_p, n_p), dtype=np.float64)
        for i in range(n_ch):
            if noise_variance[i] > 1.0e-30:
                wi = 1.0 / noise_variance[i]
                fi = jacobian[i, :].reshape(-1, 1)
                fisher_full += wi * (fi @ fi.T)

        try:
            fisher_inv = np.linalg.inv(fisher_full + 1.0e-10 * np.eye(n_p))
            leverage = np.zeros(n_ch, dtype=np.float64)
            for i in range(n_ch):
                fi = jacobian[i, :]
                if noise_variance[i] > 1.0e-30:
                    leverage[i] = fi @ fisher_inv @ fi / noise_variance[i]

            leverage = np.maximum(leverage, 1.0e-10)
            weights = leverage / np.sum(leverage)
        except np.linalg.LinAlgError:
            weights = np.ones(n_ch) / n_ch

        return weights

    def select_optimal_channels(
        self,
        wavelength_grid: np.ndarray,
        snr_grid: np.ndarray,
        jacobian: np.ndarray,
        n_select: int = 20,
    ) -> Dict:
        """
        选择最优光谱通道子集。

        考虑:
        (1) SNR 截断: 丢弃 SNR < snr_min 的通道
        (2) 信息量: 基于 Fisher 权重选择 n_select 个通道
        (3) 覆盖性: 确保波长覆盖均匀
        """
        n_ch = len(wavelength_grid)
        if n_select > n_ch:
            n_select = n_ch

        valid_mask = snr_grid >= self.snr_min
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < n_select:
            return {
                "selected_indices": valid_indices,
                "selected_wavelengths": wavelength_grid[valid_indices],
                "n_selected": len(valid_indices),
                "warnings": "有效通道数不足",
            }

        j_valid = jacobian[valid_indices, :]
        noise_var = 1.0 / (snr_grid[valid_indices] ** 2 + 1.0e-30)

        weights = self.compute_fisher_weights(j_valid, noise_var)

        # 选择权重最大的 n_select 个通道
        sorted_idx = np.argsort(weights)[::-1]
        selected_local = sorted_idx[:n_select]
        selected_global = valid_indices[selected_local]

        selected_global = np.sort(selected_global)

        return {
            "selected_indices": selected_global,
            "selected_wavelengths": wavelength_grid[selected_global],
            "weights": weights[selected_local],
            "n_selected": len(selected_global),
            "fisher_trace": float(np.trace(
                j_valid[selected_local, :].T @ j_valid[selected_local, :]
            )),
        }

    def compute_detection_delay(
        self,
        k0: float,
        sigma0: float,
        n_batch: int,
        alpha: float = 0.0027,
    ) -> Dict:
        """
        计算光谱异常检测的平均延迟 (ATS)。
        移植自 1105 func_lognormal.compute_H。

        物理意义: 当大气参数发生突变 (如火山喷发注入气溶胶) 时,
        检测系统需要多少个观测周期才能发现异常。
        """
        x, pdf = self.truncated_normal_pdf(k0, sigma0, self.c, self.resolution)
        valid = np.isfinite(pdf)
        x = x[valid]
        pdf = pdf[valid]

        cdfs_list = []
        xs_list = []
        for k in range(1, n_batch + 1):
            sum_x, sum_pdf = self.convolve_pdfs(x, pdf, k)
            sum_cdf = self.empirical_cdf(sum_x, sum_pdf)
            xs_list.append(sum_x)
            cdfs_list.append(sum_cdf)

        return {
            "n_batch": n_batch,
            "xs": xs_list,
            "cdfs": cdfs_list,
            "censoring_threshold": self.c,
        }
