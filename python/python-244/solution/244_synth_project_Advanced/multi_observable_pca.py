#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_observable_pca.py
=======================
【融合种子项目】 1048_Sami-fak_DistributedMTLSPCA (分布式多任务 SPCA)

将分布式多任务主成分分析移植到中子星多观测约束的 EoS 参数降维.

物理: 多个观测 (质量、半径、潮汐形变) 共同约束 EoS 参数空间.
数学: PCA 降维 + 多任务相关性估计.
"""

import math
import numpy as np
from typing import Tuple, Dict


def pca_decomposition(data_matrix: np.ndarray,
                       n_components: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA 分解.
    data ≈ U S V^T
    返回: 主成分 (V), 奇异值 (S), 投影 (U S)
    """
    X = data_matrix - np.mean(data_matrix, axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    if n_components is None:
        n_components = len(S)

    components = Vt[:n_components]
    singular_values = S[:n_components]
    projections = X @ components.T

    return components, singular_values, projections


def explained_variance_ratio(singular_values: np.ndarray) -> np.ndarray:
    """各主成分的方差解释率."""
    return singular_values ** 2 / np.sum(singular_values ** 2)


def multi_task_correlation(observations: Dict[str, np.ndarray]) -> np.ndarray:
    """
    计算多观测间的相关矩阵.

    observations: dict 观测名 -> 观测值向量
    """
    names = sorted(observations.keys())
    n_obs = len(names)
    corr = np.eye(n_obs)
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            x = observations[names[i]]
            y = observations[names[j]]
            n = min(len(x), len(y))
            xm = x[:n] - np.mean(x[:n])
            ym = y[:n] - np.mean(y[:n])
            den = math.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
            corr[i, j] = np.dot(xm, ym) / den if den > 0 else 0.0
            corr[j, i] = corr[i, j]
    return corr


def distributed_mean_estimation(task_data: list, weights: list = None) -> np.ndarray:
    """
    分布式多任务均值估计 (移植自 distributedMTLSPCA).
    M_cal = (sum_i w_i M_i) / (sum_i w_i)
    """
    if weights is None:
        weights = [1.0] * len(task_data)
    total_w = sum(weights)
    result = sum(w * d for w, d in zip(weights, task_data)) / total_w
    return result


def empirical_covariance(X: np.ndarray) -> np.ndarray:
    """经验协方差矩阵 Sigma = X^T X / n."""
    n = X.shape[0]
    X_centered = X - np.mean(X, axis=0)
    return X_centered.T @ X_centered / max(n - 1, 1)


def eos_parameter_reduction(mass_obs: np.ndarray, radius_obs: np.ndarray,
                             lambda_obs: np.ndarray,
                             n_pca: int = 3) -> Dict:
    """
    EoS 参数空间 PCA 降维.

    输入: 质量、半径、潮汐形变的观测样本
    输出: 降维后的参数空间和解释率
    """
    n_samples = min(len(mass_obs), len(radius_obs), len(lambda_obs))
    data = np.column_stack([
        mass_obs[:n_samples],
        radius_obs[:n_samples],
        lambda_obs[:n_samples],
    ])

    # 标准化
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-30] = 1.0
    data_norm = (data - mean) / std

    components, singular_values, projections = pca_decomposition(data_norm, n_pca)
    var_ratio = explained_variance_ratio(singular_values)
    corr = multi_task_correlation({
        'mass': mass_obs[:n_samples],
        'radius': radius_obs[:n_samples],
        'lambda': lambda_obs[:n_samples],
    })

    return {
        'components': components,
        'singular_values': singular_values,
        'projections': projections,
        'variance_ratio': var_ratio,
        'correlation_matrix': corr,
        'mean': mean,
        'std': std,
    }


# 自检
if __name__ == "__main__":
    print("=== 多观测 PCA 自检 ===")
    np.random.seed(42)
    n = 50
    mass = np.random.normal(1.4, 0.2, n)
    radius = 10.0 + 2.0 * mass + np.random.normal(0, 0.5, n)
    lam = 300.0 * (radius / 12.0) ** 5 + np.random.normal(0, 50, n)

    result = eos_parameter_reduction(mass, radius, lam, n_pca=2)
    print(f"方差解释率: {result['variance_ratio']}")
    print(f"相关矩阵:\n{result['correlation_matrix']}")

    print("\nmulti_observable_pca.py 自检通过.")
