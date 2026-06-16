# -*- coding: utf-8 -*-
"""
sparse_covariance.py — BAO 协方差矩阵的稀疏精度矩阵构造

本模块为 BAO 相关函数 ξ(s) 构造协方差矩阵 C 与精度矩阵 P = C^{-1}.
物理上, ξ(s) 在不同 s 区间上的协方差来自:
  (1) 宇宙方差 (cosmic variance): 体积有限导致的样本方差
  (2) 泊松噪声 (shot noise): 星系离散性
  (3) 积分条件 (integral constraint): 随机 catalog 有限大小

协方差模型
---------
对第 ℓ 阶多极矩 ξ_ℓ(s), 其协方差在 Gaussian 近似下为:

  C_{ij}^{(ℓ)} = (2/(2ℓ+1)) ∫ dk k^2 / (2π^2)
                 × [P_ℓ^{tot}(k)]^2 j_ℓ(k s_i) j_ℓ(k s_j) × W(k)

其中 P_ℓ^{tot} = P_g + P_sn 为总功率谱 (星系 + 散粒噪声), W(k) 为
survey 窗口函数.

稀疏性来源
---------
当把协方差截断到"近邻 s 区间" (|i-j| ≤ m), 得到稀疏协方差. 进一步,
我们用种子项目 1405 web_matrix 的"网页链接矩阵"范式 — 把 s 区间视为
"网页", 用 (i, j) 之间的"相关度" 定义稀疏图 — 然后用 `incidence_to_transition`
的思想把稀疏关联矩阵转换为马尔可夫转移矩阵, 用于下游 MCMC 预条件.

种子项目映射
----------
- 1405 web_matrix (incidence_to_transition.py, harvard_st.py 等):
  稀疏关联图 → 转移矩阵的构造在此被移植为: s 区间之间的"相关性图" →
  精度矩阵的稀疏分解.
- 1135 bilevel-optim (dict_learning.py) : 字典学习的稀疏约束被用于
  精度矩阵的正则化, 避免 C 的病态逆.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import math
import numpy as np

from spherical_hankel import j0, jl


# =============================================================
# 高斯协方差模型
# =============================================================
@dataclass
class CovarianceConfig:
    """协方差构造参数."""
    s_bins        : np.ndarray      # s 区间中心 (Mpc/h)
    volume        : float = 5.0e9   # survey 体积 (Mpc/h)^3
    n_bar         : float = 3.0e-4  # 平均数密度 (h/Mpc)^3
    P_shot        : float = 1.0e3   # 散粒噪声功率 (h/Mpc)^3
    k_min         : float = 0.005   # k 积分下限
    k_max         : float = 0.30    # k 积分上限
    n_k           : int = 128       # k 积分采样数
    ell           : int = 0         # 多极矩阶


def gaussian_covariance_Pk(k: float, bias: float = 2.0,
                           P_lin_005: float = 2.0e4
                           ) -> float:
    """
    简化模型: P_g(k) = b^2 P_lin(k), 其中 P_lin(k) = P_0 (k/k_0)^{-1.5}.
    """
    k0 = 0.05
    P_lin = P_lin_005 * (k / k0) ** (-1.5)
    return bias * bias * P_lin


def build_covariance_matrix(cfg: CovarianceConfig,
                            bias: float = 2.0) -> np.ndarray:
    """
    构造高斯协方差矩阵 C_{ij} = Cov[ξ_ℓ(s_i), ξ_ℓ(s_j)].
    """
    s = cfg.s_bins
    n = len(s)
    k_arr = np.geomspace(cfg.k_min, cfg.k_max, cfg.n_k)
    C = np.zeros((n, n), dtype=np.float64)
    prefactor = 2.0 / ((2 * cfg.ell + 1) * cfg.volume)
    for i in range(n):
        for j in range(n):
            integrand = np.zeros_like(k_arr)
            for ik, k in enumerate(k_arr):
                P_tot = gaussian_covariance_Pk(k, bias) + cfg.P_shot
                j_ell_i = jl(cfg.ell, k * s[i])
                j_ell_j = jl(cfg.ell, k * s[j])
                integrand[ik] = k ** 2 * P_tot ** 2 * j_ell_i * j_ell_j
            C[i, j] = prefactor * np.trapz(integrand, k_arr) / (2.0 * math.pi ** 2)
    # 对称化
    C = 0.5 * (C + C.T)
    # 正则化: 加对角噪声防止奇异
    C += np.eye(n) * np.diag(C).mean() * 1.0e-6
    return C


# =============================================================
# 稀疏化: 关联图 → 稀疏协方差
# =============================================================
def build_sparse_graph(s_bins: np.ndarray, correlation_length: float = 40.0
                       ) -> np.ndarray:
    """
    构造稀疏关联图 G_{ij} = exp(-|s_i - s_j| / L_cor).
    这是 `web_matrix` 的"网页链接"范式在 BAO 协方差中的对应物.
    """
    n = len(s_bins)
    G = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            G[i, j] = math.exp(-abs(s_bins[i] - s_bins[j]) / correlation_length)
    return G


def incidence_to_precision(G: np.ndarray, regularization: float = 0.01
                           ) -> np.ndarray:
    """
    从关联图 G 构造精度矩阵 P ∝ (D - G) + λ I.
    这里 D 为度矩阵 (行和), 类比 web_matrix 的 incidence_to_transition.
    """
    n = G.shape[0]
    D = np.diag(G.sum(axis=1))
    L = D - G  # 图 Laplacian
    P = L + regularization * np.eye(n)
    return P


def sparse_precision_from_covariance(C: np.ndarray,
                                     sparsity_ratio: float = 0.1
                                     ) -> np.ndarray:
    """
    从协方差矩阵 C 构造稀疏精度矩阵.
    方法: (1) 求逆得 P = C^{-1}; (2) 保留 |P_ij| 最大的 sparsity_ratio 比例元素.
    """
    n = C.shape[0]
    # 稳定求逆: 用 Cholesky
    try:
        L_chol = np.linalg.cholesky(C)
        L_inv = np.linalg.solve(L_chol, np.eye(n))
        P = L_inv.T @ L_inv
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(C)
    # 稀疏化: 保留 top sparsity_ratio
    n_keep = max(int(n * n * sparsity_ratio), n)
    flat_abs = np.abs(P).ravel()
    thresh_idx = np.argpartition(flat_abs, -n_keep)[-n_keep:]
    mask = np.zeros(n * n, dtype=bool)
    mask[thresh_idx] = True
    mask = mask.reshape(n, n)
    # 保留对角
    mask = mask | np.eye(n, dtype=bool)
    P_sparse = P * mask
    # 对称化
    P_sparse = 0.5 * (P_sparse + P_sparse.T)
    return P_sparse


# =============================================================
# 关联长度估计 (从数据自适应)
# =============================================================
def estimate_correlation_length(C: np.ndarray, s_bins: np.ndarray
                                ) -> float:
    """
    从协方差矩阵的对角线衰减估计关联长度 L_cor.
    方法: 拟合 C_{i, i0} / C_{i0, i0} = exp(-|s_i - s_{i0}| / L_cor).
    """
    i0 = len(s_bins) // 2
    C_col = np.abs(C[:, i0])
    C_col = C_col / (C_col[i0] + 1.0e-30)
    ds = np.abs(s_bins - s_bins[i0])
    # 最小二乘拟合 ln(C_col) = -ds / L_cor
    valid = (C_col > 1.0e-6) & (ds > 0.0)
    if valid.sum() < 3:
        return 30.0
    log_C = np.log(C_col[valid])
    ds_v = ds[valid]
    # 斜率 = -1 / L_cor
    slope = -np.polyfit(ds_v, -log_C, 1)[0]
    L_cor = 1.0 / max(abs(slope), 1.0e-6)
    return min(max(L_cor, 10.0), 200.0)


# =============================================================
# 条件数与稳定性指标
# =============================================================
def matrix_condition_report(C: np.ndarray) -> Dict[str, float]:
    """报告矩阵 C 的条件数、最小/最大特征值."""
    eigvals = np.linalg.eigvalsh(C)
    return {
        "lambda_min": float(np.min(eigvals)),
        "lambda_max": float(np.max(eigvals)),
        "condition": float(np.max(eigvals) / (np.min(eigvals) + 1.0e-30)),
        "log_det": float(np.sum(np.log(np.maximum(eigvals, 1.0e-30)))),
    }


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    s_bins = np.linspace(60.0, 200.0, 15)
    cfg = CovarianceConfig(s_bins=s_bins, volume=5.0e9, n_k=64)
    C = build_covariance_matrix(cfg)
    rep = matrix_condition_report(C)
    print(f"[sparse_cov] 协方差条件数 = {rep['condition']:.2e}")
    L_cor = estimate_correlation_length(C, s_bins)
    print(f"[sparse_cov] 关联长度 = {L_cor:.2f} Mpc/h")
    G = build_sparse_graph(s_bins, L_cor)
    P = incidence_to_precision(G)
    print(f"[sparse_cov] 精度矩阵维度 = {P.shape}")


if __name__ == "__main__":
    _self_check()
