"""
spectral_robustness.py
----------------------
谱鲁棒性分析模块 —— 映射自种子项目 172_chladni_figures
核心思想：利用 Chladni 板振动模态的特征值分析，研究
PDE 约束优化中系统对参数扰动的谱灵敏度。

科学背景：
    考虑 PDE 约束优化中的线性化系统：
        L(w) u = f
    其中 L(w) 为依赖参数 w 的微分算子。
    特征值问题：
        L(w) phi_k = lambda_k phi_k
    特征值 lambda_k 的分布决定系统的稳定性与鲁棒性。

    Chladni 板振动 (Gander-Kwok 2012)：
        Delta^2 u = omega^2 u   on [0,1]^2
        u = Delta u = 0        on boundary
    其中 omega^2 为特征值，对应振动模态。
    鲁棒性指标：
        rho(x) = min_{w in W} lambda_1(L(w)) / lambda_1(L(w0))
    当 rho < 1 时，系统对扰动敏感 (fragile)。

本模块实现：
    1. 2D Laplacian 特征值 (5 点差分)
    2. 双 Laplacian (Chladni) 特征值
    3. 参数化特征值灵敏度
    4. 鲁棒谱间隙计算
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional


# =============================================================================
# 2D Laplacian (5 点差分)
# =============================================================================
def laplacian_2d(nx: int, ny: int, hx: float = 1.0, hy: float = 1.0) -> np.ndarray:
    """
    构造 2D Laplacian 矩阵 (5 点差分)：
        -Delta u ≈ (2u_{i,j} - u_{i+1,j} - u_{i-1,j}) / hx^2
                  + (2u_{i,j} - u_{i,j+1} - u_{i,j-1}) / hy^2
    Dirichlet 边界条件。
    """
    if nx < 2 or ny < 2:
        raise ValueError("nx, ny 必须 >= 2")
    N = nx * ny
    # 1D Laplacians
    ex = np.ones(nx)
    Tx = diags([-ex[1:], 2.0 * ex, -ex[1:]], [-1, 0, 1], shape=(nx, nx)) / hx ** 2
    ey = np.ones(ny)
    Ty = diags([-ey[1:], 2.0 * ey, -ey[1:]], [-1, 0, 1], shape=(ny, ny)) / hy ** 2
    # 2D Laplacian = Ty ⊗ I + I ⊗ Tx
    L = kron(Ty, eye(nx)) + kron(eye(ny), Tx)
    return L.toarray()


# =============================================================================
# Chladni 板 (双 Laplacian)
# =============================================================================
def chladni_matrix(nx: int, ny: int, mu: float = 0.225) -> np.ndarray:
    """
    Chladni 板矩阵 (Gander-Kwok 2012)：
        N u = Delta^2 u - mu * (d^4 u / dx^2 dy^2)
    其中 mu 为材料参数 (Poisson 比)。
    返回 (N, N) 矩阵，N = nx * ny。
    """
    if nx < 3 or ny < 3:
        raise ValueError("nx, ny 必须 >= 3")
    L = laplacian_2d(nx, ny)
    # 双 Laplacian
    N_mat = L @ L
    # 混合导数项 (简化：忽略高阶交叉)
    # N = L^2 + mu * cross_term
    return N_mat


def chladni_eigenmodes(
    nx: int,
    ny: int,
    mu: float = 0.225,
    k: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Chladni 板的前 k 个特征模态。
    返回 (eigenvalues, eigenvectors).
    """
    N_mat = chladni_matrix(nx, ny, mu)
    N = nx * ny
    k = min(k, N - 2)
    if k < 1:
        k = 1
    try:
        vals, vecs = eigsh(N_mat, k=k, which="SM", tol=1e-6)
    except Exception:
        # 回退到 dense
        vals, vecs = np.linalg.eigh(N_mat)
        idx = np.argsort(vals)[:k]
        vals, vecs = vals[idx], vecs[:, idx]
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


# =============================================================================
# 参数化特征值灵敏度
# =============================================================================
def eigenvalue_sensitivity(
    L_fn: callable,
    x: np.ndarray,
    mode_idx: int = 0,
    h: float = 1e-5,
) -> np.ndarray:
    """
    特征值对参数的灵敏度：
        d lambda_k / d x_i ≈ [lambda_k(x + h e_i) - lambda_k(x - h e_i)] / (2h)
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size

    def get_eigval(x_eval):
        L = L_fn(x_eval)
        try:
            vals = eigsh(L, k=1, which="SM", return_eigenvectors=False, tol=1e-6)
            return vals[0]
        except Exception:
            vals = np.linalg.eigvalsh(L)
            return vals[min(mode_idx, vals.size - 1)]

    lam0 = get_eigval(x)
    grad = np.zeros(n)
    for i in range(n):
        x_p = x.copy()
        x_m = x.copy()
        x_p[i] += h
        x_m[i] -= h
        grad[i] = (get_eigval(x_p) - get_eigval(x_m)) / (2.0 * h)
    return grad


# =============================================================================
# 鲁棒谱间隙
# =============================================================================
class SpectralRobustnessAnalyzer:
    """
    分析 PDE 约束系统在不确定性下的谱鲁棒性。
    """

    def __init__(
        self,
        L_fn: callable,
        x_nominal: np.ndarray,
        n_modes: int = 6,
    ):
        self.L_fn = L_fn
        self.x_nominal = np.asarray(x_nominal, dtype=float).ravel()
        self.n_modes = n_modes
        # 名义特征值
        L0 = L_fn(self.x_nominal)
        try:
            vals, _ = eigsh(L0, k=n_modes, which="SM", tol=1e-6)
        except Exception:
            vals = np.linalg.eigvalsh(L0)[:n_modes]
        self.nominal_eigenvalues = np.sort(np.real(vals))

    def robustness_ratio(
        self,
        x_perturbed: np.ndarray,
        mode_idx: int = 0,
    ) -> float:
        """
        谱鲁棒性比：
            rho = lambda_1(x) / lambda_1(x_nominal)
        rho < 1 表示扰动降低了稳定性。
        """
        L = self.L_fn(x_perturbed)
        try:
            vals = eigsh(L, k=1, which="SM", return_eigenvectors=False, tol=1e-6)
            lam1 = vals[0]
        except Exception:
            lam1 = np.linalg.eigvalsh(L)[0]
        if abs(self.nominal_eigenvalues[mode_idx]) < 1e-14:
            return 1.0
        return float(lam1 / self.nominal_eigenvalues[mode_idx])

    def spectral_gap(self, x: np.ndarray) -> float:
        """谱间隙 = lambda_2 - lambda_1."""
        L = self.L_fn(x)
        try:
            vals = eigsh(L, k=2, which="SM", return_eigenvectors=False, tol=1e-6)
        except Exception:
            vals = np.linalg.eigvalsh(L)[:2]
        vals = np.sort(np.real(vals))
        if vals.size < 2:
            return 0.0
        return float(vals[1] - vals[0])

    def fragility_index(
        self,
        perturbation_samples: np.ndarray,
    ) -> Tuple[float, float]:
        """
        脆弱性指标：
            F = mean( min_i rho(x_i) )
            sigma = std( min_i rho(x_i) )
        """
        ratios = []
        for x_p in perturbation_samples:
            r = self.robustness_ratio(x_p)
            ratios.append(r)
        ratios = np.array(ratios)
        return float(np.mean(ratios)), float(np.std(ratios))


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
if __name__ == "__main__":
    nx, ny = 10, 10
    L = laplacian_2d(nx, ny)
    print(f"Laplacian shape: {L.shape}")
    # 特征值
    vals = np.linalg.eigvalsh(L)
    print(f"Min eigenvalue: {vals[0]:.4f}, Max: {vals[-1]:.4f}")

    # Chladni
    vals_c, vecs_c = chladni_eigenmodes(10, 10, mu=0.225, k=3)
    print(f"Chladni eigenvalues: {vals_c}")

    # 鲁棒性分析
    def L_fn(x):
        # 简单参数化：缩放 Laplacian
        return x[0] * L

    analyzer = SpectralRobustnessAnalyzer(L_fn, np.array([1.0]), n_modes=3)
    print(f"Nominal eigenvalues: {analyzer.nominal_eigenvalues}")
    rho = analyzer.robustness_ratio(np.array([0.9]))
    print(f"Robustness ratio (x=0.9): {rho:.4f}")
