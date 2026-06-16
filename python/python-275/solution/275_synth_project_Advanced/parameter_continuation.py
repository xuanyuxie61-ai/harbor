"""
parameter_continuation.py
=========================
参数延拓法追踪非厄米系统的谱演化.
融合种子项目:
  - 138_cauchy_method: Cauchy 法沿参数路径推进
  - 003_allen_cahn_pde: 参数化 ODE/PDE 系统
  - 099_blend: 参数空间插值

物理框架:
  设 H(λ) 为含参数 λ 的非厄米哈密顿量.
  追踪本征值 E_n(λ) 随 λ 的变化 → 谱流 (spectral flow).
  EP 表现为 E_n(λ_EP) = E_m(λ_EP), 导数发散.
  参数延拓: 从已知 λ_0 出发, 逐步推进到目标 λ_f.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
import nonhermitian_hamiltonian as nh


# ---------------------------------------------------------------------------
# 参数延拓: 追踪本征值路径
# ---------------------------------------------------------------------------
def track_eigenvalue_path(H_func: Callable, lambda_vals: np.ndarray,
                          band_idx: int = 0,
                          continuity: str = "min_distance") -> Tuple[np.ndarray, np.ndarray]:
    """沿参数路径 λ 追踪第 band_idx 支本征值.

    在每一步, 选择与上一步最接近的本征值以保持连续性
    (避免分支交换造成的不连续).

    Parameters
    ----------
    H_func : λ -> H(λ) 的函数.
    lambda_vals : 参数序列.
    band_idx : 初始带的索引.
    continuity : 'min_distance' 或 'max_overlap'.

    Returns
    -------
    E_path : (len(lambda_vals),) 复数本征值路径.
    overlap_path : 相邻步的重叠 (用于检测 EP).
    """
    n_lambda = len(lambda_vals)
    E_path = np.zeros(n_lambda, dtype=complex)
    overlap_path = np.zeros(n_lambda - 1)
    H0 = H_func(lambda_vals[0])
    E, psi_R = np.linalg.eig(H0)
    sorted_idx = np.argsort(E.real)
    idx = sorted_idx[min(band_idx, len(sorted_idx) - 1)]
    E_path[0] = E[idx]
    prev_v = psi_R[:, idx].copy()
    for i in range(1, n_lambda):
        H = H_func(lambda_vals[i])
        E_new, psi_R_new = np.linalg.eig(H)
        if continuity == "min_distance":
            dists = np.abs(E_new - E_path[i - 1])
            idx = np.argmin(dists)
        else:
            overlaps = np.abs(np.vdot(prev_v, psi_R_new[:, j])
                              for j in range(len(E_new)))
            idx = np.argmax(overlaps)
        E_path[i] = E_new[idx]
        overlap = abs(np.vdot(prev_v, psi_R_new[:, idx]))
        overlap_path[i - 1] = overlap
        prev_v = psi_R_new[:, idx].copy()
    return E_path, overlap_path


def cauchy_continuation(H_func: Callable, lambda_0: float,
                        lambda_f: float, n_steps: int,
                        band_idx: int = 0,
                        theta: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Cauchy theta 法参数延拓 (源自 cauchy_fixed).

    将 E(λ) 视为 ODE: dE/dλ = (dH/dλ)_{mn} / (相关矩阵元).
    简化: 直接用 Hellmann-Feynman: dE_n/dλ = ⟨ψ_L^n| dH/dλ |ψ_R^n⟩.

    Returns
    -------
    lambda_vals, E_vals : 参数与本征值序列.
    """
    lambda_vals = np.linspace(lambda_0, lambda_f, n_steps + 1)
    E_vals = np.zeros(n_steps + 1, dtype=complex)
    h = 1e-7
    H = H_func(lambda_0)
    E, psi_R = np.linalg.eig(H)
    idx = np.argsort(E.real)[band_idx]
    E_vals[0] = E[idx]
    for i in range(n_steps):
        d_lambda = lambda_vals[i + 1] - lambda_vals[i]
        H_p = H_func(lambda_vals[i] + h)
        H_m = H_func(lambda_vals[i] - h)
        dH = (H_p - H_m) / (2 * h)
        v_R = psi_R[:, idx]
        v_L_conj = np.linalg.solve(H.conj().T - np.conj(E[idx]) * np.eye(H.shape[0]),
                                   np.zeros(H.shape[0]) + 0j)
        E_current = E_vals[i]
        v_R_full = psi_R[:, idx]
        dE = np.vdot(v_R_full, dH @ v_R_full) / np.vdot(v_R_full, v_R_full)
        E_pred = E_current + theta * d_lambda * dE
        H_mid = H_func(lambda_vals[i] + theta * d_lambda)
        E_mid, psi_R_mid = np.linalg.eig(H_mid)
        dists = np.abs(E_mid - E_pred)
        idx_new = np.argmin(dists)
        E_corrector = E_mid[idx_new]
        E_vals[i + 1] = (1.0 / theta) * E_corrector + (1.0 - 1.0 / theta) * E_current
        H = H_func(lambda_vals[i + 1])
        E, psi_R = np.linalg.eig(H)
        dists = np.abs(E - E_vals[i + 1])
        idx = np.argmin(dists)
    return lambda_vals, E_vals


# ---------------------------------------------------------------------------
# 多参数扫描 (源自 Allen-Cahn 的参数管理)
# ---------------------------------------------------------------------------
def parameter_sweep_2d(H_func: Callable,
                       param1_name: str, param1_vals: np.ndarray,
                       param2_name: str, param2_vals: np.ndarray,
                       fixed_params: Dict = None,
                       band_indices: List[int] = None) -> Dict:
    """二维参数扫描, 计算谱.

    Returns
    -------
    result : dict with 'param1', 'param2', 'eigenvalues' (形状 (n1, n2, n_bands)).
    """
    if fixed_params is None:
        fixed_params = {}
    n1, n2 = len(param1_vals), len(param2_vals)
    H_test = H_func(**{param1_name: param1_vals[0],
                       param2_name: param2_vals[0], **fixed_params})
    n_bands = H_test.shape[0]
    if band_indices is None:
        band_indices = list(range(n_bands))
    E_all = np.zeros((n1, n2, len(band_indices)), dtype=complex)
    for i, p1 in enumerate(param1_vals):
        for j, p2 in enumerate(param2_vals):
            try:
                H = H_func(**{param1_name: p1, param2_name: p2, **fixed_params})
            except Exception:
                E_all[i, j, :] = np.nan + 1j * np.nan
                continue
            E = np.linalg.eigvalsh(H @ np.eye(n_bands)) if False else np.linalg.eigvals(H)
            E_sorted = E[np.argsort(E.real)]
            for b_idx, b in enumerate(band_indices):
                if b < len(E_sorted):
                    E_all[i, j, b_idx] = E_sorted[b]
    return {
        "param1_name": param1_name,
        "param2_name": param2_name,
        "param1_vals": param1_vals,
        "param2_vals": param2_vals,
        "eigenvalues": E_all,
    }


# ---------------------------------------------------------------------------
# 例外点邻域展开
# ---------------------------------------------------------------------------
def ep_puiseux_expansion(H_func: Callable, ep_params: Dict,
                         delta_range: np.ndarray,
                         branch: int = 0) -> np.ndarray:
    """EP 邻域的 Puiseux 展开.

    在二阶 EP 附近, 本征值具有平方根分支:
      E_{±}(λ) = E_EP ± c * sqrt(λ - λ_EP) + O(λ - λ_EP)

    通过计算不同 δ = λ - λ_EP 的本征值, 验证平方根行为.

    Returns
    -------
    E_branch : (len(delta_range),) 指定分支的本征值.
    """
    E_vals = np.zeros(len(delta_range), dtype=complex)
    base_params = ep_params.copy()
    param_name = list(ep_params.keys())[0]
    ep_value = ep_params[param_name]
    for i, delta in enumerate(delta_range):
        params = base_params.copy()
        params[param_name] = ep_value + delta
        try:
            H = H_func(**params)
        except Exception:
            E_vals[i] = np.nan + 1j * np.nan
            continue
        E = np.linalg.eigvals(H)
        E_sorted = E[np.argsort(E.real)]
        idx = min(branch, len(E_sorted) - 1)
        E_vals[i] = E_sorted[idx]
    return E_vals


def winding_number_around_ep(H_func: Callable,
                             center_params: Dict,
                             radius: float,
                             n_points: int = 100,
                             param_name: str = "gamma") -> float:
    """计算绕 EP 的卷绕数.

    在参数空间绕 EP 画圆, 追踪本征值, 计算:
      W = (1/2πi) ∮ d ln(E - E_EP)
    对二阶 EP, W = 1/2 (半整数卷绕).

    Returns
    -------
    winding : 卷绕数 (可为半整数).
    """
    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    E_path = np.zeros(n_points + 1, dtype=complex)
    base_val = center_params[param_name]
    for i, theta in enumerate(angles):
        params = center_params.copy()
        params[param_name] = base_val + radius * np.exp(1j * theta)
        H = H_func(**params)
        E = np.linalg.eigvals(H)
        E_sorted = E[np.argsort(np.abs(E))]
        E_path[i] = E_sorted[0]
    E_center = np.mean(E_path[:-1])
    phase_changes = np.angle((E_path[1:] - E_center) / (E_path[:-1] - E_center + 1e-15))
    total_phase = np.sum(phase_changes)
    return total_phase / (2 * np.pi)
