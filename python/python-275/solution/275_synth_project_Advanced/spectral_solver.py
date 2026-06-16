"""
spectral_solver.py
==================
非厄米矩阵的谱求解器: 本征值、本征矢、左/右本征矢、双正交归一.
物理基础:
  非厄米哈密顿量 H ≠ H^†, 本征值 E 可为复数.
  右本征矢: H |ψ_R⟩ = E |ψ_R⟩
  左本征矢: ⟨ψ_L| H = E ⟨ψ_L|, 即 H^† |ψ_L⟩ = E* |ψ_L⟩
  双正交归一: ⟨ψ_L^i | ψ_R^j⟩ = δ^{ij}
  例外点: 两个本征值与本征矢同时简并, 代数重数 > 几何重数.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List


# ---------------------------------------------------------------------------
# 完整谱分解
# ---------------------------------------------------------------------------
def compute_spectrum(H: np.ndarray,
                     sort_by: str = "real") -> Tuple[np.ndarray, np.ndarray]:
    """计算非厄米矩阵的完整本征谱.

    Parameters
    ----------
    H : (N, N) 一般复数矩阵.
    sort_by : 'real', 'imag', 'abs', 'none'.

    Returns
    -------
    E : (N,) 本征值 (复数).
    psi_R : (N, N) 右本征矢, 列向量为第 i 个本征矢.
    """
    E, psi_R = np.linalg.eig(H)
    if sort_by == "real":
        idx = np.argsort(E.real)
    elif sort_by == "imag":
        idx = np.argsort(E.imag)
    elif sort_by == "abs":
        idx = np.argsort(np.abs(E))
    else:
        idx = np.arange(len(E))
    return E[idx], psi_R[:, idx]


def compute_left_eigenvectors(H: np.ndarray,
                              E: np.ndarray,
                              psi_R: np.ndarray) -> np.ndarray:
    """计算左本征矢 (通过 H^† 的本征分解).

    左本征矢满足: H^† |ψ_L⟩ = E* |ψ_L⟩
    然后双正交归一化: ⟨ψ_L^i | ψ_R^j⟩ = δ^{ij}.

    Returns
    -------
    psi_L : (N, N) 左本征矢 (列向量), 已双正交归一.
    """
    N = H.shape[0]
    E_L, psi_L_raw = np.linalg.eig(H.conj().T)
    psi_L = np.zeros_like(psi_L_raw)
    for i in range(N):
        dists = np.abs(E_L - np.conj(E[i]))
        j = np.argmin(dists)
        psi_L[:, i] = psi_L_raw[:, j]
    overlap = psi_L.conj().T @ psi_R
    for i in range(N):
        norm = overlap[i, i]
        if abs(norm) < 1e-14:
            psi_L[:, i] = psi_R[:, i]
            overlap_mat = psi_L.conj().T @ psi_R
            for j in range(N):
                if j != i:
                    psi_L[:, i] -= overlap_mat[j, i] * psi_L[:, j]
            overlap = psi_L.conj().T @ psi_R
        else:
            psi_L[:, i] /= np.sqrt(norm)
    return psi_L


def biorthogonal_projector(psi_L: np.ndarray,
                           psi_R: np.ndarray,
                           idx: int) -> np.ndarray:
    """构造第 idx 个本征态的双正交投影算子.

    P_i = |ψ_R^i⟩⟨ψ_L^i| / ⟨ψ_L^i|ψ_R^i⟩
    在非厄米系统中, 投影算子不满足 P = P^†, 但满足 P^2 = P.
    """
    vR = psi_R[:, idx:idx + 1]
    vL = psi_L[:, idx:idx + 1].conj().T
    norm = (vL @ vR)[0, 0]
    if abs(norm) < 1e-14:
        raise ValueError(f"degenerate norm at idx={idx}")
    return vR @ vL / norm


def spectral_decomposition_residual(H: np.ndarray,
                                    E: np.ndarray,
                                    psi_R: np.ndarray) -> float:
    """谱分解残差: ||H - sum_i E_i P_i||_F / ||H||_F.

    用于检验数值精度.
    """
    N = H.shape[0]
    psi_L = compute_left_eigenvectors(H, E, psi_R)
    H_recon = np.zeros_like(H)
    for i in range(N):
        H_recon += E[i] * biorthogonal_projector(psi_L, psi_R, i)
    return np.linalg.norm(H - H_recon, 'fro') / max(np.linalg.norm(H, 'fro'), 1e-15)


# ---------------------------------------------------------------------------
# 例外点特征量
# ---------------------------------------------------------------------------
def ep_accumulation_phase(E: np.ndarray,
                          psi_R: np.ndarray,
                          path_indices: List[int]) -> float:
    """沿闭合参数路径的相位累积, 用于确认 EP 的拓扑荷.

    绕 EP 一周, 本征值交换 → 累积相位 π (对于二阶 EP).
    数学: θ = Im ∮ ⟨ψ_R| d/dλ |ψ_R⟩ dλ

    Parameters
    ----------
    E : 沿路径的本征值序列 (需预先排序以保证连续性).
    psi_R : 沿路径的右本征矢.
    path_indices : 本征值索引, 追踪哪一支.

    Returns
    -------
    phase : 累积 Berry 相位 (弧度).
    """
    phase = 0.0
    for k in range(len(path_indices) - 1):
        i = path_indices[k]
        j = path_indices[k + 1]
        if i >= psi_R.shape[1] or j >= psi_R.shape[1]:
            continue
        overlap = np.vdot(psi_R[:, i], psi_R[:, j])
        if abs(overlap) > 1e-15:
            phase += np.angle(overlap)
    return float(phase)


def chiral_decomposition(E: np.ndarray) -> Dict[str, float]:
    """计算谱的 chirality 分解.

    对粒子-空穴对称的非厄米系统, 本征值成对出现: (E, -E*).
    chirality χ = (N_+ - N_-) / (N_+ + N_-)
    其中 N_+ 为 Re(E)>0 的数量.
    """
    N_plus = int(np.sum(E.real > 0))
    N_minus = int(np.sum(E.real < 0))
    N_zero = int(np.sum(np.abs(E.real) < 1e-10))
    total = len(E)
    return {
        "N_plus": N_plus,
        "N_minus": N_minus,
        "N_zero": N_zero,
        "chirality": (N_plus - N_minus) / max(total, 1),
        "total": total,
    }


def skin_mode_localization(psi_R: np.ndarray) -> np.ndarray:
    """非厄米趋肤效应的局域化度量.

    IPR (逆参与比) = ∑_j |ψ_j|^4 / (∑_j |ψ_j|^2)^2
    IPR ~ 1/N: 扩展态;  IPR ~ O(1): 局域态 (趋肤模式).

    也计算重心位置: x_c = ∑ j |ψ_j|^2 / ∑ |ψ_j|^2.
    """
    N, n_states = psi_R.shape
    ipr = np.zeros(n_states)
    x_c = np.zeros(n_states)
    for s in range(n_states):
        rho = np.abs(psi_R[:, s]) ** 2
        norm_rho = rho / max(rho.sum(), 1e-15)
        ipr[s] = np.sum(norm_rho ** 2) * N
        x_c[s] = np.sum(np.arange(N) * norm_rho)
    return ipr, x_c
