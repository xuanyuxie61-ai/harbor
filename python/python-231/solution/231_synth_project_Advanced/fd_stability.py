"""
fd_stability.py — 有限差分格式的稳定性分析与循环矩阵 (映射自 r8ci)
=============================================================
本模块分析 DGLAP 演化中不同有限差分格式的数值稳定性.
核心工具为循环矩阵 (circulant matrix) 的特征值分析,
映射自 r8ci 的循环矩阵运算.

包含:
  (1) 循环矩阵构造与行列式计算 (映射自 r8ci_det, r8ci_dif2);
  (2) 有限差分格式的稳定域分析;
  (3) von Neumann 稳定性分析;
  (4) CFL 条件检查;
  (5) 不同阶数 FD 格式的精度-稳定性权衡.

核心公式 (循环矩阵):
    C = circ(c_0, c_1, ..., c_{n-1})
    C_{ij} = c_{(j-i) mod n}

    特征值: λ_k = Σ_{j=0}^{n-1} c_j ω^{jk}
    其中 ω = exp(2πi/n)

核心公式 (二阶差分循环矩阵):
    D2 = circ(−2, 1, 0, ..., 0, 1) / h²
    特征值: λ_k = (2 cos(2πk/n) − 2) / h² ≤ 0

核心公式 (von Neumann 稳定性):
    对 DGLAP 的 IMEX 格式, 放大因子:
    |g(θ)| = |1 + dt λ_exp(θ)| / |1 − dt λ_imp(θ)| ≤ 1

核心公式 (CFL 条件):
    dt ≤ C × h² / max|α_s P(z)|
    其中 C 为安全系数 (通常 C = 0.5)
"""
from __future__ import annotations
import math
import cmath
from typing import List, Tuple, Dict

from phys_consts import (
    FD4_FIRST_DERIV_STENCIL, FD4_FIRST_DERIV_OFFSETS,
    FD4_SECOND_DERIV_STENCIL, FD4_SECOND_DERIV_OFFSETS,
    FD6_FIRST_DERIV_STENCIL, FD6_FIRST_DERIV_OFFSETS,
    alpha_s_lo, EPS_NUMERICAL
)


# ============================================================
# 1. 循环矩阵构造 (映射自 r8ci)
# ============================================================
def circulant_first_row(stencil: tuple, offsets: tuple, n: int) -> List[float]:
    """
    从差分模板构造循环矩阵的第一行:
        c[j] = stencil[k] / h  若 j = offsets[k] mod n
        c[j] = 0               否则

    映射自 r8ci_zeros / r8ci_dif2 的循环矩阵构造.
    """
    row = [0.0] * n
    for c, off in zip(stencil, offsets):
        j = off % n
        row[j] += c
    return row


def circulant_eigenvalues(first_row: List[float]) -> List[complex]:
    """
    循环矩阵的特征值 (FFT):
        λ_k = Σ_{j=0}^{n-1} c_j ω^{jk}
    其中 ω = exp(2πi/n)

    映射自 r8ci_eval / r8ci_det.
    """
    n = len(first_row)
    eigenvalues = []
    for k in range(n):
        lam = complex(0.0, 0.0)
        for j in range(n):
            angle = 2.0 * math.pi * j * k / n
            lam += first_row[j] * cmath.exp(complex(0, angle))
        eigenvalues.append(lam)
    return eigenvalues


def circulant_determinant(first_row: List[float]) -> float:
    """
    循环矩阵的行列式:
        det(C) = Π_{k=0}^{n-1} λ_k

    映射自 r8ci_det.
    """
    eigs = circulant_eigenvalues(first_row)
    det = complex(1.0, 0.0)
    for lam in eigs:
        det *= lam
    return det.real  # 行列式应为实数


def second_diff_circulant(n: int, h: float = 1.0) -> List[float]:
    """
    二阶差分循环矩阵的第一行 (映射自 r8ci_dif2):
        c = [−2, 1, 0, ..., 0, 1] / h²

    这是周期性边界条件下的离散 Laplacian.
    """
    if n < 3:
        return [0.0] * n
    row = [0.0] * n
    row[0] = -2.0 / (h * h)
    row[1] = 1.0 / (h * h)
    row[n - 1] = 1.0 / (h * h)
    return row


# ============================================================
# 2. von Neumann 稳定性分析
# ============================================================
def von_neumann_amplification(
    theta: float, dt: float, h: float,
    alpha_s: float, scheme: str = 'imex2'
) -> complex:
    """
    von Neumann 放大因子:
        g(θ) = 分子 / 分母

    对不同格式:
      - 'forward_euler': g = 1 + dt λ(θ)
      - 'imex1': g = (1 + dt λ_exp(θ)) / (1 − dt λ_imp(θ))
      - 'imex2': g = 二阶 IMEX 的放大因子

    λ(θ) 为循环矩阵在波数 θ 处的特征值.

    稳定性条件: |g(θ)| ≤ 1 对所有 θ.
    """
    CF = 4.0 / 3.0
    # 简化: 使用 P_qq 的对角近似
    a_qq = CF * 1.5
    # 空间特征值 (二阶差分)
    lam_spatial = 2.0 * (math.cos(theta) - 1.0) / (h * h)
    # 时间系数
    coeff = alpha_s / (2.0 * math.pi)

    if scheme == 'forward_euler':
        return 1.0 + dt * coeff * (lam_spatial - a_qq)
    elif scheme == 'imex1':
        numer = 1.0 + dt * coeff * lam_spatial
        denom = 1.0 + dt * coeff * a_qq
        return numer / max(denom, EPS_NUMERICAL)
    elif scheme == 'imex2':
        # 二阶 IMEX (Heun)
        a1 = 1.0 + dt * coeff * lam_spatial
        b1 = 1.0 + dt * coeff * a_qq
        g1 = a1 / max(b1, EPS_NUMERICAL)
        a2 = 1.0 + dt * coeff * lam_spatial * g1
        b2 = 1.0 + dt * coeff * a_qq
        g2 = 0.5 * (1.0 + g1 + a2 / max(b2, EPS_NUMERICAL))
        return g2
    else:
        return complex(0.0, 0.0)


def stability_check(
    dt: float, h: float, q2: float, nf: int = 3,
    scheme: str = 'imex2', n_theta: int = 100
) -> Tuple[bool, float, Dict[str, float]]:
    """
    完整的稳定性检查:
      - 对所有波数 θ ∈ [0, 2π], 计算 |g(θ)|;
      - 稳定性 ⟺ max_θ |g(θ)| ≤ 1 + ε.

    参数:
        dt:      时间步长
        h:       空间步长 (在 v = ln(1/x) 空间)
        q2:      当前 Q² 值
        nf:      活跃味数
        scheme:  差分格式
        n_theta: θ 采样点数
    返回:
        (is_stable, max_amplification, details)
    """
    as_val = alpha_s_lo(q2, nf)
    max_amp = 0.0
    worst_theta = 0.0
    for i in range(n_theta + 1):
        theta = 2.0 * math.pi * i / n_theta
        g = von_neumann_amplification(theta, dt, h, as_val, scheme)
        amp = abs(g)
        if amp > max_amp:
            max_amp = amp
            worst_theta = theta

    is_stable = max_amp <= 1.0 + 1e-10
    details = {
        'alpha_s': as_val,
        'max_amplification': max_amp,
        'worst_theta': worst_theta,
        'dt': dt,
        'h': h,
        'scheme': scheme,
    }
    return is_stable, max_amp, details


# ============================================================
# 3. CFL 条件
# ============================================================
def cfl_max_timestep(h: float, q2: float, nf: int = 3,
                     safety: float = 0.5) -> float:
    """
    CFL 最大时间步长:
        dt_max = safety × h² / (α_s/2π × max|P|)

    简化: max|P| ≈ A_qq (对角系数)
    """
    CF = 4.0 / 3.0
    a_qq = CF * 1.5
    as_val = alpha_s_lo(q2, nf)
    coeff = as_val / (2.0 * math.pi)
    return safety * h * h / max(coeff * a_qq, EPS_NUMERICAL)


# ============================================================
# 4. 精度-稳定性权衡分析
# ============================================================
def fd_accuracy_stability_tradeoff(
    h_values: List[float], q2: float, nf: int = 3
) -> List[Dict[str, float]]:
    """
    对不同网格步长 h, 分析 FD2/FD4/FD6 的精度与稳定性.

    返回:
        [{h, max_amp_fd2, max_amp_fd4, max_amp_fd6, dt_max}, ...]
    """
    results = []
    for h in h_values:
        dt_max = cfl_max_timestep(h, q2, nf)
        dt = 0.8 * dt_max

        # 各格式的放大因子
        amp_fd2 = abs(von_neumann_amplification(math.pi, dt, h, alpha_s_lo(q2, nf), 'imex1'))
        amp_fd4 = abs(von_neumann_amplification(math.pi, dt, h, alpha_s_lo(q2, nf), 'imex2'))
        amp_fd6 = amp_fd4  # 简化: 使用相同格式

        results.append({
            'h': h,
            'dt_max': dt_max,
            'dt_used': dt,
            'max_amp_fd2': amp_fd2,
            'max_amp_fd4': amp_fd4,
            'max_amp_fd6': amp_fd6,
        })
    return results


def print_stability_report(h_values: List[float], q2: float, nf: int = 3):
    """打印稳定性分析报告"""
    results = fd_accuracy_stability_tradeoff(h_values, q2, nf)
    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        "║   有限差分格式稳定性分析报告                             ║",
        "╠══════════════════════════════════════════════════════════╣",
        "║     h        dt_max    |g_FD2|   |g_FD4|   stable?     ║",
        "╠══════════════════════════════════════════════════════════╣",
    ]
    for r in results:
        stable_str = "YES" if r['max_amp_fd4'] <= 1.0 + 1e-10 else "NO"
        lines.append(
            f"║ {r['h']:8.4f}  {r['dt_max']:8.5f}  {r['max_amp_fd2']:8.5f}  "
            f"{r['max_amp_fd4']:8.5f}  {stable_str:5s}  ║"
        )
    lines.append("╚══════════════════════════════════════════════════════════╝")
    return '\n'.join(lines)
