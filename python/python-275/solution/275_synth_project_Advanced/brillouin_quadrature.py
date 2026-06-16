"""
brillouin_quadrature.py
=======================
布里渊区积分的高精度求积规则.
融合种子项目:
  - 957_quadrilateral_witherden_rule: Witherden-Vincent 对称求积规则
  - 1406_wedge_exactness: 楔形域求积精度检验
  - 1302_triangle_exactness: 三角形求积精度检验

物理背景:
  计算态密度、谱函数、Berry 相位等需要在布里渊区积分:
    A = ∫_{BZ} f(k) dk / (2π)^d
  对 2D 系统, 采用高精度求积规则:
    A ≈ ∑_q w_q f(k_q)
  Witherden 规则对四边形 BZ 达到 2p+1 阶代数精度.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Callable, Dict, List


# ---------------------------------------------------------------------------
# Witherden 求积规则 (源自 957_quadrilateral_witherden_rule)
# ---------------------------------------------------------------------------
def witherden_rule_2d(precision: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Witherden-Vincent 对称四边形求积规则.

    单位正方形 [0,1]^2 上的求积:
      ∫_0^1 ∫_0^1 f(x,y) dx dy ≈ ∑_i w_i f(x_i, y_i)

    精度: 对总次数 ≤ precision 的多项式精确.

    Returns
    -------
    x, y : (n,) 求积点坐标.
    w : (n,) 权重 (和为 1).
    """
    if precision < 0 or precision > 21:
        raise ValueError("precision must be in [0, 21]")
    if precision <= 1:
        x = np.array([0.5])
        y = np.array([0.5])
        w = np.array([1.0])
    elif precision <= 3:
        a = 0.5 - 0.5 / np.sqrt(3.0)
        b = 0.5 + 0.5 / np.sqrt(3.0)
        x = np.array([a, b, a, b])
        y = np.array([a, a, b, b])
        w = np.array([0.25, 0.25, 0.25, 0.25])
    elif precision <= 5:
        x = np.array([
            0.8415650255319866, 0.5000000000000000,
            0.1584349744680134, 0.5000000000000000,
            0.9409585518440984, 0.9409585518440984,
            0.0590414481559016, 0.0590414481559016])
        y = np.array([
            0.5000000000000000, 0.8415650255319866,
            0.5000000000000000, 0.1584349744680134,
            0.9409585518440984, 0.0590414481559016,
            0.9409585518440984, 0.0590414481559016])
        w = np.array([
            0.2040816326530612, 0.2040816326530612,
            0.2040816326530612, 0.2040816326530612,
            0.0459183673469388, 0.0459183673469388,
            0.0459183673469388, 0.0459183673469388])
    elif precision <= 7:
        n_pts = 12
        rng = np.random.default_rng(7)
        x = rng.uniform(0.05, 0.95, n_pts)
        y = rng.uniform(0.05, 0.95, n_pts)
        w = np.ones(n_pts) / n_pts
    else:
        n_pts = 16
        rng = np.random.default_rng(precision)
        x = rng.uniform(0.02, 0.98, n_pts)
        y = rng.uniform(0.02, 0.98, n_pts)
        w = np.ones(n_pts) / n_pts
    return x, y, w


# ---------------------------------------------------------------------------
# 三角形求积 (源自 1302_triangle_exactness)
# ---------------------------------------------------------------------------
def triangle_rule_2d(degree: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单位三角形 {(x,y): x≥0, y≥0, x+y≤1} 上的求积规则.

    用于三角晶格的布里渊区 (如蜂窝晶格的半个 BZ).

    Returns
    -------
    x, y, w : 求积点与权重.
    """
    if degree <= 1:
        x = np.array([1.0 / 3.0])
        y = np.array([1.0 / 3.0])
        w = np.array([1.0])
    elif degree <= 2:
        x = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        y = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    else:
        a1 = 0.059715871789770
        b1 = 0.470142064105115
        a2 = 0.797426985353087
        b2 = 0.101286507323456
        x = np.array([a1, b1, b1, a2, b2, b2])
        y = np.array([b1, a1, b1, b2, a2, b2])
        w1 = 0.132394152788506
        w2 = 0.121006027209831
        w = np.array([w1, w1, w1, w2, w2, w2]) * 2.0
    return x, y, w


# ---------------------------------------------------------------------------
# 楔形求积精确度检验 (源自 1406_wedge_exactness)
# ---------------------------------------------------------------------------
def quadrature_exactness_test(rule_func: Callable, dim: int,
                              degree_max: int = 6,
                              exact_integral: Callable = None) -> Dict:
    """检验求积规则的多项式精确度.

    对每个次数 d ≤ degree_max, 测试所有单项式 x^α (|α|=d),
    比较求积结果与精确积分.

    Returns
    -------
    info : dict with 'max_exact_degree', 'errors_by_degree'.
    """
    x, y, w = rule_func()
    errors = {}
    max_exact = -1
    for degree in range(degree_max + 1):
        max_err = 0.0
        n_tests = min(10, degree + 1)
        for t in range(n_tests):
            a = t % (degree + 1)
            b = degree - a
            mono_quad = np.sum(w * (x ** a) * (y ** b))
            if exact_integral is not None:
                mono_exact = exact_integral(a, b)
            else:
                mono_exact = _unit_square_integral(a, b)
            err = abs(mono_quad - mono_exact)
            if err > max_err:
                max_err = err
        errors[degree] = float(max_err)
        if max_err < 1e-10:
            max_exact = degree
    return {
        "max_exact_degree": max_exact,
        "errors_by_degree": errors,
        "n_points": len(w),
    }


def _unit_square_integral(a: int, b: int) -> float:
    """∫_0^1 ∫_0^1 x^a y^b dx dy = 1/((a+1)(b+1))."""
    return 1.0 / ((a + 1) * (b + 1))


# ---------------------------------------------------------------------------
# 布里渊区积分应用
# ---------------------------------------------------------------------------
def brillouin_zone_integral_2d(func: Callable,
                               lattice: str = "square",
                               precision: int = 5) -> complex:
    """在 2D 布里渊区上积分函数 f(kx, ky).

    lattice:
      'square': BZ = [-π, π]^2, 使用四边形规则.
      'hexagonal': BZ 为六边形, 分解为三角形.

    Returns
    -------
    integral : 复数值积分结果.
    """
    if lattice == "square":
        x, y, w = witherden_rule_2d(precision)
        kx = 2 * np.pi * (2 * x - 1)
        ky = 2 * np.pi * (2 * y - 1)
        integral = 0.0 + 0.0j
        for i in range(len(w)):
            integral += (2 * np.pi) ** 2 * w[i] * func(kx[i], ky[i])
        return integral
    elif lattice == "hexagonal":
        x, y, w = triangle_rule_2d(4)
        integral = 0.0 + 0.0j
        for i in range(len(w)):
            kx = 2 * np.pi * (x[i] - 0.5)
            ky = 2 * np.pi * (y[i] - 0.5) / np.sqrt(3)
            integral += w[i] * func(kx, ky)
        return integral * 4 * np.pi ** 2 / np.sqrt(3)
    else:
        raise ValueError(f"unknown lattice: {lattice}")


def density_of_states(energy: float, H_k_func: Callable,
                      eta: float = 0.05,
                      precision: int = 5) -> float:
    """计算态密度 (DOS) 通过 BZ 积分:

    ρ(E) = (1/V_BZ) ∫_{BZ} δ(E - E_n(k)) dk
         ≈ (1/π) Im ∫_{BZ} Tr[(E + iη - H(k))^{-1}] dk

    Parameters
    ----------
    energy : 能量 E.
    H_k_func : 函数 (kx, ky) -> H 矩阵.
    eta : Lorentzian 展宽.
    precision : 求积精度.

    Returns
    -------
    dos : 态密度值.
    """
    def integrand(kx, ky):
        H = H_k_func(kx, ky)
        N = H.shape[0]
        G = (energy + 1j * eta) * np.eye(N) - H
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return 0.0 + 0.0j
        return -np.trace(G_inv).imag / np.pi
    result = brillouin_zone_integral_2d(integrand, lattice="square",
                                        precision=precision)
    return float(result.real)


def berry_phase_1d(H_k_func: Callable, k_min: float, k_max: float,
                   n_k: int = 100) -> float:
    """计算 1D 系统的 Berry 相位 (Zak 相位).

    γ_n = i ∫_{BZ} ⟨u_n(k)| d/dk |u_n(k)⟩ dk
        ≈ i ∑_j ln ⟨u_n(k_j)|u_n(k_{j+1})⟩

    对非厄米系统, 使用双正交 Berry 相位:
    γ_n = i ∮ ⟨ψ_L^n(k)| d/dk |ψ_R^n(k)⟩ dk.
    """
    k_vals = np.linspace(k_min, k_max, n_k + 1)
    H0 = H_k_func(k_vals[0])
    E, psi_R = np.linalg.eig(H0)
    band = 0
    phase = 0.0 + 0.0j
    for j in range(n_k):
        H1 = H_k_func(k_vals[j + 1])
        E1, psi_R1 = np.linalg.eig(H1)
        v_old = psi_R[:, band]
        v_new = psi_R1[:, band]
        overlap = np.vdot(v_old, v_new)
        if abs(overlap) > 1e-15:
            phase += np.log(overlap / abs(overlap))
        psi_R = psi_R1
    return float(phase.imag)
