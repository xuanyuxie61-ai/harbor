"""
stability_analysis.py
=====================

高阶有限差分格式的 von Neumann 稳定性分析与 CFL 条件计算。

融合种子项目：
    360_fd1d_heat_explicit/fd1d_heat_explicit_cfl：CFL 条件检查
    125_burgers_steady_viscous：非线性稳定性的 Newton 收敛判据

数学基础（von Neumann 稳定性分析）：
    对 FTCS 格式：
        c_i^{n+1} = c_i^n + CFL * (c_{i-1}^n - 2c_i^n + c_{i+1}^n)
    代入 Fourier 模态 c_j^n = G^n e^{i k j dx}，得放大因子：
        G(k) = 1 + CFL * (e^{-ikdx} - 2 + e^{ikdx})
             = 1 - 4 CFL sin^2(k dx / 2)
    稳定性要求 |G(k)| <= 1 对所有 k，即：
        CFL <= 0.5

    对四阶差分：
        CFL_max = 1/4 * (1 + 1/2) = 3/8 ≈ 0.375（更严格）

    反应-扩散耦合的稳定性：
        dt <= dx^2 / (2 D_eff) * 1/(1 + tau_rxn/tau_diff)
    其中 tau_rxn = 1/(k_sei * c_ref) 为反应特征时间。

作者: DA-Synthesis
"""

import math
import cmath
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


# ============================================================
#  von Neumann 放大因子计算
# ============================================================

def amplification_factor_ftcs(cfl, k_wave, dx):
    """
    FTCS 格式的 von Neumann 放大因子。

    数学形式：
        G(k) = 1 - 4 CFL sin^2(k dx / 2)

    Parameters
    ----------
    cfl : float
        库朗数 D dt / dx^2。
    k_wave : float
        波数 [1/m]。
    dx : float
        空间步长。

    Returns
    -------
    complex
        放大因子 G。
    """
    G = 1.0 - 4.0 * cfl * math.sin(k_wave * dx / 2.0) ** 2
    return complex(G, 0.0)


def amplification_factor_4th_order(cfl, k_wave, dx):
    """
    四阶差分 FTCS 格式的放大因子。

    四阶 Laplacian 的 Fourier 符号：
        lambda_4(k) = (1/dx^2) * [-2 cos(k dx) + 2 cos(2 k dx)/12
                                  - 16 cos(k dx)/12 + 30/12]
    化简后放大因子为复数形式。

    Parameters
    ----------
    cfl : float
        库朗数。
    k_wave : float
        波数。
    dx : float
        空间步长。

    Returns
    -------
    complex
        放大因子。
    """
    kd = k_wave * dx
    # 四阶二阶导数模板的 Fourier 符号
    symbol = (-cmath.exp(-2j * kd) + 16.0 * cmath.exp(-1j * kd) - 30.0
              + 16.0 * cmath.exp(1j * kd) - cmath.exp(2j * kd)) / 12.0
    G = 1.0 + cfl * symbol
    return G


def von_neumann_scan(cfl_values, k_max_modes, dx, scheme="2nd_order"):
    """
    扫描一系列 CFL 数和波数，检查 von Neumann 稳定性。

    Parameters
    ----------
    cfl_values : list[float]
        待检查的 CFL 数列表。
    k_max_modes : int
        检查的最大 Fourier 模态数。
    dx : float
        空间步长。
    scheme : str
        '2nd_order' 或 '4th_order'。

    Returns
    -------
    results : list[dict]
        每个 CFL 的稳定性分析结果。
    """
    results = []
    amp_func = (amplification_factor_ftcs if scheme == "2nd_order"
                else amplification_factor_4th_order)

    for cfl in cfl_values:
        max_abs_G = 0.0
        worst_k = 0.0
        unstable_modes = 0

        # 扫描完整布里渊区 k ∈ (0, π/dx]
        n_k = max(k_max_modes, 200)
        for m in range(1, n_k + 1):
            k_wave = m * math.pi / n_k / dx
            G = amp_func(cfl, k_wave, dx)
            abs_G = abs(G)
            if abs_G > max_abs_G:
                max_abs_G = abs_G
                worst_k = k_wave
            if abs_G > 1.0 + 1.0e-10:
                unstable_modes += 1

        is_stable = max_abs_G <= 1.0 + 1.0e-10
        results.append({
            "cfl": cfl,
            "max_abs_G": max_abs_G,
            "worst_k": worst_k,
            "unstable_modes": unstable_modes,
            "stable": is_stable,
        })

    return results


# ============================================================
#  CFL 临界值计算
# ============================================================

def critical_cfl_2nd_order():
    """
    二阶 FTCS 的临界 CFL 数。

    理论值：CFL_crit = 0.5

    Returns
    -------
    float
        临界 CFL 数。
    """
    return 0.5


def critical_cfl_4th_order():
    """
    四阶 FTCS 的临界 CFL 数。

    通过对放大因子 G(k) 求极值得到：
        CFL_crit ≈ 0.375

    Returns
    -------
    float
        临界 CFL 数。
    """
    # 数值搜索
    cfl_test = 0.0
    dcfl = 0.001
    while cfl_test < 1.0:
        cfl_test += dcfl
        # 检查最不稳定模态（k dx = π）
        kd = math.pi
        symbol = (-cmath.exp(-2j * kd) + 16.0 * cmath.exp(-1j * kd) - 30.0
                  + 16.0 * cmath.exp(1j * kd) - cmath.exp(2j * kd)) / 12.0
        G = 1.0 + cfl_test * symbol
        if abs(G) > 1.0 + 1.0e-10:
            return cfl_test - dcfl
    return cfl_test


def max_time_step_diffusion(dx, d_eff, scheme="2nd_order"):
    """
    给定空间步长和扩散系数，计算最大允许时间步长。

    数学形式：
        dt_max = CFL_crit * dx^2 / D_eff

    Parameters
    ----------
    dx : float
        空间步长 [m]。
    d_eff : float
        有效扩散系数 [m^2/s]。
    scheme : str
        差分格式。

    Returns
    -------
    float
        最大时间步长 [s]。
    """
    cfl_crit = (critical_cfl_2nd_order() if scheme == "2nd_order"
                else critical_cfl_4th_order())
    return cfl_crit * dx * dx / d_eff


# ============================================================
#  反应-扩散耦合稳定性
# ============================================================

def reaction_diffusion_stability(dx, d_eff, k_reaction, c_ref):
    """
    反应-扩散耦合系统的稳定性分析。

    线性化稳定性条件：
        dt <= dx^2 / (2 D) * 1 / (1 + Da)
    其中 Damköhler 数 Da = k_reaction * c_ref * dx^2 / D

    Parameters
    ----------
    dx : float
        空间步长。
    d_eff : float
        有效扩散系数。
    k_reaction : float
        反应速率常数 [1/s]。
    c_ref : float
        参考浓度。

    Returns
    -------
    dict
        稳定性信息（Da, dt_max, 控制因素等）。
    """
    tau_diff = dx * dx / d_eff if d_eff > 0 else float('inf')
    tau_rxn = 1.0 / (k_reaction * c_ref) if k_reaction * c_ref > 0 else float('inf')
    da = tau_diff / tau_rxn if tau_rxn > 0 else 0.0

    dt_max = 0.5 * tau_diff / (1.0 + da)

    if da < 0.1:
        controlling = "diffusion"
    elif da > 10.0:
        controlling = "reaction"
    else:
        controlling = "coupled"

    return {
        "tau_diff": tau_diff,
        "tau_rxn": tau_rxn,
        "damkohler": da,
        "dt_max": dt_max,
        "controlling": controlling,
    }


# ============================================================
#  非线性 Newton 稳定性判据（参考 burgers_steady_viscous）
# ============================================================

def newton_convergence_analysis(residual_history, tol=1.0e-8):
    """
    分析 Newton 迭代的收敛性与稳定性。

    检查：
        1. 残差是否单调递减
        2. 收敛阶（二次收敛检验）
        3. 是否发散

    Parameters
    ----------
    residual_history : list[float]
        每步 Newton 残差。
    tol : float
        收敛容限。

    Returns
    -------
    dict
        收敛性分析结果。
    """
    if len(residual_history) < 2:
        return {
            "converged": len(residual_history) > 0 and residual_history[0] < tol,
            "monotone": True,
            "quadratic": False,
            "diverged": False,
        }

    # 检查单调递减
    monotone = all(residual_history[i + 1] <= residual_history[i] * 1.1
                   for i in range(len(residual_history) - 1))

    # 检查二次收敛（近似）
    quadratic = False
    if len(residual_history) >= 3:
        rates = []
        for i in range(2, len(residual_history)):
            if residual_history[i - 1] > 1.0e-30:
                rate = (math.log(max(residual_history[i], 1.0e-30))
                        / math.log(max(residual_history[i - 1], 1.0e-30)))
                rates.append(rate)
        if rates:
            avg_rate = sum(rates) / len(rates)
            quadratic = avg_rate > 1.5  # 接近 2 为二次

    diverged = residual_history[-1] > residual_history[0] * 10.0
    converged = residual_history[-1] < tol

    return {
        "converged": converged,
        "monotone": monotone,
        "quadratic": quadratic,
        "diverged": diverged,
        "final_residual": residual_history[-1],
        "n_iterations": len(residual_history),
    }


# ============================================================
#  综合稳定性报告
# ============================================================

def generate_stability_report():
    """
    生成完整的稳定性分析报告。

    Returns
    -------
    dict
        包含各项稳定性指标。
    """
    dx = P.DX
    dt = P.DT_EXPLICIT
    d_eff = P.D_LI_SEI

    cfl_actual = d_eff * dt / (dx * dx)
    cfl_crit_2 = critical_cfl_2nd_order()
    cfl_crit_4 = critical_cfl_4th_order()

    dt_max_2 = max_time_step_diffusion(dx, d_eff, "2nd_order")
    dt_max_4 = max_time_step_diffusion(dx, d_eff, "4th_order")

    # von Neumann 扫描
    cfl_range = [0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.5, 0.51, 0.6, 0.8]
    vn_results_2 = von_neumann_scan(cfl_range, 20, dx, "2nd_order")
    vn_results_4 = von_neumann_scan(cfl_range, 20, dx, "4th_order")

    # 反应-扩散耦合
    rd_stability = reaction_diffusion_stability(
        dx, d_eff, P.K0_SEI, P.C_LI_INIT
    )

    return {
        "cfl_actual": cfl_actual,
        "cfl_crit_2nd": cfl_crit_2,
        "cfl_crit_4th": cfl_crit_4,
        "dt_max_2nd": dt_max_2,
        "dt_max_4th": dt_max_4,
        "von_neumann_2nd": vn_results_2,
        "von_neumann_4th": vn_results_4,
        "reaction_diffusion": rd_stability,
    }


if __name__ == "__main__":
    report = generate_stability_report()
    print("[stability_analysis] 稳定性报告")
    print(f"  实际 CFL = {report['cfl_actual']:.4f}")
    print(f"  二阶临界 CFL = {report['cfl_crit_2nd']:.4f}")
    print(f"  四阶临界 CFL = {report['cfl_crit_4th']:.4f}")
    print(f"  二阶 dt_max = {report['dt_max_2nd']:.6e} s")
    print(f"  四阶 dt_max = {report['dt_max_4th']:.6e} s")

    print("\n  von Neumann 扫描 (二阶):")
    for r in report['von_neumann_2nd']:
        status = "STABLE" if r['stable'] else "UNSTABLE"
        print(f"    CFL={r['cfl']:.3f}: |G|_max={r['max_abs_G']:.6f} [{status}]")

    rd = report['reaction_diffusion']
    print(f"\n  反应-扩散耦合:")
    print(f"    Damköhler 数 Da = {rd['damkohler']:.6e}")
    print(f"    控制因素: {rd['controlling']}")
