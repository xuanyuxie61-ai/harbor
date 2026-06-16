# -*- coding: utf-8 -*-
"""
conservation_monitor.py
=======================
能量守恒与 Hamiltonian 漂移监测.

物理基础:
---------
在 fast ignition 模拟中, 理想情况下总能量应守恒:
    E_total(t) = ∫∫ u(x,y,t) dx dy + E_beam_remaining(t) = const

数值上, 守恒量漂移反映格式耗散/色散误差:
    δH/H = (H(t) - H(0)) / H(0)

守恒量构造 (来自 018_arenstorf_ode/arenstorf_conserved):
    H = 0.5 · (p_x^2 + p_y^2) - 0.5 · (x^2 + y^2) - μ_1/r_2 - μ_2/r_1
    (Arenstorf 守恒量)

    类比到等离子体:
    H_plasma = E_kinetic + E_thermal + E_field - E_deposited
"""

import math


def compute_total_energy_1d(u, dx):
    """一维总能量: E = Σ u_i · Δx."""
    return sum(u) * dx


def compute_total_energy_2d(u, dx, dy):
    """二维总能量: E = ΣΣ u_{ij} · Δx · Δy."""
    total = 0.0
    for row in u:
        total += sum(row)
    return total * dx * dy


def compute_energy_moments_1d(u, x, dx, max_moment=3):
    """
    计算能量矩:
        M_k = ∫ x^k · u(x) dx

    M_0: 总能量
    M_1: 能量中心位置
    M_2: 能量展宽 (方差)
    """
    moments = [0.0] * (max_moment + 1)
    for k in range(max_moment + 1):
        for i in range(len(u)):
            moments[k] += (x[i] ** k) * u[i] * dx
    return moments


def compute_conservation_drift(energy_history):
    """
    计算守恒量漂移:
        δ(t) = (E(t) - E(0)) / E(0)

    参数:
        energy_history: 时间序列能量列表
    返回:
        dict: 漂移统计
    """
    empty = {"max_drift": 0.0, "final_drift": 0.0, "rms_drift": 0.0,
             "E0": 0.0, "E_final": 0.0, "n_steps": 0}
    if not energy_history:
        return empty

    E0 = energy_history[0]
    E_final = energy_history[-1]

    if abs(E0) < 1.0e-300:
        # 初始能量为零, 改用最终能量作参考
        if abs(E_final) < 1.0e-300:
            return empty
        drifts = [(E - E_final) / E_final for E in energy_history]
    else:
        drifts = [(E - E0) / E0 for E in energy_history]

    max_drift = max(abs(d) for d in drifts)
    final_drift = drifts[-1]
    rms_drift = math.sqrt(sum(d ** 2 for d in drifts) / len(drifts))

    return {
        "max_drift": max_drift,
        "final_drift": final_drift,
        "rms_drift": rms_drift,
        "E0": E0,
        "E_final": E_final,
        "n_steps": len(energy_history),
    }


def arenstorf_like_conserved(x, y, px, py, mu1, mu2):
    """
    Arenstorf 守恒量 (来自 018_arenstorf_ode/arenstorf_conserved).
    用于 ODE 积分器验证.

    H = 0.5(p_x^2 + p_y^2) - 0.5(x^2 + y^2) - μ_1/r_2 - μ_2/r_1

    r_1 = sqrt((x+μ_1)^2 + y^2)
    r_2 = sqrt((x-μ_2)^2 + y^2)
    """
    r1 = math.sqrt((x + mu1) ** 2 + y ** 2)
    r2 = math.sqrt((x - mu2) ** 2 + y ** 2)
    r1 = max(r1, 1.0e-15)
    r2 = max(r2, 1.0e-15)
    H = 0.5 * (px ** 2 + py ** 2) - 0.5 * (x ** 2 + y ** 2) - mu1 / r2 - mu2 / r1
    return H


def plasma_energy_conservation_check(u_history, x, dx, source_energy_history=None):
    """
    完整的等离子体能量守恒检查.

    参数:
        u_history          : 能量密度时间序列
        x                  : 空间网格
        dx                 : 步长
        source_energy_history: 源项累积注入能量 (可选)
    返回:
        dict: 守恒性诊断
    """
    energies = [compute_total_energy_1d(u, dx) for u in u_history]
    moments = [compute_energy_moments_1d(u, x, dx, max_moment=2)
               for u in u_history]

    drift = compute_conservation_drift(energies)

    result = {
        "energies": energies,
        "drift": drift,
        "M0_history": [m[0] for m in moments],
        "M1_history": [m[1] for m in moments],
        "M2_history": [m[2] for m in moments],
    }

    if source_energy_history:
        # 守恒 = 总能量 = 初始 + 注入
        balanced = []
        for i in range(len(energies)):
            expected = energies[0] + source_energy_history[i]
            balanced.append((energies[i] - expected) / max(abs(expected), 1.0e-30))
        result["balance_error"] = balanced

    return result


def print_conservation_summary(conservation_result):
    """打印守恒性摘要."""
    print("\n" + "=" * 72)
    print("能量守恒监测")
    print("=" * 72)
    drift = conservation_result["drift"]
    print("  初始能量   : {:.6e}".format(drift["E0"]))
    print("  最终能量   : {:.6e}".format(drift["E_final"]))
    print("  最大漂移   : {:.6e}".format(drift["max_drift"]))
    print("  最终漂移   : {:.6e}".format(drift["final_drift"]))
    print("  RMS 漂移   : {:.6e}".format(drift["rms_drift"]))
    print("  时间步数   : {}".format(drift["n_steps"]))

    if "balance_error" in conservation_result:
        bal = conservation_result["balance_error"]
        if bal:
            print("  源项平衡误差 (最终): {:.6e}".format(bal[-1]))

    # 能量矩演化
    M0 = conservation_result["M0_history"]
    M1 = conservation_result["M1_history"]
    M2 = conservation_result["M2_history"]
    if len(M0) >= 2:
        print("  M0 变化 (总能量): {:.4e} -> {:.4e}".format(M0[0], M0[-1]))
        if M0[0] > 0:
            centroid_initial = M1[0] / M0[0]
            centroid_final = M1[-1] / M0[-1]
            print("  能量中心 移动  : {:.4e} -> {:.4e} m".format(
                centroid_initial, centroid_final))
    print("=" * 72)
