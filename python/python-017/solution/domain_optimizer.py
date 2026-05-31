"""
畴结构能量优化模块
融合来源: 1364_tsp_descent (下降法邻域搜索) + 1266_toms178 (Hooke-Jeeves 直接搜索)

功能:
- 对多铁性材料畴结构进行能量最小化
- Hooke-Jeeves 直接搜索优化 Landau 自由能参数
- 邻域搜索（反转/转置型操作）优化畴壁构型
- 应用于寻找稳态畴结构

科学背景:
    多铁性材料的畴结构对应于 Landau 自由能的局域极小值。
    优化目标:
        min_{P, M}  F_total = ∫ f(P, M, ∇P, ∇M) dV
    其中 P, M 为离散网格上的序参量场。
"""

import numpy as np
from typing import Callable, Tuple, Optional


def best_nearby(delta: np.ndarray, point: np.ndarray, prevbest: float,
                nvars: int, f: Callable[[np.ndarray], float],
                funevals: int) -> Tuple[float, np.ndarray, int]:
    """
    Hooke-Jeeves 算法的 best_nearby 子程序。
    沿每个坐标轴方向搜索更优解。
    直接源自 toms178 中 best_nearby.m 的算法逻辑。
    """
    z = point.copy()
    minf = prevbest

    for i in range(nvars):
        # 正向搜索
        z[i] = point[i] + delta[i]
        ftmp = f(z)
        funevals += 1
        if ftmp < minf:
            minf = ftmp
        else:
            # 负向搜索
            z[i] = point[i] - delta[i]
            ftmp = f(z)
            funevals += 1
            if ftmp < minf:
                minf = ftmp
            else:
                z[i] = point[i]

    return minf, z, funevals


def hooke_jeeves(nvars: int, startpt: np.ndarray, rho: float,
                 eps: float, itermax: int,
                 f: Callable[[np.ndarray], float]) -> Tuple[int, np.ndarray]:
    """
    Hooke-Jeeves 直接搜索算法。
    源自 toms178 中 hooke.m。

    参数:
        nvars:   变量维度
        startpt: 初始猜测
        rho:     步长缩减因子 (0 < rho < 1)，建议 0.5~0.85
        eps:     收敛容差
        itermax: 最大迭代次数
        f:       目标函数

    返回:
        iters:   实际迭代次数
        endpt:   最优解估计
    """
    if not (0 < rho < 1):
        raise ValueError("rho 必须在 (0, 1) 之间")

    newx = startpt.copy()
    xbefore = startpt.copy()
    delta = np.zeros(nvars)
    for i in range(nvars):
        if startpt[i] == 0.0:
            delta[i] = rho
        else:
            delta[i] = rho * abs(startpt[i])

    funevals = 0
    steplength = rho
    iters = 0
    fbefore = f(newx)
    funevals += 1
    newf = fbefore

    while iters < itermax and eps < steplength:
        iters += 1
        newx = xbefore.copy()
        newf, newx, funevals = best_nearby(delta, newx, fbefore, nvars, f, funevals)

        keep = True
        while newf < fbefore and keep:
            # 沿改进方向加速
            for i in range(nvars):
                tmp = xbefore[i]
                xbefore[i] = newx[i]
                if newx[i] <= tmp:
                    delta[i] = -abs(delta[i])
                else:
                    delta[i] = abs(delta[i])
                newx[i] = newx[i] + newx[i] - tmp

            fbefore = newf
            newf, newx, funevals = best_nearby(delta, newx, fbefore, nvars, f, funevals)

            if fbefore <= newf:
                break

            keep = False
            for i in range(nvars):
                if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                    keep = True
                    break

        if eps <= steplength and fbefore <= newf:
            steplength *= rho
            delta *= rho

    endpt = xbefore.copy()
    return iters, endpt


def tsp_descent_style_domain_optimization(
    initial_state: np.ndarray,
    energy_func: Callable[[np.ndarray], float],
    n_variations: int = 500,
    step_size: float = 0.05
) -> Tuple[np.ndarray, float]:
    """
    基于 TSP descent 思想的畴结构邻域搜索优化。
    对序参量场进行两种局部扰动:
    1. 转置型交换: 交换两个不相邻区域 (像素块) 的序参量
    2. 反转型反转: 反转一段序参量的符号/取向

    参数:
        initial_state: 初始序参量场 (展平)
        energy_func:   能量泛函
        n_variations:  扰动次数
        step_size:     扰动幅度

    返回:
        best_state: 最优状态
        best_energy: 最优能量
    """
    n = len(initial_state)
    state = initial_state.copy()
    best_energy = energy_func(state)

    rng = np.random.default_rng(seed=42)

    for _ in range(n_variations):
        # 变体 1: 高斯型局部扰动 (类似转置)
        idx = rng.integers(0, n)
        perturbation = state.copy()
        width = max(1, n // 20)
        start = max(0, idx - width)
        end = min(n, idx + width)
        perturbation[start:end] += rng.normal(0, step_size, end - start)

        e_new = energy_func(perturbation)
        if e_new < best_energy:
            state = perturbation
            best_energy = e_new

        # 变体 2: 反转型符号翻转 (类似 reversal)
        idx1, idx2 = sorted(rng.integers(0, n, 2))
        if idx2 - idx1 < 2:
            continue
        perturbation = state.copy()
        perturbation[idx1:idx2] = -perturbation[idx1:idx2]

        e_new = energy_func(perturbation)
        if e_new < best_energy:
            state = perturbation
            best_energy = e_new

    return state, best_energy


def optimize_domain_configuration(
    P0: np.ndarray, M0: np.ndarray,
    total_energy_func: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    综合优化接口: 先执行 Hooke-Jeeves 粗优化，再执行 TSP-descent 精细搜索。

    参数:
        P0, M0: 初始极化和磁化场（二维数组，会被展平）
        total_energy_func: E(P, M) -> float
        max_iter: 最大迭代次数

    返回:
        P_opt, M_opt, E_min
    """
    shape_P = P0.shape
    shape_M = M0.shape
    P_flat = P0.flatten().copy()
    M_flat = M0.flatten().copy()

    # 联合状态
    nP = len(P_flat)
    nM = len(M_flat)

    def joint_energy(z: np.ndarray) -> float:
        P = z[:nP].reshape(shape_P)
        M = z[nM:].reshape(shape_M)
        e = total_energy_func(P, M)
        if not np.isfinite(e):
            return 1e20
        return e

    z0 = np.concatenate([P_flat, M_flat])

    # Hooke-Jeeves 粗优化
    _, z_opt = hooke_jeeves(len(z0), z0, rho=0.7, eps=1e-5,
                            itermax=max(20, max_iter // 2),
                            f=joint_energy)

    # TSP-descent 精细搜索
    z_opt, E_min = tsp_descent_style_domain_optimization(
        z_opt, joint_energy, n_variations=max_iter * 2, step_size=0.02
    )

    P_opt = z_opt[:nP].reshape(shape_P)
    M_opt = z_opt[nM:].reshape(shape_M)
    return P_opt, M_opt, E_min
