# -*- coding: utf-8 -*-
"""
plasma_diagnostics.py
=====================
等离子体相空间诊断: QC 过滤 + 最近邻沉积 + 扩散率估计.

QC 过滤 (来自 1234_playingwithgithub24_HackBio):
------------------------------------------------
类比单细胞 RNA-seq 的质量控制:
- 过滤低质量细胞 → 过滤低权重电子宏粒子
- MT% 过滤 → 高能量展 outlier 过滤
- 基因计数过滤 → 能量范围过滤
- 归一化 → 能量分布归一化
- 高变模式选择 → 相空间关键区域识别

最近邻沉积 (来自 192_closest_point_brute):
-------------------------------------------
将电子能量沉积映射到最近等离子体网格单元:
    S_j = Σ_{e∈V_j} w_e · (-dE/dx)_e / V_j
    V_j = {x : ||x - x_j|| ≤ ||x - x_k|| ∀k≠j}

扩散率估计 (来自 1158_shoh5301_Quick-MSD-Diffusivity-Calculator):
----------------------------------------------------------------
从电子均方位移 (MSD) 估计有效扩散系数:
    MSD(t) = <(x(t) - x(0))^2> = 2d · D · t  (Fick 扩散)
    D = MSD(t) / (2d · t)
    其中 d 为空间维数.

核心来源 (种子项目映射):
- 1234_HackBio: QC 过滤流程
- 192_closest_point_brute: 最近邻映射
- 1158_MSD: 扩散率计算
"""

import math


# ============================================================
# 相空间 QC 过滤 (来自 1234_playingwithgithub24)
# ============================================================
def plasma_qc_filter(particles, min_energy_ev=1.0e3, max_energy_ev=1.0e8,
                     max_divergence_rad=None, sigma_clip_factor=3.0):
    """
    等离子体相空间 QC 过滤.

    类比单细胞 RNA-seq QC (来自 1234):
    1. min_genes → min_energy: 去除低能电子
    2. max_genes → max_energy: 去除超高能 outlier
    3. MT% → divergence: 去除大角度 outlier
    4. 高变基因 → sigma clipping: 统计 outlier 去除

    参数:
        particles         : 粒子列表 (dict)
        min_energy_ev     : 最低能量阈值
        max_energy_ev     : 最高能量阈值
        max_divergence_rad: 最大发散角
        sigma_clip_factor : sigma clipping 因子
    返回:
        dict: 'passed', 'failed', 'stats'
    """
    if not particles:
        return {"passed": [], "failed": [], "stats": {}}

    passed = []
    failed = []
    reasons = {"low_energy": 0, "high_energy": 0, "large_angle": 0, "outlier": 0}

    # 第一轮: 基本阈值过滤
    for p in particles:
        E = p["energy_ev"]
        if E < min_energy_ev:
            failed.append((p, "low_energy"))
            reasons["low_energy"] += 1
            continue
        if E > max_energy_ev:
            failed.append((p, "high_energy"))
            reasons["high_energy"] += 1
            continue

        if max_divergence_rad is not None:
            # 计算发散角
            v_mag = math.sqrt(p["vx"] ** 2 + p["vy"] ** 2 + p["vz"] ** 2)
            if v_mag > 0:
                cos_theta = p["vz"] / v_mag
                cos_theta = max(-1.0, min(1.0, cos_theta))
                theta = math.acos(cos_theta)
                if theta > max_divergence_rad:
                    failed.append((p, "large_angle"))
                    reasons["large_angle"] += 1
                    continue

        passed.append(p)

    # 第二轮: sigma clipping on energy
    if len(passed) > 3:
        energies = [p["energy_ev"] for p in passed]
        mean_E = sum(energies) / len(energies)
        std_E = math.sqrt(sum((E - mean_E) ** 2 for E in energies)
                          / (len(energies) - 1))
        if std_E > 0:
            final_passed = []
            for p in passed:
                z = abs(p["energy_ev"] - mean_E) / std_E
                if z > sigma_clip_factor:
                    failed.append((p, "outlier"))
                    reasons["outlier"] += 1
                else:
                    final_passed.append(p)
            passed = final_passed

    # 归一化 (类比 sc.pp.normalize_total + log1p)
    if passed:
        total_w = sum(p["weight_j"] for p in passed)
        if total_w > 0:
            for p in passed:
                p["weight_normalized"] = p["weight_j"] / total_w
        # log 变换 (类比 sc.pp.log1p)
        for p in passed:
            p["log_energy"] = math.log1p(p["energy_ev"] / 1.0e3)

    stats = {
        "n_initial": len(particles),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "pass_rate": len(passed) / max(len(particles), 1),
        "reasons": reasons,
    }

    return {"passed": passed, "failed": failed, "stats": stats}


# ============================================================
# 最近邻沉积映射 (来自 192_closest_point_brute)
# ============================================================
def closest_point_brute_2d(px, py, grid_x, grid_y):
    """
    暴力最近邻搜索 (来自 192_closest_point_brute).

    参数:
        px, py : 查询点坐标
        grid_x : x 方向网格坐标
        grid_y : y 方向网格坐标
    返回:
        (i, j): 最近网格点索引, dist: 距离
    """
    min_dist_sq = float('inf')
    near_i, near_j = 0, 0

    for j in range(len(grid_y)):
        for i in range(len(grid_x)):
            dist_sq = (px - grid_x[i]) ** 2 + (py - grid_y[j]) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                near_i = i
                near_j = j

    return near_i, near_j, math.sqrt(min_dist_sq)


def closest_point_vectorized_2d(points, grid_x, grid_y):
    """
    向量化最近邻 (来自 192_closest_point_brute/closest_point_brute2).

    参数:
        points : list of (x, y)
        grid_x : x 网格
        grid_y : y 网格
    返回:
        list of (i, j, dist)
    """
    results = []
    for px, py in points:
        # 向量化距离计算
        min_idx = -1
        min_dist = float('inf')
        idx = 0
        for j in range(len(grid_y)):
            for i in range(len(grid_x)):
                dist = (px - grid_x[i]) ** 2 + (py - grid_y[j]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
                    idx += 1
                else:
                    idx += 1
        near_j = min_idx // len(grid_x)
        near_i = min_idx % len(grid_x)
        results.append((near_i, near_j, math.sqrt(min_dist)))
    return results


def deposit_beam_to_grid(particles, grid_x, grid_y, cell_volumes=None):
    """
    将电子束沉积到网格 (最近邻映射).

    参数:
        particles     : 过滤后粒子列表
        grid_x, grid_y: 网格坐标
        cell_volumes  : 单元体积 (可选)
    返回:
        source_matrix: ny × nx 沉积源项
    """
    nx = len(grid_x)
    ny = len(grid_y)
    source = [[0.0] * nx for _ in range(ny)]

    # 默认单元体积
    if cell_volumes is None:
        dx = grid_x[1] - grid_x[0] if nx > 1 else 1.0
        dy = grid_y[1] - grid_y[0] if ny > 1 else 1.0
        dV = dx * dy
    else:
        dV = sum(cell_volumes) / max(len(cell_volumes), 1)

    for p in particles:
        i, j, dist = closest_point_brute_2d(
            p["x"], p["y"], grid_x, grid_y)
        w = p.get("weight_normalized", p["weight_j"])
        source[j][i] += w / max(dV, 1.0e-30)

    return source


# ============================================================
# MSD 扩散率估计 (来自 1158_shoh5301)
# ============================================================
def compute_msd_1d(trajectories, max_lag=None):
    """
    均方位移 (MSD) 计算 (来自 1158_MSD).

    MSD(τ) = <(x(t+τ) - x(t))^2>_t

    参数:
        trajectories: list of list, 每条轨迹的 x(t) 序列
        max_lag     : 最大滞后步数
    返回:
        lags: 滞后时间列表
        msd : MSD 值列表
    """
    if not trajectories:
        return [], []

    n_min = min(len(traj) for traj in trajectories)
    if max_lag is None:
        max_lag = n_min // 2

    msd = [0.0] * max_lag
    count = [0] * max_lag

    for traj in trajectories:
        n = len(traj)
        for lag in range(1, max_lag):
            for t in range(n - lag):
                dx = traj[t + lag] - traj[t]
                msd[lag] += dx * dx
                count[lag] += 1

    # 归一化
    lags = list(range(max_lag))
    for k in range(1, max_lag):
        if count[k] > 0:
            msd[k] /= count[k]

    return lags, msd


def estimate_diffusion_coefficient(lags, msd, dt, dimension=1, fit_range=None):
    """
    从 MSD 估计扩散系数 (来自 1158_load.py 的 Slope 计算).

    对正常扩散:
        MSD(t) = 2d · D · t
        D = MSD / (2d · t)

    线性拟合 MSD ~ slope · t, 则 D = slope / (2d)

    参数:
        lags     : 滞后步数
        msd      : MSD 值
        dt       : 时间步长
        dimension: 空间维数
        fit_range: 拟合范围 (lag_min, lag_max)
    返回:
        dict: 'D', 'slope', 'r_squared'
    """
    if len(lags) < 2 or len(msd) < 2:
        return {"D": 0.0, "slope": 0.0, "r_squared": 0.0}

    if fit_range is None:
        i_min = max(1, len(lags) // 10)
        i_max = min(len(lags), len(lags) // 2)
    else:
        i_min, i_max = fit_range

    if i_max <= i_min:
        i_min = 1
        i_max = len(lags)

    # 最小二乘拟合: MSD = slope · t
    # t = lag · dt
    n_fit = 0
    S_tt = 0.0
    S_tm = 0.0
    S_mm = 0.0
    S_t = 0.0
    S_m = 0.0

    for k in range(i_min, i_max):
        t = lags[k] * dt
        m = msd[k]
        S_tt += t * t
        S_tm += t * m
        S_mm += m * m
        S_t += t
        S_m += m
        n_fit += 1

    if n_fit < 2:
        return {"D": 0.0, "slope": 0.0, "r_squared": 0.0}

    denom = n_fit * S_tt - S_t * S_t
    if abs(denom) < 1.0e-300:
        return {"D": 0.0, "slope": 0.0, "r_squared": 0.0}

    slope = (n_fit * S_tm - S_t * S_m) / denom
    intercept = (S_m - slope * S_t) / n_fit

    # R²
    ss_res = sum((msd[k] - (slope * lags[k] * dt + intercept)) ** 2
                 for k in range(i_min, i_max))
    mean_m = S_m / n_fit
    ss_tot = sum((msd[k] - mean_m) ** 2 for k in range(i_min, i_max))
    r_squared = 1.0 - ss_res / max(ss_tot, 1.0e-300)

    D = slope / (2.0 * dimension)

    return {
        "D": D,
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "n_fit": n_fit,
    }


def print_diagnostics_summary(qc_result, deposit_info, diffusion_result):
    """打印诊断摘要."""
    print("\n" + "=" * 72)
    print("等离子体诊断 (QC 过滤 + 沉积 + 扩散率)")
    print("=" * 72)

    if qc_result:
        stats = qc_result["stats"]
        print("  QC 过滤统计:")
        print("    初始粒子数: {}".format(stats["n_initial"]))
        print("    通过过滤  : {} ({:.1f}%)".format(
            stats["n_passed"], stats["pass_rate"] * 100))
        print("    过滤原因  : {}".format(stats["reasons"]))

    if deposit_info:
        nx = deposit_info.get("nx", 0)
        ny = deposit_info.get("ny", 0)
        source = deposit_info.get("source", [])
        if source:
            max_dep = max(max(row) for row in source)
            total_dep = sum(sum(row) for row in source)
            print("  沉积映射:")
            print("    网格尺寸  : {} × {}".format(nx, ny))
            print("    最大沉积  : {:.4e}".format(max_dep))
            print("    总沉积    : {:.4e}".format(total_dep))

    if diffusion_result:
        print("  扩散率估计 (MSD 方法):")
        print("    D_eff     : {:.4e} m^2/s".format(diffusion_result["D"]))
        print("    MSD 斜率  : {:.4e}".format(diffusion_result["slope"]))
        print("    R^2       : {:.6f}".format(diffusion_result["r_squared"]))

    print("=" * 72)
