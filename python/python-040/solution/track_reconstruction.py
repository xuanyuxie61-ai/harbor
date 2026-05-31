#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_reconstruction.py
粒子径迹重建与优化模块

融合原项目:
- 1366_tsp_moler: 旅行商问题启发式求解（探测器层间径迹关联）
- 518_hermite_cubic: Hermite 三次样条插值（径迹平滑与动量提取）
- 385_fem1d_approximate: 有限元近似（能量损失分布拟合）

在BSM信号分析中用于:
- 将对撞机各探测器层中的孤立击中点关联为完整径迹
- 使用 Hermite 样条重建连续径迹并提取曲率（动量）
- 有限元方法拟合径迹的 dE/dx 分布进行粒子鉴别
"""

import numpy as np
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# TSP 启发式径迹关联
# ---------------------------------------------------------------------------

def path_length(path: np.ndarray, dist_matrix: np.ndarray) -> float:
    """
    计算给定路径的总长度（闭合回路）。

    Parameters
    ----------
    path : np.ndarray
        城市访问顺序（0-based 索引）
    dist_matrix : np.ndarray
        距离矩阵 D(i,j)

    Returns
    -------
    float
        总路径长度
    """
    n = path.size
    total = 0.0
    for i in range(n - 1):
        total += dist_matrix[path[i], path[i + 1]]
    total += dist_matrix[path[-1], path[0]]
    return total


def tsp_track_association(
    hits_per_layer: List[np.ndarray],
    max_iter: int = 5000
) -> Tuple[np.ndarray, float]:
    """
    使用 TSP 启发式算法优化探测器层间击中点关联。

    将对撞机径迹探测器（如硅微条 tracker）的各层视为图中的节点层，
    每层可能有多个候选击中点。将候选击中点编号为 "城市"，
    层间距离定义为击中点间的欧氏距离加惩罚项：
        D(i,j) = ||r_i - r_j|| + λ |z_i - z_j|

    其中 z_i 是层坐标，惩罚项确保路径按层序前进。

    优化目标: 寻找最短的穿越所有层的击中点序列，
    对应最可能的粒子径迹。

    Parameters
    ----------
    hits_per_layer : List[np.ndarray]
        每层击中点坐标列表，每个元素形状 (n_hits, 2) 或 (n_hits, 3)
    max_iter : int
        最大迭代次数

    Returns
    -------
    best_path : np.ndarray
        最优击中点索引序列
    min_length : float
        最优路径长度
    """
    if len(hits_per_layer) < 2:
        raise ValueError("至少需要两层击中点")

    # 展平所有击中点并记录层信息
    all_hits = []
    layer_ids = []
    for lid, hits in enumerate(hits_per_layer):
        hits = np.atleast_2d(hits)
        for h in hits:
            all_hits.append(h)
            layer_ids.append(lid)

    all_hits = np.array(all_hits)
    n = all_hits.shape[0]
    layer_ids = np.array(layer_ids)

    # 构建距离矩阵
    dist = np.zeros((n, n))
    lambda_penalty = 10.0 * np.max(np.std(all_hits, axis=0)) if n > 1 else 1.0

    for i in range(n):
        for j in range(i + 1, n):
            d_spatial = np.linalg.norm(all_hits[i] - all_hits[j])
            # 层序惩罚：不允许回到已访问层（简化处理）
            d_layer = abs(layer_ids[i] - layer_ids[j])
            if d_layer == 0:
                d_spatial += 1e6  # 同层惩罚
            d_total = d_spatial + lambda_penalty * max(0, 2 - d_layer)
            dist[i, j] = d_total
            dist[j, i] = d_total

    # TSP 启发式求解（2-opt + 单点插入）
    p = np.random.permutation(n)
    best_len = path_length(p, dist)
    best_path = p.copy()

    for _ in range(max_iter):
        # 2-opt: 反转子段
        pt1 = np.random.randint(n)
        pt2 = np.random.randint(n)
        lo, hi = min(pt1, pt2), max(pt1, pt2)
        q = np.arange(n)
        q[lo:hi+1] = q[lo:hi+1][::-1]
        p_new = p[q]
        new_len = path_length(p_new, dist)
        if new_len < best_len:
            p = p_new
            best_len = new_len
            best_path = p.copy()

        # 单点插入
        pt1 = np.random.randint(n)
        pt2 = np.random.randint(n - 1) if n > 1 else 0
        q = np.delete(np.arange(n), pt1)
        q = np.insert(q, pt2, pt1)
        p_new = p[q]
        new_len = path_length(p_new, dist)
        if new_len < best_len:
            p = p_new
            best_len = new_len
            best_path = p.copy()

    return best_path, best_len


# ---------------------------------------------------------------------------
# Hermite 三次样条径迹平滑
# ---------------------------------------------------------------------------

def hermite_cubic_value(
    x1: float, f1: float, d1: float,
    x2: float, f2: float, d2: float,
    n: int, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    在区间 [x1, x2] 上求 Hermite 三次插值及其导数。

    Hermite 基函数:
        H_00(t) = 2t³ - 3t² + 1
        H_10(t) = t³ - 2t² + t
        H_01(t) = -2t³ + 3t²
        H_11(t) = t³ - t²

    其中 t = (x - x1) / (x2 - x1), h = x2 - x1

    插值函数:
        p(x) = f1 * H_00(t) + h * d1 * H_10(t)
             + f2 * H_01(t) + h * d2 * H_11(t)

    Parameters
    ----------
    x1, x2 : float
        区间端点
    f1, f2 : float
        端点函数值
    d1, d2 : float
        端点导数值
    n : int
        求值点数
    x : np.ndarray
        求值点坐标（应在 [x1, x2] 附近）

    Returns
    -------
    f, d, s, t : np.ndarray
        函数值、一阶导数、二阶导数、三阶导数
    """
    x = np.atleast_1d(x)
    h = x2 - x1
    if abs(h) < 1e-14:
        return np.full_like(x, f1), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    t = (x - x1) / h
    # 将 t 限制在 [0, 1] 外推
    t = np.clip(t, -0.5, 1.5)

    # Hermite 基函数
    h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
    h10 = t ** 3 - 2.0 * t ** 2 + t
    h01 = -2.0 * t ** 3 + 3.0 * t ** 2
    h11 = t ** 3 - t ** 2

    # 导数基函数（对 t 的导数）
    dh00 = 6.0 * t ** 2 - 6.0 * t
    dh10 = 3.0 * t ** 2 - 4.0 * t + 1.0
    dh01 = -6.0 * t ** 2 + 6.0 * t
    dh11 = 3.0 * t ** 2 - 2.0 * t

    # 二阶导数基函数
    ddh00 = 12.0 * t - 6.0
    ddh10 = 6.0 * t - 4.0
    ddh01 = -12.0 * t + 6.0
    ddh11 = 6.0 * t - 2.0

    # 三阶导数基函数
    dddh00 = 12.0 * np.ones_like(t)
    dddh10 = 6.0 * np.ones_like(t)
    dddh01 = -12.0 * np.ones_like(t)
    dddh11 = 6.0 * np.ones_like(t)

    f = f1 * h00 + h * d1 * h10 + f2 * h01 + h * d2 * h11
    d = (f1 * dh00 + h * d1 * dh10 + f2 * dh01 + h * d2 * dh11) / h
    s = (f1 * ddh00 + h * d1 * ddh10 + f2 * ddh01 + h * d2 * ddh11) / (h ** 2)
    t3 = (f1 * dddh00 + h * d1 * dddh10 + f2 * dddh01 + h * d2 * dddh11) / (h ** 3)

    return f, d, s, t3


def hermite_cubic_spline(
    xn: np.ndarray,
    fn: np.ndarray,
    dn: np.ndarray,
    x_eval: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hermite 三次样条：分段 Hermite 插值。

    对给定数据点 (xn, fn, dn) 构造连续可微的分段三次多项式，
    用于平滑重建粒子径迹。

    Parameters
    ----------
    xn : np.ndarray
        节点横坐标（必须严格递增）
    fn : np.ndarray
        节点函数值
    dn : np.ndarray
        节点导数值
    x_eval : np.ndarray
        求值点

    Returns
    -------
    f, d, s, t : np.ndarray
        函数值及各阶导数在 x_eval 处的值
    """
    xn = np.asarray(xn).ravel()
    fn = np.asarray(fn).ravel()
    dn = np.asarray(dn).ravel()
    x_eval = np.asarray(x_eval).ravel()

    nn = xn.size
    if nn < 2:
        raise ValueError("至少需要两个节点")
    if not np.all(np.diff(xn) > 0):
        raise ValueError("xn 必须严格递增")

    n_eval = x_eval.size
    f_out = np.zeros(n_eval)
    d_out = np.zeros(n_eval)
    s_out = np.zeros(n_eval)
    t_out = np.zeros(n_eval)

    # 对每个求值点定位所在区间
    for i in range(n_eval):
        xv = x_eval[i]
        # 边界外推
        if xv <= xn[0]:
            idx = 0
        elif xv >= xn[-1]:
            idx = nn - 2
        else:
            idx = np.searchsorted(xn, xv, side='right') - 1
            idx = max(0, min(idx, nn - 2))

        ff, dd, ss, tt = hermite_cubic_value(
            xn[idx], fn[idx], dn[idx],
            xn[idx + 1], fn[idx + 1], dn[idx + 1],
            1, np.array([xv])
        )
        f_out[i] = ff[0]
        d_out[i] = dd[0]
        s_out[i] = ss[0]
        t_out[i] = tt[0]

    return f_out, d_out, s_out, t_out


def estimate_momentum_from_curvature(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: Optional[np.ndarray] = None,
    magnetic_field: float = 3.8  # Tesla
) -> float:
    """
    从径迹曲率估算粒子动量。

    在均匀磁场 B 中，带电粒子的横向动量与曲率半径 R 的关系：
        p_T [GeV] = 0.3 × B [T] × R [m]

    对于三维径迹，使用最小二乘法拟合圆以提取 R：
        (x - x_c)^2 + (y - y_c)^2 = R^2

    线性化方法: 令 u = x - x̄, v = y - ȳ
        u_c = (S_uv S_vv - S_uu S_uv) / (2 (S_uv^2 - S_uu S_vv))
        v_c = (S_uu S_uv - S_uv S_vv) / (2 (S_uv^2 - S_uu S_vv))
        R = sqrt(u_c^2 + v_c^2 + (S_uu + S_vv) / n)

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        径迹在 xy 平面的坐标 [m]
    z_coords : np.ndarray or None
        z 坐标（用于计算纵向动量）
    magnetic_field : float
        磁场强度 [Tesla]

    Returns
    -------
    float
        估算的横向动量 p_T [GeV]
    """
    n = x_coords.size
    if n < 3:
        return 0.0

    x = np.asarray(x_coords)
    y = np.asarray(y_coords)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    u = x - x_mean
    v = y - y_mean

    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suv = np.sum(u * v)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)
    Suvv = np.sum(u * v ** 2)
    Svuu = np.sum(v * u ** 2)

    denom = 2.0 * (Suu * Svv - Suv ** 2)
    if abs(denom) < 1e-14:
        return 0.0

    uc = (Svv * (Suuu + Suvv) - Suv * (Svvv + Svuu)) / denom
    vc = (Suu * (Svvv + Svuu) - Suv * (Suuu + Suvv)) / denom

    R = np.sqrt(uc ** 2 + vc ** 2 + (Suu + Svv) / n)

    # 防止 R 过小导致动量发散
    R = max(R, 1e-4)

    p_T = 0.3 * magnetic_field * R

    # 若提供 z 坐标，计算总动量
    if z_coords is not None and z_coords.size >= 2:
        dz = z_coords[-1] - z_coords[0]
        dr = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
        if abs(dr) > 1e-10:
            tan_lambda = dz / dr
            p_total = p_T * np.sqrt(1.0 + tan_lambda ** 2)
            return p_total

    return p_T


# ---------------------------------------------------------------------------
# 有限元 dE/dx 拟合（基于 fem1d_approximate 思想）
# ---------------------------------------------------------------------------

def fem1d_track_fit(
    track_length: np.ndarray,
    energy_deposit: np.ndarray,
    n_nodes: int = 20,
    weight_a: float = 1.0,
    weight_d: float = 0.1,
    weight_b: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用一维有限元方法拟合径迹的能量损失分布 dE/dx。

    最小化加权泛函:
        J = w_a Σ(FEM(x_i) - E_i)^2
          + w_d Σ(FEM''(x_j))^2
          + w_b [FEM(0)^2 + FEM(L)^2]

    第一项: 数据近似条件
    第二项: 光滑性惩罚（二阶导数最小化）
    第三项: 边界条件（端点趋于零）

    Parameters
    ----------
    track_length : np.ndarray
        沿径迹的采样位置 [m]
    energy_deposit : np.ndarray
        对应位置的能量沉积 [GeV/m]
    n_nodes : int
        有限元节点数
    weight_a, weight_d, weight_b : float
        权重系数

    Returns
    -------
    node_x : np.ndarray
        有限元节点坐标
    node_c : np.ndarray
        有限元系数（即节点处的 dE/dx 值）
    """
    track_length = np.asarray(track_length).ravel()
    energy_deposit = np.asarray(energy_deposit).ravel()

    if track_length.size != energy_deposit.size:
        raise ValueError("track_length 与 energy_deposit 长度不匹配")

    n_data = track_length.size
    x_min = np.min(track_length)
    x_max = np.max(track_length)

    # 均匀节点
    node_x = np.linspace(x_min, x_max, n_nodes)

    # 数据点所在单元
    data_l = np.searchsorted(node_x, track_length, side='right') - 1
    data_l = np.clip(data_l, 0, n_nodes - 2)
    data_r = data_l + 1

    # 统计有效方程数
    is_legal = (track_length >= x_min) & (track_length <= x_max)
    eq_num = int(np.sum(is_legal)) + (n_nodes - 2) + 2

    A = np.zeros((eq_num, n_nodes))
    b_vec = np.zeros(eq_num)

    eq_i = 0
    # 近似条件
    for i in range(n_data):
        if is_legal[i]:
            l = data_l[i]
            r = data_r[i]
            h = node_x[r] - node_x[l]
            if abs(h) < 1e-14:
                A[eq_i, l] = weight_a
            else:
                A[eq_i, l] = weight_a * (node_x[r] - track_length[i]) / h
                A[eq_i, r] = weight_a * (track_length[i] - node_x[l]) / h
            b_vec[eq_i] = weight_a * energy_deposit[i]
            eq_i += 1

    # 二阶导数光滑条件（有限差分）
    for i in range(1, n_nodes - 1):
        h_left = node_x[i] - node_x[i - 1]
        h_right = node_x[i + 1] - node_x[i]
        if h_left > 1e-14 and h_right > 1e-14:
            A[eq_i, i - 1] = weight_d / h_left
            A[eq_i, i] = -weight_d / h_left - weight_d / h_right
            A[eq_i, i + 1] = weight_d / h_right
            eq_i += 1

    # 边界条件
    A[eq_i, 0] = weight_b
    b_vec[eq_i] = 0.0
    eq_i += 1

    A[eq_i, n_nodes - 1] = weight_b
    b_vec[eq_i] = 0.0
    eq_i += 1

    # 最小二乘求解（截断到实际方程数）
    A = A[:eq_i, :]
    b_vec = b_vec[:eq_i]

    # 使用 numpy 最小二乘求解
    node_c, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

    return node_x, node_c


def particle_id_from_dedx(
    dedx_samples: np.ndarray,
    momentum: float
) -> str:
    """
    基于 dE/dx 与动量的关系进行粒子鉴别。

    Bethe-Bloch 公式（简化版）:
        ⟨dE/dx⟩ ∝ (z^2 / β^2) [ln(2 m_e c^2 β^2 γ^2 / I) - β^2]

    对于相对论性粒子 (γ >> 1):
        β ≈ 1,  dE/dx ∝ ln(γ) ≈ ln(p / m)

    Parameters
    ----------
    dedx_samples : np.ndarray
        dE/dx 样本 [MeV cm^2 / g]
    momentum : float
        粒子动量 [GeV]

    Returns
    -------
    str
        粒子类型: 'electron', 'muon', 'pion', 'proton', 'unknown'
    """
    mean_dedx = np.median(dedx_samples)
    if mean_dedx <= 0.0 or momentum <= 0.0:
        return 'unknown'

    # 简化鉴别：利用 dE/dx 的最小电离区（MIP）特性
    # 电子: dE/dx 高（辐射损失），βγ 大
    # 缪子: MIP 平台 (~1.5 MeV cm^2/g)
    # 强子: 在相同动量下 βγ 较小，dE/dx 偏高

    log_mom = np.log10(momentum)
    log_dedx = np.log10(mean_dedx)

    # 简化的判别边界（基于 PID 曲线）
    if log_dedx > 0.8 - 0.3 * log_mom:
        return 'electron'
    elif abs(log_dedx - 0.2) < 0.3:
        return 'muon'
    elif log_dedx > 0.4 and momentum < 2.0:
        return 'pion'
    else:
        return 'muon'  # 默认返回缪子（BSM Z' → μ⁺μ⁻ 最灵敏）
