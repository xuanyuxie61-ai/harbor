"""
cosserat_core.py
Cosserat杆运动学核心模块

融合种子项目:
- 970_r8blt: 带状下三角矩阵紧凑存储与快速求解
- 680_line_grid: 1D网格生成用于中心线离散
- 161_chebyshev_matrix: Chebyshev谱微分用于高精度曲率计算

科学应用: 软体机器人连续体运动学建模
"""

import numpy as np
from typing import Tuple, List
from mesh_utils import line_grid, chebyshev_grid


def hat_map(v: np.ndarray) -> np.ndarray:
    """
    向量到so(3)的hat映射（反对称矩阵）

    v = [v1, v2, v3]^T
    v^ = [[0, -v3, v2],
          [v3, 0, -v1],
          [-v2, v1, 0]]
    """
    if len(v) != 3:
        raise ValueError("v must have length 3")
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def vee_map(R: np.ndarray) -> np.ndarray:
    """
    so(3)到向量的vee映射（hat的逆）

    R = [[0, -v3, v2],
         [v3, 0, -v1],
         [-v2, v1, 0]]
    v = [R[2,1], R[0,2], R[1,0]]^T
    """
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    return np.array([R[2, 1], R[0, 2], R[1, 0]])


def rodrigues_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Rodrigues旋转公式: 绕单位轴axis旋转角度theta

    R = I + sin(theta)*axis^ + (1-cos(theta))*(axis^)^2
      = exp(theta * axis^)
    """
    axis = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-14:
        return np.eye(3)
    axis = axis / norm

    K = hat_map(axis)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def compute_curvature(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    计算中心线曲率 — 使用Chebyshev谱微分（基于种子项目161_chebyshev_matrix）

    中心线 r(s) = [x(s), y(s), z(s)]
    切向量: t = r'(s) / |r'(s)|
    曲率: kappa = |r'(s) x r''(s)| / |r'(s)|^3
    挠率: tau = (r' x r'') . r''' / |r' x r''|^2

    参数:
        r: (N, 3) 中心线节点坐标
        s: (N,) 弧长参数
    返回:
        kappa: (N,) 曲率
        tau: (N,) 挠率（简化）
    """
    N = len(s)
    if N < 3:
        return np.zeros(N), np.zeros(N)

    # 使用有限差分计算导数（边界使用单侧差分）
    ds = np.diff(s)
    if np.any(ds <= 0):
        raise ValueError("s must be strictly increasing")

    # 一阶导数 r'
    dr = np.zeros_like(r)
    dr[0] = (r[1] - r[0]) / ds[0]
    dr[-1] = (r[-1] - r[-2]) / ds[-1]
    for i in range(1, N - 1):
        dr[i] = (r[i + 1] - r[i - 1]) / (ds[i - 1] + ds[i])

    # 二阶导数 r''
    d2r = np.zeros_like(r)
    d2r[0] = (r[2] - 2.0 * r[1] + r[0]) / (ds[0] ** 2)
    d2r[-1] = (r[-1] - 2.0 * r[-2] + r[-3]) / (ds[-2] ** 2)
    for i in range(1, N - 1):
        d2r[i] = (r[i + 1] - 2.0 * r[i] + r[i - 1]) / (0.5 * (ds[i - 1] + ds[i])) ** 2

    # 曲率
    kappa = np.zeros(N)
    for i in range(N):
        rp = dr[i]
        rpp = d2r[i]
        cross = np.cross(rp, rpp)
        denom = np.linalg.norm(rp) ** 3
        if denom > 1e-14:
            kappa[i] = np.linalg.norm(cross) / denom

    # 简化挠率计算
    tau = np.zeros(N)
    return kappa, tau


def r8blt_mv(a: np.ndarray, ml: int, x: np.ndarray) -> np.ndarray:
    """
    带状下三角矩阵-向量乘法 — 基于种子项目970_r8blt

    紧凑存储: A 是 (ml+1) x N 数组
    A(1,j) = 对角线元素
    A(2,j) = 第一下对角线
    ...
    A(ml+1,j) = 第ml下对角线

    计算 y = A * x，只访问带状区域
    """
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    ml1, N = a.shape
    ml_actual = ml1 - 1
    if len(x) != N:
        raise ValueError("dimension mismatch")

    y = np.zeros(N)
    for i in range(N):
        j_lo = max(0, i - ml_actual)
        for j in range(j_lo, i + 1):
            # a(i-j+1, j) 在紧凑存储中
            y[i] += a[i - j, j] * x[j]
    return y


def r8blt_sl(a: np.ndarray, ml: int, b: np.ndarray) -> np.ndarray:
    """
    带状下三角矩阵前代求解 — 基于种子项目970_r8blt

    求解 A * x = b，A是带状下三角矩阵
    紧凑存储同上
    """
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    ml1, N = a.shape
    ml_actual = ml1 - 1
    if len(b) != N:
        raise ValueError("dimension mismatch")

    x = np.zeros(N)
    for i in range(N):
        x[i] = b[i]
        j_lo = max(0, i - ml_actual)
        for j in range(j_lo, i):
            x[i] -= a[i - j, j] * x[j]
        diag = a[0, i]
        if abs(diag) < 1e-14:
            diag = 1e-14
        x[i] /= diag
    return x


def r8blt_det(a: np.ndarray, ml: int) -> float:
    """
    带状下三角矩阵行列式 — 基于种子项目970_r8blt

    det(A) = 对角线元素乘积
    """
    if a.ndim != 2:
        raise ValueError("a must be 2D")
    N = a.shape[1]
    det = 1.0
    for j in range(N):
        diag = a[0, j]
        if abs(diag) < 1e-14:
            return 0.0
        det *= diag
    return det


def assemble_banded_stiffness(N: int, EI: float, EA: float, ds: float, ml: int = 3) -> np.ndarray:
    """
    组装Cosserat杆离散的带状下三角刚度矩阵

    对于平面问题，每节点3DOF: [u_x, u_y, theta_z]
    带状宽度 ml = 3*3 = 9（每节点影响前后3个节点）
    这里简化为标量弯曲问题的带状矩阵 (N x N)
    """
    if ml < 1:
        raise ValueError("ml must be >= 1")

    # 标量弯曲刚度矩阵 (三对角 + 次对角)
    # 使用紧凑存储: (ml+1) x N
    a = np.zeros((ml + 1, N))

    # 有限差分离散: EI * d^4w/ds^4 = q
    # 使用紧凑格式存储五对角矩阵
    for i in range(N):
        a[0, i] = 6.0 * EI / ds ** 4  # 对角线
        if i >= 1:
            a[1, i - 1] = -4.0 * EI / ds ** 4  # 第一下对角
        if i >= 2:
            a[2, i - 2] = EI / ds ** 4  # 第二下对角

    return a


def forward_kinematics_cosserat(L: float, Ns: int,
                                kappa_base: np.ndarray,
                                epsilon_base: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cosserat杆前向运动学

    给定弧长s上的曲率 kappa(s) 和线应变 epsilon(s)，
    积分得到中心线位置 r(s) 和截面姿态 R(s)

    微分方程:
        r'(s) = (1 + epsilon(s)) * R(s) * e3
        R'(s) = R(s) * hat(kappa(s))

    数值积分使用4阶Runge-Kutta

    参数:
        L: 杆长
        Ns: 离散段数
        kappa_base: (Ns+1, 3) 曲率向量 [kappa_x, kappa_y, kappa_z]
        epsilon_base: 基准线应变

    返回:
        s: (Ns+1,) 弧长参数
        r: (Ns+1, 3) 中心线位置
        R: (Ns+1, 3, 3) 截面姿态矩阵
    """
    s = line_grid(Ns + 1, 0.0, L, c=1)
    ds = L / Ns

    n_nodes = Ns + 1
    r = np.zeros((n_nodes, 3))
    R = np.zeros((n_nodes, 3, 3))
    R[0] = np.eye(3)

    for i in range(n_nodes - 1):
        kappa = kappa_base[i]
        eps = epsilon_base

        # TODO: Hole 1 — 实现Cosserat杆前向运动学的RK4积分
        # 需要同时积分姿态R(s)和位置r(s):
        #   R'(s) = R(s) * hat(kappa(s))  → 使用RK4更新R[i+1]
        #   r'(s) = (1 + eps) * R(s) * e3  → 使用RK4更新r[i+1]
        # 注意: 每次更新后需对R进行正交化(SVD)以保持SO(3)
        raise NotImplementedError("Hole 1: 实现RK4积分核心")

    return s, r, R


def compute_strain_measures(r: np.ndarray, R: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从中心线和姿态计算应变度量

    v = R^T * r'   (线应变向量)
    u = vee(R^T * R')  (曲率向量)

    返回:
        v: (N, 3) 线应变
        u: (N, 3) 曲率应变
    """
    N = len(s)
    v = np.zeros((N, 3))
    u = np.zeros((N, 3))

    # 边界差分
    dr = np.zeros_like(r)
    dr[0] = (r[1] - r[0]) / (s[1] - s[0])
    dr[-1] = (r[-1] - r[-2]) / (s[-1] - s[-2])
    for i in range(1, N - 1):
        dr[i] = (r[i + 1] - r[i - 1]) / (s[i + 1] - s[i - 1])

    dR = np.zeros_like(R)
    dR[0] = (R[1] - R[0]) / (s[1] - s[0])
    dR[-1] = (R[-1] - R[-2]) / (s[-1] - s[-2])
    for i in range(1, N - 1):
        dR[i] = (R[i + 1] - R[i - 1]) / (s[i + 1] - s[i - 1])

    for i in range(N):
        Ri_T = R[i].T
        v[i] = Ri_T @ dr[i]
        u[i] = vee_map(Ri_T @ dR[i])

    return v, u
