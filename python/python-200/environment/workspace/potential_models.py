"""
potential_models.py
===================
分子间势能模型与自动微分兼容的势能函数。

本模块实现多尺度势能描述，核心包含：
1. Lennard-Jones 12-6 势：描述范德华相互作用
2. 高斯修正势：描述局域应变场的非对称响应
3. 组合势能面：支持自动微分精确计算力与Hessian

核心物理公式
------------
Lennard-Jones 势（无量纲形式）：

    V_LJ(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]

其导数（力）：

    dV_LJ/dr = 4ε · (-12σ^12/r^13 + 6σ^6/r^7)
             = 24ε/σ · [ -2(σ/r)^13 + (σ/r)^7 ]

高斯型修正势（用于描述各向异性局域场）：

    V_G(r; μ, Σ) = A · exp[ -½ (r-μ)^T Σ^{-1} (r-μ) ]

其中 Σ 为对称正定协方差矩阵，A 为振幅。

总势能（对 N 粒子体系）：

    V_total = Σ_{i<j} V_LJ(r_{ij}) + Σ_i V_G(‖r_i - r_c‖)

通过自动微分，力的计算无需手动求导：

    F_i = -∇_{r_i} V_total = -∂V_total/∂r_i

Hessian 矩阵（用于晶格动力学和弹性常数）：

    H_{ij}^{αβ} = ∂²V_total / ∂r_i^α ∂r_j^β
"""

import numpy as np
from typing import Union, List
from autodiff_core import DualScalar, HyperDualScalar
from autodiff_core import dual_exp, dual_sqrt, dual_min, dual_abs
from autodiff_core import hdual_exp, hdual_sqrt


def lennard_jones_potential(r: float, epsilon: float = 1.0,
                            sigma: float = 1.0) -> float:
    """
    标准 Lennard-Jones 12-6 势（标量版本）。
    
    V_LJ(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]
    
    参数:
        r: 粒子间距（要求 r > 0）
        epsilon: 势阱深度
        sigma:   有效直径
    
    返回:
        势能值
    """
    if r <= 0:
        raise ValueError("LJ potential requires r > 0")
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return 4.0 * epsilon * (sr12 - sr6)


def lennard_jones_force(r: float, epsilon: float = 1.0,
                        sigma: float = 1.0) -> float:
    """
    LJ 势对 r 的解析导数（力的大小）。
    
    F_r = -dV/dr = 24ε/σ [ 2(σ/r)^13 - (σ/r)^7 ]
    """
    # TODO: 实现 Lennard-Jones 解析力公式
    # 提示: 基于 V_LJ(r) = 4ε[(σ/r)^12 - (σ/r)^6] 求导
    raise NotImplementedError("Hole_1: 请补全 lennard_jones_force 的解析导数实现")


def lennard_jones_dual(r: DualScalar, epsilon: float = 1.0,
                       sigma: float = 1.0) -> DualScalar:
    """
    支持自动微分的 Lennard-Jones 势（Dual Number 版本）。
    
    输入 dual number r = r0 + ε·r'，输出 dual number V = V(r0) + ε·V'(r0)·r'。
    
    此版本在 r 趋近于 0 时自动截断，避免数值爆炸：
        r_eff = max(r, 0.8σ)
    """
    r_eff = dual_min(r, DualScalar(0.8 * sigma, 0.0))
    # 实际上我们想避免 r 太小，所以取 max
    # dual_min 语义需要反过来：当 r < 0.8σ 时取 0.8σ
    if r.val < 0.8 * sigma:
        r_eff = DualScalar(0.8 * sigma, 0.0)
    else:
        r_eff = r

    sr = DualScalar(sigma, 0.0) / r_eff
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return DualScalar(4.0 * epsilon, 0.0) * (sr12 - sr6)


def lennard_jones_hyperdual(r: HyperDualScalar, epsilon: float = 1.0,
                            sigma: float = 1.0) -> HyperDualScalar:
    """
    支持二阶自动微分的 LJ 势（Hyper-Dual 版本）。
    
    用于精确计算 Hessian 矩阵的对角元信息。
    """
    if r.f0 < 0.8 * sigma:
        r_eff = HyperDualScalar(0.8 * sigma, 0.0, 0.0, 0.0)
    else:
        r_eff = r
    sr = HyperDualScalar(sigma, 0.0, 0.0, 0.0) / r_eff
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return HyperDualScalar(4.0 * epsilon, 0.0, 0.0, 0.0) * (sr12 - sr6)


def gaussian_potential_2d(x: float, y: float,
                          xmu: float = 0.0, ymu: float = 0.0,
                          xsigma: float = 1.0, ysigma: float = 1.0,
                          A: float = 1.0,
                          corr_matrix: np.ndarray = None) -> float:
    """
    二维各向异性高斯势场。
    
    数学公式：
        V_G(x,y) = A · exp( -½ v^T · C · v )
    
    其中 v = [ (x-xmu)/xsigma, (y-ymu)/ysigma ]^T，
    C 为 2×2 相关矩阵（默认单位矩阵）。
    
    参数:
        x, y: 空间坐标
        xmu, ymu: 高斯中心
        xsigma, ysigma: 标准差
        A: 振幅
        corr_matrix: 2×2 相关矩阵（SPD），控制各向异性
    """
    vx = (x - xmu) / xsigma
    vy = (y - ymu) / ysigma
    v = np.array([vx, vy])
    if corr_matrix is None:
        C = np.eye(2)
    else:
        C = np.asarray(corr_matrix)
        # 保证正定性：做特征值截断
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    quad = float(v @ C @ v)
    return A * np.exp(-0.5 * quad)


def gaussian_potential_dual(x: DualScalar, y: DualScalar,
                            xmu: float = 0.0, ymu: float = 0.0,
                            xsigma: float = 1.0, ysigma: float = 1.0,
                            A: float = 1.0,
                            corr_matrix: np.ndarray = None) -> DualScalar:
    """
    支持自动微分的二维高斯势场（Dual Number 版本）。
    """
    vx = (x - DualScalar(xmu, 0.0)) / DualScalar(xsigma, 0.0)
    vy = (y - DualScalar(ymu, 0.0)) / DualScalar(ysigma, 0.0)
    if corr_matrix is None:
        quad = vx * vx + vy * vy
    else:
        C = np.asarray(corr_matrix)
        # 对 dual number，quad = C[0,0]*vx^2 + 2*C[0,1]*vx*vy + C[1,1]*vy^2
        quad = (DualScalar(C[0, 0], 0.0) * vx * vx
                + DualScalar(2.0 * C[0, 1], 0.0) * vx * vy
                + DualScalar(C[1, 1], 0.0) * vy * vy)
    return DualScalar(A, 0.0) * dual_exp(DualScalar(-0.5, 0.0) * quad)


def total_potential_lj(positions: np.ndarray,
                       epsilon: float = 1.0,
                       sigma: float = 1.0,
                       rcut: float = 2.5,
                       box_size: float = None) -> float:
    """
    计算 N 粒子体系的总 Lennard-Jones 势能。
    
    V_total = Σ_{i<j} V_LJ(r_{ij}) · S(r_{ij})
    
    其中 S(r) 为平滑截断函数（shifted force cutoff）：
        S(r) = 1   当 r < rcut - δ
        S(r) = (rcut - r)²(2δ + r - rcut)/δ³   当 rcut-δ ≤ r < rcut
        S(r) = 0   当 r ≥ rcut
    
    参数:
        positions: N×d 数组，N 个粒子在 d 维空间中的坐标
        epsilon, sigma: LJ 参数
        rcut: 截断半径
    
    返回:
        总势能（float）
    """
    n = positions.shape[0]
    v_total = 0.0
    delta = 0.3 * sigma  # 截断过渡区宽度
    rcut_inner = rcut - delta
    rcut_sq = rcut * rcut

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            if box_size is not None:
                rij -= box_size * np.round(rij / box_size)
            r_sq = float(np.dot(rij, rij))
            if r_sq >= rcut_sq:
                continue
            r = np.sqrt(r_sq)
            if r < 1e-12:
                continue
            # 平滑截断
            if r < rcut_inner:
                s = 1.0
            else:
                # cubic smoothing: S(r) = (rcut-r)²(2δ + r - rcut)/δ³
                dr = rcut - r
                s = dr * dr * (2.0 * delta + r - rcut) / (delta ** 3)
            v = lennard_jones_potential(r, epsilon, sigma)
            v_total += v * s
    return v_total


def total_forces_lj(positions: np.ndarray,
                    epsilon: float = 1.0,
                    sigma: float = 1.0,
                    rcut: float = 2.5,
                    box_size: float = None) -> np.ndarray:
    """
    使用解析导数计算 LJ 力场。
    
    F_i = Σ_{j≠i} [ (dV/dr) · (r_i - r_j) / r ] · S(r)
    
    返回 N×d 的力矩阵。
    """
    n, d = positions.shape
    forces = np.zeros_like(positions)
    delta = 0.3 * sigma
    rcut_inner = rcut - delta
    rcut_sq = rcut * rcut

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            if box_size is not None:
                rij -= box_size * np.round(rij / box_size)
            r_sq = float(np.dot(rij, rij))
            if r_sq >= rcut_sq or r_sq < 1e-24:
                continue
            r = np.sqrt(r_sq)
            if r < rcut_inner:
                s = 1.0
                ds = 0.0
            else:
                dr = rcut - r
                s = dr * dr * (2.0 * delta + r - rcut) / (delta ** 3)
                ds = -2.0 * dr * (2.0 * delta + r - rcut) / (delta ** 3) \
                     + dr * dr / (delta ** 3)
            f_mag = lennard_jones_force(r, epsilon, sigma)
            # f_mag = -dV/dr (力的大小，排斥为正)
            # F_i = f_mag * (r_i - r_j)/r * S(r) + V(r) * dS/dr * (r_i - r_j)/r
            factor = (f_mag * s + lennard_jones_potential(r, epsilon, sigma) * ds) / r
            f_vec = factor * rij
            forces[i] += f_vec
            forces[j] -= f_vec
    return forces


def total_potential_with_gaussian(positions: np.ndarray,
                                   epsilon: float = 1.0,
                                   sigma: float = 1.0,
                                   rcut: float = 2.5,
                                   gaussian_centers: np.ndarray = None,
                                   gaussian_params: List[dict] = None) -> float:
    """
    组合势能：Lennard-Jones + 高斯修正。
    
    V_total = Σ_{i<j} V_LJ(r_{ij}) + Σ_i Σ_k V_G( r_i - c_k )
    
    参数:
        positions: N×d 粒子坐标
        gaussian_centers: M×d 高斯中心位置
        gaussian_params: 每个高斯的参数字典列表
    """
    v = total_potential_lj(positions, epsilon, sigma, rcut)
    if gaussian_centers is not None and gaussian_params is not None:
        for i in range(positions.shape[0]):
            for k, center in enumerate(gaussian_centers):
                params = gaussian_params[k % len(gaussian_params)]
                dx = positions[i, 0] - center[0]
                dy = positions[i, 1] - center[1] if positions.shape[1] > 1 else 0.0
                v += gaussian_potential_2d(
                    dx, dy,
                    xmu=0.0, ymu=0.0,
                    xsigma=params.get('xsigma', 1.0),
                    ysigma=params.get('ysigma', 1.0),
                    A=params.get('A', 0.1),
                    corr_matrix=params.get('corr', None)
                )
    return v


def virial_stress_lj(positions: np.ndarray,
                     epsilon: float = 1.0,
                     sigma: float = 1.0,
                     rcut: float = 2.5,
                     volume: float = 1.0,
                     box_size: float = None) -> np.ndarray:
    """
    计算 LJ 体系的维里应力张量（Virial Stress Tensor）。
    
    对于 d 维系统，维里应力定义为：
    
        σ_{αβ} = (1/Ω) Σ_{i<j} [ r_{ij}^α · F_{ij}^β ]
    
    其中 Ω 为系统体积，F_{ij} 为粒子 j 对 i 的作用力。
    
    在热力学极限下，维里应力与压强 P 的关系：
    
        P = (1/d) Σ_{α} σ_{αα}
    
    返回:
        d×d 应力张量
    """
    n, d = positions.shape
    stress = np.zeros((d, d))
    delta = 0.3 * sigma
    rcut_inner = rcut - delta
    rcut_sq = rcut * rcut

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r_sq = float(np.dot(rij, rij))
            if r_sq >= rcut_sq or r_sq < 1e-24:
                continue
            r = np.sqrt(r_sq)
            if r < rcut_inner:
                s = 1.0
            else:
                dr = rcut - r
                s = dr * dr * (2.0 * delta + r - rcut) / (delta ** 3)
            f_mag = lennard_jones_force(r, epsilon, sigma)
            factor = f_mag * s / r
            for alpha in range(d):
                for beta in range(d):
                    stress[alpha, beta] += rij[alpha] * factor * rij[beta]
    return stress / volume
