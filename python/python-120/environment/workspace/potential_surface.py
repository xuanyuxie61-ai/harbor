"""
potential_surface.py
势能面 (Potential Energy Surface, PES) 计算模块

整合原项目:
  - 1381_vandermonde: Bjorck-Pereyra 算法求解 Vandermonde 线性系统
    用于多项式插值构建势能面
  - 711_mandelbrot_area: 蒙特卡洛收敛性测试思想用于 PES 积分

科学背景:
  表面催化反应中，吸附物种在金属表面的势能面 V(r) 决定了
  扩散势垒、解离路径和反应坐标。本模块使用高阶多项式插值
  结合从头算数据点构建解析 PES
"""

import numpy as np
from typing import List, Tuple


class PotentialEnergySurface:
    """
    催化剂表面势能面 (PES)
    
    采用多维多项式插值表示势能:
        V(x, y, z) = Σ_{i+j+k<=N} c_{ijk} * x^i * y^j * z^k
    
    系数 c_{ijk} 通过 Vandermonde 系统求解得到
    """

    def __init__(self, poly_degree: int = 4, length_scale: float = 1e-10):
        if poly_degree < 1:
            raise ValueError("poly_degree >= 1")
        self.poly_degree = poly_degree
        self.length_scale = length_scale
        self.coeffs = None
        self.powers = None
        self.ref_point = np.zeros(3)

    def _vandermonde_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        构建三维 Vandermonde 矩阵
        
        对于点集 {x_m, y_m, z_m}, m=1..M
        基函数: φ_{ijk}(x,y,z) = x^i * y^j * z^k
        
        矩阵元:
            A_{m, idx(i,j,k)} = x_m^i * y_m^j * z_m^k
        """
        n_points = points.shape[0]
        powers = []
        for i in range(self.poly_degree + 1):
            for j in range(self.poly_degree + 1 - i):
                for k in range(self.poly_degree + 1 - i - j):
                    powers.append((i, j, k))
        self.powers = powers
        n_basis = len(powers)
        if n_points < n_basis:
            raise ValueError(f"数据点数量 ({n_points}) 必须 >= 基函数数量 ({n_basis})")

        A = np.zeros((n_points, n_basis))
        for m in range(n_points):
            x, y, z = points[m]
            for col, (i, j, k) in enumerate(powers):
                A[m, col] = (x ** i) * (y ** j) * (z ** k)
        return A

    def fit(self, points: np.ndarray, energies: np.ndarray):
        """
        拟合势能面多项式系数
        
        求解线性系统:
            A * c = E
        
        其中 A 为 Vandermonde 矩阵，E 为能量值向量
        采用最小二乘或正则化方法求解
        """
        points = np.asarray(points, dtype=float)
        energies = np.asarray(energies, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points 形状必须为 (N, 3)")
        if energies.shape[0] != points.shape[0]:
            raise ValueError("points 和 energies 长度不匹配")

        self.ref_point = np.mean(points, axis=0)
        # 中心化并缩放坐标以提高数值稳定性
        centered = (points - self.ref_point) / self.length_scale

        A = self._vandermonde_matrix(centered)
        # Tikhonov 正则化最小二乘
        lam = 1e-8
        ATA = A.T @ A + lam * np.eye(A.shape[1])
        ATb = A.T @ energies
        self.coeffs = np.linalg.solve(ATA, ATb)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        计算给定位置的势能值
        
        V(r) = Σ c_{ijk} * (x - x0)^i * (y - y0)^j * (z - z0)^k
        """
        if self.coeffs is None:
            raise RuntimeError("必须先调用 fit()")
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != 3:
            raise ValueError("points 形状必须为 (N, 3)")

        # TODO Hole_1: 实现多项式势能面评估
        # 提示: 使用 self.ref_point, self.length_scale, self.coeffs, self.powers
        # 公式: V(r) = Σ c_{ijk} * ((x-x0)/ls)^i * ((y-y0)/ls)^j * ((z-z0)/ls)^k
        raise NotImplementedError("Hole_1: 请实现 evaluate 方法的核心计算")

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """
        计算势能梯度 ∇V = (∂V/∂x, ∂V/∂y, ∂V/∂z)
        
        公式:
            ∂V/∂x = Σ i * c_{ijk} * (x-x0)^{i-1} * (y-y0)^j * (z-z0)^k
        """
        if self.coeffs is None:
            raise RuntimeError("必须先调用 fit()")
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        centered = (points - self.ref_point) / self.length_scale
        n_points = centered.shape[0]
        grad = np.zeros((n_points, 3))

        for col, (i, j, k) in enumerate(self.powers):
            x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
            if i >= 1:
                grad[:, 0] += i * self.coeffs[col] * (x ** (i - 1)) * (y ** j) * (z ** k)
            if j >= 1:
                grad[:, 1] += j * self.coeffs[col] * (x ** i) * (y ** (j - 1)) * (z ** k)
            if k >= 1:
                grad[:, 2] += k * self.coeffs[col] * (x ** i) * (y ** j) * (z ** (k - 1))
        # 链式法则: dV/dr = dV/d(x/ls) * d(x/ls)/dr = dV/d(x/ls) / ls
        return grad / self.length_scale

    def hessian(self, points: np.ndarray) -> np.ndarray:
        """
        计算 Hessian 矩阵 H_{αβ} = ∂²V / ∂r_α ∂r_β
        
        用于确定鞍点和反应路径的曲率
        """
        if self.coeffs is None:
            raise RuntimeError("必须先调用 fit()")
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        centered = (points - self.ref_point) / self.length_scale
        n_points = centered.shape[0]
        hess = np.zeros((n_points, 3, 3))

        for col, (i, j, k) in enumerate(self.powers):
            x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
            # ∂²V/∂x²
            if i >= 2:
                hess[:, 0, 0] += i * (i - 1) * self.coeffs[col] * (x ** (i - 2)) * (y ** j) * (z ** k)
            # ∂²V/∂y²
            if j >= 2:
                hess[:, 1, 1] += j * (j - 1) * self.coeffs[col] * (x ** i) * (y ** (j - 2)) * (z ** k)
            # ∂²V/∂z²
            if k >= 2:
                hess[:, 2, 2] += k * (k - 1) * self.coeffs[col] * (x ** i) * (y ** j) * (z ** (k - 2))
            # ∂²V/∂x∂y
            if i >= 1 and j >= 1:
                hess[:, 0, 1] += i * j * self.coeffs[col] * (x ** (i - 1)) * (y ** (j - 1)) * (z ** k)
                hess[:, 1, 0] = hess[:, 0, 1]
            # ∂²V/∂x∂z
            if i >= 1 and k >= 1:
                hess[:, 0, 2] += i * k * self.coeffs[col] * (x ** (i - 1)) * (y ** j) * (z ** (k - 1))
                hess[:, 2, 0] = hess[:, 0, 2]
            # ∂²V/∂y∂z
            if j >= 1 and k >= 1:
                hess[:, 1, 2] += j * k * self.coeffs[col] * (x ** i) * (y ** (j - 1)) * (z ** (k - 1))
                hess[:, 2, 1] = hess[:, 1, 2]
        # 链式法则: d²V/dr² = d²V/d(x/ls)² / ls²
        return hess / (self.length_scale ** 2)

    def find_saddle_point_newton(self, x0: np.ndarray, tol: float = 1e-8,
                                  max_iter: int = 100) -> Tuple[np.ndarray, bool]:
        """
        使用 Newton-Raphson 方法寻找势能面的鞍点 (过渡态)
        
        鞍点条件:
            ∇V = 0
            Hessian 有且仅有一个负本征值
        
        迭代公式:
            r_{n+1} = r_n - H^{-1} * ∇V
        """
        x = np.asarray(x0, dtype=float).copy()
        for _ in range(max_iter):
            g = self.gradient(x)
            if np.linalg.norm(g) < tol:
                H = self.hessian(x)[0]
                eigvals = np.linalg.eigvalsh(H)
                n_neg = np.sum(eigvals < -1e-6)
                is_saddle = (n_neg == 1)
                return x, is_saddle
            H = self.hessian(x)[0]
            try:
                dx = -np.linalg.solve(H, g[0])
            except np.linalg.LinAlgError:
                # 使用伪逆处理奇异 Hessian
                dx = -np.linalg.lstsq(H, g[0], rcond=None)[0]
            x += dx
            # 限制步长
            if np.linalg.norm(dx) > 0.1:
                x -= dx
                x += 0.1 * dx / np.linalg.norm(dx)
        return x, False

    def estimate_activation_energy(self, r_reactant: np.ndarray,
                                    r_product: np.ndarray) -> float:
        """
        使用 NEB (Nudged Elastic Band) 近似估计活化能
        
        在反应物与产物之间线性插值，寻找最高能量点
        
        公式:
            E_a ≈ max_{t∈[0,1]} V(r_reactant + t * (r_product - r_reactant))
                  - V(r_reactant)
        """
        n_images = 50
        t_vals = np.linspace(0.0, 1.0, n_images)
        path = r_reactant[None, :] + t_vals[:, None] * (r_product - r_reactant)[None, :]
        energies = self.evaluate(path)
        e_react = energies[0]
        e_max = np.max(energies)
        return float(e_max - e_react)


def build_co_oxidation_pes_demo() -> PotentialEnergySurface:
    """
    构建 CO + O -> CO2 表面反应演示势能面
    
    采用解析模型势能，并在其上叠加一个低阶多项式修正，
    以演示 Vandermonde 插值在 PES 建模中的应用。
    
    基础模型:
        V = V_Morse(CO-Pt) + V_Morse(O-Pt) + V_Lennard-Jones(CO-O)
          + V_barrier * exp(-((x-x_TS)/σ_x)^2 - ((z-z_TS)/σ_z)^2)
    """
    from utils import morse_potential, lennard_jones_potential

    # 在规则网格上采样，确保多项式拟合稳定性
    n_per_dim = 8
    x = np.linspace(-1.5e-10, 1.5e-10, n_per_dim)
    y = np.linspace(-1.5e-10, 1.5e-10, n_per_dim)
    z = np.linspace(0.8e-10, 2.5e-10, n_per_dim)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # 计算模型能量 (eV)
    r_co_pt = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
    r_o_pt = np.sqrt((points[:, 0] - 1.5e-10) ** 2 + points[:, 1] ** 2
                     + (points[:, 2] - 0.5e-10) ** 2)
    r_co_o = np.sqrt((points[:, 0] - 1.5e-10) ** 2
                     + (points[:, 2] - 0.5e-10) ** 2)

    v_co = morse_potential(r_co_pt, d_e=1.3, a_param=2.0e10, r_e=1.85e-10)
    v_o = morse_potential(r_o_pt, d_e=2.0, a_param=2.5e10, r_e=1.20e-10)
    v_lj = lennard_jones_potential(r_co_o, epsilon=0.05, sigma=3.0e-10)

    # 过渡态势垒 (高斯型)
    v_ts = 1.0 * np.exp(-((points[:, 0] - 0.8e-10) / 0.5e-10) ** 2
                        - ((points[:, 2] - 1.5e-10) / 0.4e-10) ** 2)

    energies = v_co + v_o + v_lj + v_ts
    # 限制能量范围，避免多项式外推爆炸
    energies = np.clip(energies, -5.0, 2.0)

    # 使用低阶多项式 (degree=2) 保证数值稳定性
    pes = PotentialEnergySurface(poly_degree=2, length_scale=1e-10)
    pes.fit(points, energies)
    return pes
