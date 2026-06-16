"""
随机椭圆方程求解器 (Stochastic Elliptic PDE Solver)
=====================================================
求解随机椭圆偏微分方程:
  -d/dx[κ(x,ω) du/dx] = f(x),  x ∈ (0, L)
  u(0) = u(L) = 0  (Dirichlet 边界条件)

其中 κ(x,ω) 是随机扩散系数, 通过 KL 展开参数化。

有限差分离散化 (二阶中心差分):
  对均匀网格 x_i = i·h, i=0,...,N+1, h = L/(N+1):

  -(κ_{i+1/2} (u_{i+1} - u_i) - κ_{i-1/2} (u_i - u_{i-1})) / h² = f_i

  其中 κ_{i+1/2} = (κ_i + κ_{i+1})/2 (算术平均)
  或 κ_{i+1/2} = 2κ_i κ_{i+1} / (κ_i + κ_{i+1}) (调和平均, 更适合跳跃系数)

  这给出三对角线性系统: A u = f
  A = (1/h²) tridiag(-κ_{i-1/2}, κ_{i-1/2}+κ_{i+1/2}, -κ_{i+1/2})

收敛性:
  对于 u ∈ C⁴, 截断误差 O(h²)
  对于不连续 κ, 使用调和平均保证通量连续性

条件数:
  cond(A) ≈ 4κ_max/(π² h² κ_min)
  当 κ 变化大时, 条件数可能很大, 需要预处理。
"""

import numpy as np
from scipy import linalg as la
from typing import Optional, Tuple, Dict


class EllipticSolver1D:
    """
    1D 随机椭圆方程求解器。

    求解: -d/dx[κ(x) du/dx] = f(x)
    边界: u(0) = u_a, u(L) = u_b

    离散化方法:
      有限差分法 (FDM), 二阶精度
      使用调和平均处理系数跳跃
    """

    def __init__(
        self,
        n_spatial: int = 101,
        domain_length: float = 1.0,
        boundary_left: float = 0.0,
        boundary_right: float = 0.0,
        averaging: str = 'harmonic'
    ):
        """
        参数:
            n_spatial: 空间网格点数 (含边界)
            domain_length: 域长度 L
            boundary_left: 左边界值 u(0)
            boundary_right: 右边界值 u(L)
            averaging: 界面系数平均方式 'harmonic' 或 'arithmetic'
        """
        self.n_spatial = n_spatial
        self.L = domain_length
        self.u_left = boundary_left
        self.u_right = boundary_right
        self.averaging = averaging

        # 空间网格
        self.x = np.linspace(0.0, self.L, self.n_spatial)
        self.h = self.L / (self.n_spatial - 1)

        # 内部点数
        self.n_interior = self.n_spatial - 2

    def source_term(self, x: np.ndarray) -> np.ndarray:
        """
        源项函数 f(x):
          f(x) = 2 + sin(2πx)

        选择此源项使得即使 κ 不均匀, 解也具有足够的正则性。
        """
        return 2.0 + np.sin(2.0 * np.pi * x)

    def _interface_diffusion(
        self, kappa: np.ndarray
    ) -> np.ndarray:
        """
        计算界面处的扩散系数:

        调和平均 (Harmonic average):
          κ_{i+1/2} = 2 κ_i κ_{i+1} / (κ_i + κ_{i+1})
        适用于: 跳跃系数, 保证通量连续性
        性质: min(κ_i, κ_{i+1}) ≤ κ_{i+1/2} ≤ max(κ_i, κ_{i+1})

        算术平均 (Arithmetic average):
          κ_{i+1/2} = (κ_i + κ_{i+1}) / 2
        适用于: 光滑系数
        """
        N = kappa.size
        kappa_face = np.zeros(N - 1)
        for i in range(N - 1):
            if self.averaging == 'harmonic':
                s = kappa[i] + kappa[i + 1]
                if abs(s) < 1e-30:
                    kappa_face[i] = 1e-30
                else:
                    kappa_face[i] = 2.0 * kappa[i] * kappa[i + 1] / s
            else:  # arithmetic
                kappa_face[i] = 0.5 * (kappa[i] + kappa[i + 1])
        return kappa_face

    def assemble_system(
        self, kappa: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        组装有限差分线性系统 A u = f。

        A = (1/h²) × tridiag(a_i, b_i, c_i)
        其中:
          a_i = -κ_{i-1/2}       (次对角)
          b_i = κ_{i-1/2} + κ_{i+1/2}  (主对角)
          c_i = -κ_{i+1/2}       (超对角)

        边界条件通过修改右端项处理:
          f_1 += κ_{1/2} u_left / h²
          f_N += κ_{N+1/2} u_right / h²

        参数:
            kappa: 扩散系数在各网格点的值, shape (N_spatial,)

        返回:
            A: 三对角矩阵, shape (N_interior, N_interior)
            f: 右端项, shape (N_interior,)
        """
        n = self.n_interior
        h = self.h

        # 界面扩散系数
        kappa_face = self._interface_diffusion(kappa)  # shape (N_spatial - 1,)

        # 内部节点的界面系数
        # κ_{i-1/2} for i=1,...,N_interior corresponds to kappa_face[0],...,kappa_face[N_interior-1]
        # κ_{i+1/2} for i=1,...,N_interior corresponds to kappa_face[1],...,kappa_face[N_interior]
        k_left = kappa_face[:n]      # κ_{i-1/2}
        k_right = kappa_face[1:n + 1]  # κ_{i+1/2}

        # 三对角元素
        diag = (k_left + k_right) / h ** 2
        lower = -k_left[1:] / h ** 2   # shape (n-1,)
        upper = -k_right[:-1] / h ** 2  # shape (n-1,)

        # 组装矩阵
        A = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)

        # 右端项
        x_interior = self.x[1:-1]
        f = self.source_term(x_interior)

        # 边界条件贡献
        f[0] += k_left[0] * self.u_left / h ** 2
        f[-1] += k_right[-1] * self.u_right / h ** 2

        return A, f

    def solve(self, kappa: np.ndarray) -> np.ndarray:
        """
        求解椭圆方程。

        使用直接求解器 (LU 分解):
          对于三对角系统, 计算复杂度 O(N)
          对于一般系统, 计算复杂度 O(N³)

        参数:
            kappa: 扩散系数, shape (N_spatial,)

        返回:
            u: 解, shape (N_spatial,), 包含边界值
        """
        # 数值安全: 确保扩散系数为正
        kappa = np.maximum(kappa, 1e-10)

        A, f = self.assemble_system(kappa)

        # 求解三对角系统 (使用 Thomas 算法或 scipy)
        u_interior = la.solve_banded(
            (1, 1),
            np.vstack([
                np.concatenate([[0.0], np.diag(A, 1)]),
                np.diag(A, 0),
                np.concatenate([np.diag(A, -1), [0.0]])
            ]),
            f
        )

        # 组装完整解 (含边界)
        u = np.zeros(self.n_spatial)
        u[0] = self.u_left
        u[1:-1] = u_interior
        u[-1] = self.u_right

        return u

    def compute_flux(self, u: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """
        计算通量 q(x) = -κ(x) du/dx:
          q_{i+1/2} = -κ_{i+1/2} (u_{i+1} - u_i) / h

        通量守恒检验: dq/dx = f  ⟹  q 应满足离散守恒律
        """
        kappa_face = self._interface_diffusion(kappa)
        flux = -kappa_face * np.diff(u) / self.h
        return flux

    def compute_energy_norm(
        self, u: np.ndarray, kappa: np.ndarray
    ) -> float:
        """
        计算能量范数:
          ||u||_E = √(∫ κ(x) |u'(x)|² dx)

        对于椭圆方程, 能量范数等价于 H¹ 半范数。
        物理意义: 弹性体的应变能 ∝ ||u||_E²
        """
        du_dx = np.diff(u) / self.h
        kappa_face = self._interface_diffusion(kappa)
        integral = np.sum(kappa_face * du_dx ** 2) * self.h
        return np.sqrt(max(integral, 0.0))

    def residual_error(
        self, u: np.ndarray, kappa: np.ndarray
    ) -> np.ndarray:
        """
        计算离散残差:
          r_i = f_i - (A u)_i

        用于后验误差估计和自适应网格细化。
        """
        A, f = self.assemble_system(kappa)
        u_int = u[1:-1]
        residual = f - A @ u_int
        return residual


def solve_parametric_elliptic(
    solver: EllipticSolver1D,
    kl_expansion,
    xi: np.ndarray,
    mean_diffusion: float = 1.0,
    log_transform: bool = True
) -> np.ndarray:
    """
    求解参数化椭圆方程 (单个参数实现)。

    完整流程:
    1. 从随机变量 ξ 通过 KL 展开生成扩散系数场 κ(x)
    2. 组装并求解 -d/dx[κ(x) du/dx] = f(x)

    参数:
        solver: 椭圆求解器
        kl_expansion: KL 展开对象
        xi: 随机变量, shape (K,)
        mean_diffusion: 扩散系数均值
        log_transform: 是否使用对数变换

    返回:
        u: 解, shape (N_spatial,)
    """
    # 生成随机扩散系数场
    kappa = kl_expansion.evaluate_field(xi, mean_diffusion, log_transform)

    # 求解
    u = solver.solve(kappa)
    return u


def solve_ensemble_elliptic(
    solver: EllipticSolver1D,
    kl_expansion,
    xi_ensemble: np.ndarray,
    mean_diffusion: float = 1.0,
    log_transform: bool = True
) -> np.ndarray:
    """
    求解参数化椭圆方程 (多个参数实现)。

    参数:
        xi_ensemble: shape (N_ensemble, K)

    返回:
        solutions: shape (N_ensemble, N_spatial)
    """
    N = xi_ensemble.shape[0]
    solutions = np.zeros((N, solver.n_spatial))

    for i in range(N):
        solutions[i] = solve_parametric_elliptic(
            solver, kl_expansion, xi_ensemble[i],
            mean_diffusion, log_transform
        )

    return solutions
