"""
multigrid_poisson.py
多重网格泊松求解器模块

实现一维与二维泊松方程的多重网格 (Multigrid, MG) V-cycle 求解器。
多重网格通过在不同尺度（粗细网格）上交替进行光滑与修正，
实现与网格尺寸无关的收敛速度 O(N)。

核心数学：
    - 一维泊松方程离散:
        -u''(x) = f(x),   x in [a,b],   u(a)=ua, u(b)=ub
      
      均匀网格 x_i = a + i*h, h=(b-a)/N:
        -(u_{i-1} - 2u_i + u_{i+1}) / h^2 = f_i
        => -u_{i-1} + 2u_i - u_{i+1} = h^2 * f_i
    
    - 2D 五点差分格式:
        -(u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}) / h^2 = f_{i,j}
    
    - 光滑迭代 (Jacobi / Gauss-Seidel):
        Jacobi:   u_i^{new} = (rhs_i + u_{i-1}^{old} + u_{i+1}^{old}) / 2
        Gauss-Seidel: u_i^{new} = (rhs_i + u_{i-1}^{new} + u_{i+1}^{old}) / 2
    
    - 限制算子 (Full Weighting, 1D):
        r_j^{2h} = 0.25 * r_{2j-1}^{h} + 0.5 * r_{2j}^{h} + 0.25 * r_{2j+1}^{h}
    
    - 延拓算子 (Bilinear Interpolation, 1D):
        u_{2j}^{h}   = u_j^{2h}
        u_{2j+1}^{h} = 0.5*(u_j^{2h} + u_{j+1}^{2h})
    
    - V-cycle 算法:
        1) 在细网格上执行 nu1 次前光滑
        2) 计算残差 r = f - A*u
        3) 将残差限制到粗网格: r^{2h} = I_h^{2h} * r^h
        4) 在粗网格上求解 A^{2h} * e^{2h} = r^{2h}
           （递归调用，或直接求解若网格足够粗）
        5) 将误差延拓回细网格: e^h = I_{2h}^{h} * e^{2h}
        6) 修正: u^h = u^h + e^h
        7) 在细网格上执行 nu2 次后光滑
    
    - 收敛判据:
        ||r||_inf < tol   或   ||u^{new} - u^{old}||_inf < tol
"""

import numpy as np
from typing import Tuple, Optional, Callable
from utils import (
    gauss_seidel_sweep, restrict_fine_to_coarse,
    restrict_coarse_to_fine, is_power_of_two, EPSILON_MACHINE
)


class MultigridPoisson1D:
    """
    一维泊松方程多重网格求解器。
    
    求解:
        -u''(x) = force(x),  x in [a,b]
        u(a) = ua, u(b) = ub
    """
    def __init__(self, n: int, a: float, b: float,
                 ua: float, ub: float,
                 force_func: Callable[[np.ndarray], np.ndarray],
                 exact_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        """
        Parameters
        ----------
        n : int
            区间数，必须为2的幂次
        a, b : float
            区间端点
        ua, ub : float
            Dirichlet 边界值
        force_func : callable
            右端项函数 force(x) -> array
        exact_func : callable, optional
            精确解（用于误差分析）
        """
        if not is_power_of_two(n):
            raise ValueError(f"n must be a power of 2, got {n}")
        self.n = n
        self.a = a
        self.b = b
        self.ua = ua
        self.ub = ub
        self.force_func = force_func
        self.exact_func = exact_func
        self.x = np.linspace(a, b, n + 1)
        self.h = (b - a) / n

    def _build_rhs(self) -> np.ndarray:
        """构建右端项向量（已乘以 h^2）。"""
        rhs = np.zeros(self.n + 1)
        rhs[0] = self.ua
        rhs[self.n] = self.ub
        interior_x = self.x[1:self.n]
        rhs[1:self.n] = (self.h ** 2) * self.force_func(interior_x)
        return rhs

    def jacobi_sweep(self, n_level: int, rhs: np.ndarray, u: np.ndarray,
                     omega: float = 1.0) -> np.ndarray:
        """
        执行一次带松弛因子 omega 的 Jacobi 光滑。
        
        迭代格式:
            u_i^{new} = (1 - omega) * u_i + omega * (rhs_i + u_{i-1} + u_{i+1}) / 2
        
        Parameters
        ----------
        n_level : int
            当前层网格区间数
        rhs : np.ndarray
            右端项
        u : np.ndarray
            当前解
        omega : float
            松弛因子（0 < omega <= 1 为阻尼 Jacobi）
        
        Returns
        -------
        np.ndarray
            更新后的解
        """
        u_new = u.copy()
        for i in range(1, n_level):
            u_new[i] = (1.0 - omega) * u[i] + omega * 0.5 * (rhs[i] + u[i - 1] + u[i + 1])
        u_new[0] = self.ua
        u_new[n_level] = self.ub
        return u_new

    def _residual(self, n_level: int, rhs: np.ndarray, u: np.ndarray) -> np.ndarray:
        """计算残差 r = rhs - A*u（A为1D离散矩阵）。"""
        r = np.zeros_like(u)
        r[0] = 0.0
        r[n_level] = 0.0
        for i in range(1, n_level):
            r[i] = rhs[i] - (2.0 * u[i] - u[i - 1] - u[i + 1])
        return r

    def _v_cycle(self, n_level: int, rhs: np.ndarray, u: np.ndarray,
                 nu1: int = 2, nu2: int = 2, omega: float = 0.8) -> np.ndarray:
        """
        递归 V-cycle。
        
        Parameters
        ----------
        n_level : int
            当前层网格区间数
        rhs : np.ndarray
            右端项
        u : np.ndarray
            初始猜测
        nu1, nu2 : int
            前/后光滑次数
        omega : float
            Jacobi 松弛因子
        
        Returns
        -------
        np.ndarray
            修正后的解
        """
        if n_level <= 2:
            # 粗网格直接求解（高斯消去或多次Jacobi）
            for _ in range(50):
                u = self.jacobi_sweep(n_level, rhs, u, omega=1.0)
            return u

        # 前光滑
        for _ in range(nu1):
            u = self.jacobi_sweep(n_level, rhs, u, omega=omega)

        # 残差
        res = self._residual(n_level, rhs, u)

        # 限制到粗网格
        n_coarse = n_level // 2
        u_coarse = np.zeros(n_coarse + 1)
        rhs_coarse = np.zeros(n_coarse + 1)

        # 完全加权限制
        rhs_coarse[0] = res[0]
        rhs_coarse[n_coarse] = res[n_level]
        for j in range(1, n_coarse):
            fine_idx = 2 * j
            rhs_coarse[j] = (
                0.25 * res[fine_idx - 1]
                + 0.5 * res[fine_idx]
                + 0.25 * res[fine_idx + 1]
            )

        # 递归求解粗网格误差方程
        e_coarse = self._v_cycle(n_coarse, rhs_coarse, u_coarse, nu1, nu2, omega)

        # 延拓回细网格
        e_fine = np.zeros(n_level + 1)
        for j in range(n_coarse):
            e_fine[2 * j] = e_coarse[j]
            e_fine[2 * j + 1] = 0.5 * (e_coarse[j] + e_coarse[j + 1])
        e_fine[n_level] = e_coarse[n_coarse]

        # 修正
        u = u + e_fine

        # 后光滑
        for _ in range(nu2):
            u = self.jacobi_sweep(n_level, rhs, u, omega=omega)

        return u

    def solve(self, tol: float = 1e-6, max_iter: int = 100,
              nu1: int = 2, nu2: int = 2, omega: float = 0.8) -> Tuple[np.ndarray, int]:
        """
        使用多重网格 V-cycle 迭代求解。
        
        Parameters
        ----------
        tol : float
            残差容限
        max_iter : int
            最大迭代次数
        nu1, nu2 : int
            前/后光滑次数
        omega : float
            Jacobi 松弛因子
        
        Returns
        -------
        u : np.ndarray
            数值解
        it_num : int
            实际迭代次数
        """
        rhs = self._build_rhs()
        u = np.zeros(self.n + 1)
        u[0] = self.ua
        u[self.n] = self.ub

        for it in range(max_iter):
            u_old = u.copy()
            u = self._v_cycle(self.n, rhs, u, nu1, nu2, omega)
            res = self._residual(self.n, rhs, u)
            res_norm = np.max(np.abs(res[1:self.n]))
            change = np.max(np.abs(u - u_old))

            if res_norm < tol or change < tol * 0.1:
                return u, it + 1

        print(f"[WARNING] Multigrid 1D: max_iter={max_iter} reached, res_norm={res_norm:.3e}")
        return u, max_iter


class MultigridPoisson2D:
    """
    二维泊松方程多重网格求解器（基于五点差分格式）。
    
    求解:
        -(u_{xx} + u_{yy}) = f(x,y),   (x,y) in [0,Lx] x [0,Ly]
    
    采用标准均匀笛卡尔网格，支持 Jacobi 和 Gauss-Seidel 光滑。
    """
    def __init__(self, nx: int, ny: int, Lx: float = 1.0, Ly: float = 1.0):
        """
        Parameters
        ----------
        nx, ny : int
            x, y 方向区间数，建议为2的幂次
        Lx, Ly : float
            域尺寸
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.hx = Lx / nx
        self.hy = Ly / ny
        self.dx2 = self.hx ** 2
        self.dy2 = self.hy ** 2

    def apply_operator(self, u: np.ndarray) -> np.ndarray:
        """
        应用离散 Laplacian 算子 A*u。
        
        内部点:
            (Au)_{i,j} = -(u_{i+1,j} + u_{i-1,j} - 2u_{i,j})/dx^2
                         -(u_{i,j+1} + u_{i,j-1} - 2u_{i,j})/dy^2
        
        Parameters
        ----------
        u : np.ndarray, shape (nx+1, ny+1)
            输入场
        
        Returns
        -------
        np.ndarray
            A*u
        """
        u = np.asarray(u, dtype=float)
        Au = np.zeros_like(u)
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                Au[i, j] = (
                    -(u[i + 1, j] + u[i - 1, j] - 2.0 * u[i, j]) / self.dx2
                    -(u[i, j + 1] + u[i, j - 1] - 2.0 * u[i, j]) / self.dy2
                )
        return Au

    def jacobi_sweep_2d(self, rhs: np.ndarray, u: np.ndarray,
                        omega: float = 0.8) -> np.ndarray:
        """
        2D 阻尼 Jacobi 光滑。
        
        迭代格式:
            u_{i,j}^{new} = (1-omega)*u_{i,j} + omega * (
                (rhs_{i,j} + (u_{i+1,j}+u_{i-1,j})/dx^2 + (u_{i,j+1}+u_{i,j-1})/dy^2)
                / (2/dx^2 + 2/dy^2)
            )
        
        Parameters
        ----------
        rhs : np.ndarray
            右端项
        u : np.ndarray
            当前解
        omega : float
            松弛因子
        
        Returns
        -------
        np.ndarray
            更新后的解
        """
        u_new = u.copy()
        denom = 2.0 / self.dx2 + 2.0 / self.dy2
        for i in range(1, self.nx):
            for j in range(1, self.ny):
                u_new[i, j] = (1.0 - omega) * u[i, j] + omega * (
                    rhs[i, j]
                    + (u[i + 1, j] + u[i - 1, j]) / self.dx2
                    + (u[i, j + 1] + u[i, j - 1]) / self.dy2
                ) / denom
        return u_new

    def restrict_2d(self, u_fine: np.ndarray) -> np.ndarray:
        """
        二维完全加权限制。
        
        权重:
            角点: 1/16,  边中点: 1/8,  中心: 1/4
        """
        nf_x, nf_y = u_fine.shape
        nc_x = (nf_x - 1) // 2 + 1
        nc_y = (nf_y - 1) // 2 + 1
        u_coarse = np.zeros((nc_x, nc_y))

        for ic in range(1, nc_x - 1):
            for jc in range(1, nc_y - 1):
                ifi, jfi = 2 * ic, 2 * jc
                u_coarse[ic, jc] = (
                    0.25 * u_fine[ifi, jfi]
                    + 0.125 * (u_fine[ifi - 1, jfi] + u_fine[ifi + 1, jfi]
                               + u_fine[ifi, jfi - 1] + u_fine[ifi, jfi + 1])
                    + 0.0625 * (u_fine[ifi - 1, jfi - 1] + u_fine[ifi + 1, jfi - 1]
                                + u_fine[ifi - 1, jfi + 1] + u_fine[ifi + 1, jfi + 1])
                )

        # 边界直接复制
        u_coarse[0, :] = u_fine[0, ::2]
        u_coarse[-1, :] = u_fine[-1, ::2]
        u_coarse[:, 0] = u_fine[::2, 0]
        u_coarse[:, -1] = u_fine[::2, -1]
        return u_coarse

    def prolong_2d(self, u_coarse: np.ndarray, shape_fine: Tuple[int, int]) -> np.ndarray:
        """
        二维双线性延拓。
        """
        nc_x, nc_y = u_coarse.shape
        nf_x, nf_y = shape_fine
        u_fine = np.zeros((nf_x, nf_y))

        for ic in range(nc_x - 1):
            for jc in range(nc_y - 1):
                ifi, jfi = 2 * ic, 2 * jc
                if ifi >= nf_x or jfi >= nf_y:
                    continue
                u_fine[ifi, jfi] = u_coarse[ic, jc]
                if ifi + 2 < nf_x:
                    u_fine[ifi + 2, jfi] = u_coarse[ic + 1, jc]
                if jfi + 2 < nf_y:
                    u_fine[ifi, jfi + 2] = u_coarse[ic, jc + 1]
                if ifi + 2 < nf_x and jfi + 2 < nf_y:
                    u_fine[ifi + 2, jfi + 2] = u_coarse[ic + 1, jc + 1]
                if ifi + 1 < nf_x:
                    u_fine[ifi + 1, jfi] = 0.5 * (u_coarse[ic, jc] + u_coarse[ic + 1, jc])
                if jfi + 1 < nf_y:
                    u_fine[ifi, jfi + 1] = 0.5 * (u_coarse[ic, jc] + u_coarse[ic, jc + 1])
                if ifi + 1 < nf_x and jfi + 2 < nf_y:
                    u_fine[ifi + 1, jfi + 2] = 0.5 * (u_coarse[ic, jc + 1] + u_coarse[ic + 1, jc + 1])
                if ifi + 2 < nf_x and jfi + 1 < nf_y:
                    u_fine[ifi + 2, jfi + 1] = 0.5 * (u_coarse[ic + 1, jc] + u_coarse[ic + 1, jc + 1])
                if ifi + 1 < nf_x and jfi + 1 < nf_y:
                    u_fine[ifi + 1, jfi + 1] = 0.25 * (
                        u_coarse[ic, jc] + u_coarse[ic + 1, jc]
                        + u_coarse[ic, jc + 1] + u_coarse[ic + 1, jc + 1]
                    )

        # 边界
        u_fine[0, :] = u_fine[0, :]
        u_fine[-1, :] = u_fine[-1, :]
        u_fine[:, 0] = u_fine[:, 0]
        u_fine[:, -1] = u_fine[:, -1]
        return u_fine

    def _v_cycle_2d(self, u: np.ndarray, rhs: np.ndarray,
                    nu1: int = 2, nu2: int = 2, omega: float = 0.8,
                    min_size: int = 4) -> np.ndarray:
        """递归 2D V-cycle。"""
        nx, ny = u.shape
        nx -= 1
        ny -= 1

        if nx <= min_size or ny <= min_size:
            for _ in range(50):
                u = self.jacobi_sweep_2d(rhs, u, omega=1.0)
            return u

        # 前光滑
        for _ in range(nu1):
            u = self.jacobi_sweep_2d(rhs, u, omega=omega)

        # 残差
        res = rhs - self.apply_operator(u)

        # 限制
        rhs_coarse = self.restrict_2d(res)
        u_coarse = np.zeros_like(rhs_coarse)

        # 递归
        sub = MultigridPoisson2D(
            (rhs_coarse.shape[0] - 1), (rhs_coarse.shape[1] - 1),
            self.Lx, self.Ly
        )
        e_coarse = sub._v_cycle_2d(u_coarse, rhs_coarse, nu1, nu2, omega, min_size)

        # 延拓
        e_fine = self.prolong_2d(e_coarse, u.shape)
        u = u + e_fine

        # 后光滑
        for _ in range(nu2):
            u = self.jacobi_sweep_2d(rhs, u, omega=omega)

        return u

    def solve(self, rhs: np.ndarray, u0: Optional[np.ndarray] = None,
              tol: float = 1e-6, max_iter: int = 50,
              nu1: int = 2, nu2: int = 2, omega: float = 0.8) -> Tuple[np.ndarray, int]:
        """
        求解二维泊松方程。
        
        Parameters
        ----------
        rhs : np.ndarray, shape (nx+1, ny+1)
            右端项
        u0 : np.ndarray, optional
            初始猜测
        tol : float
            残差容限
        max_iter : int
            最大V-cycle次数
        
        Returns
        -------
        u : np.ndarray
            解
        it_num : int
            迭代次数
        """
        rhs = np.asarray(rhs, dtype=float)
        if u0 is None:
            u = np.zeros((self.nx + 1, self.ny + 1))
        else:
            u = u0.copy()

        for it in range(max_iter):
            u_old = u.copy()
            u = self._v_cycle_2d(u, rhs, nu1, nu2, omega)
            res = rhs - self.apply_operator(u)
            res_norm = np.max(np.abs(res[1:self.nx, 1:self.ny]))
            change = np.max(np.abs(u - u_old))
            if res_norm < tol and change < tol:
                return u, it + 1

        return u, max_iter
