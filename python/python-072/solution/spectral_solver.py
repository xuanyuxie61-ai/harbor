"""
spectral_solver.py
==================
谱方法求解器模块

融合种子项目：
- 1085_sine_transform: 正弦变换用于函数插值与近似
- 875_poisson_1d: 一维泊松方程 Gauss-Seidel 求解

核心内容：
1. 离散正弦变换（DST）及其逆变换
2. 使用正弦变换快速求解一维/二维 Poisson 方程
3. 正弦级数展开与插值
4. 热方程的谱求解

正弦变换：
    对于函数 f(x) 在 [0, L] 上满足 f(0) = f(L) = 0，
    其正弦级数展开为：
        f(x) = Σ_{k=1}^∞ b_k sin(kπx/L)
    其中系数：
        b_k = (2/L) ∫_0^L f(x) sin(kπx/L) dx

    离散正弦变换（DST-II）：
        b_k = √(2/(N+1)) Σ_{j=1}^N f_j sin(π k j / (N+1))

二维 Poisson 方程的谱求解：
    对于 -∇²u = f，在矩形域上 Dirichlet 边界条件下：
        u_{ij} = Σ_{k,l} û_{kl} sin(iπk/(Nx+1)) sin(jπl/(Ny+1))
    其中：
        û_{kl} = f̂_{kl} / [ (πk/Lx)² + (πl/Ly)² ]
"""

import numpy as np


class SineTransform:
    """
    离散正弦变换（DST）工具。
    基于种子项目 1085_sine_transform。
    """

    @staticmethod
    def dst_1d(f):
        """
        一维离散正弦变换（DST-II）。

        对于序列 f[1], ..., f[N]：
            b_k = √(2/(N+1)) Σ_{j=1}^N f_j sin(π k j / (N+1))

        Parameters
        ----------
        f : ndarray, shape (N,)
            输入序列，假设在边界 f[0] = f[N+1] = 0。

        Returns
        -------
        ndarray, shape (N,)
            正弦变换系数。
        """
        f = np.asarray(f)
        N = len(f)
        if N == 0:
            return np.array([])

        scale = np.sqrt(2.0 / (N + 1))
        b = np.zeros(N)

        for k in range(1, N + 1):
            for j in range(1, N + 1):
                b[k - 1] += np.sin(np.pi * k * j / (N + 1)) * f[j - 1]

        b *= scale
        return b

    @staticmethod
    def idst_1d(b):
        """
        一维逆离散正弦变换。

        f_j = √(2/(N+1)) Σ_{k=1}^N b_k sin(π k j / (N+1))

        Parameters
        ----------
        b : ndarray, shape (N,)
            正弦变换系数。

        Returns
        -------
        ndarray, shape (N,)
            逆变换结果。
        """
        b = np.asarray(b)
        N = len(b)
        if N == 0:
            return np.array([])

        scale = np.sqrt(2.0 / (N + 1))
        f = np.zeros(N)

        for j in range(1, N + 1):
            for k in range(1, N + 1):
                f[j - 1] += np.sin(np.pi * k * j / (N + 1)) * b[k - 1]

        f *= scale
        return f

    @staticmethod
    def dst_2d(f):
        """
        二维离散正弦变换。

        先对每行做 DST，再对每列做 DST。

        Parameters
        ----------
        f : ndarray, shape (Nx, Ny)
            输入场。

        Returns
        -------
        ndarray, shape (Nx, Ny)
            二维正弦变换系数。
        """
        f = np.asarray(f)
        Nx, Ny = f.shape

        # 对每行做 DST
        b = np.zeros_like(f)
        for i in range(Nx):
            b[i, :] = SineTransform.dst_1d(f[i, :])

        # 对每列做 DST
        result = np.zeros_like(f)
        for j in range(Ny):
            result[:, j] = SineTransform.dst_1d(b[:, j])

        return result

    @staticmethod
    def idst_2d(b):
        """
        二维逆离散正弦变换。

        Parameters
        ----------
        b : ndarray, shape (Nx, Ny)
            变换系数。

        Returns
        -------
        ndarray, shape (Nx, Ny)
            逆变换结果。
        """
        b = np.asarray(b)
        Nx, Ny = b.shape

        # 对每列做 IDST
        f = np.zeros_like(b)
        for j in range(Ny):
            f[:, j] = SineTransform.idst_1d(b[:, j])

        # 对每行做 IDST
        result = np.zeros_like(b)
        for i in range(Nx):
            result[i, :] = SineTransform.idst_1d(f[i, :])

        return result


class SpectralPoissonSolver:
    """
    基于正弦变换的谱方法 Poisson 方程求解器。
    基于种子项目 875_poisson_1d 和 1085_sine_transform。
    """

    def __init__(self, nx, ny, Lx=1.0, Ly=1.0):
        """
        初始化谱求解器。

        Parameters
        ----------
        nx, ny : int
            网格点数（不包括边界）。
        Lx, Ly : float
            区域尺寸。
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        # 波数
        self.kx = np.arange(1, nx + 1) * np.pi / Lx
        self.ky = np.arange(1, ny + 1) * np.pi / Ly

    def solve_2d_poisson_dirichlet(self, f):
        """
        求解二维 Poisson 方程：
            -∇²u = f,  在 [0,Lx]×[0,Ly] 上
            u = 0,     在边界上

        谱方法求解步骤：
            1. 对 f 做二维 DST 得到 f̂
            2. û_{kl} = f̂_{kl} / (kx_k² + ky_l²)
            3. 对 û 做二维 IDST 得到 u

        Parameters
        ----------
        f : ndarray, shape (nx, ny)
            右端项场（内部点，不含边界）。

        Returns
        -------
        ndarray, shape (nx, ny)
            解场 u（内部点）。
        """
        if f.shape != (self.nx, self.ny):
            raise ValueError(f"f 的形状必须为 ({self.nx}, {self.ny})")

        # 正弦变换
        f_hat = SineTransform.dst_2d(f)

        # 在频域求解
        u_hat = np.zeros_like(f_hat)
        for i in range(self.nx):
            for j in range(self.ny):
                denom = self.kx[i] ** 2 + self.ky[j] ** 2
                if denom > 1e-14:
                    u_hat[i, j] = f_hat[i, j] / denom
                else:
                    u_hat[i, j] = 0.0

        # 逆正弦变换
        u = SineTransform.idst_2d(u_hat)

        return u

    def solve_1d_poisson_dirichlet(self, f):
        """
        求解一维 Poisson 方程：
            -u'' = f,  在 [0, L] 上
            u(0) = u(L) = 0

        Parameters
        ----------
        f : ndarray, shape (n,)
            右端项（内部点）。

        Returns
        -------
        ndarray, shape (n,)
            解 u。
        """
        n = len(f)
        k = np.arange(1, n + 1) * np.pi / self.Lx

        f_hat = SineTransform.dst_1d(f)
        u_hat = np.zeros(n)

        for i in range(n):
            if k[i] ** 2 > 1e-14:
                u_hat[i] = f_hat[i] / (k[i] ** 2)

        u = SineTransform.idst_1d(u_hat)
        return u


class GaussSeidelPoisson:
    """
    Gauss-Seidel 迭代求解 Poisson 方程。
    基于种子项目 875_poisson_1d 的 gauss_seidel.m。
    """

    @staticmethod
    def gauss_seidel_1d_step(n, r, u):
        """
        一维 Gauss-Seidel 单步迭代。

        离散 Laplacian：-u'' ≈ -(u_{i-1} - 2u_i + u_{i+1}) / h² = f_i
        即：u_i = (u_{i-1} + u_{i+1} + h² f_i) / 2

        这里 r 已包含 h² f_i，边界值已固定。

        Parameters
        ----------
        n : int
            总点数（包括边界）。
        r : ndarray
            右端项（包含边界值）。
        u : ndarray
            当前解估计。

        Returns
        -------
        tuple
            (u_new, dif_l1) 新解和 L1 变化量。
        """
        u_new = u.copy()
        u_old = u.copy()

        for i in range(1, n - 1):
            u_new[i] = 0.5 * (u_new[i - 1] + u_old[i + 1] + r[i])

        dif_l1 = np.sum(np.abs(u_new[1:-1] - u_old[1:-1]))

        return u_new, dif_l1

    @staticmethod
    def solve_1d_poisson_gs(n_intervals, a, b, ua, ub, force_func,
                            max_iter=10000, tol=1e-4):
        """
        用 Gauss-Seidel 迭代求解一维泊松方程。

        方程：-u''(x) = f(x),  u(a) = ua, u(b) = ub

        Parameters
        ----------
        n_intervals : int
            区间数。
        a, b : float
            区间端点。
        ua, ub : float
            Dirichlet 边界值。
        force_func : callable
            右端项函数 f(x)。
        max_iter : int
            最大迭代次数。
        tol : float
            收敛容差。

        Returns
        -------
        tuple
            (x, u, it_num) 网格点、解和迭代次数。
        """
        n = n_intervals + 1
        x = np.linspace(a, b, n)
        h = (b - a) / n_intervals

        # 右端项
        r = np.zeros(n)
        r[0] = ua
        r[1:-1] = force_func(x[1:-1]) * h ** 2
        r[-1] = ub

        # 初始猜测
        u = np.zeros(n)
        u[0] = ua
        u[-1] = ub

        it_num = 0
        while it_num < max_iter:
            it_num += 1
            u, dif = GaussSeidelPoisson.gauss_seidel_1d_step(n, r, u)
            if dif <= tol:
                break

        return x, u, it_num

    @staticmethod
    def solve_2d_poisson_gs(nx, ny, dx, dy, f, max_iter=5000, tol=1e-6):
        """
        用 Gauss-Seidel 迭代求解二维泊松方程。

        方程：-∇²u = f,  在矩形域上，u = 0 边界。

        五点差分：
            -(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h² = f_{i,j}

        迭代格式：
            u_{i,j} = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} + h² f_{i,j}) / 4

        Parameters
        ----------
        nx, ny : int
            网格点数。
        dx, dy : float
            空间步长。
        f : ndarray, shape (nx, ny)
            右端项。
        max_iter : int
            最大迭代次数。
        tol : float
            收敛容差。

        Returns
        -------
        tuple
            (u, it_num) 解和迭代次数。
        """
        u = np.zeros((nx, ny))
        h2 = dx * dy

        # 简化：假设 dx = dy
        denom = 2.0 * (1.0 / (dx ** 2) + 1.0 / (dy ** 2))
        rhs = f.copy()

        it_num = 0
        while it_num < max_iter:
            it_num += 1
            u_old = u.copy()

            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    u[i, j] = (
                        (u[i + 1, j] + u[i - 1, j]) / (dx ** 2) +
                        (u[i, j + 1] + u[i, j - 1]) / (dy ** 2) +
                        rhs[i, j]
                    ) / denom

            dif = np.max(np.abs(u - u_old))
            if dif < tol:
                break

        return u, it_num


class SpectralHeatSolver:
    """
    基于正弦变换的热方程谱求解器。
    """

    def __init__(self, nx, ny, Lx=1.0, Ly=1.0, alpha=1.0):
        """
        初始化热方程求解器。

        Parameters
        ----------
        nx, ny : int
            内部网格点数。
        Lx, Ly : float
            区域尺寸。
        alpha : float
            热扩散系数。
        """
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.alpha = alpha

        # 波数
        self.kx = np.arange(1, nx + 1) * np.pi / Lx
        self.ky = np.arange(1, ny + 1) * np.pi / Ly

    def solve_step_spectral(self, u, dt):
        """
        用谱方法推进热方程一个时间步。

        热方程：∂u/∂t = α ∇²u

        谱域中：∂û/∂t = -α (kx² + ky²) û
        解析解：û(t+dt) = û(t) * exp(-α (kx² + ky²) dt)

        Parameters
        ----------
        u : ndarray, shape (nx, ny)
            当前温度场（Dirichlet 边界为零）。
        dt : float
            时间步长。

        Returns
        -------
        ndarray
            新时刻温度场。
        """
        # 正弦变换
        u_hat = SineTransform.dst_2d(u)

        # 频域指数衰减
        for i in range(self.nx):
            for j in range(self.ny):
                decay = np.exp(-self.alpha * (self.kx[i] ** 2 + self.ky[j] ** 2) * dt)
                u_hat[i, j] *= decay

        # 逆变换
        u_new = SineTransform.idst_2d(u_hat)
        return u_new
