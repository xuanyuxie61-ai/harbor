"""
neural_field_solver.py — 二维皮层神经场 PDE 求解器
=====================================================
融合 IFISS（有限元求解器框架思想）、square_grid（规则网格生成）、
polygon_grid（多边形区域三角化）三个项目的核心算法。

求解 Amari 神经场方程（经典皮层动力学模型）：

    τ * ∂u(r,t)/∂t = -u(r,t) + ∫_Ω K(r,r') * S(u(r',t)) dr' + I_ext(r,t)

其中：
  - u(r,t) : 位置 r=(x,y) 处时刻 t 的神经场活动（膜电位）
  - τ      : 时间常数（典型值 10-20 ms）
  - K(r,r'): 空间连接核函数（通常取墨西哥帽或高斯差分）
  - S(·)   : sigmoid 激活函数
  - I_ext  : 外部输入（如感觉刺激或 BCI 指令电流）

离散化方案：
  在 2D 空间域 Ω 上采用规则/非规则网格，空间积分用 Gauss-Legendre 求积，
  时间方向采用半隐式 Euler（对线性项隐式，非线性项显式）以保证稳定性。

空间核函数：
  采用墨西哥帽型（Mexican-hat）连接核：

    K(r) = A_e * exp(-|r|^2 / (2*σ_e^2)) - A_i * exp(-|r|^2 / (2*σ_i^2))

其中 A_e > A_i, σ_e < σ_i，表示近距离兴奋、远距离抑制。
"""

import numpy as np
from utils import sigmoid_activation, gauss_legendre_nodes_weights, rk4_step


def generate_square_grid(xlim=(-1, 1), ylim=(-1, 1), nx=64, ny=64, centering='cell'):
    """
    生成二维矩形网格点，参考 square_grid 项目算法。
    centering 选项：
        'cell'   : 格点位于网格单元中心
        'vertex' : 格点位于网格顶点
        'half'   : 半格点偏移
    返回：X, Y 均为 shape (nx, ny) 的坐标矩阵。
    """
    if centering == 'vertex':
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
    elif centering == 'cell':
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny
        x = np.linspace(xlim[0] + 0.5 * dx, xlim[1] - 0.5 * dx, nx)
        y = np.linspace(ylim[0] + 0.5 * dy, ylim[1] - 0.5 * dy, ny)
    elif centering == 'half':
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny
        x = np.linspace(xlim[0] + 0.25 * dx, xlim[1] - 0.75 * dx, nx)
        y = np.linspace(ylim[0] + 0.25 * dy, ylim[1] - 0.75 * dy, ny)
    else:
        raise ValueError(f"Unknown centering: {centering}")
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y


def generate_polygon_grid_points(vertices, n_subdiv=8):
    """
    在任意多边形内部生成三角化网格点，参考 polygon_grid 项目。
    算法：以多边形重心为第一个点，将多边形划分为 nv 个三角形
    （每个三角形由重心 + 相邻两个顶点构成），然后在每个三角形内
    用重心坐标生成均匀分布的点。

    重心坐标：(λ1, λ2, λ3) 满足 λ1+λ2+λ3 = n_subdiv, λi >= 0
    点坐标：P = (λ1*V1 + λ2*V2 + λ3*V3) / n_subdiv
    """
    vertices = np.asarray(vertices, dtype=float)
    nv = len(vertices)
    if nv < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    centroid = np.mean(vertices, axis=0)
    points = [centroid.copy()]
    for k in range(nv):
        v1 = centroid
        v2 = vertices[k]
        v3 = vertices[(k + 1) % nv]
        for i in range(n_subdiv + 1):
            for j in range(n_subdiv + 1 - i):
                l = n_subdiv - i - j
                lam1 = i / n_subdiv
                lam2 = j / n_subdiv
                lam3 = l / n_subdiv
                p = lam1 * v1 + lam2 * v2 + lam3 * v3
                # 避免重复添加重心
                if not (i == n_subdiv and j == 0 and l == 0):
                    points.append(p)
    return np.array(points, dtype=float)


def mexican_hat_kernel_2d(dx, dy, sigma_e=0.1, sigma_i=0.2, A_e=1.0, A_i=0.5):
    """
    2D 墨西哥帽连接核。
    K(x,y) = A_e * exp(-(x^2+y^2)/(2*σ_e^2)) - A_i * exp(-(x^2+y^2)/(2*σ_i^2))
    """
    r2 = dx ** 2 + dy ** 2
    return A_e * np.exp(-r2 / (2.0 * sigma_e ** 2)) - A_i * np.exp(-r2 / (2.0 * sigma_i ** 2))


class NeuralFieldSolver:
    """
    二维 Amari 神经场方程求解器。
    """

    def __init__(self, X, Y, tau=0.02, sigma_e=0.15, sigma_i=0.3,
                 A_e=1.0, A_i=0.6, theta=0.0, sigma_act=1.0):
        """
        X, Y : meshgrid 坐标矩阵，shape (nx, ny)
        """
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        self.nx, self.ny = X.shape
        self.n_points = self.nx * self.ny
        self.tau = tau
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.A_e = A_e
        self.A_i = A_i
        self.theta = theta
        self.sigma_act = sigma_act
        # 计算面积元（假设均匀网格）
        dx = np.mean(np.diff(X[:, 0])) if self.nx > 1 else 1.0
        dy = np.mean(np.diff(Y[0, :])) if self.ny > 1 else 1.0
        self.dA = abs(dx * dy)
        # 预计算连接核矩阵 K[i,j] = K(|r_i - r_j|)
        self._build_kernel_matrix()

    def _build_kernel_matrix(self):
        """预计算空间连接核的离散矩阵。"""
        nx, ny = self.nx, self.ny
        x_flat = self.X.flatten()
        y_flat = self.Y.flatten()
        n = len(x_flat)
        self.K_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            dx = x_flat[i] - x_flat
            dy = y_flat[i] - y_flat
            self.K_mat[i, :] = mexican_hat_kernel_2d(
                dx, dy, self.sigma_e, self.sigma_i, self.A_e, self.A_i)
        # 归一化
        row_sums = np.sum(self.K_mat, axis=1) * self.dA
        max_sum = np.max(np.abs(row_sums))
        if max_sum > 0:
            self.K_mat /= max_sum

    def _rhs(self, u, I_ext):
        """
        计算 du/dt 的右端项：
            rhs = (-u + K * S(u) + I_ext) / tau
        其中 * 表示空间卷积（离散矩阵-向量乘）。
        """
        n = self.n_points
        Su = sigmoid_activation(u, self.theta, self.sigma_act)
        # 离散积分：sum_j K[i,j] * S(u_j) * dA
        conv = self.K_mat @ Su * self.dA
        return (-u + conv + I_ext) / self.tau

    def simulate(self, u0, I_ext_func, t_span=(0.0, 1.0), dt=0.001, method='rk4'):
        """
        模拟神经场演化。
        I_ext_func : callable(t, X, Y) -> array shape (nx, ny)
        返回 t, u_history shape (n_t, nx, ny)
        """
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        u = np.asarray(u0, dtype=float).flatten()
        u_hist = np.zeros((n_steps + 1, self.n_points), dtype=float)
        u_hist[0] = u.copy()
        for i in range(n_steps):
            I_ext = I_ext_func(t[i], self.X, self.Y).flatten()
            if method == 'rk4':
                u = rk4_step(lambda ti, ui: self._rhs(ui, I_ext_func(ti, self.X, self.Y).flatten()),
                             t[i], u, dt)
            elif method == 'euler':
                u = u + dt * self._rhs(u, I_ext)
            else:
                raise ValueError(f"Unknown method: {method}")
            u_hist[i + 1] = u.copy()
        return t, u_hist.reshape(n_steps + 1, self.nx, self.ny)

    def compute_spatial_spectrum(self, u_field):
        """
        计算神经场的空间功率谱（2D FFT 模平方）。
        返回 kx, ky, power_spectrum
        """
        nx, ny = self.nx, self.ny
        # 假设均匀网格
        dx = np.mean(np.diff(self.X[:, 0])) if nx > 1 else 1.0
        dy = np.mean(np.diff(self.Y[0, :])) if ny > 1 else 1.0
        fft_u = np.fft.fft2(u_field)
        power = np.abs(fft_u) ** 2
        kx = np.fft.fftfreq(nx, d=dx)
        ky = np.fft.fftfreq(ny, d=dy)
        return kx, ky, power


class NeuralFieldWithGaussQuadrature:
    """
    使用 Gauss-Legendre 数值积分提高空间积分精度的神经场求解器。
    在每个网格单元内使用高斯点进行子像素积分。
    """

    def __init__(self, domain=(-1.0, 1.0, -1.0, 1.0), n_quad=8, n_grid=32,
                 tau=0.02, sigma_e=0.15, sigma_i=0.3, A_e=1.0, A_i=0.6):
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.n_quad = n_quad
        self.n_grid = n_grid
        self.tau = tau
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.A_e = A_e
        self.A_i = A_i
        # Gauss-Legendre 节点与权重
        self.xi, self.wi = gauss_legendre_nodes_weights(n_quad)
        # 网格中心点
        dx = (self.xmax - self.xmin) / n_grid
        dy = (self.ymax - self.ymin) / n_grid
        self.xc = np.linspace(self.xmin + 0.5 * dx, self.xmax - 0.5 * dx, n_grid)
        self.yc = np.linspace(self.ymin + 0.5 * dy, self.ymax - 0.5 * dy, n_grid)
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing='ij')
        self.dx = dx
        self.dy = dy
        # 预计算核矩阵（从网格中心到高斯点的映射）
        self._build_quadrature_kernel()

    def _build_quadrature_kernel(self):
        """
        构建基于高斯积分的核矩阵。
        对每个目标网格点 (i,j)，积分源域时采用高斯点：
            x_s = xc_k + dx/2 * xi_m
            y_s = yc_l + dy/2 * xi_n
            weight = dx*dy/4 * wi_m * wi_n
        """
        n = self.n_grid
        nq = self.n_quad
        self.K_quad = np.zeros((n * n, n * n), dtype=float)
        # 高斯点局部偏移
        x_offset = self.dx * 0.5 * self.xi
        y_offset = self.dy * 0.5 * self.xi
        for i in range(n):
            for j in range(n):
                idx_target = i * n + j
                xt = self.Xc[i, j]
                yt = self.Yc[i, j]
                for k in range(n):
                    for l in range(n):
                        idx_source = k * n + l
                        xs_center = self.Xc[k, l]
                        ys_center = self.Yc[k, l]
                        # 在该网格单元内积分
                        integral_val = 0.0
                        for m in range(nq):
                            for n_ in range(nq):
                                xs = xs_center + x_offset[m]
                                ys = ys_center + y_offset[n_]
                                dx_ = xt - xs
                                dy_ = yt - ys
                                k_val = mexican_hat_kernel_2d(
                                    dx_, dy_, self.sigma_e, self.sigma_i, self.A_e, self.A_i)
                                w = 0.25 * self.dx * self.dy * self.wi[m] * self.wi[n_]
                                integral_val += k_val * w
                        self.K_quad[idx_target, idx_source] = integral_val
        # 归一化行
        row_sums = np.sum(self.K_quad, axis=1)
        max_sum = np.max(np.abs(row_sums))
        if max_sum > 0:
            self.K_quad /= max_sum

    def simulate(self, u0, I_ext_func, t_span=(0.0, 1.0), dt=0.001):
        """
        I_ext_func(t, Xc, Yc) -> array shape (n_grid, n_grid)
        """
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt))
        t = np.linspace(t0, tf, n_steps + 1)
        n = self.n_grid
        u = np.asarray(u0, dtype=float).flatten()
        u_hist = np.zeros((n_steps + 1, n * n), dtype=float)
        u_hist[0] = u.copy()
        for step in range(n_steps):
            I_ext = I_ext_func(t[step], self.Xc, self.Yc).flatten()
            Su = sigmoid_activation(u, theta=0.0, sigma=1.0)
            conv = self.K_quad @ Su
            dudt = (-u + conv + I_ext) / self.tau
            u = u + dt * dudt
            u_hist[step + 1] = u.copy()
        return t, u_hist.reshape(n_steps + 1, n, n)
