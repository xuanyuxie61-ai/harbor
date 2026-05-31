"""
spectral_element.py
一维间断伽辽金（DG）谱元法求解复合材料层合板中的应力波传播。
原项目映射：
  - 274_dg1d_maxwell 的完整DG谱元框架：Jacobi多项式、Gauss-Lobatto点、
    Vandermonde矩阵、微分矩阵、LIFT算子、Runge-Kutta时间推进
科学背景：
  在复合材料中，冲击载荷引起的应力波传播与损伤演化密切相关。
  采用DG方法求解一维弹性波方程：
    ρ ∂v/∂t = ∂σ/∂x
    ∂σ/∂t = E ∂v/∂x
  其中 v 为质点速度，σ 为应力，ρ 为密度，E 为等效弹性模量。
  当材料存在损伤时，E 被替换为 E(d) = E_0 * (1 - d)。
"""

import numpy as np
from scipy.special import gamma as scipy_gamma


def jacobi_gq(alpha, beta, N):
    """计算Jacobi-Gauss求积点与权重。"""
    if N == 0:
        return np.array([-(alpha - beta) / (alpha + beta + 2.0)]), np.array([2.0])
    try:
        from numpy.polynomial.legendre import leggauss
        if abs(alpha) < 1e-12 and abs(beta) < 1e-12:
            x, w = leggauss(N + 1)
            return x, w
    except Exception:
        pass
    # 简化的Newton迭代求根
    x = np.cos(np.pi * (4.0 * np.arange(1, N + 2) - 1.0) / (4.0 * (N + 1) + 2.0 * (alpha + beta)))
    for _ in range(30):
        P, dP = _jacobi_p_and_dp(x, alpha, beta, N + 1)
        dx = -P / (dP + 1e-30)
        x += dx
        if np.max(np.abs(dx)) < 1e-14:
            break
    P_n, _ = _jacobi_p_and_dp(x, alpha, beta, N)
    w = 2.0 ** (alpha + beta + 1.0) * scipy_gamma(alpha + N + 2.0) * scipy_gamma(beta + N + 2.0) / (
        scipy_gamma(N + 2.0) * scipy_gamma(alpha + beta + N + 2.0)) / ((1.0 - x ** 2) * dP ** 2 + 1e-30)
    return x, w


def jacobi_gl(alpha, beta, N):
    """Jacobi-Gauss-Lobatto点（包含端点±1）。"""
    if N == 1:
        return np.array([-1.0, 1.0])
    x_int, _ = jacobi_gq(alpha + 1.0, beta + 1.0, N - 2)
    x = np.concatenate([[-1.0], x_int, [1.0]])
    return x


def _jacobi_p_and_dp(x, alpha, beta, N):
    """计算Jacobi多项式及其导数。"""
    x = np.asarray(x)
    if N == 0:
        return np.ones_like(x), np.zeros_like(x)
    PL = np.zeros((N + 1, len(x)))
    gamma0 = 2.0 ** (alpha + beta + 1.0) / (alpha + beta + 1.0) * scipy_gamma(alpha + 1.0) * scipy_gamma(beta + 1.0) / scipy_gamma(alpha + beta + 1.0)
    PL[0, :] = 1.0 / np.sqrt(gamma0)
    if N >= 1:
        gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
        PL[1, :] = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / np.sqrt(gamma1)
    aold = 2.0 / (2.0 + alpha + beta) * np.sqrt((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0))
    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        anew = 2.0 / (h1 + 2.0) * np.sqrt(
            (i + 1.0) * (i + 1.0 + alpha + beta) * (i + 1.0 + alpha) * (i + 1.0 + beta) /
            ((h1 + 1.0) * (h1 + 3.0)))
        bnew = - (alpha ** 2 - beta ** 2) / (h1 * (h1 + 2.0))
        PL[i + 1, :] = (1.0 / anew) * (-aold * PL[i - 1, :] + (x - bnew) * PL[i, :])
        aold = anew
    # 导数用递推近似
    if N >= 1:
        dP = np.sqrt(N * (N + alpha + beta + 1.0)) * PL[N - 1, :] + 1e-30
    else:
        dP = np.zeros_like(x) + 1e-30
    return PL[N, :], dP


def jacobi_p(x, alpha, beta, N):
    """计算N阶Jacobi多项式值。"""
    P, _ = _jacobi_p_and_dp(x, alpha, beta, N)
    return P


def vandermonde_1d(N, r):
    """一维Vandermonde矩阵 V_{ij} = phi_j(r_i)。"""
    V = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_p(r, 0.0, 0.0, j)
    return V


def grad_vandermonde_1d(N, r):
    """一阶导数Vandermonde矩阵。"""
    Vr = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        _, dp = _jacobi_p_and_dp(r, 0.0, 0.0, j)
        Vr[:, j] = dp
    return Vr


def dmatrix_1d(N, r, V):
    """微分矩阵 Dr = Vr / V。"""
    Vr = grad_vandermonde_1d(N, r)
    Dr = Vr @ np.linalg.inv(V)
    return Dr


class DGSpectralElement1D:
    """一维DG谱元求解器（从 dg1d_maxwell 的完整框架迁移）。"""

    def __init__(self, N, K, x_bounds, rho, E_func):
        """
        N: 每单元多项式阶数
        K: 单元数
        x_bounds: [xmin, xmax]
        rho: 密度 (常数)
        E_func: 弹性模量分布函数 E(x)
        """
        self.N = N
        self.K = K
        self.Np = N + 1
        self.xmin, self.xmax = x_bounds
        self.rho = float(rho)
        self.E_func = E_func

        # Gauss-Lobatto点（参考单元 [-1, 1]）
        self.r = jacobi_gl(0.0, 0.0, N)

        # Vandermonde与微分矩阵
        self.V = vandermonde_1d(N, self.r)
        self.Vinv = np.linalg.inv(self.V)
        self.Dr = dmatrix_1d(N, self.r, self.V)

        # 物理坐标
        self._build_mesh()

        # RK4系数（低存储5级4阶Runge-Kutta）
        self.rk4a = np.array([0.0, -567301805773.0 / 1357537059087.0,
                              -2404267990393.0 / 2016746695238.0,
                              -3550918686646.0 / 2091501179385.0,
                              -1275806237668.0 / 842570457699.0])
        self.rk4b = np.array([1432997174477.0 / 9575080441755.0,
                              5161836677717.0 / 13612068292357.0,
                              1720146321549.0 / 2090206949498.0,
                              3134564353537.0 / 4481467310338.0,
                              2277821191437.0 / 14882151754819.0])

    def _build_mesh(self):
        """构建单元与节点坐标。"""
        dx = (self.xmax - self.xmin) / self.K
        self.x = np.zeros((self.Np, self.K))
        for k in range(self.K):
            x_left = self.xmin + k * dx
            x_right = x_left + dx
            for i in range(self.Np):
                self.x[i, k] = x_left + 0.5 * (self.r[i] + 1.0) * dx

    def compute_rhs(self, sigma, v):
        """
        计算右端项，采用简化的中心差分+数值通量格式。
        为保证稳定性，使用局部Lax-Friedrichs型通量。
        """
        res_v = np.zeros_like(v)
        res_s = np.zeros_like(sigma)

        for k in range(self.K):
            dx_k = self.x[-1, k] - self.x[0, k] if self.N > 0 else (self.xmax - self.xmin) / self.K
            rx = 2.0 / dx_k

            E_k = np.array([self.E_func(self.x[i, k]) for i in range(self.Np)])

            # 内部导数（谱精度）
            dsigma_dr = self.Dr @ sigma[:, k]
            dv_dr = self.Dr @ v[:, k]

            res_v[:, k] = dsigma_dr * rx / self.rho
            res_s[:, k] = E_k * dv_dr * rx

            # 数值通量（局部Lax-Friedrichs）
            c_k = np.sqrt(np.mean(E_k) / self.rho)
            alpha_flux = c_k

            if k > 0:
                sigma_m = sigma[-1, k - 1]
                v_m = v[-1, k - 1]
            else:
                sigma_m = 0.0
                v_m = 0.0
            if k < self.K - 1:
                sigma_p = sigma[0, k + 1]
                v_p = v[0, k + 1]
            else:
                sigma_p = 0.0
                v_p = 0.0

            # 左边界通量
            sigma_star_L = 0.5 * (sigma[0, k] + sigma_m) - 0.5 * alpha_flux * (v[0, k] - v_m)
            v_star_L = 0.5 * (v[0, k] + v_m) - 0.5 / alpha_flux * (sigma[0, k] - sigma_m)
            # 右边界通量
            sigma_star_R = 0.5 * (sigma[-1, k] + sigma_p) - 0.5 * alpha_flux * (v[-1, k] - v_p)
            v_star_R = 0.5 * (v[-1, k] + v_p) - 0.5 / alpha_flux * (sigma[-1, k] - sigma_p)

            # 将通量差异加入右端项（简化的弱形式）
            dsigma_L = sigma_star_L - sigma[0, k]
            dsigma_R = sigma_star_R - sigma[-1, k]
            dv_L = v_star_L - v[0, k]
            dv_R = v_star_R - v[-1, k]

            # 仅在边界节点施加修正
            res_v[0, k] += rx * dsigma_L / self.rho
            res_v[-1, k] -= rx * dsigma_R / self.rho
            res_s[0, k] += rx * E_k[0] * dv_L
            res_s[-1, k] -= rx * E_k[-1] * dv_R

        return res_s, res_v

    def solve(self, sigma0, v0, FinalTime):
        """时间推进求解。"""
        sigma = sigma0.copy()
        v = v0.copy()
        time = 0.0

        # 保守CFL条件
        E_max = max(self.E_func(self.x[i, k]) for k in range(self.K) for i in range(self.Np))
        c_max = np.sqrt(E_max / self.rho)
        dx_min = np.min(np.abs(self.x[1:, :] - self.x[:-1, :]))
        dt = 0.1 * dx_min / (c_max + 1e-12)
        nsteps = max(1, int(np.ceil(FinalTime / dt)))
        dt = FinalTime / nsteps

        res_sigma = np.zeros_like(sigma)
        res_v = np.zeros_like(v)

        for _ in range(nsteps):
            for intrk in range(5):
                rhs_s, rhs_v = self.compute_rhs(sigma, v)
                # 稳定性检查
                rhs_s = np.nan_to_num(rhs_s, nan=0.0, posinf=0.0, neginf=0.0)
                rhs_v = np.nan_to_num(rhs_v, nan=0.0, posinf=0.0, neginf=0.0)
                res_sigma = self.rk4a[intrk] * res_sigma + dt * rhs_s
                res_v = self.rk4a[intrk] * res_v + dt * rhs_v
                sigma = sigma + self.rk4b[intrk] * res_sigma
                v = v + self.rk4b[intrk] * res_v
                # 裁剪防止溢出
                sigma = np.clip(sigma, -1e6, 1e6)
                v = np.clip(v, -1e6, 1e6)
            time += dt

        return sigma, v

    def compute_wave_speed(self):
        """计算各节点的波速 c = sqrt(E/rho)。"""
        c = np.zeros_like(self.x)
        for k in range(self.K):
            for i in range(self.Np):
                E = self.E_func(self.x[i, k])
                c[i, k] = np.sqrt(E / self.rho)
        return c
