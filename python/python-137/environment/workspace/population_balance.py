# -*- coding: utf-8 -*-
"""
population_balance.py

博士级人口平衡方程 (PBE) 求解器

融合原项目算法：
- 685_line_nco_rule 的 Newton-Cotes Open 一维求积
- 818_normal_ode 的高斯初始分布与 ODE 验证思想
- 436_flame_exact 的刚性 ODE 精确解思想

科学应用场景：
人口平衡方程描述结晶过程中晶体尺寸分布 (CSD) 的时空演化：

    ∂f(L,t)/∂t + ∂[G(L,t,σ)·f(L,t)]/∂L = B(σ,t)·δ(L-L_0)

其中：
    f(L,t) : 晶体尺寸分布函数 (#/(m³·m))
    G(L,t,σ) : 尺寸依赖生长速率 (m/s)
    B(σ,t) : 成核率 (#/(m³·s))
    L_0 : 临界核尺寸 (m)

本模块实现了基于矩方法的降阶求解，以及基于特征线法的数值求解。

核心公式：
1. 第 j 阶矩定义：μ_j(t) = ∫_0^∞ L^j f(L,t) dL
2. 矩方程：
   dμ_j/dt = j·∫_0^∞ L^{j-1} G(L,σ) f(L,t) dL + B(σ,t)·L_0^j
3. 质量平衡：
   dc/dt = -3·ρ_c·k_v·∫_0^∞ L² G(L,σ) f(L,t) dL = -3·ρ_c·k_v·μ_2·G_eff
4. 平均尺寸：L_{mean} = μ_1 / μ_0
5. 变异系数：CV = √(μ_2/μ_0 - (μ_1/μ_0)²) / (μ_1/μ_0)
"""

import numpy as np
from scipy.integrate import solve_ivp
from growth_kinetics import size_dependent_growth
from nucleation_model import total_nucleation_rate


def newton_cotes_open_weights(n, a=0.0, b=1.0):
    """
    计算 Newton-Cotes Open (NCO) 求积规则的节点和权重。

    数学原理：
        在区间 [a,b] 内取 n 个等距内点（不含端点）：
        x_i = a + i·h,  h = (b-a)/(n+1),  i=1,...,n

        权重通过构造 Lagrange 基函数 L_i(x) 并积分得到：
        w_i = ∫_a^b L_i(x) dx

        使用 Newton 前向差分格式计算多项式系数，再求反导数。

    参数：
        n : int
            节点数
        a, b : float
            积分区间

    返回：
        x : ndarray, shape (n,)
        w : ndarray, shape (n,)
    """
    if n <= 0:
        return np.array([]), np.array([])
    x = np.array([((n - i + 1) * a + i * b) / (n + 1) for i in range(1, n + 1)], dtype=float)

    # 使用 Lagrange 插值构造基函数并积分
    w = np.zeros(n, dtype=float)
    for i in range(n):
        # 构造 L_i(x) = ∏_{j≠i} (x - x_j) / (x_i - x_j)
        # 转换为标准多项式系数
        poly = np.array([1.0])
        denom = 1.0
        for j in range(n):
            if i == j:
                continue
            # 乘以 (x - x_j)
            poly = np.convolve(poly, [1.0, -x[j]])
            denom *= (x[i] - x[j])

        # 积分：∫ poly dx = Σ_k poly_k · x^{k+1} / (k+1)
        antideriv = poly / np.arange(1, len(poly) + 1, dtype=float)
        # 在 a, b 处求值
        val_b = np.polyval(antideriv[::-1], b)
        val_a = np.polyval(antideriv[::-1], a)
        w[i] = (val_b - val_a) / denom

    return x, w


def quadrature_integrate(func, a, b, n=64):
    """
    使用 Newton-Cotes Open 规则数值积分。

    参数：
        func : callable
        a, b : float
        n : int
            节点数

    返回：
        integral : float
    """
    x, w = newton_cotes_open_weights(n, a, b)
    if x.size == 0:
        return 0.0
    f_vals = func(x)
    f_vals = np.asarray(f_vals, dtype=float)
    return float(np.dot(w, f_vals))


class PopulationBalanceSolver:
    """
    基于矩方法的人口平衡方程求解器。
    """

    def __init__(self, params):
        """
        参数：
            params : dict
                - rho_c: 晶体密度 (kg/m³)
                - kv: 体积形状因子
                - L0: 临界核尺寸 (m)
                - k_g0: 生长前置因子 (m/s)
                - E_g: 生长活化能 (J/mol)
                - g_exp: 生长过饱和度指数
                - alpha: 尺寸依赖系数 (1/m)
                - beta: 尺寸依赖指数
                - A_prefactor: 成核前置因子
                - kb_sec: 二级成核系数
                - b_exp: 成核过饱和度指数
                - j_exp: 悬浮密度指数
                - H_diss: 溶解焓 (J/mol)
                - S_diss: 溶解熵 (J/(mol·K))
                - c0: 初始浓度
                - T0: 初始温度 (K)
                - V: 结晶器体积 (m³)
        """
        self.p = params
        self.n_moments = 6  # 计算前 6 阶矩

    def _rhs_moments(self, t, y, T_func):
        """
        矩方程的右端项。

        y = [μ_0, μ_1, ..., μ_{n-1}, c]
        """
        # TODO: Implement the Method of Moments RHS for the Population Balance Equation.
        #
        # Steps required:
        # 1. Extract mu = y[:n] and c = y[n], where n = self.n_moments.
        # 2. Compute temperature T = T_func(t).
        # 3. Compute supersaturation sigma from c, T, H_diss, S_diss via cooling_profile.supersaturation.
        #    Clamp sigma >= 0.
        # 4. Compute magma density MT = rho_c * kv * mu[3].
        # 5. Compute total nucleation rate B using total_nucleation_rate(...).
        # 6. Compute mean crystal size L_mean = mu[1] / mu[0] (guard against mu[0] == 0).
        # 7. Compute growth rate G using size_dependent_growth(L_mean, sigma, T, ...).
        # 8. Build dydt array of length n+1:
        #    - dμ_0/dt = B
        #    - dμ_j/dt = j * G * corr * μ_{j-1} + B * L0^j  for j > 0
        #      where corr = (1 + alpha * L_mean) ** beta
        #    - dc/dt   = -3 * rho_c * kv * μ_2 * G_eff
        #      where G_eff = size_dependent_growth(L_mean, sigma, T, ...)
        # 9. Return dydt.
        raise NotImplementedError("Hole 2: _rhs_moments is not implemented.")

    def solve(self, t_span, T_func, y0=None, method='RK45', rtol=1e-6, atol=1e-9):
        """
        求解人口平衡方程。

        参数：
            t_span : tuple (t0, tf)
            T_func : callable
                温度函数 T(t)
            y0 : ndarray, optional
                初始条件 [μ_0, ..., μ_5, c]
            method : str
            rtol, atol : float

        返回：
            sol : OdeSolution
        """
        n = self.n_moments
        if y0 is None:
            # 初始高斯分布：f(L,0) = N·exp(-(L-L_c)²/(2σ_L²))
            # 对应矩量：μ_j = ∫ L^j f(L) dL
            N0 = 1e10  # 初始晶种数密度
            Lc = 50e-6  # 初始平均尺寸 50 μm
            sigma_L = 10e-6  # 标准差 10 μm
            mu0 = np.zeros(n, dtype=float)
            for j in range(n):
                # 高斯分布的矩：μ_j = N · Σ_{k=0}^{floor(j/2)} C(j,2k)·(2k-1)!!·σ^{2k}·L_c^{j-2k}
                from math import comb, factorial
                s = 0.0
                for k in range(0, j // 2 + 1):
                    double_fact = 1.0
                    for m in range(1, k + 1):
                        double_fact *= (2.0 * m - 1.0)
                    s += comb(j, 2 * k) * double_fact * (sigma_L ** (2 * k)) * (Lc ** (j - 2 * k))
                mu0[j] = N0 * s
            y0 = np.zeros(n + 1, dtype=float)
            y0[:n] = mu0
            y0[n] = self.p['c0']

        y0 = np.asarray(y0, dtype=float)

        def rhs(t, y):
            return self._rhs_moments(t, y, T_func)

        sol = solve_ivp(rhs, t_span, y0, method=method,
                        dense_output=True, rtol=rtol, atol=atol,
                        max_step=(t_span[1] - t_span[0]) / 2000)
        return sol

    def get_moments_at_time(self, sol, t):
        """
        获取指定时刻的矩量。
        """
        t = float(t)
        t0, tf = sol.t[0], sol.t[-1]
        t = max(t0, min(t, tf))
        y = sol.sol(t)
        n = self.n_moments
        mu = y[:n]
        c = y[n]
        return mu, c

    def get_csd_parameters(self, sol, t):
        """
        从矩量反推 CSD 参数（假设对数正态分布）。

        对数正态分布：f(L) = N/(L·σ√(2π)) · exp(-(lnL - μ)²/(2σ²))
        矩关系：
            μ_j = N · exp(j·μ + j²·σ²/2)
        由 μ_0, μ_1, μ_2 可解：
            σ² = ln(μ_2/μ_0) - 2·ln(μ_1/μ_0)
            μ = ln(μ_1/μ_0) - σ²/2
            N = μ_0
        """
        mu, c = self.get_moments_at_time(sol, t)
        N = mu[0]
        if N <= 0:
            return {'N': 0.0, 'mu_ln': 0.0, 'sigma_ln': 1.0, 'L_mean': 0.0, 'CV': 0.0}

        r1 = mu[1] / N
        r2 = mu[2] / N

        # 边界处理：确保 r2 > r1²
        val = r2 / (r1 ** 2)
        val = max(val, 1.0 + 1e-10)
        sigma_ln_sq = np.log(val)
        sigma_ln = np.sqrt(sigma_ln_sq)
        mu_ln = np.log(r1) - 0.5 * sigma_ln_sq
        L_mean = r1
        CV = np.sqrt(np.exp(sigma_ln_sq) - 1.0)

        return {
            'N': N,
            'mu_ln': mu_ln,
            'sigma_ln': sigma_ln,
            'L_mean': L_mean,
            'CV': CV
        }
