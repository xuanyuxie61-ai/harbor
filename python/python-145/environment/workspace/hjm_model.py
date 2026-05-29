"""
hjm_model.py
============
博士级多因子 Heath-Jarrow-Morton (HJM) 利率期限结构模型

本模块实现完整的 HJM 框架，将随机动力学（Lorenz-96、Duffing、Oregonator）
耦合为期限结构的多因子波动率驱动，结合多项式混沌展开进行不确定性量化。

数学理论
--------
Heath-Jarrow-Morton 框架 (1992):
    设 f(t, T) 为时刻 t 观察到的、在 T 时刻生效的瞬时前向利率。
    在风险中性测度 Q 下，其演化满足:

        df(t, T) = α(t, T) dt + Σ_{i=1}^d σ_i(t, T, f) dW_i(t)

    无套利条件（HJM 漂移限制）:
        α(t, T) = Σ_{i=1}^d σ_i(t, T, f) ∫_t^T σ_i(t, s, f) ds

Musiela 参数化（s = T - t，剩余期限）:
    r_t(s) = f(t, t + s)
    dr_t(s) = (∂r_t(s)/∂s + α̃(t, s)) dt + Σ σ̃_i(t, s) dW_i(t)

多因子波动率结构（本模型采用）:
    σ_1(t, s) = σ_0 * exp(-κ_1 s)          （水平因子，Vasicek 型）
    σ_2(t, s) = σ_0 * s * exp(-κ_2 s)      （斜率因子）
    σ_3(t, s) = σ_chaos(t) * g(s)          （混沌驱动因子）

其中 σ_chaos(t) 由 Lorenz-96、Duffing 和 Oregonator 系统耦合投影得到。

多项式混沌展开:
    r_t(s; ξ) = Σ_{|α|≤p} r_{t,α}(s) He_α(ξ)
    其中 ξ ~ N(0, I_d) 为标准正态随机向量。

债券定价:
    P(t, T) = exp(-∫_0^{T-t} r_t(s) ds)
    B(t) = exp(∫_0^t r_u(0) du)    （货币市场账户）
"""

import numpy as np
import stochastic_dynamics as sd
import polynomial_chaos_uq as pc
import special_functions as sf
import time_stepping as ts


class HJMMultiFactorModel:
    """
    多因子 HJM 期限结构模型。

    Attributes
    ----------
    n_factors : int
        波动率因子数。
    sigma0 : float
        基准波动率水平。
    kappa : np.ndarray
        指数衰减参数。
    lorenz_params : tuple
        Lorenz-96 参数。
    duffing_params : tuple
        Duffing 参数。
    oregonator_params : tuple
        Oregonator 参数。
    pc_degree : int
        多项式混沌展开阶数。
    pc_dim : int
        混沌展开随机维度。
    """

    def __init__(self, n_factors=3, sigma0=0.02, kappa=None,
                 lorenz_n=8, lorenz_force=8.0,
                 duffing_alpha=1.0, duffing_beta=5.0,
                 duffing_gamma=8.0, duffing_delta=0.02, duffing_omega=0.5,
                 oregonator_f=1.0,
                 pc_degree=3, pc_dim=3):
        """
        初始化 HJM 多因子模型。

        Parameters
        ----------
        n_factors : int
            波动率因子数。
        sigma0 : float
            基准波动率，必须 >= 0。
        kappa : np.ndarray or None
            指数衰减参数，shape (n_factors,)；None 时使用默认值。
        pc_degree : int
            混沌展开阶数。
        pc_dim : int
            随机维度。
        """
        if sigma0 < 0.0:
            raise ValueError("HJMMultiFactorModel: sigma0 必须非负")
        if n_factors < 1:
            raise ValueError("HJMMultiFactorModel: n_factors 必须至少为 1")

        self.n_factors = n_factors
        self.sigma0 = sigma0
        self.kappa = kappa if kappa is not None else np.array([0.1, 0.3, 0.5])
        self.kappa = np.asarray(self.kappa, dtype=float)
        if self.kappa.shape[0] < n_factors:
            self.kappa = np.pad(self.kappa, (0, n_factors - self.kappa.shape[0]),
                                constant_values=0.5)

        # Lorenz-96 参数
        self.lorenz_n = lorenz_n
        self.lorenz_force = lorenz_force
        _, _, _, _, self.lorenz_y0, _ = sd.lorenz96_parameters(
            n=lorenz_n, force=lorenz_force)

        # Duffing 参数
        self.duffing_alpha = duffing_alpha
        self.duffing_beta = duffing_beta
        self.duffing_gamma = duffing_gamma
        self.duffing_delta = duffing_delta
        self.duffing_omega = duffing_omega
        _, _, _, _, _, _, self.duffing_y0, _ = sd.duffing_parameters(
            alpha=duffing_alpha, beta=duffing_beta, gamma=duffing_gamma,
            delta=duffing_delta, omega=duffing_omega)

        # Oregonator 参数
        self.oregonator_f = oregonator_f
        _, _, _, _, _, self.oregonator_y0, _ = sd.oregonator_parameters(f=oregonator_f)

        # 多项式混沌
        self.pc_degree = pc_degree
        self.pc_dim = pc_dim
        self.multi_indices = pc.generate_multi_indices(pc_dim, pc_degree)

    def volatility_structure(self, t, s, lorenz_state, duffing_state, oregonator_state):
        """
        计算多因子波动率结构 σ_i(t, s)。

        公式:
            σ_1(t, s) = σ_0 * exp(-κ_1 s)
            σ_2(t, s) = σ_0 * s * exp(-κ_2 s)
            σ_3(t, s) = σ_coupled * h(s)
        其中 σ_coupled 由 multi_factor_coupling 投影得到。

        Parameters
        ----------
        t : float
            当前时间。
        s : float
            剩余期限，必须 >= 0。
        lorenz_state : np.ndarray
            Lorenz-96 状态。
        duffing_state : np.ndarray
            Duffing 状态。
        oregonator_state : np.ndarray
            Oregonator 状态。

        Returns
        -------
        np.ndarray, shape (n_factors,)
            各因子波动率。
        """
        if s < 0.0:
            raise ValueError("volatility_structure: s 必须非负")

        sigma = np.zeros(self.n_factors, dtype=float)

        # 标准 HJM 因子（指数型）
        sigma[0] = self.sigma0 * np.exp(-self.kappa[0] * s)
        if self.n_factors > 1:
            sigma[1] = self.sigma0 * s * np.exp(-self.kappa[1] * s)

        # 混沌耦合因子
        if self.n_factors > 2:
            sigma_chaos = sd.multi_factor_coupling(
                t, lorenz_state, duffing_state, oregonator_state,
                n_factors=1, coupling_matrix=np.array([[0.0, 0.0, 1.0]]))
            # 引入期限衰减使长期波动率趋于稳定
            sigma[2] = sigma_chaos[0] * self.sigma0 * np.exp(-self.kappa[2] * s)

        # 数值鲁棒性
        sigma = np.clip(sigma, 0.0, 1.0)
        return sigma

    def drift_term(self, t, s, lorenz_state, duffing_state, oregonator_state):
        """
        HJM 无套利漂移项。

        公式:
            α(t, s) = Σ_i σ_i(t, s) ∫_0^s σ_i(t, u) du

        Parameters
        ----------
        t : float
            当前时间。
        s : float
            剩余期限。
        lorenz_state : np.ndarray
            Lorenz-96 状态。
        duffing_state : np.ndarray
            Duffing 状态。
        oregonator_state : np.ndarray
            Oregonator 状态。

        Returns
        -------
        float
            漂移项 α(t, s)。
        """
        if s <= 0.0:
            return 0.0

        import term_structure_pde as tsp
        sigma_funcs = []
        for i in range(self.n_factors):
            def make_sigma(i_factor):
                def sigma_i(t_, s_):
                    return self.volatility_structure(t_, s_, lorenz_state,
                                                      duffing_state, oregonator_state)[i_factor]
                return sigma_i
            sigma_funcs.append(make_sigma(i))

        return tsp.musiela_drift(sigma_funcs, s, t)

    def evolve_stochastic_dynamics(self, t, dt, lorenz_y, duffing_y, oregonator_y):
        """
        推进随机动力学系统一个时间步。

        使用 RK3 分别积分 Lorenz-96、Duffing 和 Oregonator。

        Parameters
        ----------
        t : float
            当前时间。
        dt : float
            时间步长。
        lorenz_y : np.ndarray
            Lorenz-96 当前状态。
        duffing_y : np.ndarray
            Duffing 当前状态。
        oregonator_y : np.ndarray
            Oregonator 当前状态。

        Returns
        -------
        tuple
            (lorenz_new, duffing_new, oregonator_new)
        """
        # Lorenz-96
        def lorenz_rhs(tt, yy):
            return sd.lorenz96_deriv(tt, yy, force=self.lorenz_force)
        lorenz_new = ts.rk3_step(lorenz_rhs, t, lorenz_y, dt)

        # Duffing
        def duffing_rhs(tt, yy):
            return sd.duffing_deriv(tt, yy,
                                    alpha=self.duffing_alpha,
                                    beta=self.duffing_beta,
                                    gamma=self.duffing_gamma,
                                    delta=self.duffing_delta,
                                    omega=self.duffing_omega)
        duffing_new = ts.rk3_step(duffing_rhs, t, duffing_y, dt)

        # Oregonator: 使用更小步长多次积分以保持稳定性
        eta1, eta2, q, f, _, _, _ = sd.oregonator_parameters(f=self.oregonator_f)

        def oregonator_rhs(tt, yy):
            return sd.oregonator_deriv(tt, yy, eta1, eta2, q, f)

        oregonator_new = oregonator_y.copy()
        n_sub = max(1, int(np.ceil(dt / 0.01)))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            oregonator_new = ts.rk3_step(oregonator_rhs, t, oregonator_new, dt_sub)
            # 数值鲁棒性
            if np.any(np.isnan(oregonator_new)) or np.any(np.isinf(oregonator_new)):
                oregonator_new = np.array([1.0, 1.0, 1.0], dtype=float)
                break
            oregonator_new = np.clip(oregonator_new, -50.0, 50.0)

        # 全局数值检查
        lorenz_new = np.nan_to_num(lorenz_new, nan=0.0, posinf=50.0, neginf=-50.0)
        duffing_new = np.nan_to_num(duffing_new, nan=0.0, posinf=50.0, neginf=-50.0)
        oregonator_new = np.nan_to_num(oregonator_new, nan=1.0, posinf=50.0, neginf=-50.0)

        return lorenz_new, duffing_new, oregonator_new

    def simulate_path(self, T_grid, f_init, t_max, dt,
                      nu=0.001, mu_func=None, forcing_func=None):
        """
        模拟一条前向利率曲线路径。

        结合 PDE 输运-扩散与随机动力学驱动。

        Parameters
        ----------
        T_grid : np.ndarray
            期限网格。
        f_init : np.ndarray
            初始前向利率曲线。
        t_max : float
            终止时间。
        dt : float
            时间步长。
        nu : float
            扩散系数。
        mu_func : callable or None
            漂移速度；None 时使用 μ(T) = -0.01。
        forcing_func : callable or None
            外部强迫；None 时使用 F=0。

        Returns
        -------
        t_history : np.ndarray
            时间序列。
        f_history : np.ndarray
            前向利率历史。
        dynamics_history : list
            动力学状态历史。
        """
        import term_structure_pde as tsp
        from scipy import sparse as sp
        from scipy.sparse.linalg import spsolve

        T_grid = np.asarray(T_grid, dtype=float)
        f_init = np.asarray(f_init, dtype=float)
        N = len(T_grid)
        n_steps = int(np.ceil(t_max / dt))
        dt = t_max / n_steps

        if mu_func is None:
            def mu_func_default(T):
                return -0.01
            mu_func = mu_func_default

        if forcing_func is None:
            def forcing_default(t, T):
                return 0.0
            forcing_func = forcing_default

        t_history = np.zeros(n_steps + 1, dtype=float)
        f_history = np.zeros((n_steps + 1, N), dtype=float)
        dynamics_history = []

        f = f_init.copy()
        lorenz_y = self.lorenz_y0.copy()
        duffing_y = self.duffing_y0.copy()
        oregonator_y = self.oregonator_y0.copy()

        t_history[0] = 0.0
        f_history[0, :] = f
        dynamics_history.append((lorenz_y.copy(), duffing_y.copy(), oregonator_y.copy()))

        for step in range(n_steps):
            t = step * dt
            t_new = (step + 1) * dt

            # 更新随机动力学
            lorenz_y, duffing_y, oregonator_y = self.evolve_stochastic_dynamics(
                t, dt, lorenz_y, duffing_y, oregonator_y)
            dynamics_history.append((lorenz_y.copy(), duffing_y.copy(), oregonator_y.copy()))

            # 更新波动率函数引用
            def sigma_factory_update(i):
                def sigma_i(t_, s_):
                    return self.volatility_structure(t_, s_, lorenz_y, duffing_y, oregonator_y)[i]
                return sigma_i

            sigma_funcs = [sigma_factory_update(i) for i in range(self.n_factors)]

            # 推进 PDE 一个时间步
            _, A_fd, rhs_f = tsp.forward_rate_pde_rhs(
                t, T_grid, f, nu, mu_func, forcing_func, sigma_funcs)

            # TODO HOLE_2: 实现隐式后向 Euler 时间推进格式。
            # 已知 forward_rate_pde_rhs 返回空间离散矩阵 A_fd 和强迫项 rhs_f，
            # 满足 dfdt ≈ A_fd @ f + rhs_f。
            # 需要构造并求解：
            #   (I - dt * A_fd) * f_new = f + dt * rhs_f
            # 同时施加 Dirichlet 边界条件：
            #   f_new[0]  = max(f[0], 0.01)   (短期利率下界)
            #   f_new[-1] = max(f[-1], 0.02)  (长期利率下界)
            # 最后对 f_new 做非负截断：f = clip(f_new, 0.0, None)
            raise NotImplementedError("HOLE_2: 隐式时间推进格式尚未实现")

            t_history[step + 1] = t_new
            f_history[step + 1, :] = f

        return t_history, f_history, dynamics_history
