# -*- coding: utf-8 -*-
"""
inverse_model.py
裂隙介质渗透率参数反演模块

融合种子项目：
    - 809_nonlin_regula: Regula Falsi（假位法）非线性求根
    - 053_asa266: Dirichlet 分布最大似然估计

在水文地质反演问题中，我们经常需要：
    1. 从示踪试验突破曲线反演等效渗透率
    2. 从水头数据估计裂隙传导系数
    3. 从多井试验数据反演渗透率张量

核心数学模型：

    前向模型（一维对流-弥散解析解，Ogata-Banks）：
        C(x,t) = C_0/2 * [erfc((x - vt)/(2√(Dt))) 
                          + exp(vx/D) * erfc((x + vt)/(2√(Dt)))]
    
    目标函数（最小二乘）：
        J(k) = Σ [C_obs(t_i) - C_sim(t_i; k)]²
    
    等效渗透率与平均流速关系：
        v = K_eq i / n_e
        K_eq = (ρ g b²)/(12 μ) （裂隙的 Cubic Law）

    对于单裂隙示踪试验，穿透时间与渗透率关系：
        t_b ≈ L / v = L n_e / (K_eq i)
    
    反演即为求解：
        f(K) = t_sim(K) - t_obs = 0
"""

import numpy as np
from typing import Tuple, List, Callable, Optional
from scipy.special import erfc


class InverseModel:
    """
    裂隙介质渗透率参数反演器

    结合 Regula Falsi 求根法和 Dirichlet 分布的参数估计，
    实现从观测数据到物理参数的稳健反演。
    """

    def __init__(self):
        self.rho = 1000.0
        self.g = 9.81
        self.mu = 1.0e-3

    @staticmethod
    def regula_falsi(f: Callable[[float], float],
                     a: float, b: float,
                     tol: float = 1.0e-10,
                     max_iter: int = 100) -> Tuple[float, int]:
        """
        Regula Falsi（假位法）求根

        基于 nonlin_regula 的算法：
            c = (a * f(b) - b * f(a)) / (f(b) - f(a))
            
            若 f(c) 与 f(a) 同号：
                a = c, f(a) = f(c)
            否则：
                b = c, f(b) = f(c)

        Parameters
        ----------
        f : callable
            目标函数
        a, b : float
            初始区间端点（需满足 f(a)*f(b) < 0）
        tol : float
            收敛容差
        max_iter : int
            最大迭代次数

        Returns
        -------
        tuple
            (根, 迭代次数)
        """
        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            raise ValueError("f(a) 和 f(b) 必须异号")

        for it in range(max_iter):
            if abs(b - a) < tol:
                break

            # 假位法迭代点
            if abs(fb - fa) < 1e-30:
                break
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)

            if abs(fc) < tol:
                return c, it + 1

            if np.sign(fc) == np.sign(fa):
                a = c
                fa = fc
            else:
                b = c
                fb = fc

        return (a + b) / 2.0, max_iter

    def invert_permeability_from_travel_time(self,
                                              t_obs: float,
                                              L: float,
                                              i_hydraulic: float,
                                              n_e: float = 1.0,
                                              b_guess: float = 1.0e-4,
                                              b_min: float = 1.0e-6,
                                              b_max: float = 1.0e-2) -> dict:
        """
        从示踪剂穿透时间反演裂隙开度和等效渗透率

        模型：
            t_b = L * n_e / (K_eq * i)
            K_eq = ρ g b² / (12 μ)

        反演目标：求 b 使得 t_sim(b) = t_obs

        Parameters
        ----------
        t_obs : float
            观测穿透时间 [s]
        L : float
            传输距离 [m]
        i_hydraulic : float
            水力梯度 [-]
        n_e : float
            有效孔隙度
        b_guess : float
            开度初始猜测 [m]
        b_min, b_max : float
            开度搜索区间 [m]

        Returns
        -------
        dict
            反演结果
        """
        if t_obs <= 0 or L <= 0 or i_hydraulic <= 0:
            raise ValueError("物理参数必须为正")

        def travel_time_error(b):
            if b <= 0:
                return float('inf')
            K_eq = (self.rho * self.g * b ** 2) / (12.0 * self.mu)
            v = K_eq * i_hydraulic / n_e
            if v <= 0:
                return float('inf')
            t_sim = L / v
            return t_sim - t_obs

        # 确保区间有根
        f_min = travel_time_error(b_min)
        f_max = travel_time_error(b_max)

        if f_min * f_max > 0:
            # 扩展搜索区间
            if f_min > 0 and f_max > 0:
                # 两者都正，需要更大的 b
                b_max = b_max * 10.0
                f_max = travel_time_error(b_max)
            elif f_min < 0 and f_max < 0:
                # 两者都负，需要更小的 b
                b_min = max(b_min / 10.0, 1.0e-8)
                f_min = travel_time_error(b_min)

            if f_min * f_max > 0:
                # 使用最小二乘近似
                b_opt = b_guess
                err = float('inf')
                for b_test in np.logspace(np.log10(b_min), np.log10(b_max), 100):
                    e = abs(travel_time_error(b_test))
                    if e < err:
                        err = e
                        b_opt = b_test
                K_opt = (self.rho * self.g * b_opt ** 2) / (12.0 * self.mu)
                return {
                    'aperture': b_opt,
                    'permeability': K_opt,
                    'iterations': -1,
                    'converged': False,
                    'method': 'grid_search'
                }

        b_inv, it = self.regula_falsi(travel_time_error, b_min, b_max)
        K_inv = (self.rho * self.g * b_inv ** 2) / (12.0 * self.mu)

        return {
            'aperture': b_inv,
            'permeability': K_inv,
            'iterations': it,
            'converged': it < 100,
            'method': 'regula_falsi'
        }

    def invert_dispersivity_from_breakthrough(self,
                                               times: np.ndarray,
                                               concentrations: np.ndarray,
                                               L: float,
                                               v: float,
                                               C0: float = 1.0) -> dict:
        """
        从突破曲线反演纵向弥散度

        使用 Ogata-Banks 解析解匹配：
            C(x,t)/C0 = 0.5 * erfc((x - vt)/(2√(Dt)))
        
        目标：最小化 Σ [C_obs - C_sim(D)]²

        Parameters
        ----------
        times : np.ndarray
            时间数组 [s]
        concentrations : np.ndarray
            观测浓度 [kg/m³]
        L : float
            传输距离 [m]
        v : float
            平均流速 [m/s]
        C0 : float
            注入浓度

        Returns
        -------
        dict
            反演结果
        """
        if len(times) != len(concentrations):
            raise ValueError("times 和 concentrations 长度必须相同")

        def objective(D):
            if D <= 0:
                return 1e20
            C_sim = 0.5 * C0 * erfc((L - v * times) / (2.0 * np.sqrt(D * np.maximum(times, 1e-10))))
            return np.sum((concentrations - C_sim) ** 2)

        # 一维搜索最优 D
        D_best = 1.0e-5
        obj_best = objective(D_best)

        for D_test in np.logspace(-8, -2, 100):
            obj = objective(D_test)
            if obj < obj_best:
                obj_best = obj
                D_best = D_test

        alpha_L = D_best / v if v > 0 else 0.0

        return {
            'dispersion_coefficient': D_best,
            'longitudinal_dispersivity': alpha_L,
            'objective_value': obj_best,
            'method': 'least_squares_grid'
        }

    @staticmethod
    def dirichlet_estimate_moments(samples: np.ndarray,
                                    alpha_min: float = 1.0e-5,
                                    max_iter: int = 100,
                                    tol: float = 1.0e-8) -> dict:
        """
        Dirichlet 分布参数的最大似然估计

        基于 dirichlet_estimate 的 Newton-Raphson 算法：
            
            Dirichlet PDF:
                p(x|α) = Γ(Σα_k) / ΠΓ(α_k) * Π x_k^{α_k - 1}
            
            对数似然：
                l(α) = ln Γ(Σα_k) - Σ ln Γ(α_k) + Σ (α_k - 1) ln x_k
            
            Newton 更新：
                α^{(t+1)} = α^{(t)} + H^{-1} ∇l

        在水文地质中的应用：
            - 多相流中各相体积分数的分布建模
            - 裂隙网络中不同方向裂隙比例的统计推断

        Parameters
        ----------
        samples : np.ndarray
            样本矩阵 (n_samples, k)，每行和为 1
        alpha_min : float
            参数下限
        max_iter : int
            最大迭代次数
        tol : float
            收敛容差

        Returns
        -------
        dict
            估计结果
        """
        from scipy.special import digamma, polygamma

        samples = np.asarray(samples)
        if samples.ndim != 2:
            raise ValueError("samples 必须为二维数组")

        n, k = samples.shape
        if n <= k:
            raise ValueError("样本数必须大于维度")

        # 检查约束
        row_sums = np.sum(samples, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ValueError("每行样本的和必须近似为 1")
        if np.any(samples <= 0):
            raise ValueError("所有样本分量必须为正")

        # 矩估计初始化
        means = np.mean(samples, axis=0)
        vars_comp = np.var(samples, axis=0)

        # 使用矩估计初始化 alpha
        s = np.sum(means * (1.0 - means) / (vars_comp + 1e-20))
        alpha0 = means * (s - 1.0)
        alpha0 = np.maximum(alpha0, alpha_min)

        # Newton-Raphson 迭代
        log_x = np.mean(np.log(samples), axis=0)
        alpha = alpha0.copy()

        for it in range(max_iter):
            alpha_sum = np.sum(alpha)

            # 梯度
            g = digamma(alpha_sum) - digamma(alpha) + log_x

            # Hessian 对角线
            h_diag = -polygamma(1, alpha)
            h_off = polygamma(1, alpha_sum)

            # 简化 Newton 步（使用 Woodbury 公式近似逆 Hessian）
            z = h_diag
            b = h_off

            # 近似更新
            denom = 1.0 / z
            c = np.sum(denom)
            if c * b > -1.0:
                inv_h_times_g = g / z - denom * np.sum(g / z) / (1.0 / b + c)
            else:
                inv_h_times_g = g / z

            alpha_new = alpha - inv_h_times_g
            alpha_new = np.maximum(alpha_new, alpha_min)

            if np.max(np.abs(alpha_new - alpha)) < tol:
                alpha = alpha_new
                break

            alpha = alpha_new

        # 计算对数似然
        from scipy.special import gammaln
        ll = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((alpha - 1.0) * log_x)

        return {
            'alpha': alpha,
            'log_likelihood': float(ll),
            'iterations': it + 1,
            'converged': it < max_iter - 1
        }

    def calibrate_dual_porosity(self,
                                 t_obs: np.ndarray,
                                 C_obs: np.ndarray,
                                 L: float,
                                 phi_m: float = 0.01,
                                 phi_im: float = 0.05) -> dict:
        """
        双孔隙介质模型的参数标定

        使用简化的单速率质量交换模型：
            C_m 为可动区浓度，C_im 为不动区浓度
            
            φ_m ∂C_m/∂t + φ_im ∂C_im/∂t + v φ_m ∂C_m/∂x = 0
            
            φ_im ∂C_im/∂t = α (C_m - C_im)

        反演参数：α（质量交换系数）

        Parameters
        ----------
        t_obs : np.ndarray
            观测时间 [s]
        C_obs : np.ndarray
            观测浓度 [kg/m³]
        L : float
            传输距离 [m]
        phi_m : float
            可动区孔隙度
        phi_im : float
            不动区孔隙度

        Returns
        -------
        dict
            标定结果
        """
        if len(t_obs) != len(C_obs):
            raise ValueError("t_obs 和 C_obs 长度必须相同")

        # 简化的双孔隙解析近似
        # 使用经验公式匹配早期和晚期数据
        t_early = t_obs[t_obs <= np.percentile(t_obs, 20)]
        C_early = C_obs[t_obs <= np.percentile(t_obs, 20)]
        t_late = t_obs[t_obs >= np.percentile(t_obs, 80)]
        C_late = C_obs[t_obs >= np.percentile(t_obs, 80)]

        # 计算尾部衰减速率作为质量交换的指标
        if len(t_late) > 1 and np.all(C_late > 0):
            logC = np.log(C_late)
            dt = t_late - np.mean(t_late)
            slope = np.sum(dt * (logC - np.mean(logC))) / np.sum(dt ** 2)
            alpha_est = -slope * phi_im if slope < 0 else 1.0e-5
        else:
            alpha_est = 1.0e-5

        return {
            'mass_transfer_rate': max(alpha_est, 1.0e-8),
            'mobile_porosity': phi_m,
            'immobile_porosity': phi_im,
            'method': 'empirical_tail_fitting'
        }
