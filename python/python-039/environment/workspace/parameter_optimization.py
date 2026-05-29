"""
parameter_optimization.py
QGP状态方程与输运参数优化

基于种子项目:
- 836_opt_quadratic: 二次插值优化

物理应用:
1. 通过拟合实验数据优化QGP状态方程参数 c_s²
2. 优化比剪切粘滞系数 η/s
3. 优化热化时间 τ₀
4. 优化初始能量密度剖面参数

优化模型:
对于目标函数 f(x)，给定三个初始点 x1, x2, x3，
构造二次插值多项式 p(x) = a x² + b x + c，
通过 p'(x*) = 0 求得极值点 x* = -b / (2a)。
"""

import numpy as np
from typing import Callable, Tuple, Optional, Dict


class QuadraticOptimizer:
    """
    基于二次插值的单变量优化器。
    """

    def __init__(self, max_iter: int = 50, x_tol: float = 1e-6,
                 f_tol: float = 1e-8):
        """
        初始化优化器参数。

        Parameters
        ----------
        max_iter : int
            最大迭代次数
        x_tol : float
            x收敛容差
        f_tol : float
            函数值收敛容差
        """
        self.max_iter = max_iter
        self.x_tol = x_tol
        self.f_tol = f_tol

    def optimize(self, f: Callable[[float], float],
                 x1: float, x2: float, x3: float) -> Tuple[float, int, float]:
        """
        使用二次插值寻找函数的临界点。

        算法:
        1. 在三点 (x1, f1), (x2, f2), (x3, f3) 上构造二次多项式
        2. 解Vandermonde方程组得到系数
        3. 极值点 x* = -b/(2a)
        4. 用x*替换最差的点，迭代

        Parameters
        ----------
        f : Callable
            目标函数
        x1, x2, x3 : float
            三个初始点（必须互不相同）

        Returns
        -------
        x_opt : float
            最优x值
        iters : int
            迭代次数
        f_opt : float
            最优函数值
        """
        # 边界检查
        if abs(x1 - x2) < 1e-15 or abs(x2 - x3) < 1e-15 or abs(x1 - x3) < 1e-15:
            raise ValueError("三个初始点必须互不相同")

        x = [x1, x2, x3]
        fx = [f(xi) for xi in x]

        for it in range(1, self.max_iter + 1):
            # 排序：f(x1) <= f(x2) <= f(x3)
            idx = np.argsort(fx)
            x = [x[i] for i in idx]
            fx = [fx[i] for i in idx]

            # 构造Vandermonde矩阵并求解二次多项式系数
            V = np.array([
                [x[0] ** 2, x[0], 1.0],
                [x[1] ** 2, x[1], 1.0],
                [x[2] ** 2, x[2], 1.0]
            ])
            try:
                p = np.linalg.solve(V, fx)
            except np.linalg.LinAlgError:
                break

            a, b, c = p
            # 极值点
            if abs(a) < 1e-15:
                # 线性情况，取中点
                x_star = (x[0] + x[2]) / 2.0
            else:
                x_star = -b / (2.0 * a)

            f_star = f(x_star)

            # 收敛判断
            if abs(x_star - x[1]) < self.x_tol and abs(f_star - fx[1]) < self.f_tol:
                return x_star, it, f_star

            # 替换最差的点
            if f_star < fx[0]:
                x[2] = x[1]
                fx[2] = fx[1]
                x[1] = x_star
                fx[1] = f_star
            elif f_star < fx[1]:
                x[2] = x[1]
                fx[2] = fx[1]
                x[1] = x_star
                fx[1] = f_star
            elif f_star < fx[2]:
                x[2] = x_star
                fx[2] = f_star
            else:
                # 收缩
                x[2] = (x[1] + x[2]) / 2.0
                fx[2] = f(x[2])

        # 返回最佳点
        idx_best = np.argmin(fx)
        return x[idx_best], self.max_iter, fx[idx_best]


class QGPParameterFit:
    """
    QGP物理参数的实验数据拟合。
    """

    def __init__(self, optimizer: Optional[QuadraticOptimizer] = None):
        """
        初始化拟合器。

        Parameters
        ----------
        optimizer : QuadraticOptimizer
            优化器实例
        """
        self.optimizer = optimizer if optimizer is not None else QuadraticOptimizer()

    def fit_eta_over_s(self, v2_data: np.ndarray,
                       pt_bins: np.ndarray,
                       centrality: str = '0-5%') -> Tuple[float, float]:
        """
        通过拟合椭圆流 v₂(p_t) 曲线优化 η/s。

        简化模型: v₂ ∝ κ · ε₂ · f(η/s)
        其中 f(η/s) = 1 / (1 + α (η/s - 0.08)) 描述粘滞抑制效应

        Parameters
        ----------
        v2_data : np.ndarray
            实验v₂数据
        pt_bins : np.ndarray
            p_t bin中心
        centrality : str
            中心度

        Returns
        -------
        eta_s_best : float
            最优η/s
        chi2_min : float
            最小χ²
        """
        # 模拟的理论曲线
        def theory_v2(eta_s):
            alpha = 5.0
            base_v2 = 0.05 * np.tanh(pt_bins / 2.0)
            suppression = 1.0 / (1.0 + alpha * max(eta_s - 0.08, 0.0))
            return base_v2 * suppression

        def chi2(eta_s):
            theory = theory_v2(eta_s)
            err = v2_data * 0.1 + 0.001
            chi2_val = np.sum(((v2_data - theory) / err) ** 2)
            return chi2_val

        # 在合理范围内优化
        x_opt, iters, chi2_min = self.optimizer.optimize(chi2, 0.0, 0.08, 0.5)
        x_opt = np.clip(x_opt, 0.0, 1.0)
        return float(x_opt), float(chi2_min)

    def fit_cs2(self, mean_pt_data: float,
                T_range: Tuple[float, float] = (0.15, 0.4)) -> Tuple[float, float]:
        """
        通过平均横动量拟合声速平方 c_s²。

        在QGP中，集体流使p_t谱产生蓝移，蓝移程度与c_s²相关。
        近似关系: ⟨p_t⟩ ≈ ⟨p_t⟩_thermal · [1 + α (c_s² - 1/3)]
        其中 α ~ 1.5 为流增强系数。

        Parameters
        ----------
        mean_pt_data : float
            实验平均p_t [GeV]
        T_range : Tuple[float, float]
            温度范围

        Returns
        -------
        cs2_best : float
            最优c_s²
        residual : float
            残差
        """
        T_mid = (T_range[0] + T_range[1]) / 2.0
        pt_thermal = 2.1 * T_mid  # 无质量粒子的热平均p_t
        alpha_flow = 1.5

        def model_mean_pt(cs2):
            if cs2 < 0.01:
                return 1e6
            return pt_thermal * (1.0 + alpha_flow * (cs2 - 1.0 / 3.0))

        def residual(cs2):
            return (model_mean_pt(cs2) - mean_pt_data) ** 2

        x_opt, iters, res = self.optimizer.optimize(residual, 0.1, 1.0 / 3.0, 0.5)
        x_opt = np.clip(x_opt, 0.05, 0.5)
        return float(x_opt), float(res)

    def fit_tau0(self, dNch_deta_data: float,
                 epsilon0: float = 15.0,
                 area: float = 150.0) -> Tuple[float, float]:
        """
        通过带电粒子快度密度拟合初始热化时间 τ₀。

        Bjorken公式: dN_ch/dη = (4/9) τ₀ s(τ₀) A_⊥

        Parameters
        ----------
        dNch_deta_data : float
            实验 dN_ch/dη
        epsilon0 : float
            初始能量密度 [GeV/fm³]
        area : float
            重叠面积 [fm²]

        Returns
        -------
        tau0_best : float
            最优τ₀ [fm/c]
        residual : float
            残差
        """
        # HOLE 2: Compute initial temperature T0 and entropy density s0

        def model_dn(tau0):
            return (4.0 / 9.0) * tau0 * s0 * area

        def residual(tau0):
            if tau0 < 0.1:
                return 1e6
            return (model_dn(tau0) - dNch_deta_data) ** 2

        x_opt, iters, res = self.optimizer.optimize(residual, 0.2, 0.6, 2.0)
        x_opt = np.clip(x_opt, 0.1, 5.0)
        return float(x_opt), float(res)

    def fit_all_parameters(self, v2_data: np.ndarray,
                           mean_pt: float,
                           dNch_deta: float) -> Dict[str, float]:
        """
        联合拟合所有QGP参数。

        Parameters
        ----------
        v2_data : np.ndarray
            v₂数据
        mean_pt : float
            平均p_t
        dNch_deta : float
            快度密度

        Returns
        -------
        Dict[str, float]
            拟合参数字典
        """
        pt_bins = np.linspace(0.5, 5.0, len(v2_data))

        eta_s, chi2_v2 = self.fit_eta_over_s(v2_data, pt_bins)
        cs2, res_pt = self.fit_cs2(mean_pt)
        tau0, res_dn = self.fit_tau0(dNch_deta)

        return {
            'eta_over_s': eta_s,
            'cs2': cs2,
            'tau0': tau0,
            'chi2_v2': chi2_v2,
            'residual_pt': res_pt,
            'residual_dNch': res_dn
        }
