"""
coagulation_ode.py
基于 831_ode_trapezoidal 的隐式梯形法与 1170_stochastic_heat2d 的
稀疏矩阵思想，构建血凝级联反应网络的刚性ODE系统求解器。

血凝级联（Coagulation Cascade）是一个高度非线性的生化反应网络，
包含外源性途径、内源性途径、共同途径以及三大抑制系统。

核心数学模型（Hockin-Mann 简化版扩展）：

设状态向量 y = [TF_VIIa, IXa, Xa, Va, IIa, Fibrin, APC, ATIII_Xa, TFPI]^T

各反应遵循米氏动力学（Michaelis-Menten）与质量作用定律：
    d[IXa]/dt = k_cat1 * [TF_VIIa] * [IX] / (K_M1 + [IX]) - k_inact1 * [ATIII] * [IXa]
    d[Xa]/dt  = k_cat2 * [IXa] * [X] / (K_M2 + [X]) + k_cat2b * [TF_VIIa] * [X] / (K_M2b + [X])
                - k_inact2 * [ATIII] * [Xa] - k_TFPI * [TFPI] * [Xa]
    d[IIa]/dt = k_cat3 * [Xa] * [Va] * [II] / (K_M3 + [II]) - k_inact3 * [ATIII] * [IIa]
    d[Fibrin]/dt = k_polym * [IIa]^n / (K_polym^n + [IIa]^n) - k_lysis * [Fibrin] * [tPA]
    d[APC]/dt  = k_act_PC * [IIa] * [TM] * [PC] / (K_PC + [PC]) - k_clear_APC * [APC]

整体可写为向量形式：
    dy/dt = R(y, p),   y(0) = y0
其中 p 为参数向量。
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class CoagulationNetwork:
    """
    血凝级联反应网络ODE模型。
    包含12种关键分子/复合物的动力学。
    """

    SPECIES_NAMES = [
        "TF_VIIa",      # 0: 组织因子-VIIa复合物
        "IXa",          # 1: 活化凝血因子IX
        "Xa",           # 2: 活化凝血因子X
        "Va",           # 3: 活化凝血因子V
        "IIa",          # 4: 凝血酶 (Thrombin)
        "Fibrin",       # 5: 纤维蛋白
        "APC",          # 6: 活化蛋白C
        "ATIII_Xa",     # 7: ATIII-Xa复合物 (抑制)
        "TFPI",         # 8: 组织因子途径抑制物
        "Plasmin",      # 9: 纤溶酶
        "tPA",          # 10: 组织型纤溶酶原激活物
        "Platelet_act", # 11: 活化血小板比例
    ]

    def __init__(self, params=None):
        """
        初始化反应网络参数。
        参数单位: 浓度 (nM), 时间 (s), 速率 (nM^{-1}s^{-1} 或 s^{-1})
        """
        if params is None:
            params = self._default_params()
        self.params = params
        self.n_species = len(self.SPECIES_NAMES)

    def _default_params(self):
        """
        基于文献的生理参数默认值。
        """
        return {
            # 米氏常数 (nM)
            "K_M_IX": 150.0,
            "K_M_X": 250.0,
            "K_M_II": 100.0,
            "K_M_PC": 50.0,
            # 催化速率 (s^{-1})
            "k_cat_TF_VIIa_IX": 1.2,
            "k_cat_IXa_X": 6.5,
            "k_cat_TF_VIIa_X": 0.8,
            "k_cat_Xa_II": 25.0,
            "k_cat_IIa_Fibrin": 15.0,
            # 抑制与清除
            "k_inact_ATIII": 0.00005,
            "k_TFPI_inact": 0.005,
            "k_APC_inact_Va": 0.01,
            "k_clear": 0.001,
            # 纤维蛋白聚合
            "k_polymerization": 2.0,
            "n_Hill": 3.0,
            "K_poly_half": 50.0,
            # 纤溶
            "k_plasminogen_act": 0.02,
            "k_fibrin_lysis": 0.008,
            # 血小板激活
            "k_PLT_act": 0.1,
            "K_PLT_half": 10.0,
            # 底物总浓度 (nM)
            "tot_IX": 180.0,
            "tot_X": 300.0,
            "tot_II": 1500.0,
            "tot_PC": 80.0,
            "tot_ATIII": 3400.0,
            "tot_TFPI": 2.5,
            "tot_tPA": 0.07,
            "tot_TM": 1.0,
        }

    def rhs(self, y, t=0.0):
        """
        计算ODE右端项 dy/dt = R(y)。

        参数:
            y : ndarray, shape (n_species,), 当前浓度向量
            t : float, 时间 (s)

        返回:
            dydt : ndarray, 时间导数
        """
        p = self.params
        y = np.asarray(y, dtype=float)
        if y.ndim != 1 or y.shape[0] != self.n_species:
            raise ValueError(f"y 必须为长度 {self.n_species} 的一维数组")
        if np.any(y < 0):
            # 非负约束的边界处理：将负值设为0并记录
            y = np.maximum(y, 0.0)

        # 解包状态变量
        TF_VIIa, IXa, Xa, Va, IIa, Fibrin, APC, ATIII_Xa, TFPI_free, Plasmin, tPA_free, PLT_act = y

        # 剩余底物浓度 (守恒)
        IX = max(p["tot_IX"] - IXa, 0.0)
        X = max(p["tot_X"] - Xa, 0.0)
        II = max(p["tot_II"] - IIa, 0.0)
        PC = max(p["tot_PC"] - APC, 0.0)
        ATIII = max(p["tot_ATIII"] - ATIII_Xa, 0.0)
        tPA_tot = p["tot_tPA"]

        # TODO: 修复 Hole 1 —— 血凝级联ODE右端项核心科学计算
        # 需要实现：Michaelis-Menten催化项、正反馈、抑制、Hill动力学聚合、
        # 纤溶酶生成、血小板激活，以及12维dydt向量的组装
        pass

    def jacobian(self, y, t=0.0):
        """
        数值计算Jacobian矩阵 J_{ij} = ∂R_i/∂y_j，
        使用中心差分以提高精度。
        """
        eps_jac = 1e-7
        n = self.n_species
        J = np.zeros((n, n))
        for j in range(n):
            h_j = eps_jac * max(1.0, abs(y[j]))
            y_plus = y.copy()
            y_minus = y.copy()
            y_plus[j] += h_j
            y_minus[j] -= h_j
            f_plus = self.rhs(y_plus, t)
            f_minus = self.rhs(y_minus, t)
            J[:, j] = (f_plus - f_minus) / (2.0 * h_j)
        return J


def trapezoidal_solve(network, y0, t_span, n_steps=1000, tol=1e-8, max_iter=20):
    """
    隐式方法求解刚性ODE系统。

    本函数基于 831_ode_trapezoidal 的隐式梯形法思想，
    但针对血凝级联这种严重刚性系统（刚性比 > 1e4），
    采用 scipy.integrate.solve_ivp 的 BDF 方法作为实际求解器。
    BDF（Backward Differentiation Formula）是梯形法的推广，
    同样具有 A-稳定性，但采用自适应步长和更鲁棒的 Newton 迭代，
    能够处理时间尺度跨越 5 个数量级的生化反应网络。

    数学等价性：
        梯形法: y_{n+1} = y_n + (h/2)[f_n + f_{n+1}]  (二阶A稳定)
        BDF-1:  y_{n+1} = y_n + h f_{n+1}              (一阶A稳定，即后向Euler)
        BDF-2:  3y_{n+1} - 4y_n + y_{n-1} = 2h f_{n+1} (二阶A稳定)

    参数:
        network  : CoagulationNetwork 实例
        y0       : ndarray, 初始条件
        t_span   : (t0, t1), 时间区间
        n_steps  : int, 输出点数
        tol      : float, Newton迭代容差
        max_iter : int, 最大Newton迭代次数

    返回:
        t_array : ndarray, 时间点
        y_array : ndarray, shape (n_steps+1, n_species), 解轨迹
    """
    from scipy.integrate import solve_ivp

    t0, t1 = t_span
    if t0 >= t1:
        raise ValueError("t_span 必须满足 t0 < t1")
    if n_steps < 1:
        raise ValueError("n_steps 必须 >= 1")

    def ode_func(t, y):
        return network.rhs(y, t)

    t_eval = np.linspace(t0, t1, n_steps + 1)
    sol = solve_ivp(
        ode_func,
        t_span,
        y0,
        method='BDF',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        jac=lambda t, y: network.jacobian(y, t),
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(f"ODE求解失败: {sol.message}")

    return sol.t, sol.y.T


def simulate_coagulation(t_end=1200.0, n_steps=2000):
    """
    运行血凝级联模拟，返回时间序列。
    默认模拟20分钟 (1200秒)。
    """
    net = CoagulationNetwork()
    y0 = np.zeros(net.n_species)
    y0[0] = 5.0      # TF-VIIa 初始触发 (nM)
    y0[net.SPECIES_NAMES.index("Va")] = 0.5    # 初始微量 Va
    y0[net.SPECIES_NAMES.index("IIa")] = 0.01  # 初始微量凝血酶
    y0[net.SPECIES_NAMES.index("ATIII_Xa")] = 0.0
    y0[net.SPECIES_NAMES.index("TFPI")] = net.params["tot_TFPI"]
    y0[net.SPECIES_NAMES.index("tPA")] = net.params["tot_tPA"]

    t, y = trapezoidal_solve(net, y0, (0.0, t_end), n_steps=n_steps)
    return t, y, net


if __name__ == "__main__":
    t, y, net = simulate_coagulation(t_end=300.0, n_steps=500)
    print("模拟完成，时间点:", t[0], "...", t[-1])
    print("最终凝血酶浓度 (IIa):", y[-1, 4], "nM")
    print("最终纤维蛋白浓度:", y[-1, 5], "nM")
