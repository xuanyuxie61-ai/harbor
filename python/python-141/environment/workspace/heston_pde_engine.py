"""
Heston随机波动率PDE引擎
========================
融合种子项目的核心算法:
  - 975_r8ccs   : CCS稀疏矩阵存储与运算（通过sparse_matrix_ccs模块）
  - 210_continuation : 延拓法参数分支追踪思想
  - 328_ellipse : 椭圆截断域概念
  - 1168_stla_to_tri_surface_fast / 382_fem_to_xml : 网格数据快速解析与结构化

科学问题:
---------
在Heston随机波动率模型下，衍生品价格 V(S,v,t) 满足二维对流-扩散-反应PDE:

    ∂V/∂t + L_{Heston} V = 0

其中Heston算子:
    L_{Heston} = ½ v S² ∂²/∂S² + ρσvS ∂²/∂S∂v + ½ σ²v ∂²/∂v²
               + rS ∂/∂S + κ(θ-v) ∂/∂v - r·I

变量域:
    S ∈ [0, S_max],  v ∈ [0, v_max],  t ∈ [0, T]

采用非均匀网格与ADI（交替方向隐式）格式离散:
    (I - θ_x A_x)(I - θ_y A_y) V^{n+1} = (I + (1-θ_x)A_x)(I + (1-θ_y)A_y) V^n

其中 θ_x = θ_y = 0.5 对应Craig-Sneyd修正格式，具有二阶时间精度。

边界条件:
---------
1. S = 0（资产价格归零）:  V(0,v,t) = 0  （看涨期权）
2. S = S_max:  V(S_max,v,t) = S_max - K·e^{-r(T-t)}
3. v = 0:  退化边界，PDE降维为:
     ∂V/∂t + rS ∂V/∂S + κθ ∂V/∂v - rV = 0
4. v = v_max:  Neumann条件 ∂V/∂v = 0（波动率饱和）

Feller条件:
    2κθ ≥ σ²
若满足，则v=0为entrance boundary，自然边界条件数值稳定；
若不满足，需特殊处理零边界的吸收/反射效应。
"""

import numpy as np
from math import log, exp, sqrt, pi, erf
from sparse_matrix_ccs import SparseMatrixCCS
from gmres_iterative import gmres_dense
from special_math_utils import MeshDataManager, ellipse_area_matrix


class HestonPDESolver:
    """
    Heston PDE有限差分解器，支持非均匀网格与ADI时间推进。
    """

    def __init__(self, S_max, v_max, T, r, kappa, theta, sigma, rho,
                 n_S=80, n_v=40, n_t=100, scheme='CS'):
        """
        参数:
        ------
        S_max, v_max : float, 空间截断上界
        T            : float, 到期时间
        r            : float, 无风险利率
        kappa        : float, 均值回归速率
        theta        : float, 长期波动率均值
        sigma        : float, 波动率的波动率（vol-of-vol）
        rho          : float, 资产与波动率的相关系数
        n_S, n_v     : int, 空间网格数
        n_t          : int, 时间步数
        scheme       : str, 'CS'=Craig-Sneyd ADI, 'DO'=Douglas ADI
        """
        self.S_max = float(S_max)
        self.v_max = float(v_max)
        self.T = float(T)
        self.r = float(r)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.n_S = n_S
        self.n_v = n_v
        self.n_t = n_t
        self.scheme = scheme

        # Feller条件检查
        self.feller_ratio = 2.0 * kappa * theta / (sigma ** 2)
        if self.feller_ratio < 1.0:
            # 在v=0附近加密网格
            pass

        # 生成非均匀网格
        self._generate_grid()
        # 构建离散算子
        self._build_operators()

    def _generate_grid(self):
        """
        生成非均匀空间网格。

        S方向: 在S=K附近加密（使用对数变换后的均匀分布）
        v方向: 在v=0附近加密（使用指数拉伸）
        """
        # S方向: 对数拉伸，在S=0附近更密
        # 使用变换 ξ = sinh^{-1}(c·S/S_max) / sinh^{-1}(c)
        c_s = 5.0
        xi_s = np.linspace(0.0, 1.0, self.n_S + 1)
        self.S_grid = self.S_max * np.sinh(c_s * xi_s) / np.sinh(c_s)
        self.S_grid[0] = 0.0  # 强制零点
        self.S_grid[-1] = self.S_max
        self.dS = np.diff(self.S_grid)

        # v方向: 指数拉伸
        c_v = 8.0
        xi_v = np.linspace(0.0, 1.0, self.n_v + 1)
        self.v_grid = self.v_max * (np.exp(c_v * xi_v) - 1.0) / (np.exp(c_v) - 1.0)
        self.v_grid[0] = 0.0
        self.v_grid[-1] = self.v_max
        self.dv = np.diff(self.v_grid)

        self.dt = self.T / self.n_t

    def _build_operators(self):
        """
        构建有限差分算子矩阵。

        对每个内部点(i,j)，对应(S_i, v_j)，构造行索引:
            idx = j * (n_S + 1) + i

        差分模板（非均匀网格）:
            ∂²V/∂S² ≈ 2/(ΔS_{i-1}+ΔS_i) [ V_{i+1}/ΔS_i - V_i(1/ΔS_i+1/ΔS_{i-1}) + V_{i-1}/ΔS_{i-1} ]
            ∂²V/∂v² ≈ 类似
            ∂²V/∂S∂v ≈ 交叉导数（九点格式）
            ∂V/∂S   ≈ 迎风/中心差分（根据对流项符号选择）
        """
        n_total = (self.n_S + 1) * (self.n_v + 1)
        self.n_total = n_total

        # 用字典临时存储非零元
        entries = {}

        def add_entry(row, col, val):
            key = (row, col)
            entries[key] = entries.get(key, 0.0) + val

        for j in range(1, self.n_v):  # v内部点
            vj = self.v_grid[j]
            dv_j = self.dv[j]
            dv_jm1 = self.dv[j - 1]
            dv_avg = 0.5 * (dv_j + dv_jm1)

            for i in range(1, self.n_S):  # S内部点
                Si = self.S_grid[i]
                dS_i = self.dS[i]
                dS_im1 = self.dS[i - 1]
                dS_avg = 0.5 * (dS_i + dS_im1)
                idx = j * (self.n_S + 1) + i

                # 二阶导数 ∂²V/∂S² 系数: ½ v S²
                coeff_SS = 0.5 * vj * Si ** 2
                alpha_S = 2.0 / (dS_im1 * dS_i * (dS_im1 + dS_i))
                add_entry(idx, idx, -coeff_SS * alpha_S * (dS_im1 + dS_i))
                add_entry(idx, idx + 1, coeff_SS * alpha_S * dS_im1)
                add_entry(idx, idx - 1, coeff_SS * alpha_S * dS_i)

                # 二阶导数 ∂²V/∂v² 系数: ½ σ² v
                coeff_vv = 0.5 * self.sigma ** 2 * vj
                alpha_v = 2.0 / (dv_jm1 * dv_j * (dv_jm1 + dv_j))
                add_entry(idx, idx, -coeff_vv * alpha_v * (dv_jm1 + dv_j))
                add_entry(idx, idx + (self.n_S + 1), coeff_vv * alpha_v * dv_jm1)
                add_entry(idx, idx - (self.n_S + 1), coeff_vv * alpha_v * dv_j)

                # 交叉导数 ∂²V/∂S∂v 系数: ρσvS
                # 使用九点格式（混合二阶差分）
                coeff_Sv = self.rho * self.sigma * vj * Si
                if abs(coeff_Sv) > 1e-12:
                    cross = coeff_Sv / (4.0 * dS_avg * dv_avg)
                    add_entry(idx, idx + 1 + (self.n_S + 1), cross)
                    add_entry(idx, idx + 1 - (self.n_S + 1), -cross)
                    add_entry(idx, idx - 1 + (self.n_S + 1), -cross)
                    add_entry(idx, idx - 1 - (self.n_S + 1), cross)

                # 对流项 ∂V/∂S 系数: rS
                coeff_S1 = self.r * Si
                if coeff_S1 >= 0:
                    # 向前差分（上风）
                    add_entry(idx, idx, -coeff_S1 / dS_i)
                    add_entry(idx, idx + 1, coeff_S1 / dS_i)
                else:
                    add_entry(idx, idx, coeff_S1 / dS_im1)
                    add_entry(idx, idx - 1, -coeff_S1 / dS_im1)

                # 对流项 ∂V/∂v 系数: κ(θ-v)
                coeff_v1 = self.kappa * (self.theta - vj)
                if coeff_v1 >= 0:
                    add_entry(idx, idx, -coeff_v1 / dv_j)
                    add_entry(idx, idx + (self.n_S + 1), coeff_v1 / dv_j)
                else:
                    add_entry(idx, idx, coeff_v1 / dv_jm1)
                    add_entry(idx, idx - (self.n_S + 1), -coeff_v1 / dv_jm1)

                # 反应项: -r
                add_entry(idx, idx, -self.r)

        # 边界条件
        # S=0: V=0 (Dirichlet)
        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1)
            add_entry(idx, idx, 1.0)

        # S=S_max: V = S_max - K·e^{-r(T-t)} (非齐次Dirichlet，时间依赖，在时间步处理)
        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1) + self.n_S
            add_entry(idx, idx, 1.0)

        # v=0: 退化边界处理（简化：使用一维对流扩散方程离散）
        for i in range(1, self.n_S):
            idx = i
            Si = self.S_grid[i]
            dS_i = self.dS[i]
            dS_im1 = self.dS[i - 1]
            # rS ∂V/∂S + κθ ∂V/∂v - rV ≈ 0
            # 对S用中心差分，对v用向前差分（一阶）
            coeff_SS = 0.0  # v=0时扩散项消失
            # S方向一阶导
            add_entry(idx, idx + 1, self.r * Si / (dS_i + dS_im1))
            add_entry(idx, idx - 1, -self.r * Si / (dS_i + dS_im1))
            # v方向向前差分 (κθ)
            dv0 = self.dv[0]
            add_entry(idx, idx, -self.kappa * self.theta / dv0)
            add_entry(idx, idx + (self.n_S + 1), self.kappa * self.theta / dv0)
            # 反应项
            add_entry(idx, idx, -self.r)

        # v=v_max: Neumann ∂V/∂v = 0
        for i in range(self.n_S + 1):
            idx = self.n_v * (self.n_S + 1) + i
            add_entry(idx, idx, 1.0)
            if self.n_v >= 1:
                add_entry(idx, idx - (self.n_S + 1), -1.0)

        # 构造稠密矩阵（中小规模用；大规模应改用稀疏格式+GMRES）
        self.A_dense = np.zeros((n_total, n_total), dtype=np.float64)
        for (row, col), val in entries.items():
            self.A_dense[row, col] = val

    def _apply_boundary_rhs(self, V, t, K):
        """生成边界条件对应的右端项修正。"""
        rhs = np.zeros(self.n_total, dtype=np.float64)
        tau = self.T - t
        disc = exp(-self.r * tau)

        # S=0: V=0
        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1)
            rhs[idx] = 0.0

        # S=S_max: V = S_max - K·e^{-r(T-t)}
        for j in range(self.n_v + 1):
            idx = j * (self.n_S + 1) + self.n_S
            rhs[idx] = self.S_max - K * disc

        # v=0: 已包含在矩阵中，右端项为0
        # v=v_max: Neumann, 右端项为0
        return rhs

    def solve_european_call(self, K):
        """
        求解欧式看涨期权价格曲面 V(S,v,0)。

        终端条件:
            V(S,v,T) = max(S - K, 0)

        返回:
        ------
        ndarray, 形状 (n_v+1, n_S+1), 价格曲面
        """
        if K <= 0:
            raise ValueError("行权价K必须为正")

        # 初始化终端条件
        V = np.zeros(self.n_total, dtype=np.float64)
        for j in range(self.n_v + 1):
            for i in range(self.n_S + 1):
                idx = j * (self.n_S + 1) + i
                Si = self.S_grid[i]
                V[idx] = max(Si - K, 0.0)

        # 时间反向推进
        # TODO: 实现时间反向推进求解PDE
        # 需要从终端条件 V(S,v,T) = max(S-K, 0) 出发，逆向时间推进到 t=0
        # 推荐格式: 隐式欧拉 (I - dt·A) V^{n+1} = V^n + dt·rhs_bc
        # 关键步骤:
        #   1. 对每个时间步，应用边界条件生成右端项修正 rhs_bc
        #   2. 构造 M = I - dt * A_dense, rhs = V + dt * rhs_bc
        #   3. 对边界行（S=0, S=S_max, v=v_max的Neumann）直接替换矩阵行和rhs值
        #   4. 求解线性系统 M·V_new = rhs（小规模用np.linalg.solve，大规模用gmres_dense）
        #   5. 将最终解reshape为 (n_v+1, n_S+1) 的价格曲面
        raise NotImplementedError("Hole_2: 需要实现PDE时间推进与边界处理")

    def price_at_spot(self, V_surface, S0, v0):
        """
        从价格曲面插值得到指定(S0, v0)处的期权价格。

        使用双线性插值：
            V ≈ Σ_{α,β∈{0,1}} w_{αβ} · V(S_{i+α}, v_{j+β})
            w_{αβ} = |S0 - S_{i+1-α}| · |v0 - v_{j+1-β}| / (ΔS·Δv)
        """
        if S0 < 0 or S0 > self.S_max or v0 < 0 or v0 > self.v_max:
            raise ValueError("(S0, v0)超出网格范围")

        # 找到包围(S0, v0)的网格单元
        i = np.searchsorted(self.S_grid, S0) - 1
        i = max(0, min(i, self.n_S - 1))
        j = np.searchsorted(self.v_grid, v0) - 1
        j = max(0, min(j, self.n_v - 1))

        S0_l, S0_r = self.S_grid[i], self.S_grid[i + 1]
        v0_l, v0_r = self.v_grid[j], self.v_grid[j + 1]
        dS_cell = S0_r - S0_l
        dv_cell = v0_r - v0_l
        if dS_cell < 1e-15 or dv_cell < 1e-15:
            return V_surface[j, i]

        w_00 = (S0_r - S0) * (v0_r - v0) / (dS_cell * dv_cell)
        w_10 = (S0 - S0_l) * (v0_r - v0) / (dS_cell * dv_cell)
        w_01 = (S0_r - S0) * (v0 - v0_l) / (dS_cell * dv_cell)
        w_11 = (S0 - S0_l) * (v0 - v0_l) / (dS_cell * dv_cell)

        price = (w_00 * V_surface[j, i] +
                 w_10 * V_surface[j, i + 1] +
                 w_01 * V_surface[j + 1, i] +
                 w_11 * V_surface[j + 1, i + 1])
        return price


def heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0,
                               n_S=60, n_v=30, n_t=80):
    """
    便捷的Heston欧式看涨期权定价函数。

    参数:
    ------
    S0, K, T, r : float, 标准期权参数
    kappa, theta, sigma, rho, v0 : float, Heston参数
    n_S, n_v, n_t : int, 网格参数

    返回:
    ------
    float, 期权价格
    """
    if S0 <= 0 or K <= 0 or T <= 0:
        raise ValueError("S0, K, T必须为正")
    S_max = max(3.0 * K, 4.0 * S0)
    v_max = max(5.0 * theta, 3.0 * v0, 1.0)
    solver = HestonPDESolver(S_max, v_max, T, r, kappa, theta, sigma, rho,
                             n_S=n_S, n_v=n_v, n_t=n_t)
    V_surface = solver.solve_european_call(K)
    return solver.price_at_spot(V_surface, S0, v0)


def heston_pde_greeks(S0, K, T, r, kappa, theta, sigma, rho, v0,
                      d_param=1e-4):
    """
    使用有限差分计算PDE-based Greeks。

    Delta = ∂V/∂S0
    Vega  = ∂V/∂v0
    Theta = -∂V/∂T
    Rho   = ∂V/∂r
    """
    V0 = heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0)

    # Delta
    dS = max(S0 * d_param, 1e-3)
    V_up = heston_european_call_price(S0 + dS, K, T, r, kappa, theta, sigma, rho, v0,
                                      n_S=50, n_v=25, n_t=60)
    V_down = heston_european_call_price(S0 - dS, K, T, r, kappa, theta, sigma, rho, v0,
                                        n_S=50, n_v=25, n_t=60)
    delta = (V_up - V_down) / (2.0 * dS)

    # Vega (对v0)
    dv = max(v0 * d_param, 1e-4)
    V_up = heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, v0 + dv,
                                      n_S=50, n_v=25, n_t=60)
    V_down = heston_european_call_price(S0, K, T, r, kappa, theta, sigma, rho, max(v0 - dv, 1e-6),
                                        n_S=50, n_v=25, n_t=60)
    vega = (V_up - V_down) / (2.0 * dv)

    # Theta
    dT = max(T * d_param, 1e-4)
    if T > dT:
        V_T = heston_european_call_price(S0, K, T - dT, r, kappa, theta, sigma, rho, v0,
                                         n_S=50, n_v=25, n_t=60)
        theta_greek = -(V0 - V_T) / dT
    else:
        theta_greek = 0.0

    # Rho
    dr = 1e-4
    V_up = heston_european_call_price(S0, K, T, r + dr, kappa, theta, sigma, rho, v0,
                                      n_S=50, n_v=25, n_t=60)
    V_down = heston_european_call_price(S0, K, T, r - dr, kappa, theta, sigma, rho, v0,
                                        n_S=50, n_v=25, n_t=60)
    rho_greek = (V_up - V_down) / (2.0 * dr)

    return {
        'price': V0,
        'delta': delta,
        'vega': vega,
        'theta': theta_greek,
        'rho': rho_greek
    }
