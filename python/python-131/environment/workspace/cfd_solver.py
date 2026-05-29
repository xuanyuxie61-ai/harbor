"""
cfd_solver.py
=============
CFD-PBM 耦合求解器主模块。

将两流体模型、群体平衡方程、反应动力学与催化剂优化耦合，
实现浆态床气泡柱反应器的稳态/准稳态数值模拟。

求解流程
--------
1. 网格生成：调用 reactor_mesh 生成圆柱结构化网格。
2. 入口条件：调用 stochastic_inlet 生成随机入口数据。
3. 流动场初始化：气含率 α_g、气速 u_g、液速 u_l、压力 p。
4. 非线性耦合求解：
   a. 动量方程残差计算（momentum_equations）
   b. 用 Newton 法 / 定点迭代求解局部代数方程组（nonlinear_solver）
   c. 相间动量交换更新
5. 群体平衡方程：QMOM 更新气泡尺寸分布矩（population_balance）
6. 反应动力学：基于更新的气含率与比表面积计算转化率
7. 催化剂优化：评估当前温度场下的催化剂分布价值（catalyst_optimization）
8. 稳态浓度场：幂法迭代求解物种浓度（numerical_linear_algebra）
9. 收敛判定与误差分析

核心公式
--------
1. 气含率-滑移速度耦合：
       α_g = u_g / (C_0 u_g/α_g + u_∞)
   可解析化简为：α_g = u_g / (C_0 u_g + u_∞ α_g) · α_g
   实际求解采用 Newton 法处理隐式关系。

2. 比表面积（Interfacial Area Concentration）：
       a_i = 6 α_g / d_32
       d_32 = m_3 / m_2   (Sauter 平均直径)

3. 传质速率：
       N_A = k_L a_i (C_A^* - C_A)
       k_L = √(4 D_A u_slip / (π d_32))

4. Fischer-Tropsch 反应速率（简化双曲型）：
       r_FT = k_FT exp(-E_a/RT) · P_CO^a · P_H2^b / (1 + K_CO P_CO)^2

5. 能量平衡（稳态）：
       ρ_m C_p (u_l · ∇T) = ∇·(k_eff ∇T) + (-ΔH) r_FT

6. 稳态浓度方程（离散化后）：
       (u_l/Δz + k_L a_i + k_r) C_A = (u_l/Δz) C_{A,in} + k_L a_i C_A^*
       写成矩阵形式 A c = b，用幂法/松弛迭代求解。
"""

import numpy as np

from reactor_mesh import generate_cylindrical_mesh, mesh_quality_report
from stochastic_inlet import generate_inlet_conditions, generate_perturbed_profile
from nonlinear_solver import (fixed_point_iteration, newton_solver,
                               reactor_algebraic_residual, reactor_jacobian)
from momentum_equations import (HartmannFlow, interphase_momentum_exchange,
                                 effective_viscosity_slurry)
from population_balance import (poisson_nucleation_events, qmom_integrate_pbe,
                                 breakage_frequency_lehr, wheeler_algorithm)
from catalyst_optimization import optimize_catalyst_loading
from numerical_linear_algebra import (steady_state_concentration_solver,
                                       power_iteration_eigenvector,
                                       estimate_condition_number)
from spectral_quadrature import gauss_legendre_integral, chebyshev_eval
from reactor_operations import reactor_operation_timeline


class SlurryBubbleColumnReactor:
    """
    浆态床气泡柱反应器耦合求解器。
    """

    def __init__(self, R=0.15, H=3.0, Nr=10, Nz=30,
                 rho_l=800.0, rho_g=20.0, mu_l=0.002,
                 sigma=0.072, g=9.81,
                 T_in=523.0, P_in=2.5e6,
                 u_g_in=0.05, alpha_s=0.25,
                 k_FT=5.8e2, Ea=60000.0, dH_FT=-165e3,
                 Cp_mix=2300.0, k_eff=0.35):
        """
        初始化反应器参数。
        """
        self.R = R
        self.H = H
        self.Nr = Nr
        self.Nz = Nz
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.mu_l = mu_l
        self.sigma = sigma
        self.g = g
        self.T_in = T_in
        self.P_in = P_in
        self.u_g_in = u_g_in
        self.alpha_s = alpha_s
        self.k_FT = k_FT
        self.Ea = Ea
        self.dH_FT = dH_FT
        self.Cp_mix = Cp_mix
        self.k_eff = k_eff

        # 网格
        self.nodes, self.elements = generate_cylindrical_mesh(R, H, Nr, Nz)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

        # 轴向切片数 = Nz+1
        self.n_axial = Nz + 1
        self.dz = H / Nz

        # 场变量（轴向平均值，简化 1D+径向均匀模型）
        self.alpha_g = np.ones(self.n_axial) * 0.25
        self.u_l = np.zeros(self.n_axial)
        self.p = np.ones(self.n_axial) * P_in
        self.T = np.ones(self.n_axial) * T_in

        # PBE 矩（初始单分散气泡 d=5mm）
        V0 = np.pi / 6.0 * (5e-3) ** 3
        self.moments = np.array([1.0, V0, V0 ** 2, V0 ** 3])
        self.moments_hist = [self.moments.copy()]

        # 入口条件
        self.inlet_data = None

        # 收敛历史
        self.convergence_history = []

    def setup_inlet_conditions(self, n_samples=100, seed=42):
        """
        生成随机入口条件。
        """
        self.inlet_data = generate_inlet_conditions(
            n_samples=n_samples,
            T_mean=self.T_in,
            T_std=3.0,
            yCO_mean=0.30,
            yH2_mean=0.60,
            y_std=0.015,
            Q_mean=np.pi * self.R ** 2 * self.u_g_in,
            Q_std=0.0005,
            seed=seed
        )

    def compute_flow_field(self, max_iter=50, tol=1e-6):
        """
        求解稳态流动场（气含率、液速、压力）。
        使用定点迭代耦合各轴向段。
        """
        for it in range(1, max_iter + 1):
            alpha_g_old = self.alpha_g.copy()

            for j in range(self.n_axial):
                # TODO: 构建 params 字典、设置初始猜测 x0，调用 Newton 法求解局部代数方程
                # 并解析 x_sol 结果更新 self.alpha_g[j]、self.u_l[j] 和 self.p[j]
                # 注意：params 的键名与结构必须与 reactor_algebraic_residual 期望一致
                raise NotImplementedError("Hole 2: 请实现 compute_flow_field 的循环体")

            diff = np.linalg.norm(self.alpha_g - alpha_g_old, ord=np.inf)
            self.convergence_history.append(diff)
            if diff < tol:
                return True, it, diff

        return False, max_iter, diff

    def update_pbe_moments(self, dt=0.1, n_steps=50):
        """
        用 QMOM 更新气泡尺寸分布矩。
        若数值不稳定，保持上一时刻矩不变。
        """
        try:
            t_arr, m_hist = qmom_integrate_pbe(
                self.moments, (0.0, dt * n_steps), dt,
                n_nodes=2, rho_l=self.rho_l, sigma=self.sigma, epsilon=0.05
            )
            m_new = m_hist[-1]
            if not np.all(np.isfinite(m_new)) or np.any(m_new < 0):
                # 数值发散，保持原矩
                return
            self.moments = m_new
            self.moments_hist.append(self.moments.copy())
        except Exception:
            pass

    def compute_sauter_diameter(self):
        """
        计算 Sauter 平均直径 d_32 = m_3 / m_2。
        """
        m2 = self.moments[2]
        m3 = self.moments[3]
        if not np.isfinite(m2) or not np.isfinite(m3) or m2 < 1e-30:
            return 5.0e-3  # 默认值 5 mm
        d32 = m3 / m2
        # 物理约束：1 mm ~ 50 mm
        if d32 < 1e-4 or d32 > 0.05 or not np.isfinite(d32):
            return 5.0e-3
        return d32

    def compute_interfacial_area(self):
        """
        计算比表面积 a_i = 6 α_g / d_32。
        """
        d32 = self.compute_sauter_diameter()
        a_i = 6.0 * self.alpha_g / max(d32, 1e-9)
        return a_i

    def compute_temperature_profile(self):
        """
        基于反应放热与对流传热计算稳态温度分布。
        简化模型：轴向一维能量平衡。
        引入催化剂有效因子 η（Thiele 模数与内扩散限制）。
        """
        R_gas = 8.314
        k = self.k_FT * np.exp(-self.Ea / (R_gas * self.T))
        z = np.arange(self.n_axial) * self.dz
        # CO 浓度（摩尔分数 -> mol/m³）
        C_total = self.P_in / (R_gas * self.T)
        C_CO = 0.30 * np.exp(-0.5 * z / self.H) * C_total

        # 有效因子（大颗粒内扩散限制，η ≈ 0.05-0.15）
        eta_eff = 0.08
        r_FT = eta_eff * k * C_CO  # [mol/(m³·s)]

        # 能量方程：ρ Cp u dT/dz = (-dH) r_FT - Q_cool
        # 简化：忽略轴向导热，考虑对流与反应源项
        rho_m = self.alpha_g * self.rho_g + (1.0 - self.alpha_g) * self.rho_l
        u_l_clip = np.clip(np.abs(self.u_l), 1e-6, None)

        T_new = np.zeros(self.n_axial)
        T_new[0] = self.T_in
        for j in range(1, self.n_axial):
            # 反应放热项 [W/m³]
            q_rxn = (-self.dH_FT) * r_FT[j]
            # 简化的换热管束移热（线性近似）
            q_cool = 500.0 * (T_new[j - 1] - self.T_in)
            dTdz = (q_rxn - q_cool) / (rho_m[j] * self.Cp_mix * u_l_clip[j])
            T_new[j] = T_new[j - 1] + dTdz * self.dz

        # 边界处理：最高温度限制（催化剂烧结温度 ~ 623 K）
        T_new = np.clip(T_new, self.T_in, 623.0)
        self.T = T_new

    def compute_species_concentration(self):
        """
        用稳态迭代求解 CO 轴向浓度分布。
        """
        n = self.n_axial
        A = np.zeros((n, n))
        b = np.zeros(n)

        a_i = self.compute_interfacial_area()
        # 传质系数 k_L 简化模型
        d32 = self.compute_sauter_diameter()
        u_slip = 0.23  # m/s
        D_CO = 2.5e-9  # m²/s
        k_L = np.sqrt(4.0 * D_CO * u_slip / (np.pi * max(d32, 1e-9)))

        # 整体传质效率（包含液固传质与内扩散限制）
        eta_mt = 0.001
        kLa = eta_mt * k_L * a_i
        k_r = self.k_FT * np.exp(-self.Ea / (8.314 * self.T))

        yCO_in = 0.30
        for j in range(n):
            u = max(abs(self.u_l[j]), 1e-6)
            adv = u / self.dz
            kLa_j = kLa[j] if np.isfinite(kLa[j]) else 0.0
            k_r_j = k_r[j] if np.isfinite(k_r[j]) else 0.0
            if j == 0:
                A[j, j] = adv + kLa_j + k_r_j
                b[j] = adv * yCO_in
            else:
                A[j, j - 1] = -adv
                A[j, j] = adv + kLa_j + k_r_j
                b[j] = 0.0

        # 检查矩阵是否包含 NaN/Inf
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            # 退化：纯对流
            c = np.full(n, yCO_in * np.exp(-np.arange(n) * 0.1))
            return c, 0.0, 0, True

        # 条件数检查
        try:
            cond = estimate_condition_number(A)
            if cond > 1e12:
                A = A + 1e-8 * np.eye(n)
        except Exception:
            A = A + 1e-8 * np.eye(n)

        c, res, it, conv = steady_state_concentration_solver(
            A, b, alpha_relax=0.6, max_iter=500, tol=1e-8
        )
        return c, res, it, conv

    def evaluate_catalyst_distribution(self):
        """
        评估当前温度场下催化剂的最优分布。
        """
        n_seg = min(self.n_axial, 10)
        T_profile = np.interp(
            np.linspace(0, self.n_axial - 1, n_seg),
            np.arange(self.n_axial),
            self.T
        )
        result = optimize_catalyst_loading(
            W_total=50.0,
            n_segments=n_seg,
            T_profile=T_profile,
            Q_gas=np.pi * self.R ** 2 * self.u_g_in,
            method='brute_force'
        )
        return result

    def run_simulation(self, verbose=False):
        """
        执行完整的耦合模拟流程。
        """
        if self.inlet_data is None:
            self.setup_inlet_conditions()

        # 步骤 1：流动场
        conv_flow, it_flow, diff_flow = self.compute_flow_field()
        if verbose:
            print(f"[Flow] Converged={conv_flow}, it={it_flow}, diff={diff_flow:.3e}")

        # 步骤 2：PBE 矩更新
        self.update_pbe_moments()
        if verbose:
            print(f"[PBE] Moments: {self.moments}")

        # 步骤 3：温度场
        self.compute_temperature_profile()
        if verbose:
            print(f"[Energy] T_max={self.T.max():.2f} K, T_min={self.T.min():.2f} K")

        # 步骤 4：物种浓度
        c_CO, res_CO, it_CO, conv_CO = self.compute_species_concentration()
        if verbose:
            print(f"[Species] CO outlet={c_CO[-1]:.4f}, res={res_CO:.3e}, conv={conv_CO}")

        # 步骤 5：催化剂优化
        cat_result = self.evaluate_catalyst_distribution()
        if verbose:
            print(f"[Catalyst] Max value={cat_result['max_value']:.4f}")

        # 步骤 6：网格质量
        mesh_report = mesh_quality_report(self.nodes, self.elements)
        if verbose:
            print(f"[Mesh] Jacobian min={mesh_report['jacobian_min']:.3e}, "
                  f"neg_count={mesh_report['jacobian_negative_count']}")

        # 步骤 7：Hartmann 基准验证（用于动量方程数值验证）
        hart = HartmannFlow(G=1.0, Ha=2.0, Re=10.0, Rm=6.0)
        y_test = np.linspace(-0.9, 0.9, 5)
        ur, br = hart.residual_check(y_test)
        hartmann_error = max(np.max(np.abs(ur)), np.max(np.abs(br)))
        if verbose:
            print(f"[Hartmann] Residual max={hartmann_error:.3e}")

        # 步骤 8：操作时间线
        timeline = reactor_operation_timeline(
            (2024, 1, 1), (2024, 12, 31)
        )
        if verbose:
            print(f"[Ops] Operating days={timeline['total_days']}, "
                  f"max cycles={timeline['max_cycles']}")

        results = {
            'converged_flow': conv_flow,
            'flow_iterations': it_flow,
            'moments': self.moments.copy(),
            'sauter_diameter': self.compute_sauter_diameter(),
            'interfacial_area_mean': float(np.mean(self.compute_interfacial_area())),
            'temperature_max': float(self.T.max()),
            'temperature_min': float(self.T.min()),
            'CO_outlet': float(c_CO[-1]),
            'CO_conversion': float(1.0 - c_CO[-1] / max(c_CO[0], 1e-12)),
            'species_residual': res_CO,
            'catalyst_result': cat_result,
            'mesh_report': mesh_report,
            'hartmann_benchmark_error': float(hartmann_error),
            'operational_timeline': timeline,
            'inlet_statistics': self.inlet_data['statistics'] if self.inlet_data else None,
        }
        return results
