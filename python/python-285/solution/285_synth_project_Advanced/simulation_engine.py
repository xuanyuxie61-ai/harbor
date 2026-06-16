"""
模拟引擎模块
=============
统一调度所有子模块, 执行完整的多铁性磁电耦合模拟.

本模块负责:
    1. 初始化所有物理场 (P, M)
    2. 构建网格拓扑
    3. 运行时间演化
    4. 收集输出数据
    5. 后处理分析

对应种子项目的集成:
    755_mesh_etoe → mesh_domain_topology (网格邻接)
    972_r8but → banded_solver (上三角求解)
    984_r8lt → banded_solver (下三角求解)
    855_pdflib → stochastic_thermal (随机采样)
    1186_GlacierWeilin → phase_classifier (分类树)
    1136_SonyResearch → magnetoelectric_connector (跨场耦合)
    486_gray_scott → lgd_free_energy (反应扩散类比)
    1234_HackBio → order_parameter_analysis (PCA降维)
    1223_gev26 → magnetoelectric_bounds (LP界)
    1306_triangle → brillouin_zone (BZ采样)
    1270_PAC → md_trajectory_analysis (轨迹分析)
    1147_Hilbert-Smith → entropy_trace (熵迹分析)
    088_biharmonic → high_order_fd (高阶差分)
    434_fisher → time_integrator (FTCS格式)
    1058_dndimitri → multi_task_surrogate (多任务代理)
"""

import numpy as np
from multiferroic_constants import (BiFeO3LGD, BiFeO3Lattice,
                                     SimulationConfig, EPSILON_0, MU_0)
from lgd_free_energy import LGDFreeEnergyFunctional
from high_order_fd import (compact_laplacian_2d, sixth_order_second_derivative,
                            von_neumann_stability_limit, biharmonic_2d)
from banded_solver import (BandedMatrixUpper, BandedMatrixLower,
                            thomas_algorithm, solve_2d_adi)
from mesh_domain_topology import DomainMeshTopology
from stochastic_thermal import ThermalNoiseGenerator
from magnetoelectric_connector import MagnetoelectricConnector
from brillouin_zone import BrillouinZoneSampler
from phase_classifier import MultiferroicPhaseClassifier
from magnetoelectric_bounds import MagnetoelectricBounds
from entropy_trace import MagnetoelectricEntropyTrace
from md_trajectory_analysis import MDTrajectoryAnalyzer
from order_parameter_analysis import OrderParameterAnalyzer


class MultiferroicSimulationEngine:
    """
    多铁性磁电耦合模拟引擎.

    集成 LGD 自由能、高阶有限差分、带状求解器、
    热噪声、跨场耦合等模块, 执行完整模拟.
    """

    def __init__(self, config=None):
        """
        参数:
            config: SimulationConfig 实例. 若为 None 使用默认.
        """
        self.config = config if config is not None else SimulationConfig()

        # 物理参数
        self.lattice = BiFeO3Lattice()
        self.lgd = BiFeO3LGD()

        # 验证稳定性
        self.config.validate_stability(self.lgd, verbose=True)

        # 自由能泛函
        self.free_energy = LGDFreeEnergyFunctional(self.lgd)

        # 连接器
        self.connector = MagnetoelectricConnector(
            alpha_ME=self.lgd.alpha_ME_linear,
            gamma_ME=self.lgd.gamma_ME_biquadratic,
            coupling_mode='bilinear'
        )

        # 时间积分器
        from time_integrator import TimeIntegrator
        self.integrator = TimeIntegrator(
            self.config, self.lgd, self.free_energy, self.connector
        )

        # 热噪声
        self.noise_gen = ThermalNoiseGenerator(
            self.config.nx, self.config.ny,
            self.config.dx, self.config.dy, self.config.dt,
            self.config.temperature, self.lgd.L_P,
            seed=self.config.seed
        )

        # 网格拓扑
        self.mesh = DomainMeshTopology(
            self.config.nx, self.config.ny,
            self.config.Lx, self.config.Ly
        )

        # 布里渊区
        self.bz = BrillouinZoneSampler(
            lattice_type='rhombohedral',
            a_lattice=self.lattice.a_cubic,
            n_divisions=6
        )

        # 相分类器
        self.phase_classifier = MultiferroicPhaseClassifier()

        # 磁电界
        self.me_bounds = MagnetoelectricBounds(
            chi_e=100.0, chi_m=0.01, T=self.config.temperature
        )

        # 熵迹分析
        self.entropy_analyzer = MagnetoelectricEntropyTrace(
            n_modes=128, alpha_gaussian=0.5, seed=self.config.seed
        )

        # MD 分析
        self.md_analyzer = MDTrajectoryAnalyzer()

        # 序参量分析
        self.op_analyzer = OrderParameterAnalyzer()

        # 场初始化
        self.P = None
        self.M = None

        # 历史记录
        self.history = {
            'P': [], 'M': [], 'energy': [], 'step': [],
            'P_avg': [], 'M_avg': [], 'sync_strength': [],
        }

    # ============================================================
    # 初始化
    # ============================================================

    def initialize_fields(self, mode='random_ferroelectric'):
        """
        初始化极化和磁化场.

        模式:
            'random_ferroelectric': 随机铁电畴
            'single_domain': 单畴态
            'vortex': 涡旋态
            'stripe': 条纹畴

        参数:
            mode: 初始化模式
        """
        nx, ny = self.config.nx, self.config.ny
        P0 = self.lattice.P_spontaneous
        M0 = self.lattice.M_saturation * 0.01  # 弱铁磁分量

        rng = np.random.RandomState(self.config.seed)

        if mode == 'random_ferroelectric':
            # 随机畴结构: 8个等价 [111] 方向
            directions = np.array([
                [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
                [-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
            ]) / np.sqrt(3.0)

            # 随机分配畴
            domain_map = rng.randint(0, 8, size=(nx, ny))
            self.P = np.zeros((nx, ny, 3))
            for i in range(nx):
                for j in range(ny):
                    self.P[i, j] = P0 * directions[domain_map[i, j]]

            # 加入小扰动
            self.P += rng.randn(nx, ny, 3) * P0 * 0.05

        elif mode == 'single_domain':
            self.P = np.zeros((nx, ny, 3))
            self.P[..., 2] = P0  # [001] 方向

        elif mode == 'vortex':
            x = np.linspace(-1, 1, nx)
            y = np.linspace(-1, 1, ny)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            r = np.sqrt(xx ** 2 + yy ** 2)
            r = np.maximum(r, 1e-10)
            theta = np.arctan2(yy, xx)

            self.P = np.zeros((nx, ny, 3))
            self.P[..., 0] = P0 * np.cos(theta) * np.exp(-r / 0.3)
            self.P[..., 1] = P0 * np.sin(theta) * np.exp(-r / 0.3)
            self.P[..., 2] = P0 * 0.3 * np.exp(-r ** 2 / 0.1)

        elif mode == 'stripe':
            x = np.linspace(0, 4 * np.pi, nx)
            self.P = np.zeros((nx, ny, 3))
            self.P[..., 0] = P0 * np.sin(x)[:, np.newaxis]
            self.P[..., 2] = P0 * np.cos(x)[:, np.newaxis] * 0.5

        else:
            self.P = rng.randn(nx, ny, 3) * P0 * 0.1

        # 磁化初始化 (弱铁磁, canted antiferromagnetic)
        self.M = np.zeros((nx, ny, 3))
        self.M[..., 0] = M0 * (1.0 + 0.1 * rng.randn(nx, ny))
        self.M[..., 1] = M0 * 0.1 * rng.randn(nx, ny)
        self.M[..., 2] = M0 * 0.5

    # ============================================================
    # 主模拟循环
    # ============================================================

    def run(self, n_steps=None, output_interval=None):
        """
        运行时间演化模拟.

        参数:
            n_steps: 总步数 (默认使用 config)
            output_interval: 输出间隔 (默认使用 config)
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        if output_interval is None:
            output_interval = self.config.output_interval

        T = self.config.temperature

        print("=" * 60)
        print("  多铁性材料磁电耦合模拟")
        print(f"  网格: {self.config.nx}×{self.config.ny}")
        print(f"  温度: {T} K")
        print(f"  时间步长: {self.config.dt:.3e} s")
        print(f"  总步数: {n_steps}")
        print(f"  格式: {self.config.scheme}")
        print(f"  FD 精度: O(h^{self.config.fd_order})")
        print("=" * 60)

        # 初始能量
        E_init = self.free_energy.total_free_energy(
            self.P, self.M, T,
            self.config.dx, self.config.dy, self.config.fd_order
        )
        print(f"  初始自由能: {E_init:.6e} J/m")

        # 主循环
        for step in range(n_steps):
            # 生成热噪声
            noise_P = self.noise_gen.gaussian_noise()
            noise_M = self.noise_gen.gaussian_noise(
                shape=(self.config.nx, self.config.ny, 3)
            ) * 0.01

            # 时间步进
            if self.config.scheme == 'FTCS':
                self.P, self.M = self.integrator.step_ftcs(
                    self.P, self.M, T, noise_P, noise_M
                )
            elif self.config.scheme == 'semi_implicit':
                self.P, self.M = self.integrator.step_semi_implicit(
                    self.P, self.M, T, 0.5, noise_P, noise_M
                )
            elif self.config.scheme == 'RK4':
                self.P, self.M = self.integrator.step_rk4(
                    self.P, self.M, T, noise_P, noise_M
                )

            # 记录
            if step % output_interval == 0:
                self._record_state(step)

            # 边界条件
            self._apply_boundary_conditions()

        # 最终能量
        E_final = self.free_energy.total_free_energy(
            self.P, self.M, T,
            self.config.dx, self.config.dy, self.config.fd_order
        )
        print(f"  最终自由能: {E_final:.6e} J/m")
        print(f"  能量变化: {(E_final - E_init):.6e} J/m")

        # 输出稳定性信息
        S = self.config.stability_number(self.lgd)
        print(f"  CFL 数: {S:.4f}")

        print("=" * 60)

    def _record_state(self, step):
        """记录当前状态."""
        T = self.config.temperature
        energy_comps = self.free_energy.extract_energy_components(
            self.P, self.M, T,
            self.config.dx, self.config.dy, self.config.fd_order
        )

        sync = self.connector.synchronization_strength(self.P, self.M)

        self.history['step'].append(step)
        self.history['P_avg'].append(energy_comps['P_avg'].copy())
        self.history['M_avg'].append(energy_comps['M_avg'].copy())
        self.history['energy'].append(energy_comps['f_total'])
        self.history['sync_strength'].append(sync)

        print(f"  Step {step:5d}: F = {energy_comps['f_total']:.4e} J/m, "
              f"|P| = {energy_comps['P_magnitude']:.4e}, "
              f"|M| = {energy_comps['M_magnitude']:.4e}, "
              f"sync = {sync:.4f}")

    def _apply_boundary_conditions(self):
        """施加边界条件."""
        if self.config.P_boundary == 'periodic':
            # np.roll 已隐式处理周期性
            pass
        elif self.config.P_boundary == 'dirichlet':
            self.P[0, :, :] = 0.0
            self.P[-1, :, :] = 0.0
            self.P[:, 0, :] = 0.0
            self.P[:, -1, :] = 0.0

    # ============================================================
    # 后处理分析
    # ============================================================

    def run_postprocessing(self):
        """
        运行完整后处理分析.

        包括:
        1. 畴结构分析
        2. 序参量 PCA
        3. 相分类
        4. 磁电界估计
        5. 熵迹分析
        6. BZ 积分
        7. MD 轨迹分析
        """
        print("\n" + "=" * 60)
        print("  后处理分析")
        print("=" * 60)

        results = {}

        # 1. 畴结构
        results['domain'] = self._analyze_domains()

        # 2. 序参量 PCA
        results['pca'] = self._analyze_pca()

        # 3. 相分类
        results['phase'] = self._classify_phase()

        # 4. 磁电界
        results['me_bounds'] = self._compute_me_bounds()

        # 5. 熵迹
        results['entropy'] = self._analyze_entropy()

        # 6. BZ 积分
        results['bz'] = self._analyze_bz()

        # 7. MD 分析
        results['md'] = self._analyze_md_trajectory()

        # 8. 拓扑荷
        results['topology'] = self._compute_topology()

        return results

    def _analyze_domains(self):
        """畴结构分析."""
        wall_mask = self.mesh.identify_domain_walls(self.P)
        wall_fraction = np.mean(wall_mask)

        stats = self.mesh.mesh_statistics()

        print(f"\n  [畴结构分析]")
        print(f"  畴壁面积分数: {wall_fraction:.4f}")
        print(f"  网格 Euler 特征数: {stats['euler_characteristic']}")
        print(f"  边界边数: {stats['n_boundary_edges']}")

        return {
            'wall_fraction': wall_fraction,
            'mesh_stats': stats,
        }

    def _analyze_pca(self):
        """序参量 PCA 分析."""
        if len(self.history['P_avg']) < 3:
            print("\n  [PCA] 数据不足, 跳过")
            return {}

        P_history = np.array(self.history['P_avg'])
        M_history = np.array(self.history['M_avg'])

        # 组合序参量
        combined = np.column_stack([P_history, M_history])

        if combined.shape[0] < 3:
            return {}

        pc_data = self.op_analyzer.fit_pca(combined)

        print(f"\n  [序参量 PCA]")
        print(f"  主成分数: {pc_data.shape[1]}")
        if self.op_analyzer.explained_variance_ratio is not None:
            print(f"  方差解释比: "
                  f"{self.op_analyzer.explained_variance_ratio}")

        return {
            'n_components': pc_data.shape[1],
            'explained_variance_ratio':
                self.op_analyzer.explained_variance_ratio,
        }

    def _classify_phase(self):
        """相分类."""
        phase_info = self.phase_classifier.classify_parameters(
            self.config.temperature
        )

        print(f"\n  [相分类]")
        print(f"  当前相: {phase_info['phase_name']}")
        print(f"  置信度: {phase_info['confidence']:.2f}")

        return phase_info

    def _compute_me_bounds(self):
        """磁电界估计."""
        bounds = self.me_bounds.compute_lp_bounds()

        print(f"\n  [磁电系数界]")
        print(f"  Cauchy-Schwarz 界: ±{bounds['cs_bound']:.4e} s/m")
        print(f"  LP 上界: {bounds['alpha_max']:.4e} s/m")
        print(f"  LP 下界: {bounds['alpha_min']:.4e} s/m")
        print(f"  顶点数: {bounds['n_vertices']}")

        return bounds

    def _analyze_entropy(self):
        """熵迹分析."""
        t_values = np.logspace(-2, 2, 30)
        t_vals, theta = self.entropy_analyzer.compute_entropy_trace(t_values)

        beta, r2 = self.entropy_analyzer.scaling_exponent(t_vals, theta)

        print(f"\n  [熵迹分析]")
        print(f"  标度指数 β = {beta:.4f}")
        print(f"  拟合 R² = {r2:.4f}")

        return {
            'scaling_exponent': beta,
            'r_squared': r2,
            'theta_values': theta,
        }

    def _analyze_bz(self):
        """布里渊区分析."""
        # 磁子色散
        k_path = self.bz.k_points[:50]
        omega = self.bz.compute_magnon_dispersion(k_path)

        # 态密度
        E_grid, dos = self.bz.density_of_states(omega)

        print(f"\n  [布里渊区分析]")
        print(f"  k 点数: {len(self.bz.k_points)}")
        print(f"  磁子能量范围: [{np.min(omega):.4e}, {np.max(omega):.4e}] J")
        print(f"  DOS 峰值: {np.max(dos):.4e}")

        return {
            'magnon_energy_range': (float(np.min(omega)),
                                    float(np.max(omega))),
            'dos_peak': float(np.max(dos)),
        }

    def _analyze_md_trajectory(self):
        """MD 轨迹分析."""
        traj = self.md_analyzer.generate_synthetic_trajectory(
            n_steps=500, T=self.config.temperature,
            seed=self.config.seed
        )

        dist_stats = self.md_analyzer.analyze_distances(traj)
        energy_stats = self.md_analyzer.analyze_energies(traj)

        print(f"\n  [MD 轨迹分析]")
        print(f"  Fe-O 距离: {dist_stats['d_FeO_angstrom']['mean']:.3f} "
              f"± {dist_stats['d_FeO_angstrom']['std']:.3f} Å")
        print(f"  Bi-O 距离: {dist_stats['d_BiO_angstrom']['mean']:.3f} "
              f"± {dist_stats['d_BiO_angstrom']['std']:.3f} Å")
        print(f"  总能: {energy_stats['E_total_eV']['mean']:.3f} "
              f"± {energy_stats['E_total_eV']['std']:.3f} eV")

        return {
            'distance_stats': dist_stats,
            'energy_stats': energy_stats,
        }

    def _compute_topology(self):
        """拓扑荷计算."""
        Q, q_density = self.mesh.compute_topological_charge(self.P)

        print(f"\n  [拓扑分析]")
        print(f"  总拓扑荷 Q = {Q:.4f}")

        return {
            'total_charge': float(Q),
            'charge_density_rms': float(np.sqrt(np.mean(q_density ** 2))),
        }

    # ============================================================
    # 汇总报告
    # ============================================================

    def generate_summary(self, results):
        """
        生成模拟结果汇总报告.

        参数:
            results: run_postprocessing 返回的结果字典

        返回:
            summary: 文本报告
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  多铁性材料磁电耦合模拟 - 结果汇总")
        lines.append("=" * 60)

        lines.append(f"\n1. 模拟参数:")
        lines.append(f"   网格: {self.config.nx}×{self.config.ny}")
        lines.append(f"   温度: {self.config.temperature} K")
        lines.append(f"   时间步长: {self.config.dt:.3e} s")
        lines.append(f"   FD 精度: O(h^{self.config.fd_order})")

        if 'domain' in results:
            lines.append(f"\n2. 畴结构:")
            lines.append(f"   畴壁分数: "
                         f"{results['domain']['wall_fraction']:.4f}")

        if 'phase' in results:
            lines.append(f"\n3. 相分类:")
            lines.append(f"   当前相: {results['phase']['phase_name']}")

        if 'me_bounds' in results:
            lines.append(f"\n4. 磁电系数界:")
            mb = results['me_bounds']
            lines.append(f"   CS 界: ±{mb['cs_bound']:.4e} s/m")
            lines.append(f"   LP 上界: {mb['alpha_max']:.4e} s/m")

        if 'entropy' in results:
            lines.append(f"\n5. 熵迹分析:")
            lines.append(f"   标度指数: {results['entropy']['scaling_exponent']:.4f}")

        if 'topology' in results:
            lines.append(f"\n6. 拓扑荷:")
            lines.append(f"   Q = {results['topology']['total_charge']:.4f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
