"""
qgp_spiking_source.py — 喷注能量沉积: 脉冲源模型
=====================================================

融合种子项目: 1039_gaspar-c_replay-simulations (脉冲神经网络模拟)

本模块模拟高能部分子 (喷注) 穿越 QGP 介质时的能量沉积,
类比于脉冲神经网络中的脉冲事件.

物理背景:
----------
高能 parton (夸克/胶子) 在 QGP 中传播时经历:
1. 碰撞能量损失 (collisional): dE/dx ~ alpha_s^2 * T^2 * ln(E/T)
2. 辐射能量损失 (radiative): dE/dx ~ alpha_s * qhat * L
   (BDMPS-Z 机制, L 为介质长度)

能量沉积作为源项加入守恒律:
    d_mu T^{mu nu} = J^nu(x, t)

其中 J^nu 为喷注能量-动量沉积率.

脉冲源模型 (源自 1039_replay-simulations):
-------------------------------------------
类比于脉冲神经元的时间离散发放:
    J(x, t) = sum_k E_k * delta(x - x_jet(t_k)) * delta(t - t_k)

在离散化中:
    J[i, j] = E_dep * Gaussian(x_i - x_jet, y_j - y_jet; sigma)

其中 sigma ~ 0.3 fm 为沉积宽度 (Debye 屏蔽尺度).

喷注轨迹:
    x_jet(t) = x_0 + v_jet * (t - t_0)
    v_jet ~ c = 1 (光锥)

能量损失模型:
    dE/dt = -qhat * L / (2*E) * E  (BDMPS-Z)
    或简化: dE/dt = -kappa * sqrt(e(x_jet)) * E^alpha

脉冲检测 (源自 detect_peaks):
    从能量沉积时间序列中检测脉冲事件,
    使用阈值交叉和峰检测算法.
"""

import numpy as np
from qgp_config import NumericalParams, HBAR_C, ALPHA_S, N_COLOR
from qgp_grid import QGPGrid


class JetQuenchingSource:
    """
    喷注淬火能量沉积源

    模拟高能部分子在 QGP 中的传播和能量损失,
    产生时空依赖的能量-动量源项.

    Attributes:
        grid: 计算网格
        n_jets: 喷注数量
        jet_positions: 喷注当前位置 (n_jets, 2)
        jet_directions: 喷注方向 (n_jets, 2)
        jet_energies: 喷注能量 (n_jets,)
        deposition_history: 沉积历史
    """

    def __init__(self, grid: QGPGrid):
        """
        Args:
            grid: 计算网格
        """
        self.grid = grid
        self.n_jets = 0
        self.jet_positions = []
        self.jet_directions = []
        self.jet_energies = []
        self.deposition_history = []

        # 沉积参数
        self.sigma_dep = 0.3  # 沉积高斯宽度 (fm)
        self.kappa_collisional = 0.5  # 碰撞能量损失系数
        self.kappa_radiative = 1.0  # 辐射能量损失系数

    def add_jet(self, position: tuple, direction: tuple,
                 energy: float):
        """
        添加一个喷注

        Args:
            position: 初始位置 (x_0, y_0) in fm
            direction: 方向单位向量 (dx, dy)
            energy: 初始能量 (GeV)
        """
        # 归一化方向
        d = np.sqrt(direction[0]**2 + direction[1]**2)
        if d < 1.0e-10:
            direction = (1.0, 0.0)
        else:
            direction = (direction[0]/d, direction[1]/d)

        self.jet_positions.append(np.array(position))
        self.jet_directions.append(np.array(direction))
        self.jet_energies.append(energy)
        self.n_jets += 1

    def energy_loss_rate(self, energy: float, local_edensity: float,
                          tau: float) -> float:
        """
        喷注能量损失率 dE/dt

        组合碰撞和辐射能量损失:

        碰撞损失 (Bjorken 模型):
            dE/dx|_coll = (4*pi*alpha_s^2 / 3) * T^2 * CF *
                          ln(12*T/m_D) * (1 + alpha_s/(2*pi))

        辐射损失 (BDMPS-Z):
            dE/dx|_rad = alpha_s * CF * sqrt(qhat * E)
            qhat = jet_quenching_parameter(T)

        简化模型:
            dE/dt = -kappa * sqrt(e_local) * (1 + E/E_0)

        Args:
            energy: 喷注当前能量 (GeV)
            local_edensity: 局部能量密度 (GeV/fm^3)
            tau: 当前时间 (fm/c)

        Returns:
            dE/dt (GeV/fm), 负值表示损失
        """
        T_local = (local_edensity / max(NumericalParams.SIGMA_SB, 1.0e-10))**0.25
        T_local = max(T_local, 1.0e-6)

        # 碰撞损失
        CF = (N_COLOR**2 - 1) / (2.0 * N_COLOR)
        dE_coll = -self.kappa_collisional * 4.0 * np.pi * ALPHA_S**2 / 3.0 * \
                  T_local**2 * CF * np.log(max(12.0 * T_local / 0.5, 2.0))

        # 辐射损失 (BDMPS-Z 简化)
        from qgp_eos import QGPEquationOfState
        eos = QGPEquationOfState()
        qhat = eos.jet_quenching_parameter(np.array([T_local]))[0]
        dE_rad = -self.kappa_radiative * ALPHA_S * CF * \
                 np.sqrt(qhat * max(energy, 0.1))

        return dE_coll + dE_rad

    def propagate_jet(self, dt: float, energy_density_field: np.ndarray,
                       tau: float):
        """
        推进所有喷注一个时间步

        对每个喷注:
        1. 查找局部能量密度 (双线性插值)
        2. 计算能量损失
        3. 更新喷注位置和能量
        4. 记录沉积能量

        Args:
            dt: 时间步长 (fm/c)
            energy_density_field: 能量密度场 (含鬼单元)
            tau: 当前时间
        """
        ng = self.grid.ng
        for j in range(self.n_jets):
            pos = self.jet_positions[j]
            dire = self.jet_directions[j]
            E = self.jet_energies[j]

            if E <= 0.01:  # 能量耗尽
                continue

            # 局部能量密度 (最近邻插值)
            ix = int(round((pos[0] - self.grid.x[ng]) / self.grid.dx)) + ng
            iy = int(round((pos[1] - self.grid.y[ng]) / self.grid.dy)) + ng
            ix = np.clip(ix, ng, ng + self.grid.nx - 1)
            iy = np.clip(iy, ng, ng + self.grid.ny - 1)
            e_local = energy_density_field[iy, ix]

            # 能量损失
            dEdt = self.energy_loss_rate(E, e_local, tau)
            dE = dEdt * dt  # 负值
            dE = max(dE, -E * 0.5)  # 单步不超过 50%

            # 更新
            self.jet_energies[j] = E + dE
            self.jet_positions[j] = pos + dire * dt  # v ~ c = 1

            # 记录沉积
            self.deposition_history.append({
                'time': tau,
                'jet_id': j,
                'position': pos.copy(),
                'energy_deposited': -dE,
                'remaining_energy': self.jet_energies[j],
            })

    def compute_source_term(self, tau: float) -> np.ndarray:
        """
        计算当前时刻的能量-动量沉积源项

        J_E(x, y) = sum_j (-dE/dt)_j *
                     G(x - x_j, y - y_j; sigma_dep)

        其中 G 为二维高斯:
            G(dx, dy; sigma) = exp(-(dx^2+dy^2)/(2*sigma^2)) / (2*pi*sigma^2)

        Args:
            tau: 当前时间

        Returns:
            源项 J_E (内部区域 ny, nx)
        """
        ng = self.grid.ng
        ny, nx = self.grid.ny, self.grid.nx
        X = self.grid.X[ng:ng+ny, ng:ng+nx]
        Y = self.grid.Y[ng:ng+ny]

        source = np.zeros((ny, nx))

        for j in range(self.n_jets):
            if self.jet_energies[j] <= 0.01:
                continue

            pos = self.jet_positions[j]
            # 高斯沉积
            r2 = (X - pos[0])**2 + (Y - pos[1])**2
            sigma = self.sigma_dep
            G = np.exp(-r2 / (2.0 * sigma**2)) / (2.0 * np.pi * sigma**2)

            # 沉积率 (从历史中获取最近一步的 dE)
            dE = 0.0
            for rec in reversed(self.deposition_history):
                if rec['jet_id'] == j:
                    dE = rec['energy_deposited']
                    break

            source += dE * G

        return source

    def detect_deposition_peaks(self, time_window: tuple = None) -> list:
        """
        检测能量沉积脉冲事件 (源自 1039 detect_peaks)

        算法:
        1. 对每个喷注, 提取 dE/dt 时间序列
        2. 使用阈值法检测脉冲:
            - dE/dt > threshold
            - 局部极大值
        3. 返回脉冲时间、位置、幅度

        Args:
            time_window: (t_start, t_end) 检测时间窗

        Returns:
            [{'time': float, 'jet_id': int, 'amplitude': float,
              'position': array}, ...]
        """
        if not self.deposition_history:
            return []

        # 按喷注分组
        by_jet = {}
        for rec in self.deposition_history:
            jid = rec['jet_id']
            if jid not in by_jet:
                by_jet[jid] = []
            by_jet[jid].append(rec)

        peaks = []
        threshold = 0.01  # GeV/fm 阈值

        for jid, records in by_jet.items():
            # 时间序列
            times = [r['time'] for r in records]
            amplitudes = [r['energy_deposited'] for r in records]

            if len(times) < 3:
                continue

            for i in range(1, len(amplitudes) - 1):
                # 局部极大值
                if (amplitudes[i] > amplitudes[i-1] and
                    amplitudes[i] > amplitudes[i+1] and
                    amplitudes[i] > threshold):
                    peaks.append({
                        'time': times[i],
                        'jet_id': jid,
                        'amplitude': amplitudes[i],
                        'position': records[i]['position'],
                    })

        return peaks

    def total_deposited_energy(self) -> float:
        """总沉积能量"""
        return sum(r['energy_deposited'] for r in self.deposition_history)

    def total_remaining_energy(self) -> float:
        """总剩余能量"""
        return sum(self.jet_energies)

    def jet_R_AA(self, initial_energy: float = None) -> float:
        """
        核修正因子 R_AA

        R_AA = dN_AA / (N_coll * dN_pp)
        ~ E_final / E_initial (简化)

        R_AA < 1: 喷注淬火 (能量损失)
        R_AA ~ 1: 无介质效应

        Args:
            initial_energy: 初始能量 (若为 None, 使用所有喷注初始能量之和)

        Returns:
            R_AA
        """
        if initial_energy is None:
            initial_energy = sum(self.jet_energies) + self.total_deposited_energy()

        if initial_energy < 1.0e-10:
            return 0.0

        return self.total_remaining_energy() / initial_energy
