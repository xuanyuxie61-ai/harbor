"""
分子动力学轨迹后处理模块
=========================
对应种子项目: 1270_zruan_PAC_desensitization_eLife2022
  (MD 轨迹距离/能量分析 → 多铁性 MD 模拟后处理)

物理背景:
    多铁性材料的原子尺度模拟 (分子动力学或第一性原理 MD)
    产生大量轨迹数据, 需要后处理提取物理量:

    1. Fe-O 八面体畸变:
        - Fe-O 距离: 表征八面体倾斜
        - O-Fe-O 键角: 表征八面体变形
        - 与铁电极化直接关联

    2. 磁有序参数:
        - Fe-Fe 距离与超交换角
        - 自旋关联函数 <Sᵢ·Sⱼ>

    3. 能量分量:
        - 库仑能, 范德华能, 键伸缩能
        - 与 LGD 自由能各分量对应

方法论 (对应种子项目 1270):
    1. 读取轨迹数据 (距离和能量时间序列)
    2. 统计: 均值, 标准差, 分布
    3. 对比: 不同条件 (温度, 掺杂, 应力) 下的差异

核心公式:
    自旋关联函数:
        C(r) = <S(0)·S(r)> / <|S|²>
    对于反铁磁体:
        C(r) ~ (-1)^n · exp(-r/ξ) / r^η

    Debye-Waller 因子:
        W = ½ <u²> · k²
    其中 <u²> 为原子均方位移.
"""

import numpy as np


class MDTrajectoryAnalyzer:
    """
    多铁性材料 MD 轨迹后处理器.

    分析距离时间序列、能量时间序列,
    提取结构有序参数和热力学量.
    """

    def __init__(self, n_atoms=40, a_lattice=3.96e-10):
        """
        参数:
            n_atoms: 原子数
            a_lattice: 晶格常数 (m)
        """
        self.n_atoms = n_atoms
        self.a = a_lattice

    # ============================================================
    # 合成 MD 轨迹数据
    # ============================================================

    def generate_synthetic_trajectory(self, n_steps=1000, T=300.0,
                                      seed=285):
        """
        生成 BiFeO3 的合成 MD 轨迹.

        基于物理模型生成:
        1. Fe-O 距离: 约 2.0 Å, 带热涨落
        2. Bi-O 距离: 约 2.5 Å, 带铁电偏移
        3. 能量: 库仑 + 短程 + 键伸缩

        物理模型:
            d_FeO(t) = d₀ + A·cos(ωt) + σ·ξ(t)
            其中 ξ(t) 为 Ornstein-Uhlenbeck 过程

        参数:
            n_steps: 时间步数
            T: 温度 (K)
            seed: 随机种子

        返回:
            trajectory: dict 包含各物理量的时间序列
        """
        from multiferroic_constants import K_BOLTZMANN

        rng = np.random.RandomState(seed)

        dt = 1e-15  # 1 fs
        time = np.arange(n_steps) * dt * 1e12  # ps

        # 热涨落幅度
        sigma_d = np.sqrt(K_BOLTZMANN * T / 100.0) * 1e10  # Å

        # Fe-O 距离 (约 2.0 Å + 热涨落)
        d_FeO = 2.0 + sigma_d * self._ou_process(n_steps, 0.01, rng)

        # Bi-O 距离 (铁电偏移, 约 2.5 Å)
        d_BiO = 2.5 + 0.1 * np.sin(2 * np.pi * time / 5.0) + \
            sigma_d * 1.5 * self._ou_process(n_steps, 0.005, rng)

        # O-Fe-O 键角 (理想 90°, 有畸变)
        angle_OFeO = 90.0 + 3.0 * self._ou_process(n_steps, 0.02, rng)

        # Fe-Fe 距离
        d_FeFe = self.a * 1e10 * np.sqrt(3) + \
            sigma_d * 0.5 * self._ou_process(n_steps, 0.01, rng)

        # 能量分量 (eV/atom)
        E_coulomb = -15.0 + 0.5 * self._ou_process(n_steps, 0.005, rng)
        E_vdw = -0.5 + 0.1 * self._ou_process(n_steps, 0.01, rng)
        E_bond = 2.0 + sigma_d ** 2 * 0.01 + \
            0.2 * self._ou_process(n_steps, 0.02, rng)
        E_total = E_coulomb + E_vdw + E_bond

        # 极化 (C/m²)
        P_x = 0.3 + 0.1 * self._ou_process(n_steps, 0.003, rng)
        P_y = 0.3 + 0.1 * self._ou_process(n_steps, 0.003, rng)
        P_z = 0.8 + 0.15 * self._ou_process(n_steps, 0.003, rng)

        trajectory = {
            'time_ps': time,
            'd_FeO_angstrom': d_FeO,
            'd_BiO_angstrom': d_BiO,
            'angle_OFeO_degree': angle_OFeO,
            'd_FeFe_angstrom': d_FeFe,
            'E_coulomb_eV': E_coulomb,
            'E_vdw_eV': E_vdw,
            'E_bond_eV': E_bond,
            'E_total_eV': E_total,
            'P_x': P_x,
            'P_y': P_y,
            'P_z': P_z,
        }

        return trajectory

    def _ou_process(self, n_steps, theta, rng):
        """
        Ornstein-Uhlenbeck 过程.

        dX = -θ·X·dt + σ·dW

        用于模拟具有记忆效应的热涨落.

        参数:
            n_steps: 步数
            theta: 回复率
            rng: 随机数生成器

        返回:
            X: OU 过程样本
        """
        sigma = np.sqrt(2 * theta)
        X = np.zeros(n_steps)
        dt = 0.1
        for i in range(1, n_steps):
            dW = rng.randn() * np.sqrt(dt)
            X[i] = X[i - 1] - theta * X[i - 1] * dt + sigma * dW
        return X

    # ============================================================
    # 距离分析 (对应种子项目 1270 read_dist_xvg)
    # ============================================================

    def analyze_distances(self, trajectory):
        """
        分析距离时间序列.

        对应种子项目 1270 的距离分析:
        计算均值、标准差、分布.

        参数:
            trajectory: MD 轨迹字典

        返回:
            stats: 统计信息字典
        """
        stats = {}

        for key in ['d_FeO_angstrom', 'd_BiO_angstrom',
                    'angle_OFeO_degree', 'd_FeFe_angstrom']:
            data = trajectory[key]
            stats[key] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'rmsd': np.sqrt(np.mean((data - np.mean(data)) ** 2)),
            }

        return stats

    # ============================================================
    # 能量分析 (对应种子项目 1270 read_energy_xvg)
    # ============================================================

    def analyze_energies(self, trajectory):
        """
        分析能量时间序列.

        对应种子项目 1270 的能量分析:
        提取各能量分量, 计算统计量.

        参数:
            trajectory: MD 轨迹字典

        返回:
            energy_stats: 能量统计字典
        """
        stats = {}

        for key in ['E_coulomb_eV', 'E_vdw_eV',
                    'E_bond_eV', 'E_total_eV']:
            data = trajectory[key]
            stats[key] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'fluctuation': np.std(data) ** 2,
            }

        # 热容估算: C_V = (<E²> - <E>²) / (k_B·T²)
        E = trajectory['E_total_eV']
        from multiferroic_constants import K_BOLTZMANN, E_CHARGE
        kT = K_BOLTZMANN * 300.0 / E_CHARGE  # eV
        C_V = (np.mean(E ** 2) - np.mean(E) ** 2) / kT
        stats['heat_capacity_eV_per_K'] = C_V

        return stats

    # ============================================================
    # 自旋关联函数
    # ============================================================

    def spin_correlation_function(self, spin_trajectory, max_lag=100):
        """
        计算自旋关联函数 C(t).

        C(t) = <S(0)·S(t)> / <|S|²>

        参数:
            spin_trajectory: 自旋时间序列, shape (n_steps, 3)
            max_lag: 最大延迟步数

        返回:
            lags: 延迟步数数组
            C_t: 关联函数
        """
        n_steps = len(spin_trajectory)
        max_lag = min(max_lag, n_steps - 1)

        S_norm_sq = np.mean(np.sum(spin_trajectory ** 2, axis=1))
        if S_norm_sq < 1e-30:
            return np.arange(max_lag), np.zeros(max_lag)

        C_t = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                C_t[lag] = 1.0
            else:
                S0 = spin_trajectory[:-lag]
                St = spin_trajectory[lag:]
                dot_product = np.sum(S0 * St, axis=1)
                C_t[lag] = np.mean(dot_product) / S_norm_sq

        return np.arange(max_lag), C_t

    # ============================================================
    # 功率谱分析
    # ============================================================

    def power_spectrum(self, signal, dt=1e-15):
        """
        计算信号的功率谱密度.

        S(f) = |FFT(signal)|² / N

        用于识别:
        - 声子模式 (特征频率)
        - 磁子模式
        - 畴壁振动频率

        参数:
            signal: 时间序列
            dt: 时间步长 (s)

        返回:
            freq: 频率数组 (Hz)
            power: 功率谱密度
        """
        N = len(signal)
        signal_centered = signal - np.mean(signal)

        fft_vals = np.fft.rfft(signal_centered)
        power = np.abs(fft_vals) ** 2 / N

        freq = np.fft.rfftfreq(N, d=dt)

        return freq, power

    # ============================================================
    # 均方位移
    # ============================================================

    def mean_square_displacement(self, positions, max_lag=None):
        """
        计算均方位移 MSD(t).

        MSD(t) = <|r(t₀+t) - r(t₀)|²>

        用于判断扩散行为:
            MSD ~ t: 扩散
            MSD ~ t²: 弹道
            MSD ~ const: 局域化

        参数:
            positions: 位置时间序列, shape (n_steps, 3)
            max_lag: 最大延迟

        返回:
            lags: 延迟数组
            msd: MSD 值
        """
        n_steps = len(positions)
        if max_lag is None:
            max_lag = n_steps // 4

        max_lag = min(max_lag, n_steps - 1)
        msd = np.zeros(max_lag)

        for lag in range(max_lag):
            if lag == 0:
                msd[lag] = 0.0
            else:
                diff = positions[lag:] - positions[:-lag]
                msd[lag] = np.mean(np.sum(diff ** 2, axis=1))

        return np.arange(max_lag), msd
