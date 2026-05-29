"""
bci_decoder.py — 脑机接口神经信号解码核心引擎
===============================================
整合多尺度神经模型（neural_mass_ode.py）、神经场动力学（neural_field_solver.py）、
谱分析（spectral_signal_analysis.py）、连接组拓扑（connectome_topology.py）、
稳定性分析（stability_and_roots.py）与电极采样（electrode_sampling.py），
构建端到端的 BCI 信号解码流程。

解码流程：
1. 多通道 E-I 神经振荡器生成模拟 LFP 信号
2. 神经场 PDE 求解器计算空间活动分布
3. 电极阵列采样获取离散时空信号矩阵 S(t,e)
4. Chebyshev 谱分解提取时频特征
5. Gauss-Legendre 积分计算信号能量泛函
6. 连接组扩散分析识别信息源区域
7. 稳定性分析验证闭环系统安全
8. 综合解码输出运动意图向量

核心数学：
---
**解码目标泛函：**

给定电极记录 S ∈ R^{T×E}，解码目标是找到运动参数向量 m ∈ R^d：

    m* = argmin_m  ( ||S - Φ(m)||_F^2 + λ * R(m) )

其中 Φ(m) 为前向神经模型，R(m) = ||Dm||^2 为光滑正则项，
D 为离散 Laplacian。

---
**时空特征张量：**

对每个电极 e 的信号 s_e(t)，构造特征向量：
    f_e = [μ_e, σ_e, E_e, ω_e, κ_e]
其中
    μ_e : 均值（Gauss-Legendre 积分）
    σ_e : 标准差
    E_e : L2 能量泛函
    ω_e : 主导 Chebyshev 模式频率
    κ_e : 偏度（三阶矩）

---
**意图解码映射：**
使用线性解码器（最优线性估计）：
    m = W · vec(F)
其中 W 通过最小二乘从训练数据估计。
"""

import numpy as np
from neural_mass_ode import MultiPopulationArray, EIOscillator, AIRPopulationDynamics
from neural_field_solver import NeuralFieldSolver, generate_square_grid
from spectral_signal_analysis import (
    ChebyshevSpectrumAnalyzer, GaussLegendreSignalIntegrator,
    SpatialPaduaSampler, generate_padua_points
)
from connectome_topology import BrainConnectomeGraph, ConnectomePercolationBridge
from stability_and_roots import BCIStabilityAnalyzer
from electrode_sampling import ElectrodeArray, CorticalSurfaceGeometry
from utils import softmax_stable


class SyntheticBCISignalGenerator:
    """
    合成 BCI 信号生成器：模拟多电极记录到的神经信号。
    """

    def __init__(self,
                 n_electrodes=64,
                 n_channels=8,
                 t_span=(0.0, 2.0),
                 dt=0.001,
                 field_nx=32,
                 random_state=42):
        np.random.seed(random_state)
        self.n_electrodes = n_electrodes
        self.n_channels = n_channels
        self.t_span = t_span
        self.dt = dt
        self.field_nx = field_nx
        # 初始化组件
        self.ei_array = MultiPopulationArray(
            n_channels=n_channels,
            omega=2.0 * np.pi * 10.0,  # 10 Hz alpha 节律
            sawtooth_amp=0.8)
        self.geometry = CorticalSurfaceGeometry(curvature_radius=80.0, patch_radius=5.0)
        self.electrode_array = ElectrodeArray(n_electrodes=n_electrodes, geometry=self.geometry)
        self.electrode_array.generate_layout(layout='cvt')
        # 神经场
        X, Y = generate_square_grid(xlim=(-5, 5), ylim=(-5, 5),
                                    nx=field_nx, ny=field_nx, centering='cell')
        self.neural_field = NeuralFieldSolver(
            X, Y, tau=0.015, sigma_e=1.0, sigma_i=2.0, A_e=1.0, A_i=0.5)
        # 连接组
        self.connectome = BrainConnectomeGraph(n_regions=30, connection_prob=0.12,
                                               weight_dist='lognormal', random_state=random_state)

    def generate_lfp_signals(self):
        """
        生成多通道 LFP 信号。
        返回 t, lfp_channels shape (n_channels, n_t)
        """
        # TODO_HOLE_3: implement multi-channel LFP signal generation
        # Use self.ei_array (a MultiPopulationArray) to simulate E-I dynamics
        # and extract LFP channels. Must return (t, lfp_channels).
        pass

    def generate_neural_field_signal(self, t, lfp_channels):
        """
        用神经场方程生成空间活动分布，并在电极位置采样。
        外部输入由 LFP 通道信号驱动。
        """
        n_t = len(t)
        # 简化：使用第一个通道的 LFP 作为外部输入调制
        u0 = np.zeros((self.field_nx, self.field_nx), dtype=float)
        # 在中心区域添加初始高斯脉冲
        cx, cy = self.field_nx // 2, self.field_nx // 2
        sigma_init = 3.0
        for i in range(self.field_nx):
            for j in range(self.field_nx):
                dx = i - cx
                dy = j - cy
                u0[i, j] = 0.5 * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_init ** 2))

        def I_ext_func(ti, X, Y):
            # 外部输入：时间调制高斯 + LFP 驱动
            idx = int(np.clip(ti / self.t_span[1] * (n_t - 1), 0, n_t - 1))
            lfp_drive = lfp_channels[0, idx] if lfp_channels.shape[0] > 0 else 0.0
            # 空间高斯调制
            r2 = X ** 2 + Y ** 2
            return 0.3 * lfp_drive * np.exp(-r2 / 8.0)

        t_field, u_hist = self.neural_field.simulate(
            u0, I_ext_func, t_span=self.t_span, dt=self.dt * 5, method='euler')
        return t_field, u_hist

    def sample_at_electrodes(self, u_hist, t_field):
        """
        在电极位置采样神经场活动。
        返回 electrode_signals shape (n_electrodes, n_t_field)
        """
        n_t = u_hist.shape[0]
        signals = np.zeros((self.n_electrodes, n_t), dtype=float)
        pos = self.electrode_array.positions
        # 将电极 (x,y) 坐标映射到网格索引
        xlim = (-5, 5)
        ylim = (-5, 5)
        nx = self.field_nx
        for e in range(self.n_electrodes):
            x, y = pos[e, 0], pos[e, 1]
            # 映射到网格索引
            ix = int(np.clip((x - xlim[0]) / (xlim[1] - xlim[0]) * nx, 0, nx - 1))
            iy = int(np.clip((y - ylim[0]) / (ylim[1] - ylim[0]) * nx, 0, nx - 1))
            signals[e, :] = u_hist[:, ix, iy]
        return signals

    def generate_full_dataset(self):
        """
        生成完整的合成 BCI 数据集：
        返回字典包含 t, lfp, neural_field, electrode_signals
        """
        t, lfp = self.generate_lfp_signals()
        t_field, u_hist = self.generate_neural_field_signal(t, lfp)
        electrode_signals = self.sample_at_electrodes(u_hist, t_field)
        return {
            'time': t,
            'lfp_channels': lfp,
            'field_time': t_field,
            'field_history': u_hist,
            'electrode_signals': electrode_signals,
            'electrode_positions': self.electrode_array.positions,
            'connectome': self.connectome
        }


class BCIFeatureExtractor:
    """
    BCI 特征提取器：从电极信号中提取高维特征。
    """

    def __feature_init__(self, n_cheb_modes=32, n_gl_points=32):
        self.cheb_analyzer = ChebyshevSpectrumAnalyzer(n_modes=n_cheb_modes)
        self.gl_integrator = GaussLegendreSignalIntegrator(n_points=n_gl_points)

    def __init__(self, n_cheb_modes=32, n_gl_points=32):
        self.cheb_analyzer = ChebyshevSpectrumAnalyzer(n_modes=n_cheb_modes)
        self.gl_integrator = GaussLegendreSignalIntegrator(n_points=n_gl_points)

    def extract_temporal_features(self, signal, t):
        """
        提取单个电极信号的时域特征。
        """
        # Gauss-Legendre 统计矩
        moments = self.gl_integrator.signal_moments(signal, t)
        # 能量泛函
        energy = self.gl_integrator.signal_energy_functional(signal, t, alpha=2.0)
        # Chebyshev 谱特征
        cheb = self.cheb_analyzer.analyze(signal, t_min=t[0], t_max=t[-1])
        # 组合特征向量
        features = np.array([
            moments['mean'],
            moments['std'],
            moments['skewness'],
            moments['kurtosis'],
            energy,
            cheb['dominant_mode'],
            cheb['energy'],
            cheb['dc_component']
        ], dtype=float)
        return features

    def extract_spatial_features(self, electrode_signals, positions):
        """
        提取空间特征：信号的空间梯度、空间自相关系数。
        """
        n_elec = electrode_signals.shape[0]
        # 空间自相关系数（简化：最近邻相关）
        spatial_corr = 0.0
        count = 0
        for i in range(n_elec):
            for j in range(i + 1, n_elec):
                d = np.linalg.norm(positions[i] - positions[j])
                if 0 < d < 2.5:  # 近邻
                    sig_i = electrode_signals[i]
                    sig_j = electrode_signals[j]
                    # 皮尔逊相关
                    c = np.corrcoef(sig_i, sig_j)[0, 1]
                    if not np.isnan(c):
                        spatial_corr += c
                        count += 1
        avg_spatial_corr = spatial_corr / max(count, 1)
        # 信号功率的空间方差
        power = np.var(electrode_signals, axis=1)
        power_variance = np.var(power)
        return np.array([avg_spatial_corr, power_variance], dtype=float)

    def extract_all_features(self, dataset):
        """
        从数据集中提取所有电极的特征矩阵。
        返回 features shape (n_electrodes, n_features_per_electrode)
        和全局空间特征。
        """
        electrode_signals = dataset['electrode_signals']
        positions = dataset['electrode_positions']
        # 电极信号使用 field_time（神经场时间步）
        t = dataset['field_time']
        n_elec = electrode_signals.shape[0]
        per_elec_features = []
        for e in range(n_elec):
            feats = self.extract_temporal_features(electrode_signals[e], t)
            per_elec_features.append(feats)
        per_elec_features = np.array(per_elec_features, dtype=float)
        spatial_feats = self.extract_spatial_features(electrode_signals, positions)
        return per_elec_features, spatial_feats


class BCIDecoder:
    """
    BCI 解码器：将神经特征映射到运动意图。
    使用最优线性估计 + 正则化。
    """

    def __init__(self, n_motor_dims=2, regularization=0.1):
        """
        n_motor_dims : 运动参数维度（如 2D 速度向量）
        """
        self.n_motor_dims = n_motor_dims
        self.lambda_reg = regularization
        self.W = None
        self.feature_mean = None
        self.feature_std = None

    def _flatten_features(self, per_elec_features, spatial_features):
        """将特征展平为向量。"""
        return np.concatenate([
            per_elec_features.flatten(),
            spatial_features.flatten()
        ])

    def fit(self, feature_list, motor_target_list):
        """
        从训练数据拟合解码器权重。
        feature_list : list of (per_elec_features, spatial_features)
        motor_target_list : list of motor vectors
        """
        X = []
        Y = []
        for feats, motor in zip(feature_list, motor_target_list):
            per_elec, spatial = feats
            x_vec = self._flatten_features(per_elec, spatial)
            X.append(x_vec)
            Y.append(motor)
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        # 标准化
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        self.feature_std[self.feature_std < 1e-10] = 1.0
        X_norm = (X - self.feature_mean) / self.feature_std
        # 岭回归：W = (X^T X + λI)^{-1} X^T Y
        n_features = X_norm.shape[1]
        XtX = X_norm.T @ X_norm + self.lambda_reg * np.eye(n_features)
        try:
            self.W = np.linalg.solve(XtX, X_norm.T @ Y)
        except np.linalg.LinAlgError:
            self.W = np.linalg.lstsq(XtX, X_norm.T @ Y, rcond=None)[0]
        return self

    def decode(self, per_elec_features, spatial_features):
        """解码运动意图。"""
        if self.W is None:
            raise RuntimeError("Decoder has not been fitted yet.")
        x_vec = self._flatten_features(per_elec_features, spatial_features)
        x_norm = (x_vec - self.feature_mean) / self.feature_std
        motor = x_norm @ self.W
        return motor


class BCIPipeline:
    """
    完整的 BCI 解码流水线：生成数据 → 提取特征 → 解码 → 稳定性验证。
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.generator = SyntheticBCISignalGenerator(random_state=random_state)
        self.feature_extractor = BCIFeatureExtractor()
        self.decoder = BCIDecoder(n_motor_dims=2, regularization=0.5)
        self.stability_analyzer = None

    def run_single_trial(self, motor_target=None):
        """
        运行单试次解码流程。
        motor_target : 目标运动向量（2D），若为 None 则随机生成
        """
        if motor_target is None:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 2.0)
            motor_target = np.array([speed * np.cos(angle), speed * np.sin(angle)])

        # 1. 生成合成神经信号
        dataset = self.generator.generate_full_dataset()

        # 2. 稳定性分析
        ei_osc = EIOscillator(omega=2.0 * np.pi * 10.0)
        self.stability_analyzer = BCIStabilityAnalyzer(ei_osc, feedback_gain=0.3)
        stability = self.stability_analyzer.analyze_open_loop_stability()

        # 3. 连接组信息传播分析
        bridge = ConnectomePercolationBridge(self.generator.connectome, grid_shape=(16, 16))
        spread_info = bridge.analyze_information_spread(seed_region=0, steps=30)

        # 4. 特征提取
        per_elec_feats, spatial_feats = self.feature_extractor.extract_all_features(dataset)

        # 5. 返回结果
        return {
            'dataset': dataset,
            'motor_target': motor_target,
            'stability': stability,
            'information_spread': spread_info,
            'per_electrode_features': per_elec_feats,
            'spatial_features': spatial_feats
        }

    def run_training_and_decoding(self, n_train=20):
        """
        运行训练-测试流程：生成训练数据拟合解码器，然后在测试数据上评估。
        """
        # 训练阶段
        train_features = []
        train_targets = []
        for _ in range(n_train):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 2.0)
            target = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            trial = self.run_single_trial(motor_target=target)
            train_features.append((trial['per_electrode_features'], trial['spatial_features']))
            train_targets.append(target)
        self.decoder.fit(train_features, train_targets)

        # 测试阶段
        n_test = 10
        errors = []
        for _ in range(n_test):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 2.0)
            target = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            trial = self.run_single_trial(motor_target=target)
            decoded = self.decoder.decode(trial['per_electrode_features'],
                                           trial['spatial_features'])
            error = np.linalg.norm(decoded - target)
            errors.append(error)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        return {
            'mean_decoding_error': mean_error,
            'std_decoding_error': std_error,
            'test_errors': errors,
            'decoder_weights_shape': self.decoder.W.shape if self.decoder.W is not None else None
        }

    def run_full_analysis(self):
        """
        运行完整的分析流程，返回所有科学指标。
        """
        # 单次试次详细分析
        trial = self.run_single_trial()
        dataset = trial['dataset']
        stability = trial['stability']
        spread = trial['information_spread']

        # 解码性能
        decode_result = self.run_training_and_decoding(n_train=15)

        # 汇总所有指标
        results = {
            'signal_summary': {
                'n_electrodes': dataset['electrode_signals'].shape[0],
                'n_timepoints': dataset['electrode_signals'].shape[1],
                'lfp_peak_amplitude': float(np.max(np.abs(dataset['lfp_channels']))),
                'field_peak_amplitude': float(np.max(np.abs(dataset['field_history'])))
            },
            'stability_analysis': {
                'equilibrium_EI': stability['equilibrium'].tolist(),
                'jacobian_eigenvalues_real': [float(ev.real) for ev in stability['eigenvalues']],
                'jacobian_eigenvalues_imag': [float(ev.imag) for ev in stability['eigenvalues']],
                'mu_1': stability['mu_1'],
                'mu_2': stability['mu_2'],
                'mu_inf': stability['mu_inf'],
                'cauchy_root_bound': stability['cauchy_root_bound'],
                'is_stable': stability['is_stable']
            },
            'connectome_analysis': {
                'fiedler_value': self.generator.connectome.compute_fiedler_value(),
                'max_cluster_size': spread['max_cluster_size'],
                'is_spanning': spread['is_spanning']
            },
            'decoding_performance': decode_result,
            'electrode_geometry': {
                'n_electrodes': self.generator.electrode_array.n_electrodes,
                'spatial_coverage': self.generator.electrode_array.compute_spatial_coverage(),
                'n_triangles': len(self.generator.electrode_array.triangles) if self.generator.electrode_array.triangles is not None else 0
            }
        }
        return results
