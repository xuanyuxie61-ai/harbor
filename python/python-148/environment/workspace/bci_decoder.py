
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

        self.ei_array = MultiPopulationArray(
            n_channels=n_channels,
            omega=2.0 * np.pi * 10.0,
            sawtooth_amp=0.8)
        self.geometry = CorticalSurfaceGeometry(curvature_radius=80.0, patch_radius=5.0)
        self.electrode_array = ElectrodeArray(n_electrodes=n_electrodes, geometry=self.geometry)
        self.electrode_array.generate_layout(layout='cvt')

        X, Y = generate_square_grid(xlim=(-5, 5), ylim=(-5, 5),
                                    nx=field_nx, ny=field_nx, centering='cell')
        self.neural_field = NeuralFieldSolver(
            X, Y, tau=0.015, sigma_e=1.0, sigma_i=2.0, A_e=1.0, A_i=0.5)

        self.connectome = BrainConnectomeGraph(n_regions=30, connection_prob=0.12,
                                               weight_dist='lognormal', random_state=random_state)

    def generate_lfp_signals(self):



        pass

    def generate_neural_field_signal(self, t, lfp_channels):
        n_t = len(t)

        u0 = np.zeros((self.field_nx, self.field_nx), dtype=float)

        cx, cy = self.field_nx // 2, self.field_nx // 2
        sigma_init = 3.0
        for i in range(self.field_nx):
            for j in range(self.field_nx):
                dx = i - cx
                dy = j - cy
                u0[i, j] = 0.5 * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma_init ** 2))

        def I_ext_func(ti, X, Y):

            idx = int(np.clip(ti / self.t_span[1] * (n_t - 1), 0, n_t - 1))
            lfp_drive = lfp_channels[0, idx] if lfp_channels.shape[0] > 0 else 0.0

            r2 = X ** 2 + Y ** 2
            return 0.3 * lfp_drive * np.exp(-r2 / 8.0)

        t_field, u_hist = self.neural_field.simulate(
            u0, I_ext_func, t_span=self.t_span, dt=self.dt * 5, method='euler')
        return t_field, u_hist

    def sample_at_electrodes(self, u_hist, t_field):
        n_t = u_hist.shape[0]
        signals = np.zeros((self.n_electrodes, n_t), dtype=float)
        pos = self.electrode_array.positions

        xlim = (-5, 5)
        ylim = (-5, 5)
        nx = self.field_nx
        for e in range(self.n_electrodes):
            x, y = pos[e, 0], pos[e, 1]

            ix = int(np.clip((x - xlim[0]) / (xlim[1] - xlim[0]) * nx, 0, nx - 1))
            iy = int(np.clip((y - ylim[0]) / (ylim[1] - ylim[0]) * nx, 0, nx - 1))
            signals[e, :] = u_hist[:, ix, iy]
        return signals

    def generate_full_dataset(self):
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

    def __feature_init__(self, n_cheb_modes=32, n_gl_points=32):
        self.cheb_analyzer = ChebyshevSpectrumAnalyzer(n_modes=n_cheb_modes)
        self.gl_integrator = GaussLegendreSignalIntegrator(n_points=n_gl_points)

    def __init__(self, n_cheb_modes=32, n_gl_points=32):
        self.cheb_analyzer = ChebyshevSpectrumAnalyzer(n_modes=n_cheb_modes)
        self.gl_integrator = GaussLegendreSignalIntegrator(n_points=n_gl_points)

    def extract_temporal_features(self, signal, t):

        moments = self.gl_integrator.signal_moments(signal, t)

        energy = self.gl_integrator.signal_energy_functional(signal, t, alpha=2.0)

        cheb = self.cheb_analyzer.analyze(signal, t_min=t[0], t_max=t[-1])

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
        n_elec = electrode_signals.shape[0]

        spatial_corr = 0.0
        count = 0
        for i in range(n_elec):
            for j in range(i + 1, n_elec):
                d = np.linalg.norm(positions[i] - positions[j])
                if 0 < d < 2.5:
                    sig_i = electrode_signals[i]
                    sig_j = electrode_signals[j]

                    c = np.corrcoef(sig_i, sig_j)[0, 1]
                    if not np.isnan(c):
                        spatial_corr += c
                        count += 1
        avg_spatial_corr = spatial_corr / max(count, 1)

        power = np.var(electrode_signals, axis=1)
        power_variance = np.var(power)
        return np.array([avg_spatial_corr, power_variance], dtype=float)

    def extract_all_features(self, dataset):
        electrode_signals = dataset['electrode_signals']
        positions = dataset['electrode_positions']

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

    def __init__(self, n_motor_dims=2, regularization=0.1):
        self.n_motor_dims = n_motor_dims
        self.lambda_reg = regularization
        self.W = None
        self.feature_mean = None
        self.feature_std = None

    def _flatten_features(self, per_elec_features, spatial_features):
        return np.concatenate([
            per_elec_features.flatten(),
            spatial_features.flatten()
        ])

    def fit(self, feature_list, motor_target_list):
        X = []
        Y = []
        for feats, motor in zip(feature_list, motor_target_list):
            per_elec, spatial = feats
            x_vec = self._flatten_features(per_elec, spatial)
            X.append(x_vec)
            Y.append(motor)
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)

        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)
        self.feature_std[self.feature_std < 1e-10] = 1.0
        X_norm = (X - self.feature_mean) / self.feature_std

        n_features = X_norm.shape[1]
        XtX = X_norm.T @ X_norm + self.lambda_reg * np.eye(n_features)
        try:
            self.W = np.linalg.solve(XtX, X_norm.T @ Y)
        except np.linalg.LinAlgError:
            self.W = np.linalg.lstsq(XtX, X_norm.T @ Y, rcond=None)[0]
        return self

    def decode(self, per_elec_features, spatial_features):
        if self.W is None:
            raise RuntimeError("Decoder has not been fitted yet.")
        x_vec = self._flatten_features(per_elec_features, spatial_features)
        x_norm = (x_vec - self.feature_mean) / self.feature_std
        motor = x_norm @ self.W
        return motor


class BCIPipeline:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.generator = SyntheticBCISignalGenerator(random_state=random_state)
        self.feature_extractor = BCIFeatureExtractor()
        self.decoder = BCIDecoder(n_motor_dims=2, regularization=0.5)
        self.stability_analyzer = None

    def run_single_trial(self, motor_target=None):
        if motor_target is None:
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 2.0)
            motor_target = np.array([speed * np.cos(angle), speed * np.sin(angle)])


        dataset = self.generator.generate_full_dataset()


        ei_osc = EIOscillator(omega=2.0 * np.pi * 10.0)
        self.stability_analyzer = BCIStabilityAnalyzer(ei_osc, feedback_gain=0.3)
        stability = self.stability_analyzer.analyze_open_loop_stability()


        bridge = ConnectomePercolationBridge(self.generator.connectome, grid_shape=(16, 16))
        spread_info = bridge.analyze_information_spread(seed_region=0, steps=30)


        per_elec_feats, spatial_feats = self.feature_extractor.extract_all_features(dataset)


        return {
            'dataset': dataset,
            'motor_target': motor_target,
            'stability': stability,
            'information_spread': spread_info,
            'per_electrode_features': per_elec_feats,
            'spatial_features': spatial_feats
        }

    def run_training_and_decoding(self, n_train=20):

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

        trial = self.run_single_trial()
        dataset = trial['dataset']
        stability = trial['stability']
        spread = trial['information_spread']


        decode_result = self.run_training_and_decoding(n_train=15)


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
