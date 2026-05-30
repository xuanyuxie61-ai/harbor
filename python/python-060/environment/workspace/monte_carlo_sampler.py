
import numpy as np
from typing import Tuple, Optional, Dict


class RandomDataGenerator:

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def normal_square(self, n: int, d: int) -> np.ndarray:
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")
        return self.rng.standard_normal((n, d))

    def uniform_in_hypercube(self, n: int, d: int,
                              a: float = 0.0, b: float = 1.0) -> np.ndarray:
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")
        return a + (b - a) * self.rng.random((n, d))

    def uniform_in_hypersphere(self, n: int, d: int,
                                radius: float = 1.0) -> np.ndarray:
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")
        x = self.rng.standard_normal((n, d))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        u = self.rng.random((n, 1)) ** (1.0 / d)
        return radius * u * x / (norms + 1e-30)

    def direction_uniform_nd(self, n: int, d: int) -> np.ndarray:
        x = self.rng.standard_normal((n, d))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-30)

    def normal_circular(self, n: int) -> np.ndarray:
        r = np.sqrt(-2.0 * np.log(self.rng.random(n)))
        theta = 2.0 * np.pi * self.rng.random(n)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    def latin_hypercube(self, n: int, d: int) -> np.ndarray:
        if n <= 0 or d <= 0:
            raise ValueError("n > 0 且 d > 0")

        result = np.zeros((n, d))
        for i in range(d):
            perm = self.rng.permutation(n)
            result[:, i] = (perm + self.rng.random(n)) / n

        return result


class EmissionProfileSampler:

    def __init__(self):
        self.rng = np.random.default_rng(123)

    def gaussian_emission_profile(self, z_km: np.ndarray,
                                   peak_height: float = 0.0,
                                   width: float = 5.0,
                                   total_emission: float = 1.0e6) -> np.ndarray:
        profile = np.exp(-((z_km - peak_height) / width) ** 2)
        profile = profile / (np.sum(profile) + 1e-30)
        return total_emission * profile

    def multi_source_profile(self, z_km: np.ndarray,
                              sources: list) -> np.ndarray:
        profile = np.zeros_like(z_km)
        for src in sources:
            p = self.gaussian_emission_profile(
                z_km,
                peak_height=src.get('height', 0.0),
                width=src.get('width', 5.0),
                total_emission=src.get('total', 1e6)
            )
            profile += p
        return profile

    def sample_profile_parameters(self, n_samples: int) -> Dict:
        samples = {
            'n2o_peak_height_km': self.rng.normal(0.0, 2.0, n_samples),
            'n2o_width_km': self.rng.uniform(3.0, 8.0, n_samples),
            'cfc11_peak_height_km': self.rng.normal(0.0, 1.5, n_samples),
            'cfc11_width_km': self.rng.uniform(2.0, 5.0, n_samples),
            'nox_aircraft_peak_km': self.rng.normal(11.0, 1.0, n_samples),
            'nox_aircraft_width_km': self.rng.uniform(1.5, 3.0, n_samples),
        }
        return samples

    def perturbed_profile(self, base_profile: np.ndarray,
                          relative_std: float = 0.1) -> np.ndarray:
        noise = self.rng.lognormal(0.0, relative_std, len(base_profile))
        perturbed = base_profile * noise
        return np.clip(perturbed, 0.0, 1e20)


class CorrelatedParameterSampler:

    def __init__(self, n_params: int = 10):
        self.n_params = n_params
        self.rng = np.random.default_rng(456)

    def build_covariance_matrix(self, sigmas: np.ndarray,
                                 correlations: Optional[np.ndarray] = None) -> np.ndarray:
        if len(sigmas) != self.n_params:
            raise ValueError("sigmas 长度不匹配")

        Sigma = np.diag(sigmas ** 2)

        if correlations is not None:
            if correlations.shape != (self.n_params, self.n_params):
                raise ValueError("correlations 形状不匹配")
            for i in range(self.n_params):
                for j in range(i + 1, self.n_params):
                    Sigma[i, j] = correlations[i, j] * sigmas[i] * sigmas[j]
                    Sigma[j, i] = Sigma[i, j]


        eigvals = np.linalg.eigvalsh(Sigma)
        if np.min(eigvals) < 1e-14:
            Sigma += (1e-12 - np.min(eigvals)) * np.eye(self.n_params)

        return Sigma

    def sample(self, mu: np.ndarray, Sigma: np.ndarray,
               n_samples: int = 100) -> np.ndarray:
        if len(mu) != self.n_params:
            raise ValueError("mu 长度不匹配")

        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:

            Sigma_reg = Sigma + 1e-10 * np.eye(self.n_params)
            L = np.linalg.cholesky(Sigma_reg)

        Z = self.rng.standard_normal((n_samples, self.n_params))
        X = mu + Z @ L.T
        return X

    def lognormal_sample(self, mu_log: np.ndarray, Sigma: np.ndarray,
                         n_samples: int = 100) -> np.ndarray:
        X_normal = self.sample(mu_log, Sigma, n_samples)
        return np.exp(X_normal)


class OzoneMonteCarloExperiment:

    def __init__(self, n_ensemble: int = 500):
        self.n_ensemble = n_ensemble
        self.random_gen = RandomDataGenerator(seed=789)
        self.emission_sampler = EmissionProfileSampler()
        self.param_sampler = CorrelatedParameterSampler(n_params=8)

    def run_parameter_perturbation_experiment(self) -> Dict:

        param_names = [
            'k_O_O2_M', 'k_O_O3', 'k_NO_O3', 'k_Cl_O3',
            'k_OH_O3', 'J_O2', 'J_O3', 'Kzz_scale'
        ]
        mu = np.array([-33.5, -11.7, -11.7, -10.5,
                       -11.8, -10.0, -2.0, 0.0])
        sigma_rel = np.array([0.15, 0.10, 0.12, 0.20,
                              0.15, 0.10, 0.08, 0.25])


        corr = np.eye(8)
        corr[0, 1] = corr[1, 0] = 0.3
        corr[2, 3] = corr[3, 2] = 0.2
        corr[5, 6] = corr[6, 5] = 0.4

        Sigma = self.param_sampler.build_covariance_matrix(sigma_rel, corr)
        samples = self.param_sampler.sample(mu, Sigma, self.n_ensemble)


        o3_columns = []
        for i in range(self.n_ensemble):
            params = 10.0 ** samples[i]

            o3 = 300.0 * (params[5] / 1e-10) ** 0.3 * \
                 (params[6] / 1e-2) ** (-0.2) * \
                 (params[7] / 1.0) ** (-0.15)
            o3_columns.append(o3)

        o3_columns = np.array(o3_columns)

        return {
            'param_names': param_names,
            'samples': samples,
            'o3_columns': o3_columns,
            'o3_mean': np.mean(o3_columns),
            'o3_std': np.std(o3_columns),
            'o3_ci_95': (np.percentile(o3_columns, 2.5),
                        np.percentile(o3_columns, 97.5)),
            'o3_min': np.min(o3_columns),
            'o3_max': np.max(o3_columns)
        }

    def run_emission_uncertainty_experiment(self, z_km: np.ndarray) -> Dict:
        n2o_profiles = []
        for _ in range(self.n_ensemble):
            height = self.random_gen.rng.normal(0.0, 2.0)
            width = self.random_gen.rng.uniform(3.0, 8.0)
            profile = self.emission_sampler.gaussian_emission_profile(
                z_km, peak_height=height, width=width)
            n2o_profiles.append(profile)

        n2o_profiles = np.array(n2o_profiles)

        return {
            'n2o_mean_profile': np.mean(n2o_profiles, axis=0),
            'n2o_std_profile': np.std(n2o_profiles, axis=0),
            'n2o_q05': np.percentile(n2o_profiles, 5, axis=0),
            'n2o_q95': np.percentile(n2o_profiles, 95, axis=0),
        }
