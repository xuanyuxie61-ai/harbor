
import numpy as np


class MonteCarloIntegrator:

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(int(seed))

    def integrate(self, f_func, a, b, n=10000):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = a.size
        if b.size != d:
            raise ValueError("a and b must have the same size.")
        if np.any(b <= a):
            raise ValueError("Upper bounds must exceed lower bounds.")
        if n <= 0:
            raise ValueError("Number of samples must be positive.")

        volume = np.prod(b - a)
        samples = np.random.rand(n, d)
        samples = a + samples * (b - a)

        values = np.array([f_func(samples[i]) for i in range(n)])
        estimate = volume * np.mean(values)
        std_err = volume * np.std(values, ddof=1) / np.sqrt(n)
        return estimate, std_err

    def integrate_batch(self, f_func, a, b, n=10000):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = a.size
        if b.size != d:
            raise ValueError("a and b must have the same size.")
        volume = np.prod(b - a)
        samples = np.random.rand(n, d)
        samples = a + samples * (b - a)
        values = f_func(samples)
        estimate = volume * np.mean(values)
        std_err = volume * np.std(values, ddof=1) / np.sqrt(n)
        return estimate, std_err


class MonteCarloUQ:

    def __init__(self, params, n_samples=500):
        from thm_model import THMParameters
        if not isinstance(params, THMParameters):
            raise TypeError("params must be THMParameters.")
        self.params = params
        self.n_samples = int(n_samples)

    def sample_parameters(self):
        samples = []
        for _ in range(self.n_samples):
            s = {
                "porosity": np.clip(np.random.normal(self.params.porosity, 0.02), 0.01, 0.5),
                "permeability": np.clip(np.random.lognormal(np.log(self.params.matrix_permeability), 0.5),
                                         1.0e-16, 1.0e-10),
                "thermal_conductivity_rock": np.clip(
                    np.random.normal(self.params.thermal_conductivity_rock, 0.3), 0.5, 5.0),
                "young_modulus": np.clip(
                    np.random.normal(self.params.young_modulus, 5.0e9), 5.0e9, 80.0e9),
                "biot_coefficient": np.clip(
                    np.random.normal(self.params.biot_coefficient, 0.05), 0.0, 1.0),
            }
            samples.append(s)
        return samples

    def estimate_heat_extraction(self, temperature_field_func):
        samples = self.sample_parameters()
        extractions = []
        for s in samples:
            T_field = temperature_field_func(s)

            extraction = np.mean(self.params.T_initial - T_field)
            extractions.append(extraction)

        extractions = np.array(extractions)
        mean_ext = np.mean(extractions)
        std_ext = np.std(extractions, ddof=1)
        se = std_ext / np.sqrt(self.n_samples)
        ci_lower = mean_ext - 1.96 * se
        ci_upper = mean_ext + 1.96 * se
        return mean_ext, std_ext, ci_lower, ci_upper

    def estimate_effective_permeability(self, fracture_aperture_samples):
        from risk_fracture import effective_permeability_from_fracture_network
        k_effs = []
        for a_samples in fracture_aperture_samples:
            k_eff = effective_permeability_from_fracture_network(
                a_samples, fracture_density=2.0,
                matrix_perm=self.params.matrix_permeability
            )
            k_effs.append(k_eff)
        k_effs = np.array(k_effs)
        return np.mean(k_effs), np.std(k_effs, ddof=1)


def mc_integral_thermal_energy(n_samples=5000, T_mean=400.0, T_std=20.0,
                                rho_eff=2500.0, cp_eff=1000.0, volume=1.0e7):
    T_samples = np.random.normal(T_mean, T_std, n_samples)
    T_samples = np.clip(T_samples, 273.15, 1273.15)
    energies = rho_eff * cp_eff * volume * T_samples
    mean_E = np.mean(energies)
    std_E = np.std(energies, ddof=1) / np.sqrt(n_samples)
    return mean_E, std_E
