
import numpy as np
from typing import List, Tuple, Dict
from utils import r8_uniform_01, erf_approx
from wimp_physics import differential_rate, annual_modulated_rate, total_events_in_range






class ReproducibleRNG:

    def __init__(self, seed: int = 123456789):
        self.seed = int(seed) % 2147483647
        if self.seed == 0:
            self.seed = 1

    def uniform(self) -> float:
        r, self.seed = r8_uniform_01(self.seed)
        return r

    def randn(self) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        while u1 <= 1.0e-15:
            u1 = self.uniform()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def choice(self, weights: np.ndarray) -> int:
        weights = np.asarray(weights, dtype=float)
        if np.sum(weights) <= 0.0:
            return 0
        cdf = np.cumsum(weights / np.sum(weights))
        u = self.uniform()
        return int(np.searchsorted(cdf, u))

    def exponential(self, scale: float) -> float:
        u = self.uniform()
        while u <= 1.0e-15:
            u = self.uniform()
        return -scale * np.log(u)






def detection_efficiency(
    er_kev: float,
    threshold_kev: float = 0.5,
    saturation_kev: float = 100.0,
    sigma_e_kev: float = 0.1,
) -> float:
    if er_kev <= 0.0:
        return 0.0
    arg_th = (er_kev - threshold_kev) / (np.sqrt(2.0) * sigma_e_kev)
    arg_sat = (saturation_kev - er_kev) / (np.sqrt(2.0) * sigma_e_kev)
    eps_th = 0.5 * (1.0 + erf_approx(arg_th))
    eps_sat = 0.5 * (1.0 + erf_approx(arg_sat))
    return eps_th * eps_sat


def apply_energy_resolution(
    er_true_kev: float,
    fano_factor: float,
    epsilon_eV: float,
    rng: ReproducibleRNG,
) -> float:
    if er_true_kev <= 0.0:
        return 0.0
    N_e = er_true_kev * 1000.0 / epsilon_eV
    sigma_e_kev = np.sqrt(fano_factor * N_e) * epsilon_eV / 1000.0
    if sigma_e_kev < 1.0e-6:
        return er_true_kev
    noise = rng.randn() * sigma_e_kev
    return max(er_true_kev + noise, 0.0)






def generate_wimp_events(
    n_events_target: int,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
    e_min_kev: float,
    e_max_kev: float,
    detector_radius_m: float,
    detector_thickness_m: float,
    rng: ReproducibleRNG,
    apply_modulation: bool = True,
) -> List[Dict]:
    events = []
    if n_events_target <= 0:
        return events


    n_scan = 200
    e_scan = np.linspace(e_min_kev, e_max_kev, n_scan)
    rates = np.array([
        differential_rate(e, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)
        for e in e_scan
    ])
    max_rate = float(np.max(rates)) * 1.2
    if max_rate <= 0.0:
        return events

    max_attempts = n_events_target * 10000
    attempts = 0

    while len(events) < n_events_target and attempts < max_attempts:
        attempts += 1

        e_trial = e_min_kev + (e_max_kev - e_min_kev) * rng.uniform()

        rate = differential_rate(e_trial, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)

        if rate > max_rate:
            max_rate = rate * 1.2
            continue
        u = rng.uniform() * max_rate
        if u > rate:
            continue


        if apply_modulation:

            t_trial = 365.25 * rng.uniform()
            mod_factor = 1.0 + 0.05 * np.cos(2.0 * np.pi * (t_trial - 152.0) / 365.25)
            if rng.uniform() > mod_factor / 1.06:
                continue
        else:
            t_trial = 365.25 * rng.uniform()


        r_pos = detector_radius_m * np.sqrt(rng.uniform())
        theta = 2.0 * np.pi * rng.uniform()
        x = r_pos * np.cos(theta)
        y = r_pos * np.sin(theta)
        z = detector_thickness_m * rng.uniform()


        eps = detection_efficiency(e_trial)
        if rng.uniform() > eps:
            continue


        e_obs = apply_energy_resolution(e_trial, fano_factor=0.15, epsilon_eV=3.0, rng=rng)
        if e_obs < e_min_kev:
            continue

        events.append({
            "type": "WIMP",
            "energy_true": float(e_trial),
            "energy_obs": float(e_obs),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "time_day": float(t_trial),
        })

    return events






def generate_background_events(
    n_events_target: int,
    e_min_kev: float,
    e_max_kev: float,
    detector_radius_m: float,
    detector_thickness_m: float,
    rng: ReproducibleRNG,
    gamma_rate_per_day: float = 5.0,
    neutron_rate_per_day: float = 0.5,
    beta_rate_per_day: float = 2.0,
) -> List[Dict]:
    events = []
    if n_events_target <= 0:
        return events


    rates = np.array([gamma_rate_per_day, neutron_rate_per_day, beta_rate_per_day])
    labels = ["gamma", "neutron", "beta"]

    max_attempts = n_events_target * 10000
    attempts = 0

    while len(events) < n_events_target and attempts < max_attempts:
        attempts += 1
        bg_type = labels[rng.choice(rates)]

        if bg_type == "gamma":

            E0 = 10.0
            e_trial = e_min_kev + (e_max_kev - e_min_kev) * rng.uniform()
            prob = np.exp(-e_trial / E0)
            if rng.uniform() > prob:
                continue
        elif bg_type == "neutron":

            if rng.uniform() < 0.8:
                e_trial = e_min_kev + (e_max_kev - e_min_kev) * rng.uniform()
            else:
                e_trial = 30.0 + 5.0 * rng.randn()
                if e_trial < e_min_kev or e_trial > e_max_kev:
                    continue
        else:
            e_trial = 10.0 + 2.0 * rng.randn()
            if e_trial < e_min_kev or e_trial > e_max_kev:
                continue


        eps = detection_efficiency(e_trial)
        if rng.uniform() > eps:
            continue


        e_obs = apply_energy_resolution(e_trial, fano_factor=0.15, epsilon_eV=3.0, rng=rng)
        if e_obs < e_min_kev:
            continue


        r_pos = detector_radius_m * np.sqrt(rng.uniform())
        theta = 2.0 * np.pi * rng.uniform()
        x = r_pos * np.cos(theta)
        y = r_pos * np.sin(theta)
        z = detector_thickness_m * rng.uniform()
        t_trial = 365.25 * rng.uniform()

        events.append({
            "type": bg_type,
            "energy_true": float(e_trial),
            "energy_obs": float(e_obs),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "time_day": float(t_trial),
        })

    return events






if __name__ == "__main__":
    rng = ReproducibleRNG(seed=42)


    assert detection_efficiency(0.1) < detection_efficiency(10.0)
    assert 0.0 <= detection_efficiency(50.0) <= 1.0


    e_obs = apply_energy_resolution(10.0, 0.15, 3.0, rng)
    assert e_obs >= 0.0


    events = generate_wimp_events(
        50, m_chi_gev=50.0, sigma_pb=1.0, a_mass=73.0,
        target_mass_kg=10.0, exposure_days=365.0,
        e_min_kev=0.5, e_max_kev=50.0,
        detector_radius_m=0.05, detector_thickness_m=0.02,
        rng=rng,
    )
    assert len(events) > 0, "WIMP 事件生成失败"
    for ev in events:
        assert ev["type"] == "WIMP"
        assert e_min_kev <= ev["energy_obs"] <= e_max_kev * 2.0


    bg = generate_background_events(
        50, 0.5, 50.0, 0.05, 0.02, rng,
    )
    assert len(bg) > 0

    print("monte_carlo_generator.py: 所有自测通过")
