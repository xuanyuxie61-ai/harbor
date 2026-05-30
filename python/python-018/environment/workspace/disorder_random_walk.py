
import numpy as np
from typing import Tuple, List, Optional


class MajoranaDisorderRandomWalk:

    def __init__(self, n_sites: int, disorder_strength: float,
                 delta: float, t: float = 1.0, mu0: float = 0.0,
                 rng_seed: Optional[int] = None):
        self.n = n_sites
        self.W = disorder_strength
        self.delta = delta
        self.t = t
        self.mu0 = mu0

        self.rng = np.random.RandomState(rng_seed)
        self._generate_disorder_potential()

    def _generate_disorder_potential(self) -> None:
        self.disorder = self.W * (2.0 * self.rng.rand(self.n) - 1.0)
        self.mu_profile = self.mu0 + self.disorder

    def _andreev_reflection_probability(self, energy: float) -> float:
        if abs(self.delta) < 1e-15:
            return 0.0
        return (self.delta ** 2) / (energy ** 2 + self.delta ** 2 + 1e-15)

    def _localization_length_estimate(self, site: int) -> float:
        xi0 = 2.0 * abs(self.t) / (abs(self.delta) + 1e-15)
        reduction = 1.0 + (self.W / (abs(self.delta) + 1e-15)) ** 2
        return xi0 / reduction

    def simulate_andreev_random_walk(self, num_steps: int,
                                     num_walks: int,
                                     energy: float = 0.0) -> Tuple[np.ndarray,
                                                                    np.ndarray]:
        if num_steps < 1 or num_walks < 1:
            raise ValueError("步数和walker数必须为正")

        p_andreev = self._andreev_reflection_probability(energy)
        p_normal = 1.0 - p_andreev

        d2_ave = np.zeros(num_steps + 1)
        d2_max = np.zeros(num_steps + 1)

        for walk in range(num_walks):
            x = 0.0
            y = 0.0

            for step in range(1, num_steps + 1):
                r = self.rng.rand()
                if r < p_andreev:

                    y = -y + np.pi

                    xi = self._localization_length_estimate(int(abs(x)) % self.n)
                    x += self.rng.normal(0.0, xi * 0.1)
                else:

                    dx = 1.0 if self.rng.rand() < 0.5 else -1.0
                    x += dx
                    y += self.rng.normal(0.0, 0.1)

                d2 = x * x + y * y
                d2_ave[step] += d2
                d2_max[step] = max(d2_max[step], d2)

        d2_ave /= num_walks

        return d2_ave, d2_max

    def compute_participation_ratio(self, wavefunction: np.ndarray) -> float:
        if wavefunction is None or len(wavefunction) == 0:
            return 0.0

        wf = np.asarray(wavefunction, dtype=np.float64)
        norm_sq = np.sum(wf * wf)
        if norm_sq < 1e-15:
            return 0.0

        ipr = np.sum(wf ** 4) / (norm_sq ** 2)
        return float(ipr)

    def disorder_averaged_correlation(self, num_realizations: int,
                                       max_distance: int) -> np.ndarray:
        if max_distance < 0 or max_distance >= self.n:
            max_distance = self.n // 2

        corr = np.zeros(max_distance + 1)
        counts = np.zeros(max_distance + 1)

        for _ in range(num_realizations):
            self._generate_disorder_potential()


            for r in range(max_distance + 1):
                for i in range(self.n - r):
                    j = i + r
                    val = self.mu_profile[i] * self.mu_profile[j]
                    corr[r] += val
                    counts[r] += 1


        mask = counts > 0
        corr[mask] /= counts[mask]

        if abs(corr[0]) > 1e-15:
            corr /= corr[0]

        return corr

    def localization_length_scaling(self, w_vals: np.ndarray,
                                     num_realizations: int = 50) -> np.ndarray:
        xi_eff = np.zeros_like(w_vals)
        original_W = self.W

        for idx, w in enumerate(w_vals):
            if w < 1e-10:
                xi_eff[idx] = float(self.n)
                continue

            self.W = w
            iprs = []
            for _ in range(num_realizations):
                self._generate_disorder_potential()

                var = np.var(self.mu_profile)
                if var > 1e-15:
                    iprs.append(1.0 / var)

            if iprs:
                xi_eff[idx] = np.mean(iprs)

        self.W = original_W
        return xi_eff


def demo():
    walker = MajoranaDisorderRandomWalk(
        n_sites=50, disorder_strength=0.5, delta=0.8, t=1.0, mu0=0.0,
        rng_seed=42
    )
    d2_ave, d2_max = walker.simulate_andreev_random_walk(
        num_steps=100, num_walks=500, energy=0.0
    )
    print("Andreev Random Walk <r^2>(t=100):", d2_ave[-1])
    print("Max r^2(t=100):", d2_max[-1])

    corr = walker.disorder_averaged_correlation(
        num_realizations=20, max_distance=20
    )
    print("Disorder correlation C(r=0):", corr[0])
    print("Disorder correlation C(r=5):", corr[5])


if __name__ == "__main__":
    from typing import Optional
    demo()
