
import numpy as np


class PlateletMonteCarlo:

    def __init__(self, n_platelets=1000, local_iia=20.0, lambda_penalty=0.01, seed=42):
        if n_platelets < 2:
            raise ValueError("n_platelets 必须 >= 2")
        self.n = n_platelets
        self.local_iia = max(local_iia, 0.0)
        self.lambda_pen = lambda_penalty
        self.rng = np.random.default_rng(seed)

    def _generate_potentials(self):
        alpha = 1.0 + 0.1 * self.local_iia
        beta = 2.0
        alpha = max(alpha, 0.1)
        beta = max(beta, 0.1)
        x = self.rng.beta(alpha, beta, size=self.n)
        return x

    def optimal_stopping_strategy(self, potentials, skip_num=None):
        if skip_num is None:
            skip_num = int(round(self.n / np.e))
        skip_num = max(0, min(skip_num, self.n - 1))

        if skip_num == 0:
            return 0, potentials[0]

        threshold = np.max(potentials[:skip_num])
        selected_idx = -1
        selected_val = potentials[-1]

        for i in range(skip_num, self.n):
            if potentials[i] > threshold:
                selected_idx = i
                selected_val = potentials[i]
                break

        return selected_idx, selected_val

    def simulate_clot_formation(self, n_trials=500, skip_num=None):
        scores = np.zeros(n_trials)
        for t in range(n_trials):
            pots = self._generate_potentials()
            idx, val = self.optimal_stopping_strategy(pots, skip_num)



            if idx >= 0:
                n_adhered = idx + 1
                avg_pot = np.mean(pots[:idx + 1])
            else:
                n_adhered = self.n
                avg_pot = np.mean(pots)

            score = avg_pot * np.sqrt(n_adhered) - self.lambda_pen * (n_adhered - self.n / 3.0) ** 2
            scores[t] = score

        return np.mean(scores), scores

    def find_optimal_skip(self, n_trials=300):
        best_skip = 0
        best_score = -np.inf
        results = []
        for skip in range(0, self.n, max(1, self.n // 50)):
            mean_score, _ = self.simulate_clot_formation(n_trials=n_trials, skip_num=skip)
            results.append((skip, mean_score))
            if mean_score > best_score:
                best_score = mean_score
                best_skip = skip
        return best_skip, best_score, results


def demo_optimal_stopping():
    print("=" * 60)
    print("血小板粘附最优停止策略蒙特卡洛模拟")
    print("=" * 60)

    for iia in [5.0, 20.0, 50.0]:
        mc = PlateletMonteCarlo(n_platelets=500, local_iia=iia, seed=42)
        best_skip, best_score, results = mc.find_optimal_skip(n_trials=200)
        theoretical = int(round(500 / np.e))
        print(f"\n局部 IIa = {iia} nM:")
        print(f"  蒙特卡洛最优 skip = {best_skip}, 平均评分 = {best_score:.4f}")
        print(f"  理论最优 skip (N/e) = {theoretical}")

    return mc


if __name__ == "__main__":
    demo_optimal_stopping()
