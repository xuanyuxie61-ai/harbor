
import numpy as np


class CVTKeyframeSampler:

    def __init__(self, num_generators=20, max_iter=50, domain_bounds=None):
        self.num_generators = max(int(num_generators), 2)
        self.max_iter = max(int(max_iter), 1)
        self.domain_bounds = domain_bounds
        self.generators = None
        self.energies = []

    def fit(self, feature_points, weights=None):
        points = np.asarray(feature_points, dtype=np.float64)
        N, d = points.shape
        if N == 0:
            self.generators = np.zeros((self.num_generators, d))
            return self.generators, np.array([])

        if weights is None:
            weights = np.ones(N, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            weights = np.maximum(weights, 1e-12)


        if self.domain_bounds is None:
            mins = np.min(points, axis=0)
            maxs = np.max(points, axis=0)
            padding = (maxs - mins) * 0.1
            mins -= padding
            maxs += padding

            for dim_idx in range(d):
                if maxs[dim_idx] - mins[dim_idx] < 1e-8:
                    maxs[dim_idx] = mins[dim_idx] + 1.0
        else:
            mins = np.full(d, self.domain_bounds[0])
            maxs = np.full(d, self.domain_bounds[1])


        generators = self._kmeans_plus_plus(points, self.num_generators)

        self.energies = []
        for it in range(self.max_iter):

            labels = self._assign_voronoi(points, generators)


            new_generators = np.zeros_like(generators)
            for i in range(self.num_generators):
                mask = (labels == i)
                if np.any(mask):
                    w = weights[mask]
                    p = points[mask]
                    new_generators[i] = np.sum(p * w[:, None], axis=0) / np.sum(w)
                else:

                    new_generators[i] = points[np.random.choice(N)]


            new_generators = np.clip(new_generators, mins, maxs)


            energy = self._compute_energy(points, generators, labels, weights)
            self.energies.append(energy)


            motion = np.mean(np.sum((new_generators - generators) ** 2, axis=1))
            generators = new_generators
            if motion < 1e-12:
                break

        self.generators = generators
        labels = self._assign_voronoi(points, generators)
        return generators, labels

    @staticmethod
    def _kmeans_plus_plus(points, k):
        N, d = points.shape
        centers = np.zeros((k, d), dtype=np.float64)
        centers[0] = points[np.random.randint(N)]
        for i in range(1, k):
            dists = np.min(np.sum((points[:, None, :] - centers[None, :i, :]) ** 2, axis=2), axis=1)
            probs = dists / (np.sum(dists) + 1e-12)
            idx = np.random.choice(N, p=probs)
            centers[i] = points[idx]
        return centers

    @staticmethod
    def _assign_voronoi(points, generators):
        dists = np.sum((points[:, None, :] - generators[None, :, :]) ** 2, axis=2)
        return np.argmin(dists, axis=1)

    @staticmethod
    def _compute_energy(points, generators, labels, weights):
        energy = 0.0
        for i in range(generators.shape[0]):
            mask = (labels == i)
            if np.any(mask):
                diff = points[mask] - generators[i]
                energy += np.sum(weights[mask] * np.sum(diff ** 2, axis=1))
        return energy


class OptimalStoppingKeyframeSelector:

    def __init__(self, total_frames=None):
        self.total_frames = total_frames

    @staticmethod
    def optimal_skip_ratio(total_frames):
        if total_frames is None or total_frames <= 0:
            return 1.0 / np.e
        return min(int(total_frames / np.e) / max(total_frames, 1), 0.999)

    def select_keyframes(self, information_gains, total_frames=None):
        gains = np.asarray(information_gains, dtype=np.float64)
        T = len(gains)
        if T == 0:
            return []

        if total_frames is None:
            total_frames = T

        skip_num = max(1, int(total_frames / np.e))
        skip_num = min(skip_num, T - 1)

        selected = []
        if skip_num >= T:
            return [np.argmax(gains)]


        reference_max = np.max(gains[:skip_num]) if skip_num > 0 else -np.inf

        for i in range(skip_num, T):
            if gains[i] > reference_max:
                selected.append(i)

                reference_max = gains[i]


        if not selected:
            selected.append(int(np.argmax(gains[skip_num:])) + skip_num)

        return selected

    def simulate_strategy(self, deck_size=100, trial_num=500):
        correct = 0
        for _ in range(trial_num):
            cards = np.random.permutation(deck_size) + 1
            skip = max(1, int(deck_size / np.e))
            skip_max = np.max(cards[:skip]) if skip > 0 else -np.inf

            choice = cards[-1]
            for i in range(skip, deck_size):
                if cards[i] > skip_max:
                    choice = cards[i]
                    break

            if choice == deck_size:
                correct += 1

        success_rate = correct / trial_num
        theoretical = 1.0 / np.e
        return success_rate, theoretical


class InformationGainEstimator:

    @staticmethod
    def compute_fisher_information(pose, landmarks, sigma_obs=0.1):
        sigma_obs = max(float(sigma_obs), 1e-12)
        x, y, theta = pose

        info_mat = np.zeros((3, 3), dtype=np.float64)
        for lm in landmarks:
            mx, my = lm
            dx = mx - x
            dy = my - y
            r2 = dx * dx + dy * dy
            if r2 < 1e-12:
                continue
            r = np.sqrt(r2)



            H = np.array([[-dx / r, -dy / r, 0],
                          [dy / r2, -dx / r2, -1]], dtype=np.float64)
            Sigma_inv = np.eye(2) / (sigma_obs ** 2)
            info_mat += H.T @ Sigma_inv @ H


        det_info = np.linalg.det(info_mat)
        trace_info = np.trace(info_mat)

        gain = trace_info + np.log(max(det_info, 1e-12) + 1.0)
        return gain
