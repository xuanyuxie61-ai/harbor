
import numpy as np


class RandomWalkLoopDetector:

    def __init__(self, sigma_odometry=0.1, window_size=10, threshold_ratio=2.0):
        self.sigma_odometry = max(float(sigma_odometry), 1e-12)
        self.window_size = max(int(window_size), 1)
        self.threshold_ratio = max(float(threshold_ratio), 0.1)

    def compute_loop_likelihood(self, trajectory):
        T = len(trajectory)
        if T < 2:
            return np.zeros((T, T)), []

        positions = np.array([p[0:2] for p in trajectory], dtype=np.float64)
        likelihood = np.zeros((T, T), dtype=np.float64)

        for i in range(T):
            for j in range(i + self.window_size, T):
                dt = j - i
                actual_dist = np.linalg.norm(positions[i] - positions[j])
                expected_dist = self.sigma_odometry * np.sqrt(dt)

                if actual_dist < 1e-12:
                    score = 10.0
                else:
                    score = expected_dist / actual_dist

                likelihood[i, j] = score
                likelihood[j, i] = score


        candidates = []
        for i in range(T):
            for j in range(i + self.window_size, T):
                if likelihood[i, j] > self.threshold_ratio:
                    candidates.append((i, j, likelihood[i, j]))


        candidates.sort(key=lambda x: x[2], reverse=True)
        return likelihood, candidates


class RandomSearchAssociator:

    def __init__(self, max_candidates=50, max_trials=20, match_threshold=0.8):
        self.max_candidates = max(int(max_candidates), 1)
        self.max_trials = max(int(max_trials), 1)
        self.match_threshold = max(float(match_threshold), 0.0)

    def search_matches(self, current_scan, candidate_scans, similarity_func):
        n = len(candidate_scans)
        if n == 0:
            return None, 0.0


        num_samples = min(self.max_trials, n)
        indices = np.random.choice(n, size=num_samples, replace=False)

        best_idx = None
        best_score = -np.inf
        for idx in indices:
            score = similarity_func(current_scan, candidate_scans[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < self.match_threshold:
            return None, best_score
        return best_idx, best_score

    def simulate_find_probability(self, total_candidates, true_matches, num_trials, simulation_count=500):
        found_count = 0
        for _ in range(simulation_count):

            match_positions = set(np.random.choice(total_candidates, size=true_matches, replace=False))

            sampled = set(np.random.choice(total_candidates, size=min(num_trials, total_candidates), replace=False))
            if sampled & match_positions:
                found_count += 1

        empirical_prob = found_count / simulation_count

        from math import comb
        try:
            theoretical = 1.0 - comb(total_candidates - true_matches, num_trials) / comb(total_candidates, num_trials)
        except (ValueError, ZeroDivisionError):
            theoretical = empirical_prob

        return empirical_prob, theoretical


class PermutationOrthogonalityChecker:

    @staticmethod
    def check_rotation_orthogonality(points_a, points_b, correspondence):
        if not correspondence:
            return False, np.eye(2), np.zeros(2), np.inf

        pts_a = np.array([points_a[i] for i, _ in correspondence], dtype=np.float64)
        pts_b = np.array([points_b[j] for _, j in correspondence], dtype=np.float64)

        if pts_a.shape[0] < 2:
            return False, np.eye(2), np.zeros(2), np.inf

        mu_a = np.mean(pts_a, axis=0)
        mu_b = np.mean(pts_b, axis=0)
        a_centered = pts_a - mu_a
        b_centered = pts_b - mu_b

        H = a_centered.T @ b_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T


        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = mu_b - R @ mu_a


        ortho_err = np.linalg.norm(R.T @ R - np.eye(2), ord='fro')


        reproj = b_centered - a_centered @ R.T
        reproj_err = np.mean(np.sum(reproj ** 2, axis=1))

        is_valid = (ortho_err < 0.1) and (reproj_err < 1.0)

        return is_valid, R, t, ortho_err

    @staticmethod
    def demonstrate_permutation_property(n=10):
        p1 = np.random.permutation(n) + 1
        p2 = np.random.permutation(n) + 1

        xy = np.vstack([p1, p2]).astype(np.float64)
        xy = xy - n / 2.0

        angle = np.pi / 4.0
        A = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]], dtype=np.float64)
        xy_rot = A @ xy

        dot_product = np.dot(xy_rot[0, :], xy_rot[1, :])
        return dot_product


class IntegratedLoopClosureDetector:

    def __init__(self, rw_sigma=0.1, rw_window=10, rw_threshold=2.0,
                 rs_max_candidates=50, rs_trials=20, rs_threshold=0.8):
        self.rw_detector = RandomWalkLoopDetector(rw_sigma, rw_window, rw_threshold)
        self.rs_associator = RandomSearchAssociator(rs_max_candidates, rs_trials, rs_threshold)
        self.ortho_checker = PermutationOrthogonalityChecker()

    def detect(self, trajectory, scans, scan_similarity_func):

        _, candidates = self.rw_detector.compute_loop_likelihood(trajectory)

        closures = []
        checked_pairs = set()

        for i, j, rw_score in candidates[:20]:
            pair = tuple(sorted((i, j)))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)



            candidate_scans = []
            candidate_indices = []
            for idx in range(len(scans)):
                if abs(idx - i) > self.rw_detector.window_size:
                    candidate_scans.append(scans[idx])
                    candidate_indices.append(idx)

            match_rel_idx, match_score = self.rs_associator.search_matches(
                scans[i], candidate_scans, scan_similarity_func
            )

            if match_rel_idx is None or candidate_indices[match_rel_idx] != j:
                continue



            pts_i = scans[i]
            pts_j = scans[j]
            if pts_i.shape[0] == 0 or pts_j.shape[0] == 0:
                continue

            corr = []
            for pi, p in enumerate(pts_i[:min(20, len(pts_i))]):
                dists = np.sum((pts_j - p) ** 2, axis=1)
                corr.append((pi, np.argmin(dists)))

            is_valid, R, t, ortho_err = self.ortho_checker.check_rotation_orthogonality(
                pts_i, pts_j, corr
            )

            if is_valid:

                theta = np.arctan2(R[1, 0], R[0, 0])
                transform = np.array([t[0], t[1], theta], dtype=np.float64)
                closures.append({
                    'from': i,
                    'to': j,
                    'transform': transform,
                    'score': rw_score * match_score
                })

        return closures
