
import numpy as np


class BeamStatisticsAnalyzer:

    def __init__(self, dim: int = 3):
        if dim < 2:
            raise ValueError("维度必须 >= 2")
        self.dim = dim

    @staticmethod
    def sample_positive_hypersphere(dim: int) -> np.ndarray:
        x = np.abs(np.random.randn(dim))
        norm = np.linalg.norm(x)
        if norm < 1e-15:

            x = np.zeros(dim)
            x[0] = 1.0
            return x
        return x / norm

    def compute_distance_stats(self, n_samples: int = 5000) -> tuple:
        if n_samples < 2:
            raise ValueError("n_samples 必须 >= 2")

        distances = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            p = self.sample_positive_hypersphere(self.dim)
            q = self.sample_positive_hypersphere(self.dim)
            distances[i] = np.linalg.norm(p - q)

        mu = float(np.mean(distances))
        if n_samples > 1:
            var = float(np.sum((distances - mu) ** 2) / (n_samples - 1))
        else:
            var = 0.0
        return mu, var

    def analyze_beam_coverage(self, beam_directions: np.ndarray) -> dict:
        beam_directions = np.asarray(beam_directions, dtype=np.float64)
        if beam_directions.ndim != 2 or beam_directions.shape[1] != self.dim:
            raise ValueError(f"beam_directions 形状应为 (n_beams, {self.dim})")

        n_beams = beam_directions.shape[0]
        if n_beams < 2:
            return {
                'n_beams': n_beams,
                'min_angle_deg': 0.0,
                'mean_angle_deg': 0.0,
                'coverage_uniformity': 0.0,
            }


        norms = np.linalg.norm(beam_directions, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        beam_directions = beam_directions / norms


        distances = []
        angles = []
        for i in range(n_beams):
            for j in range(i + 1, n_beams):
                d = np.linalg.norm(beam_directions[i] - beam_directions[j])
                distances.append(d)

                sin_half = np.clip(d / 2.0, 0.0, 1.0)
                theta = 2.0 * np.arcsin(sin_half)
                angles.append(theta)

        distances = np.array(distances)
        angles = np.array(angles)


        mean_dist = np.mean(distances)
        std_dist = np.std(distances, ddof=1)
        uniformity = mean_dist / (std_dist + 1e-12)

        return {
            'n_beams': n_beams,
            'min_angle_deg': float(np.degrees(np.min(angles))),
            'max_angle_deg': float(np.degrees(np.max(angles))),
            'mean_angle_deg': float(np.degrees(np.mean(angles))),
            'std_angle_deg': float(np.degrees(np.std(angles, ddof=1))),
            'mean_chord_distance': float(mean_dist),
            'coverage_uniformity': float(uniformity),
        }

    def generate_optimal_fan_beams(
        self,
        n_beams: int,
        max_opening_angle_deg: float = 60.0,
        azimuth_deg: float = 0.0
    ) -> np.ndarray:
        if n_beams < 1:
            raise ValueError("波束数必须 >= 1")

        max_theta = np.radians(max_opening_angle_deg)
        azimuth = np.radians(azimuth_deg)


        thetas = np.linspace(0.0, max_theta, n_beams)

        directions = np.zeros((n_beams, 3), dtype=np.float64)
        for i, theta in enumerate(thetas):


            directions[i, 0] = np.sin(theta) * np.cos(azimuth)
            directions[i, 1] = np.sin(theta) * np.sin(azimuth)
            directions[i, 2] = np.cos(theta)

        return directions
