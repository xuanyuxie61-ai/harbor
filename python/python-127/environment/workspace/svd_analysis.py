
import numpy as np


class PatientSVDAnalyzer:

    def __init__(self):
        self.U = None
        self.S = None
        self.Vt = None
        self.mean_vector = None
        self.n_patients = 0
        self.n_features = 0

    def fit(self, data_matrix):
        data_matrix = np.asarray(data_matrix, dtype=float)
        if data_matrix.ndim != 2:
            raise ValueError("data_matrix 必须为二维数组")

        self.n_features, self.n_patients = data_matrix.shape


        self.mean_vector = np.mean(data_matrix, axis=1)
        A_centered = data_matrix - self.mean_vector[:, np.newaxis]


        self.U, self.S, self.Vt = np.linalg.svd(A_centered, full_matrices=False)

    def get_principal_components(self, n_components):
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        n = min(n_components, self.U.shape[1])
        return self.U[:, :n]

    def get_singular_values(self):
        if self.S is None:
            raise RuntimeError("必须先调用 fit()")
        return self.S.copy()

    def explained_variance_ratio(self):
        if self.S is None:
            raise RuntimeError("必须先调用 fit()")
        total = np.sum(self.S**2)
        if total < 1e-15:
            return np.zeros_like(self.S)
        return (self.S**2) / total

    def cumulative_variance_ratio(self):
        return np.cumsum(self.explained_variance_ratio())

    def project(self, patient_vector, n_components):
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        patient_vector = np.asarray(patient_vector, dtype=float)
        centered = patient_vector - self.mean_vector
        pcs = self.get_principal_components(n_components)
        coeffs = pcs.T @ centered
        return coeffs

    def reconstruct(self, coeffs):
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        coeffs = np.asarray(coeffs, dtype=float)
        n_components = len(coeffs)
        pcs = self.get_principal_components(n_components)
        reconstructed = pcs @ coeffs + self.mean_vector
        return reconstructed

    def low_rank_approximation(self, rank):
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        k = min(rank, len(self.S))
        A_approx = (self.U[:, :k] * self.S[:k]) @ self.Vt[:k, :]
        A_approx += self.mean_vector[:, np.newaxis]
        return A_approx

    def compression_ratio(self, n_components):
        if self.U is None:
            raise RuntimeError("必须先调用 fit()")
        original = self.n_features * self.n_patients
        compressed = self.n_features * n_components + self.n_patients * n_components + self.n_features
        return original / compressed


def generate_synthetic_patient_data(n_patients=50, n_features=200,
                                     n_modes=5, noise_level=0.05):
    np.random.seed(42)

    true_modes = np.random.randn(n_features, n_modes)
    q, _ = np.linalg.qr(true_modes)
    true_modes = q[:, :n_modes]


    coeffs = np.random.randn(n_modes, n_patients)


    data = true_modes @ coeffs
    data += noise_level * np.random.randn(n_features, n_patients)

    return data, true_modes


def electrode_config_optimization_svd(electrode_responses, n_components=3):
    responses = np.asarray(electrode_responses, dtype=float)
    analyzer = PatientSVDAnalyzer()
    analyzer.fit(responses.T)

    pcs = analyzer.get_principal_components(n_components)
    sv = analyzer.get_singular_values()[:n_components]


    optimal_direction = pcs[:, 0]


    importance_scores = np.abs(optimal_direction) * sv[0]

    return optimal_direction, importance_scores
