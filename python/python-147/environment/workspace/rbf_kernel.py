
import numpy as np


def rbf_phi1(r, r0):
    return np.sqrt(r ** 2 + r0 ** 2)


def rbf_phi2(r, r0):
    return 1.0 / np.sqrt(r ** 2 + r0 ** 2 + 1e-12)


def rbf_phi3(r, r0):
    v = np.zeros_like(r)
    mask = r > 1e-12
    v[mask] = r[mask] ** 2 * np.log(r[mask] / r0 + 1e-12)
    return v


def rbf_phi4(r, r0):
    return np.exp(-0.5 * r ** 2 / (r0 ** 2 + 1e-12))


def compute_pairwise_distance(X1, X2):
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must be 2D arrays")
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of columns")


    sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    cross = X1 @ X2.T
    D2 = sq1 + sq2 - 2.0 * cross

    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)


def rbf_interpolation_weights(X_data, f_data, r0, phi_type='gaussian'):
    phi_map = {
        'multiquadric': rbf_phi1,
        'inverse_mq': rbf_phi2,
        'thin_plate': rbf_phi3,
        'gaussian': rbf_phi4,
    }
    if phi_type not in phi_map:
        raise ValueError(f"Unknown phi_type: {phi_type}")
    phi = phi_map[phi_type]

    nd = X_data.shape[0]
    D = compute_pairwise_distance(X_data, X_data)
    A = phi(D, r0)


    A += 1e-10 * np.eye(nd)
    cond_num = np.linalg.cond(A)
    w = np.linalg.solve(A, f_data)
    return w, cond_num


def rbf_interpolate(X_data, w, r0, X_query, phi_type='gaussian'):
    phi_map = {
        'multiquadric': rbf_phi1,
        'inverse_mq': rbf_phi2,
        'thin_plate': rbf_phi3,
        'gaussian': rbf_phi4,
    }
    phi = phi_map[phi_type]
    D = compute_pairwise_distance(X_query, X_data)
    Aq = phi(D, r0)
    return Aq @ w


class RBFKernelLayer:

    def __init__(self, n_centers, input_dim, r0=1.0, phi_type='gaussian',
                 learnable_centers=False, seed=42):
        rng = np.random.default_rng(seed)
        self.n_centers = n_centers
        self.input_dim = input_dim
        self.r0 = float(r0)
        self.phi_type = phi_type
        self.learnable_centers = learnable_centers


        self.centers = rng.uniform(-1.0, 1.0, size=(n_centers, input_dim))

        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_centers), size=(n_centers, 1))
        self.b = np.zeros(1)

    def forward(self, X):
        D = compute_pairwise_distance(X, self.centers)
        phi_map = {
            'multiquadric': rbf_phi1,
            'inverse_mq': rbf_phi2,
            'thin_plate': rbf_phi3,
            'gaussian': rbf_phi4,
        }
        phi = phi_map[self.phi_type]
        Phi = phi(D, self.r0)
        return Phi @ self.W + self.b
