
import numpy as np
from typing import Callable, Optional






def rbf_multiquadric(r: np.ndarray, r0: float = 1.0) -> np.ndarray:
    return np.sqrt(r * r + r0 * r0)


def rbf_inverse_multiquadric(r: np.ndarray, r0: float = 1.0) -> np.ndarray:
    return 1.0 / np.sqrt(r * r + r0 * r0)


def rbf_thin_plate_spline(r: np.ndarray, r0: float = 1.0) -> np.ndarray:
    result = np.zeros_like(r)
    mask = r > 1e-14
    result[mask] = r[mask] * r[mask] * np.log(r[mask] / r0)
    return result


def rbf_gaussian(r: np.ndarray, r0: float = 1.0) -> np.ndarray:
    return np.exp(-0.5 * r * r / (r0 * r0))


RBF_KERNELS = {
    'multiquadric': rbf_multiquadric,
    'inverse_multiquadric': rbf_inverse_multiquadric,
    'thin_plate_spline': rbf_thin_plate_spline,
    'gaussian': rbf_gaussian,
}






def compute_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    p1 = np.asarray(points1, dtype=np.float64)
    p2 = np.asarray(points2, dtype=np.float64)

    diff = p1[:, np.newaxis, :] - p2[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


class RBFInterpolator:
    def __init__(self, centers: np.ndarray, values: np.ndarray,
                 kernel: str = 'multiquadric', r0: float = 1.0,
                 add_polynomial: bool = True):
        self.centers = np.asarray(centers, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.kernel_name = kernel
        self.kernel = RBF_KERNELS[kernel]
        self.r0 = float(r0)
        self.add_polynomial = add_polynomial
        self.dim = self.centers.shape[1]
        self.n_centers = self.centers.shape[0]
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        self._compute_weights()

    def _compute_weights(self):
        D = compute_distance_matrix(self.centers, self.centers)
        Phi = self.kernel(D, self.r0)
        if self.add_polynomial:

            P = np.hstack([np.ones((self.n_centers, 1)), self.centers])

            n = self.n_centers
            m = P.shape[1]
            A = np.zeros((n + m, n + m), dtype=np.float64)
            A[:n, :n] = Phi
            A[:n, n:] = P
            A[n:, :n] = P.T

            B = np.zeros((n + m, self.values.shape[1]), dtype=np.float64)
            B[:n, :] = self.values
            try:
                W = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B, rcond=None)[0]
            self.weights = W[:n, :]
            self.poly_weights = W[n:, :]
        else:
            try:
                self.weights = np.linalg.solve(Phi, self.values)
            except np.linalg.LinAlgError:
                self.weights = np.linalg.lstsq(Phi, self.values, rcond=None)[0]
            self.poly_weights = None

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        D = compute_distance_matrix(points, self.centers)
        Phi = self.kernel(D, self.r0)
        vals = Phi @ self.weights
        if self.add_polynomial and self.poly_weights is not None:
            P = np.hstack([np.ones((points.shape[0], 1)), points])
            vals += P @ self.poly_weights
        return vals.squeeze()

    def gradient(self, points: np.ndarray, h: float = 1e-7) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        n_pts, dim = points.shape
        grad = np.zeros((n_pts, dim), dtype=np.float64)
        v0 = self.evaluate(points)
        if v0.ndim == 0:
            v0 = np.array([v0])
        for d in range(dim):
            pts_plus = points.copy()
            pts_plus[:, d] += h
            vp = self.evaluate(pts_plus)
            if vp.ndim == 0:
                vp = np.array([vp])
            grad[:, d] = (vp - v0) / h
        return grad






def rbf_troubled_cell_indicator(element_data: np.ndarray,
                                neighbor_data: np.ndarray,
                                r0: float = 1.0) -> float:
    n_local = len(element_data)
    n_neighbor = len(neighbor_data)
    if n_local < 2 or n_neighbor < 2:
        return 0.0

    x_local = np.linspace(-0.5, 0.5, n_local)
    x_neighbor = np.linspace(-1.5, 1.5, n_neighbor)
    x_all = np.concatenate([x_neighbor, x_local])
    y_all = np.concatenate([neighbor_data, element_data])

    try:
        rbf = RBFInterpolator(x_all.reshape(-1, 1), y_all,
                              kernel='multiquadric', r0=r0, add_polynomial=False)
    except Exception:
        return 0.0

    h = 1e-4
    v0 = rbf.evaluate(np.array([[0.0]]))
    vp = rbf.evaluate(np.array([[h]]))
    vm = rbf.evaluate(np.array([[-h]]))
    second_deriv = abs((vp - 2 * v0 + vm) / (h * h))

    data_range = np.max(y_all) - np.min(y_all)
    if data_range < 1e-14:
        return 0.0
    indicator = min(1.0, second_deriv[0] / (data_range + 1e-30))
    return indicator


def rbf_weno_reconstruction(stencils: list,
                            linear_weights: Optional[np.ndarray] = None,
                            r0: float = 1.0) -> np.ndarray:
    if not stencils:
        raise ValueError("Empty stencil list.")

    interpolants = []
    smoothness = []
    for centers, values in stencils:
        centers = np.asarray(centers, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if centers.ndim == 1:
            centers = centers.reshape(-1, 1)
        interp = RBFInterpolator(centers, values, kernel='gaussian',
                                  r0=r0, add_polynomial=False)
        interpolants.append(interp)

        smoothness.append(np.sum(interp.weights ** 2))

    epsilon = 1e-6
    smoothness = np.array(smoothness, dtype=np.float64)
    if linear_weights is None:
        linear_weights = np.ones(len(stencils)) / len(stencils)
    alpha = linear_weights / ((smoothness + epsilon) ** 2)
    omega = alpha / (alpha.sum() + 1e-30)
    return omega, interpolants
