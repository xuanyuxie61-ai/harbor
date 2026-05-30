
import numpy as np
from typing import Tuple, List


class WeylHamiltonian:
    
    def __init__(self, model_type: str = "linear", hbar: float = 1.0, v_f: float = 1.0):
        if model_type not in ("linear", "tight_binding"):
            raise ValueError(f"不支持的模型类型: {model_type}")
        self.model_type = model_type
        self.hbar = hbar
        self.v_f = v_f
        

        self.sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        self.sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        self.sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        self.I2 = np.eye(2, dtype=complex)
        

        if model_type == "tight_binding":
            self.m0 = 0.5
            self.m1 = -0.3
            self.m2 = -0.2
            self.A = 0.4
            self.B1 = 0.3
            self.B2 = 0.1
    
    def build_hamiltonian(self, k: np.ndarray) -> np.ndarray:
        k = np.atleast_2d(k)
        n_points = k.shape[0]
        
        if self.model_type == "linear":

            H = np.zeros((n_points, 2, 2), dtype=complex)
            for i in range(n_points):
                H[i] = self.hbar * self.v_f * (
                    k[i, 0] * self.sigma_x +
                    k[i, 1] * self.sigma_y +
                    k[i, 2] * self.sigma_z
                )
        else:








            raise NotImplementedError("Hole_1: 紧束缚模型哈密顿量构建待实现")
        
        if n_points == 1:
            return H[0]
        return H
    
    def eigenproblem(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H = self.build_hamiltonian(k)
        
        if H.ndim == 2:

            energies, eigenvectors = np.linalg.eigh(H)
            return energies, eigenvectors
        
        n_points = H.shape[0]
        energies = np.zeros((n_points, 2))
        eigenvectors = np.zeros((n_points, 2, 2), dtype=complex)
        
        for i in range(n_points):
            e, v = np.linalg.eigh(H[i])
            energies[i] = e
            eigenvectors[i] = v
        
        return energies, eigenvectors
    
    def d_vectors(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k = np.atleast_2d(k)
        n_points = k.shape[0]
        
        if self.model_type == "linear":
            d0 = np.zeros(n_points)
            d_vec = self.hbar * self.v_f * k
        else:
            H = self.build_hamiltonian(k)
            d0 = np.zeros(n_points)
            d_vec = np.zeros((n_points, 3))
            for i in range(n_points):
                d0[i] = 0.5 * np.trace(H[i]).real
                d_vec[i, 0] = 0.5 * np.trace(H[i] @ self.sigma_x).real
                d_vec[i, 1] = 0.5 * np.trace(H[i] @ self.sigma_y).real
                d_vec[i, 2] = 0.5 * np.trace(H[i] @ self.sigma_z).real
        
        return d0, d_vec
    
    def monomial_expansion_value(self, k: np.ndarray, exponents: np.ndarray,
                                  coefficients: np.ndarray) -> np.ndarray:
        k = np.atleast_2d(k)
        n_points = k.shape[0]
        n_terms = exponents.shape[0]
        
        if coefficients.shape[0] != n_terms:
            raise ValueError("系数数量与指数项数量不匹配")
        
        values = np.zeros(n_points)
        for j in range(n_points):
            val = 0.0
            for i in range(n_terms):

                monomial = 1.0
                for dim in range(3):
                    e = int(exponents[i, dim])
                    if e > 0:
                        monomial *= k[j, dim] ** e
                val += coefficients[i] * monomial
            values[j] = val
        
        return values
    
    def weyl_node_position_linear(self) -> np.ndarray:
        return np.zeros(3)
    
    def find_weyl_nodes_tight_binding(self, grid_size: int = 64,
                                       bz_bounds: np.ndarray = None) -> np.ndarray:
        if bz_bounds is None:
            bz_bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
        

        kx = np.linspace(bz_bounds[0, 0], bz_bounds[0, 1], grid_size)
        ky = np.linspace(bz_bounds[1, 0], bz_bounds[1, 1], grid_size)
        kz = np.linspace(bz_bounds[2, 0], bz_bounds[2, 1], grid_size)
        
        d0, d_vec = self.d_vectors(
            np.array([[x, y, z] for x in kx for y in ky for z in kz])
        )
        d_norm = np.linalg.norm(d_vec, axis=1)
        

        threshold = 0.1 * np.max(d_norm)
        candidate_idx = np.where(d_norm < threshold)[0]
        

        candidates = np.array([[x, y, z] for x in kx for y in ky for z in kz])[candidate_idx]
        
        if len(candidates) == 0:
            return np.zeros((0, 3))
        

        nodes = []
        used = set()
        dk = np.array([
            (bz_bounds[0, 1] - bz_bounds[0, 0]) / grid_size,
            (bz_bounds[1, 1] - bz_bounds[1, 0]) / grid_size,
            (bz_bounds[2, 1] - bz_bounds[2, 0]) / grid_size
        ])
        merge_dist = 2.0 * np.linalg.norm(dk)
        
        for i, c in enumerate(candidates):
            if i in used:
                continue
            cluster = [c]
            used.add(i)
            for j in range(i + 1, len(candidates)):
                if j not in used and np.linalg.norm(c - candidates[j]) < merge_dist:
                    cluster.append(candidates[j])
                    used.add(j)

            nodes.append(np.mean(cluster, axis=0))
        
        return np.array(nodes)


def band_gap(energies: np.ndarray) -> np.ndarray:
    if energies.ndim != 2 or energies.shape[1] != 2:
        raise ValueError("energies必须是(N,2)数组")
    return energies[:, 1] - energies[:, 0]


def velocity_operator(ham: WeylHamiltonian, k: np.ndarray,
                       delta: float = 1e-6) -> np.ndarray:
    v = np.zeros((3, 2, 2), dtype=complex)
    for i in range(3):
        kp = k.copy()
        km = k.copy()
        kp[i] += delta
        km[i] -= delta
        v[i] = (ham.build_hamiltonian(kp) - ham.build_hamiltonian(km)) / (2.0 * delta * ham.hbar)
    return v
