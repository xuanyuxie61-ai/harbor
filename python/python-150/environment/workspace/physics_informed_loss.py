
import numpy as np
from typing import Callable


class PhysicsInformedLoss:

    def __init__(self, lambda_energy: float = 1.0,
                 lambda_force: float = 10.0,
                 lambda_charge: float = 5.0):
        self.lambda_energy = lambda_energy
        self.lambda_force = lambda_force
        self.lambda_charge = lambda_charge

    def energy_invariance_loss(self, energy_func: Callable,
                               coords: np.ndarray) -> float:
        n_atoms = coords.shape[0]

        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        u = np.random.randn(3)
        u = u - np.dot(u, v) * v
        u = u / np.linalg.norm(u)
        w = np.cross(v, u)
        R = np.column_stack([u, w, v])
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        t = np.random.randn(3) * 0.1
        coords_rot = (coords @ R.T) + t.reshape(1, 3)

        e1 = energy_func(coords)
        e2 = energy_func(coords_rot)
        return float((e1 - e2) ** 2)

    def force_consistency_loss(self, energy_func: Callable,
                               coords: np.ndarray) -> float:
        delta = 1e-4
        n_atoms = coords.shape[0]
        grad_fd = np.zeros_like(coords)
        e0 = energy_func(coords)
        for i in range(n_atoms):
            for d in range(3):
                coords_plus = coords.copy()
                coords_plus[i, d] += delta
                e_plus = energy_func(coords_plus)
                grad_fd[i, d] = (e_plus - e0) / delta

        return float(np.sum(grad_fd ** 2))

    def charge_conservation_loss(self, predicted_charges: np.ndarray,
                                 target_total: float) -> float:
        return float((np.sum(predicted_charges) - target_total) ** 2)

    def total_physics_loss(self, energy_func: Callable,
                           coords: np.ndarray,
                           predicted_charges: np.ndarray,
                           target_total_charge: float) -> float:
        loss_e = self.energy_invariance_loss(energy_func, coords)
        loss_f = self.force_consistency_loss(energy_func, coords)
        loss_q = self.charge_conservation_loss(predicted_charges, target_total_charge)
        return (self.lambda_energy * loss_e +
                self.lambda_force * loss_f +
                self.lambda_charge * loss_q)


def schrodinger_residual_energy(wavefunction_vals: np.ndarray,
                                potential_vals: np.ndarray,
                                laplacian_wf: np.ndarray,
                                hbar: float = 1.0,
                                mass: float = 1.0) -> float:
    kinetic = - (hbar ** 2) / (2.0 * mass) * laplacian_wf
    residual = kinetic + potential_vals * wavefunction_vals

    E_expect = np.sum(wavefunction_vals * residual)
    residual_norm = np.sum((residual - E_expect * wavefunction_vals) ** 2)
    return float(residual_norm)
