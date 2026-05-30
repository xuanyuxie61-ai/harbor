
import numpy as np
from typing import Tuple, Optional
from numeric_utils import safe_divide, soft_cutoff, distance_matrix_pbc


class ForceField:
    
    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        rcutoff: float = 2.5,
        fene_k: float = 30.0,
        fene_R0: float = 1.5,
        angle_k: float = 5.0,
        angle_theta0: float = np.pi,
    ):
        if epsilon <= 0 or sigma <= 0:
            raise ValueError("epsilon, sigma 必须 > 0")
        if rcutoff <= sigma:
            raise ValueError("rcutoff 必须 > sigma")
        if fene_k <= 0 or fene_R0 <= 0:
            raise ValueError("fene_k, fene_R0 必须 > 0")
        
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcutoff = rcutoff
        self.fene_k = fene_k
        self.fene_R0 = fene_R0
        self.angle_k = angle_k
        self.angle_theta0 = angle_theta0
        

        self._compute_lj_shift()
    
    def _compute_lj_shift(self):
        sr = self.sigma / self.rcutoff
        sr6 = sr ** 6
        sr12 = sr6 ** 2
        self.u_shift = 4.0 * self.epsilon * (sr12 - sr6)
    
    def lj_potential(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r)
        u = np.zeros_like(r, dtype=float)
        
        mask = (r > 1e-10) & (r <= self.rcutoff)
        if np.any(mask):
            sr = self.sigma / r[mask]
            sr6 = sr ** 6
            sr12 = sr6 ** 2
            u[mask] = 4.0 * self.epsilon * (sr12 - sr6) - self.u_shift
        
        return u
    
    def lj_force_scalar(self, r: float) -> float:
        if r <= 1e-10 or r > self.rcutoff:
            return 0.0
        sr = self.sigma / r
        sr7 = sr ** 7
        sr13 = sr ** 13
        return 24.0 * self.epsilon / self.sigma * (2.0 * sr13 - sr7)
    
    def lj_forces_vector(self, positions: np.ndarray, box: np.ndarray) -> np.ndarray:
        N = positions.shape[0]
        forces = np.zeros_like(positions)
        
        for i in range(N):
            for j in range(i + 1, N):
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                
                if r > 1e-10 and r <= self.rcutoff:
                    f_mag = self.lj_force_scalar(r)
                    f_vec = f_mag * dr / r
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        return forces
    
    def fene_potential(self, r: float) -> float:
        if r >= self.fene_R0:

            return 1e10
        if r <= 0:
            return 0.0
        ratio = r / self.fene_R0
        return -0.5 * self.fene_k * self.fene_R0 ** 2 * np.log(1.0 - ratio ** 2)
    
    def fene_force_scalar(self, r: float) -> float:
        if r >= self.fene_R0 or r <= 0:
            return 0.0
        ratio_sq = (r / self.fene_R0) ** 2
        return -self.fene_k * r / (1.0 - ratio_sq)
    
    def fene_forces(self, positions: np.ndarray, box: np.ndarray, chain_starts: np.ndarray) -> np.ndarray:
        N = positions.shape[0]
        forces = np.zeros_like(positions)
        beads_per_chain = chain_starts[1] - chain_starts[0] if len(chain_starts) > 1 else N
        
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for i in range(start, end - 1):
                j = i + 1
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                
                if r > 1e-10:
                    f_mag = self.fene_force_scalar(r)
                    f_vec = f_mag * dr / r
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        return forces
    
    def angle_potential(self, theta: float) -> float:
        dtheta = theta - self.angle_theta0
        return self.angle_k * dtheta ** 2
    
    def angle_forces(self, positions: np.ndarray, box: np.ndarray, chain_starts: np.ndarray) -> np.ndarray:
        N = positions.shape[0]
        forces = np.zeros_like(positions)
        beads_per_chain = chain_starts[1] - chain_starts[0] if len(chain_starts) > 1 else N
        
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for j in range(start + 1, end - 1):
                i = j - 1
                k = j + 1
                

                rji = positions[i] - positions[j]
                rji = rji - box * np.rint(rji / box)
                rjk = positions[k] - positions[j]
                rjk = rjk - box * np.rint(rjk / box)
                
                norm_ji = np.linalg.norm(rji)
                norm_jk = np.linalg.norm(rjk)
                
                if norm_ji < 1e-10 or norm_jk < 1e-10:
                    continue
                

                cos_theta = np.dot(rji, rjk) / (norm_ji * norm_jk)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                

                dtheta = theta - self.angle_theta0
                


                force_mag = 2.0 * self.angle_k * dtheta
                

                if norm_ji > 1e-10:
                    perp_i = rji - np.dot(rji, rjk) / (norm_jk ** 2) * rjk
                    perp_norm = np.linalg.norm(perp_i)
                    if perp_norm > 1e-10:
                        forces[i] += force_mag * perp_i / perp_norm
                        forces[j] -= force_mag * perp_i / perp_norm
                

                if norm_jk > 1e-10:
                    perp_k = rjk - np.dot(rjk, rji) / (norm_ji ** 2) * rji
                    perp_norm = np.linalg.norm(perp_k)
                    if perp_norm > 1e-10:
                        forces[k] += force_mag * perp_k / perp_norm
                        forces[j] -= force_mag * perp_k / perp_norm
        
        return forces
    
    def compute_total_forces(
        self,
        positions: np.ndarray,
        box: np.ndarray,
        chain_starts: np.ndarray,
    ) -> np.ndarray:
        f_lj = self.lj_forces_vector(positions, box)
        f_fene = self.fene_forces(positions, box, chain_starts)
        f_angle = self.angle_forces(positions, box, chain_starts)
        
        return f_lj + f_fene + f_angle
    
    def total_potential_energy(
        self,
        positions: np.ndarray,
        box: np.ndarray,
        chain_starts: np.ndarray,
    ) -> float:
        N = positions.shape[0]
        beads_per_chain = chain_starts[1] - chain_starts[0] if len(chain_starts) > 1 else N
        

        u_lj = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                u_lj += self.lj_potential(r)
        

        u_fene = 0.0
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for i in range(start, end - 1):
                j = i + 1
                dr = positions[i] - positions[j]
                dr = dr - box * np.rint(dr / box)
                r = np.linalg.norm(dr)
                u_fene += self.fene_potential(r)
        

        u_angle = 0.0
        for c in range(len(chain_starts)):
            start = chain_starts[c]
            end = min(start + beads_per_chain, N)
            for j in range(start + 1, end - 1):
                i = j - 1
                k = j + 1
                rji = positions[i] - positions[j]
                rji = rji - box * np.rint(rji / box)
                rjk = positions[k] - positions[j]
                rjk = rjk - box * np.rint(rjk / box)
                
                norm_ji = np.linalg.norm(rji)
                norm_jk = np.linalg.norm(rjk)
                if norm_ji > 1e-10 and norm_jk > 1e-10:
                    cos_t = np.dot(rji, rjk) / (norm_ji * norm_jk)
                    cos_t = np.clip(cos_t, -1.0, 1.0)
                    theta = np.arccos(cos_t)
                    u_angle += self.angle_potential(theta)
        
        return u_lj + u_fene + u_angle
