
import numpy as np
from typing import Tuple, Optional


class CVTSampler:
    
    def __init__(
        self,
        n_generators: int = 64,
        n_samples: int = 5000,
        max_iter: int = 50,
        box: np.ndarray = None,
        tol: float = 1e-5,
    ):
        if n_generators < 1:
            raise ValueError("n_generators 必须 >= 1")
        if n_samples < n_generators:
            raise ValueError("n_samples 必须 >= n_generators")
        if max_iter < 1:
            raise ValueError("max_iter 必须 >= 1")
        
        self.n_generators = n_generators
        self.n_samples = n_samples
        self.max_iter = max_iter
        self.box = np.array(box if box is not None else [10.0, 10.0, 10.0])
        self.tol = tol
        

        self.generators = np.random.rand(n_generators, 3) * self.box
        

        self.energy_history = []
    
    def _free_volume_density(
        self,
        samples: np.ndarray,
        polymer_positions: np.ndarray,
        exclusion_radius: float = 1.0,
    ) -> np.ndarray:
        M = samples.shape[0]
        N = polymer_positions.shape[0]
        
        sigma = exclusion_radius / 2.0
        sigma_sq = sigma ** 2
        

        density = np.zeros(M)
        

        batch_size = 1000
        for i in range(0, M, batch_size):
            batch = samples[i:i+batch_size]

            diff = batch[:, np.newaxis, :] - polymer_positions[np.newaxis, :, :]
            diff = diff - self.box * np.rint(diff / self.box)
            dist_sq = np.sum(diff ** 2, axis=2)
            

            occupancy = np.sum(np.exp(-dist_sq / (2.0 * sigma_sq)), axis=1)
            

            density[i:i+batch_size] = np.exp(-occupancy)
        

        dmax = np.max(density)
        if dmax > 1e-15:
            density = density / dmax
        

        density = np.clip(density, 1e-10, 1.0)
        
        return density
    
    def _find_closest(self, samples: np.ndarray) -> np.ndarray:
        M = samples.shape[0]
        indices = np.zeros(M, dtype=int)
        
        for i in range(M):
            diff = samples[i] - self.generators
            diff = diff - self.box * np.rint(diff / self.box)
            dist_sq = np.sum(diff ** 2, axis=1)
            indices[i] = np.argmin(dist_sq)
        
        return indices
    
    def _cvt_energy(self, density: np.ndarray, samples: np.ndarray, indices: np.ndarray) -> float:
        energy = 0.0
        for k in range(len(samples)):
            diff = samples[k] - self.generators[indices[k]]
            diff = diff - self.box * np.rint(diff / self.box)
            energy += density[k] * np.sum(diff ** 2)
        
        return energy / len(samples)
    
    def iterate(
        self,
        polymer_positions: np.ndarray,
        exclusion_radius: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        for it in range(self.max_iter):

            samples = np.random.rand(self.n_samples, 3) * self.box
            

            density = self._free_volume_density(samples, polymer_positions, exclusion_radius)
            

            indices = self._find_closest(samples)
            

            energy = self._cvt_energy(density, samples, indices)
            self.energy_history.append(energy)
            

            new_generators = np.zeros_like(self.generators)
            mass = np.zeros(self.n_generators)
            
            for k in range(self.n_samples):
                i = indices[k]
                new_generators[i] += density[k] * samples[k]
                mass[i] += density[k]
            

            for i in range(self.n_generators):
                if mass[i] > 1e-15:
                    new_generators[i] = new_generators[i] / mass[i]
                else:

                    new_generators[i] = np.random.rand(3) * self.box
            


            for d in range(3):
                mask_low = new_generators[:, d] < 0
                mask_high = new_generators[:, d] > self.box[d]
                new_generators[mask_low, d] = -new_generators[mask_low, d]
                new_generators[mask_high, d] = 2 * self.box[d] - new_generators[mask_high, d]
            

            new_generators = np.clip(new_generators, 0.0, self.box)
            

            max_displacement = np.max(np.linalg.norm(new_generators - self.generators, axis=1))
            self.generators = new_generators
            
            if max_displacement < self.tol:
                break
        

        volumes = self._estimate_voronoi_volumes(polymer_positions, exclusion_radius)
        
        return self.generators.copy(), volumes
    
    def _estimate_voronoi_volumes(
        self,
        polymer_positions: np.ndarray,
        exclusion_radius: float = 1.0,
    ) -> np.ndarray:
        n_test = 100000
        test_samples = np.random.rand(n_test, 3) * self.box
        
        indices = self._find_closest(test_samples)
        counts = np.bincount(indices, minlength=self.n_generators)
        
        box_volume = np.prod(self.box)
        volumes = counts / n_test * box_volume
        
        return volumes
    
    def free_volume_fraction(
        self,
        polymer_positions: np.ndarray,
        van_der_waals_radius: float = 0.8,
    ) -> float:
        N = polymer_positions.shape[0]
        box_volume = np.prod(self.box)
        

        v_particle = (4.0 / 3.0) * np.pi * van_der_waals_radius ** 3
        occupied_volume = N * v_particle
        


        eta = occupied_volume / box_volume
        if eta >= 1.0:
            return 0.0
        


        fv = 1.0 - eta
        
        return float(max(0.0, min(fv, 1.0)))
    
    def structural_order_parameter(self) -> float:
        if len(self.energy_history) < 2:
            return 0.0
        
        e_initial = self.energy_history[0]
        e_final = self.energy_history[-1]
        
        if abs(e_initial) < 1e-15:
            return 0.0
        
        eta = 1.0 - e_final / e_initial
        return float(np.clip(eta, 0.0, 1.0))
