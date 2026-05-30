
import numpy as np
from utils import check_finite_array, safe_divide, robust_sqrt


class SprayDistributionCVT:
    
    def __init__(self,
                 chamber_radius: float = 0.15,
                 chamber_length: float = 0.60,
                 n_droplets: int = 500,
                 droplet_diameter_mean: float = 80.0e-6,
                 droplet_diameter_std: float = 20.0e-6,
                 gas_temperature: float = 3000.0,
                 gas_pressure: float = 7.0e6):
        
        self.R_c = chamber_radius
        self.L_c = chamber_length
        self.n_droplets = n_droplets
        self.d_0 = droplet_diameter_mean
        self.d_std = droplet_diameter_std
        self.T_g = gas_temperature
        self.P_g = gas_pressure
        

        self.positions = None
        self.velocities = None
        self.diameters = None
        self.masses = None
        

        self.energy_history = []
    
    def generate_initial_distribution(self, seed: int = 42) -> np.ndarray:
        rng = np.random.RandomState(seed)
        

        lambda_z = 1.0 / (0.15 * self.L_c)
        z = rng.exponential(1.0 / lambda_z, self.n_droplets)
        z = np.clip(z, 0.0, self.L_c)
        


        r_norm = rng.beta(2.0, 2.0, self.n_droplets)
        r = 0.1 * self.R_c + 0.8 * self.R_c * r_norm
        r = np.clip(r, 0.0, self.R_c)
        

        theta = 2.0 * np.pi * rng.rand(self.n_droplets)
        
        self.positions = np.column_stack([z, r, theta])
        

        u_z = 15.0 + 10.0 * rng.randn(self.n_droplets)
        u_r = 2.0 * rng.randn(self.n_droplets)
        u_theta = 5.0 * rng.randn(self.n_droplets)
        self.velocities = np.column_stack([u_z, u_r, u_theta])
        

        n_rr = 3.5
        d_0_rr = self.d_0 * (np.log(2.0)) ** (1.0 / n_rr)
        u = rng.rand(self.n_droplets)
        d = d_0_rr * (-np.log(1.0 - u)) ** (1.0 / n_rr)
        self.diameters = np.clip(d, 10e-6, 200e-6)
        

        rho_l = 807.0
        self.masses = (np.pi / 6.0) * rho_l * self.diameters ** 3
        
        return self.positions
    
    def _find_closest_generator(self, samples: np.ndarray, generators: np.ndarray) -> np.ndarray:
        n_samples = samples.shape[0]
        n_gen = generators.shape[0]
        


        xs = samples[:, 1:2] * np.cos(samples[:, 2:3])
        ys = samples[:, 1:2] * np.sin(samples[:, 2:3])
        zs = samples[:, 0:1]
        

        xg = generators[:, 1:2] * np.cos(generators[:, 2:3])
        yg = generators[:, 1:2] * np.sin(generators[:, 2:3])
        zg = generators[:, 0:1]
        

        dx = xs.T - xg
        dy = ys.T - yg
        dz = zs.T - zg
        
        dist_sq = dx ** 2 + dy ** 2 + dz ** 2
        nearest = np.argmin(dist_sq, axis=0)
        min_dist_sq = np.min(dist_sq, axis=0)
        
        return nearest, min_dist_sq
    
    def cvt_iterate(self, n_samples: int = 10000) -> tuple:
        if self.positions is None:
            raise RuntimeError("Call generate_initial_distribution first.")
        
        generators = self.positions.copy()
        n_gen = generators.shape[0]
        

        rng = np.random.RandomState(123)
        z_samp = rng.rand(n_samples) * self.L_c
        r_samp = self.R_c * np.sqrt(rng.rand(n_samples))
        theta_samp = 2.0 * np.pi * rng.rand(n_samples)
        samples = np.column_stack([z_samp, r_samp, theta_samp])
        

        nearest, min_dist_sq = self._find_closest_generator(samples, generators)
        

        new_generators = np.zeros_like(generators)
        counts = np.zeros(n_gen)
        energy = 0.0
        
        for j in range(n_samples):
            idx = nearest[j]
            new_generators[idx] += samples[j]
            counts[idx] += 1
            energy += min_dist_sq[j]
        

        for j in range(n_gen):
            if counts[j] > 0:
                new_generators[j] /= counts[j]
            else:

                new_generators[j] = generators[j]
        

        new_generators[:, 2] = new_generators[:, 2] % (2.0 * np.pi)
        

        new_generators[:, 0] = np.clip(new_generators[:, 0], 0.0, self.L_c)
        new_generators[:, 1] = np.clip(new_generators[:, 1], 0.0, self.R_c)
        

        it_diff = np.sqrt(np.sum((new_generators - generators) ** 2, axis=1)).sum()
        energy = energy / n_samples
        
        self.positions = new_generators
        self.energy_history.append(energy)
        
        return it_diff, energy
    
    def optimize_distribution(self, n_iterations: int = 50, n_samples: int = 10000,
                              tolerance: float = 1e-6) -> dict:
        if self.positions is None:
            self.generate_initial_distribution()
        
        for it in range(n_iterations):
            it_diff, energy = self.cvt_iterate(n_samples=n_samples)
            
            if it_diff < tolerance:
                break
        
        return {
            "iterations": it + 1,
            "final_energy": energy,
            "energy_history": np.array(self.energy_history),
            "final_positions": self.positions,
            "mean_diameter": float(np.mean(self.diameters)),
            "std_diameter": float(np.std(self.diameters))
        }
    
    def compute_evaporation_rate(self) -> np.ndarray:
        if self.diameters is None:
            raise RuntimeError("Droplets not initialized.")
        
        T_b = 500.0
        L_v = 2.13e6
        k_g = 0.08
        C_p_g = 1800.0
        rho_l = 807.0
        

        B = C_p_g * (self.T_g - T_b) / L_v
        B = np.clip(B, 0.01, 10.0)
        
        K = 8.0 * k_g * np.log(1.0 + B) / (rho_l * C_p_g)
        
        return K
    
    def simulate_droplet_lifetime(self, dt: float = 1.0e-5, n_steps: int = 1000) -> dict:
        if self.positions is None or self.diameters is None:
            raise RuntimeError("Droplets not initialized.")
        
        d_current = self.diameters.copy()
        pos_current = self.positions.copy()
        vel_current = self.velocities.copy()
        
        K = self.compute_evaporation_rate()
        

        rho_g = 15.0
        mu_g = 8.5e-5
        

        d_history = [d_current.copy()]
        pos_history = [pos_current.copy()]
        
        for step in range(n_steps):

            d_sq = d_current ** 2 - K * dt
            d_current = np.sqrt(np.maximum(d_sq, 0.0))
            

            evaporated = d_current < 1e-9
            


            u_rel = 50.0 - vel_current[:, 0]
            u_rel = np.clip(u_rel, -500.0, 500.0)
            Re = rho_g * np.abs(u_rel) * d_current / mu_g
            Re = np.clip(Re, 1e-6, 1000.0)
            

            C_d = np.where(Re < 1.0, 24.0 / Re,
                           24.0 / Re * (1.0 + 0.15 * Re ** 0.687))
            C_d = np.clip(C_d, 0.0, 500.0)
            

            d_safe = np.maximum(d_current, 1e-9)
            a_z = (3.0 / 4.0) * (rho_g / 807.0) * (C_d / d_safe) * \
                  np.abs(u_rel) * u_rel
            a_z = np.clip(a_z, -1e6, 1e6)
            
            vel_current[:, 0] += a_z * dt
            pos_current[:, 0] += vel_current[:, 0] * dt
            

            pos_current[:, 0] = np.clip(pos_current[:, 0], 0.0, self.L_c)
            pos_current[:, 1] = np.clip(pos_current[:, 1], 0.0, self.R_c)
            

            pos_current[evaporated] = pos_history[0][evaporated]
            
            if step % 100 == 0:
                d_history.append(d_current.copy())
                pos_history.append(pos_current.copy())
        
        return {
            "final_diameters": d_current,
            "final_positions": pos_current,
            "evaporation_fraction": float(np.mean(d_current < 1e-9)),
            "diameter_history": d_history,
            "position_history": pos_history
        }
    
    def compute_spray_statistics(self) -> dict:
        if self.positions is None:
            raise RuntimeError("Droplets not initialized.")
        

        if self.diameters is not None:
            d32 = np.sum(self.diameters ** 3) / np.sum(self.diameters ** 2)
        else:
            d32 = self.d_0
        

        n_bins = 20
        z_bins = np.linspace(0, self.L_c, n_bins + 1)
        bin_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
        concentrations = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (self.positions[:, 0] >= z_bins[i]) & (self.positions[:, 0] < z_bins[i+1])
            if np.sum(mask) > 0:
                bin_volume = np.pi * self.R_c ** 2 * (z_bins[i+1] - z_bins[i])
                concentrations[i] = np.sum(self.masses[mask]) / bin_volume if self.masses is not None else np.sum(mask) / bin_volume
        
        return {
            "sauter_mean_diameter": float(d32),
            "n_droplets": self.n_droplets,
            "mean_axial_position": float(np.mean(self.positions[:, 0])),
            "std_axial_position": float(np.std(self.positions[:, 0])),
            "mean_radial_position": float(np.mean(self.positions[:, 1])),
            "concentration_profile": concentrations,
            "bin_centers": bin_centers
        }


if __name__ == "__main__":
    spray = SprayDistributionCVT(n_droplets=200)
    spray.generate_initial_distribution()
    
    result = spray.optimize_distribution(n_iterations=30, n_samples=5000)
    print(f"CVT converged in {result['iterations']} iterations")
    print(f"Final energy: {result['final_energy']:.6e}")
    
    lifetime = spray.simulate_droplet_lifetime(n_steps=500)
    print(f"Evaporation fraction: {lifetime['evaporation_fraction']:.3f}")
    
    stats = spray.compute_spray_statistics()
    print(f"Sauter mean diameter: {stats['sauter_mean_diameter']*1e6:.2f} μm")
