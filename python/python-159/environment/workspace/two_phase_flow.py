
import numpy as np
from utils import check_finite_array, safe_divide, robust_sqrt


class StokesDropletFlow:
    
    def __init__(self,
                 droplet_radius: float = 40.0e-6,
                 free_stream_velocity: float = 30.0,
                 gas_viscosity: float = 8.5e-5,
                 gas_density: float = 15.0,
                 liquid_viscosity: float = 1.0e-3,
                 surface_tension: float = 0.02):
        
        self.R_d = droplet_radius
        self.U_inf = free_stream_velocity
        self.mu_g = gas_viscosity
        self.rho_g = gas_density
        self.mu_l = liquid_viscosity
        self.sigma = surface_tension
        

        self.Re = self.rho_g * self.U_inf * (2 * self.R_d) / self.mu_g
        self.Ca = self.mu_g * self.U_inf / self.sigma
        self.We = self.rho_g * self.U_inf ** 2 * (2 * self.R_d) / self.sigma
    
    def stokes_drag_coefficient(self) -> float:
        lam = safe_divide(self.mu_l, self.mu_g, default=100.0)
        lam = np.clip(lam, 1e-6, 1e6)
        C_s = (2.0 / 3.0 + lam) / (1.0 + lam)
        return float(C_s)
    
    def stokes_drag_force(self) -> float:
        C_s = self.stokes_drag_coefficient()
        F_d = 6.0 * np.pi * self.mu_g * self.R_d * self.U_inf * C_s
        return float(F_d)
    
    def velocity_field_stokes(self, x: np.ndarray, y: np.ndarray) -> tuple:
        C_s = self.stokes_drag_coefficient()
        
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        

        r = np.maximum(r, self.R_d * 1.001)
        

        eta = self.R_d / r
        u_r = self.U_inf * np.cos(theta) * (1.0 - 1.5 * eta + 0.5 * eta ** 3) * C_s
        u_theta = -self.U_inf * np.sin(theta) * (1.0 - 0.75 * eta - 0.25 * eta ** 3) * C_s
        

        u = u_r * np.cos(theta) - u_theta * np.sin(theta)
        v = u_r * np.sin(theta) + u_theta * np.cos(theta)
        
        return u, v
    
    def pressure_field_stokes(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        C_s = self.stokes_drag_coefficient()
        r = np.sqrt(x ** 2 + y ** 2)
        r = np.maximum(r, self.R_d * 1.001)
        theta = np.arctan2(y, x)
        
        dp = -1.5 * self.mu_g * self.U_inf * self.R_d * np.cos(theta) / (r ** 2) * C_s
        return dp
    
    def shear_stress_distribution(self, n_points: int = 100) -> tuple:
        theta = np.linspace(0, np.pi, n_points)
        lam = safe_divide(self.mu_l, self.mu_g, default=100.0)
        lam = np.clip(lam, 1e-6, 1e6)
        
        tau = 1.5 * self.mu_g * self.U_inf * np.sin(theta) / self.R_d * (1.0 / (1.0 + lam))
        return theta, tau
    
    def compute_nusselt_number(self, Prandtl: float = 0.7) -> float:
        Re_d = max(self.Re, 1e-10)
        Nu = 2.0 + 0.6 * (Re_d ** 0.5) * (Prandtl ** (1.0 / 3.0))
        return float(Nu)
    
    def compute_sherwood_number(self, Schmidt: float = 1.0) -> float:
        Re_d = max(self.Re, 1e-10)
        Sh = 2.0 + 0.6 * (Re_d ** 0.5) * (Schmidt ** (1.0 / 3.0))
        return float(Sh)
    
    def basset_history_force(self, velocity_history: np.ndarray, dt: float) -> float:
        nu = safe_divide(self.mu_g, self.rho_g, default=1e-5)
        
        n = len(velocity_history)
        if n < 2:
            return 0.0
        
        F_basset = 0.0
        for k in range(n - 1):
            dU_dt = (velocity_history[-1] - velocity_history[k]) / ((n - 1 - k) * dt)
            tau = (n - 1 - k) * dt
            if tau > 0:
                F_basset += dU_dt / np.sqrt(np.pi * nu * tau) * dt
        
        F_basset *= 6.0 * np.pi * self.mu_g * self.R_d ** 2
        return float(F_basset)


class TwoPhaseFlowSolver:
    
    def __init__(self,
                 chamber_geometry,
                 n_z: int = 100,
                 gas_velocity_inlet: float = 50.0,
                 gas_temperature_inlet: float = 500.0):
        
        self.geo = chamber_geometry
        self.n_z = n_z
        self.z = np.linspace(0, self.geo.L_c, n_z)
        self.dz = self.z[1] - self.z[0]
        
        self.u_g = np.ones(n_z) * gas_velocity_inlet
        self.T_g = np.ones(n_z) * gas_temperature_inlet
        self.rho_g = np.ones(n_z) * 15.0
        self.P_g = np.ones(n_z) * 7.0e6
        

        self.d_droplet = np.ones(n_z) * 80e-6
        self.u_d = np.ones(n_z) * 30.0
        self.n_droplet = np.ones(n_z) * 1e8
    
    def solve_steady_1d(self, droplet_source: np.ndarray = None) -> dict:
        if droplet_source is None:
            droplet_source = np.zeros(self.n_z)
            droplet_source[:10] = 1e8
        

        for i in range(1, self.n_z):
            A = self.geo.area_at_z(self.z[i])
            A_prev = self.geo.area_at_z(self.z[i-1])
            

            stokes = StokesDropletFlow(
                droplet_radius=self.d_droplet[i-1] / 2.0,
                free_stream_velocity=self.u_g[i-1] - self.u_d[i-1]
            )
            K = stokes.compute_evaporation_rate() if hasattr(stokes, 'compute_evaporation_rate') else 1e-8

            k_g = 0.08
            B = 1.5
            K = 8.0 * k_g * np.log(1.0 + B) / (807.0 * 1800.0)
            

            d_sq_new = self.d_droplet[i-1] ** 2 - K * self.dz / max(self.u_d[i-1], 1.0)
            self.d_droplet[i] = np.sqrt(max(d_sq_new, 0.0))
            

            F_d = stokes.stokes_drag_force()
            m_d = (np.pi / 6.0) * 807.0 * self.d_droplet[i-1] ** 3
            du_d = F_d / max(m_d, 1e-15) * self.dz / max(self.u_d[i-1], 1.0)
            self.u_d[i] = self.u_d[i-1] + du_d
            

            mass_source = droplet_source[i] * (np.pi / 6.0) * 807.0 * \
                          (self.d_droplet[i-1] ** 3 - self.d_droplet[i] ** 3)
            self.rho_g[i] = self.rho_g[i-1] * self.u_g[i-1] * A_prev / \
                            (max(self.u_g[i-1], 1.0) * A) + mass_source / (A * self.dz)
            self.rho_g[i] = np.clip(self.rho_g[i], 1.0, 50.0)
            

            self.u_g[i] = self.u_g[i-1] * (A_prev / A) * (self.rho_g[i-1] / self.rho_g[i])
            self.u_g[i] = np.clip(self.u_g[i], 1.0, 500.0)
        
        return {
            "z": self.z,
            "gas_velocity": self.u_g,
            "gas_density": self.rho_g,
            "droplet_diameter": self.d_droplet,
            "droplet_velocity": self.u_d,
            "evaporation_length": self._find_evaporation_length()
        }
    
    def _find_evaporation_length(self) -> float:
        evaporated = np.where(self.d_droplet < 1e-9)[0]
        if len(evaporated) > 0:
            return float(self.z[evaporated[0]])
        return float(self.z[-1])


if __name__ == "__main__":
    from geometry_model import CombustionChamberGeometry
    
    geo = CombustionChamberGeometry()
    flow = TwoPhaseFlowSolver(geo, n_z=200)
    result = flow.solve_steady_1d()
    
    print(f"Evaporation length: {result['evaporation_length']:.4f} m")
    print(f"Max gas velocity: {np.max(result['gas_velocity']):.2f} m/s")
    
    stokes = StokesDropletFlow()
    print(f"Stokes drag coefficient: {stokes.stokes_drag_coefficient():.4f}")
    print(f"Drag force: {stokes.stokes_drag_force():.6e} N")
    print(f"Nusselt number: {stokes.compute_nusselt_number():.3f}")
