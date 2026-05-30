
import numpy as np
from typing import Tuple, Optional, Callable


class TemperatureProtocol:
    
    def __init__(
        self,
        t0: float = 0.0,
        T_initial: float = 2.0,
        T_final: float = 0.1,
        t_stop: float = 1000.0,
        protocol: str = "linear",
    ):
        if t_stop <= t0:
            raise ValueError("t_stop 必须 > t0")
        if T_final <= 0:
            raise ValueError("T_final 必须 > 0")
        
        self.t0 = t0
        self.T_initial = T_initial
        self.T_final = T_final
        self.t_stop = t_stop
        self.protocol = protocol
        

        self._setup_protocol()
    
    def _setup_protocol(self):
        if self.protocol == "linear":

            self.cooling_rate = (self.T_initial - self.T_final) / (self.t_stop - self.t0)
        
        elif self.protocol == "step":

            self.n_steps = max(3, int((self.t_stop - self.t0) / 100.0))
            self.step_times = np.linspace(self.t0, self.t_stop, self.n_steps + 1)
            self.step_temps = np.linspace(self.T_initial, self.T_final, self.n_steps + 1)
        
        elif self.protocol == "logarithmic":


            self.alpha = -np.log(0.01) / (self.t_stop - self.t0)
        
        else:
            raise ValueError(f"不支持的协议: {self.protocol}")
    
    def temperature(self, t: float) -> float:
        if t <= self.t0:
            return self.T_initial
        if t >= self.t_stop:
            return self.T_final
        
        if self.protocol == "linear":
            T = self.T_initial - self.cooling_rate * (t - self.t0)
            return max(T, self.T_final)
        
        elif self.protocol == "step":

            idx = np.searchsorted(self.step_times, t, side='right') - 1
            idx = max(0, min(idx, len(self.step_temps) - 1))
            return self.step_temps[idx]
        
        elif self.protocol == "logarithmic":
            T = self.T_final + (self.T_initial - self.T_final) * np.exp(
                -self.alpha * (t - self.t0)
            )
            return max(T, self.T_final)
        
        return self.T_final
    
    def cooling_rate_at(self, t: float) -> float:
        if t <= self.t0 or t >= self.t_stop:
            return 0.0
        
        if self.protocol == "linear":
            return -self.cooling_rate
        
        elif self.protocol == "step":
            return 0.0
        
        elif self.protocol == "logarithmic":
            dT = -(self.T_initial - self.T_final) * self.alpha * np.exp(
                -self.alpha * (t - self.t0)
            )
            return dT
        
        return 0.0
    
    def get_schedule(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        times = np.linspace(self.t0, self.t_stop, n_points)
        temps = np.array([self.temperature(t) for t in times])
        return times, temps


class AndersenThermostat:
    
    def __init__(self, collision_frequency: float = 0.1, random_seed: int = 42):
        if collision_frequency < 0:
            raise ValueError("collision_frequency 必须 >= 0")
        self.nu = collision_frequency
        self.rng = np.random.RandomState(random_seed)
    
    def apply(
        self,
        velocities: np.ndarray,
        masses: np.ndarray,
        temperature: float,
        dt: float,
    ) -> np.ndarray:
        if temperature <= 0:
            return velocities
        
        new_velocities = velocities.copy()
        N = velocities.shape[0]
        

        p_collision = self.nu * dt
        if p_collision > 1.0:
            p_collision = 1.0
        
        sigma = np.sqrt(temperature / masses)
        
        for i in range(N):
            if self.rng.rand() < p_collision:

                new_velocities[i] = self.rng.normal(0.0, sigma[i], 3)
        
        return new_velocities


class BerendsenThermostat:
    
    def __init__(self, tau: float = 0.5):
        if tau <= 0:
            raise ValueError("tau 必须 > 0")
        self.tau = tau
    
    def apply(
        self,
        velocities: np.ndarray,
        masses: np.ndarray,
        target_temperature: float,
        dt: float,
    ) -> np.ndarray:
        if target_temperature <= 0:
            return velocities
        

        ndof = 3 * velocities.shape[0] - 3
        if ndof <= 0:
            return velocities
        
        ek = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
        T_inst = 2.0 * ek / ndof
        
        if T_inst < 1e-15:
            return velocities
        

        lambda_scale = np.sqrt(1.0 + (dt / self.tau) * (target_temperature / T_inst - 1.0))
        

        lambda_scale = max(0.5, min(2.0, lambda_scale))
        
        return velocities * lambda_scale
