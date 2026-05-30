
import numpy as np
from typing import Callable, Tuple, Optional


class VelocityVerletIntegrator:
    
    def __init__(self, dt: float = 0.001):
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        self.dt = dt
    
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        box: np.ndarray,
        force_func: Callable,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = self.dt
        N = positions.shape[0]
        

        inv_m = 1.0 / masses[:, np.newaxis]
        

        v_half = velocities + 0.5 * dt * forces * inv_m
        

        new_positions = positions + dt * v_half
        

        new_positions = new_positions % box
        

        new_forces = force_func(new_positions)
        

        if np.any(~np.isfinite(new_forces)):

            new_forces = forces.copy()
        

        new_velocities = v_half + 0.5 * dt * new_forces * inv_m
        

        max_vel = 100.0 / dt
        vel_norm = np.linalg.norm(new_velocities, axis=1)
        if np.any(vel_norm > max_vel):
            scale = np.where(vel_norm > max_vel, max_vel / vel_norm, 1.0)
            new_velocities = new_velocities * scale[:, np.newaxis]
        
        return new_positions, new_velocities, new_forces
    
    def cfl_constraint(self, positions: np.ndarray, velocities: np.ndarray, box: np.ndarray) -> bool:
        max_vel = np.max(np.linalg.norm(velocities, axis=1))
        min_box = np.min(box)
        
        if max_vel < 1e-15:
            return True
        
        courant = self.dt * max_vel / min_box
        return courant < 0.1


def rk23_step(
    y: np.ndarray,
    t: float,
    dt: float,
    f: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    if dt <= 0:
        raise ValueError("rk23_step: dt 必须 > 0")
    
    k1 = dt * f(t, y)
    k2 = dt * f(t + dt, y + k1)
    k3 = dt * f(t + 0.5 * dt, y + 0.25 * k1 + 0.25 * k2)
    
    y2 = y + 0.5 * (k1 + k2)
    y3 = y + (k1 + k2 + 4.0 * k3) / 6.0
    
    error = np.abs(y3 - y2)
    

    tol = 1e-6
    max_err = np.max(error)
    if max_err < 1e-15:
        dt_suggested = 2.0 * dt
    else:
        dt_suggested = dt * min(2.0, max(0.25, 0.9 * (tol / max_err) ** (1.0 / 3.0)))
    
    return y3, error, dt_suggested


def rk23_integrate(
    y0: np.ndarray,
    tspan: Tuple[float, float],
    n_steps: int,
    f: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0, t1 = tspan
    dt = (t1 - t0) / n_steps
    m = len(y0)
    
    t_arr = np.zeros(n_steps + 1)
    y_arr = np.zeros((n_steps + 1, m))
    e_arr = np.zeros((n_steps + 1, m))
    
    t_arr[0] = t0
    y_arr[0, :] = y0
    e_arr[0, :] = 0.0
    
    for i in range(n_steps):
        y_next, err, _ = rk23_step(y_arr[i], t_arr[i], dt, f)
        t_arr[i + 1] = t_arr[i] + dt
        y_arr[i + 1, :] = y_next
        e_arr[i + 1, :] = err
    
    return t_arr, y_arr, e_arr


class NoseHooverIntegrator:
    
    def __init__(self, dt: float = 0.001, Q: float = 10.0):
        if dt <= 0 or Q <= 0:
            raise ValueError("dt 和 Q 必须 > 0")
        self.dt = dt
        self.Q = Q
        self.xi = 0.0
    
    def step(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        forces: np.ndarray,
        masses: np.ndarray,
        box: np.ndarray,
        target_temperature: float,
        force_func: Callable,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = self.dt
        N = positions.shape[0]
        ndof = 3 * N - 3
        inv_m = 1.0 / masses[:, np.newaxis]
        

        ek = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
        

        def bath_ode(t, xi_vec):
            xi_val = xi_vec[0]


            return np.array([(2.0 * ek - ndof * target_temperature) / self.Q])
        
        xi_arr, _, _ = rk23_integrate(np.array([self.xi]), (0.0, dt), 1, bath_ode)
        self.xi = float(xi_arr[-1, 0])
        

        v_half = velocities + 0.5 * dt * forces * inv_m - 0.5 * dt * self.xi * velocities
        
        new_positions = positions + dt * v_half
        new_positions = new_positions % box
        
        new_forces = force_func(new_positions)
        if np.any(~np.isfinite(new_forces)):
            new_forces = forces.copy()
        
        new_velocities = v_half + 0.5 * dt * new_forces * inv_m - 0.5 * dt * self.xi * v_half
        

        max_vel = 100.0 / dt
        vel_norm = np.linalg.norm(new_velocities, axis=1)
        if np.any(vel_norm > max_vel):
            scale = np.where(vel_norm > max_vel, max_vel / vel_norm, 1.0)
            new_velocities = new_velocities * scale[:, np.newaxis]
        
        return new_positions, new_velocities, new_forces
