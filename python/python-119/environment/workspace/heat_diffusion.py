
import numpy as np
from typing import Tuple, Callable, Optional
from numeric_utils import safe_divide


class HeatDiffusion1D:
    
    def __init__(
        self,
        L: float = 10.0,
        nx: int = 101,
        alpha: float = 0.1,
        dt: float = 0.001,
    ):
        if L <= 0 or nx < 3 or alpha <= 0 or dt <= 0:
            raise ValueError("参数必须满足: L>0, nx>=3, alpha>0, dt>0")
        
        self.L = L
        self.nx = nx
        self.alpha = alpha
        self.dt = dt
        self.dx = L / (nx - 1)
        

        self.cfl = alpha * dt / (self.dx ** 2)
        if self.cfl >= 0.5:

            self.dt = 0.45 * (self.dx ** 2) / alpha
            self.cfl = alpha * self.dt / (self.dx ** 2)
        
        self.x = np.linspace(0, L, nx)
        self.T = np.ones(nx)
    
    def solve_step(
        self,
        T_left: float,
        T_right: float,
        heat_source: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        T_new = np.zeros_like(self.T)
        

        for i in range(1, self.nx - 1):
            diffusion = self.cfl * (self.T[i-1] - 2.0 * self.T[i] + self.T[i+1])
            source = 0.0
            if heat_source is not None:
                source = self.dt * heat_source[i]
            T_new[i] = self.T[i] + diffusion + source
        

        T_new[0] = T_left
        T_new[-1] = T_right
        
        self.T = T_new
        return self.T.copy()
    
    def solve_steady(
        self,
        T_left: float,
        T_right: float,
        heat_source: Optional[np.ndarray] = None,
        max_iter: int = 10000,
        tol: float = 1e-8,
    ) -> np.ndarray:
        self.T = np.linspace(T_left, T_right, self.nx)
        
        for it in range(max_iter):
            T_old = self.T.copy()
            self.solve_step(T_left, T_right, heat_source)
            
            diff = np.max(np.abs(self.T - T_old))
            if diff < tol:
                break
        
        return self.T.copy()
    
    def thermal_gradient(self) -> np.ndarray:
        grad = np.zeros(self.nx)
        grad[1:-1] = (self.T[2:] - self.T[:-2]) / (2.0 * self.dx)
        grad[0] = (self.T[1] - self.T[0]) / self.dx
        grad[-1] = (self.T[-1] - self.T[-2]) / self.dx
        return grad


class HeatDiffusion2DFEM:
    
    def __init__(
        self,
        Lx: float = 10.0,
        Ly: float = 10.0,
        nx: int = 21,
        ny: int = 21,
        alpha: float = 0.1,
    ):
        if nx < 2 or ny < 2:
            raise ValueError("nx, ny 必须 >= 2")
        
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        

        self.X = np.zeros((nx, ny))
        self.Y = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                self.X[i, j] = i * self.dx
                self.Y[i, j] = j * self.dy
        

        self.T = np.ones((nx, ny))
    
    def _laplacian_5point(self, T: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(T)
        

        lap[1:-1, 1:-1] = (
            (T[:-2, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[2:, 1:-1]) / (self.dx ** 2)
            + (T[1:-1, :-2] - 2.0 * T[1:-1, 1:-1] + T[1:-1, 2:]) / (self.dy ** 2)
        )
        
        return lap
    
    def solve_backward_euler_step(
        self,
        dt: float,
        boundary_T: Callable[[np.ndarray, np.ndarray], np.ndarray],
        heat_source: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        

        rhs = self.T.copy()
        if heat_source is not None:
            rhs += dt * heat_source
        

        T_new = self.T.copy()
        

        cfl_x = self.alpha * dt / (self.dx ** 2)
        cfl_y = self.alpha * dt / (self.dy ** 2)
        
        if cfl_x + cfl_y > 0.5:

            dt = 0.45 / (self.alpha * (1.0 / self.dx ** 2 + 1.0 / self.dy ** 2))
        
        max_iter = 5000
        tol = 1e-8
        
        for it in range(max_iter):
            T_old = T_new.copy()
            

            lap = self._laplacian_5point(T_old)
            T_new = rhs + dt * self.alpha * lap
            

            T_boundary = boundary_T(self.X, self.Y)
            T_new[0, :] = T_boundary[0, :]
            T_new[-1, :] = T_boundary[-1, :]
            T_new[:, 0] = T_boundary[:, 0]
            T_new[:, -1] = T_boundary[:, -1]
            
            diff = np.max(np.abs(T_new - T_old))
            if diff < tol:
                break
        
        self.T = T_new
        return self.T.copy()
    
    def solve_steady_jacobi(
        self,
        boundary_T: Callable[[np.ndarray, np.ndarray], np.ndarray],
        heat_source: Optional[np.ndarray] = None,
        max_iter: int = 10000,
        tol: float = 1e-8,
    ) -> np.ndarray:
        self.T = np.ones((self.nx, self.ny))
        

        T_boundary = boundary_T(self.X, self.Y)
        self.T[0, :] = T_boundary[0, :]
        self.T[-1, :] = T_boundary[-1, :]
        self.T[:, 0] = T_boundary[:, 0]
        self.T[:, -1] = T_boundary[:, -1]
        

        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                self.T[i, j] = (
                    self.T[i, 0] * (self.ny - 1 - j) / (self.ny - 1)
                    + self.T[i, -1] * j / (self.ny - 1)
                ) * 0.5 + (
                    self.T[0, j] * (self.nx - 1 - i) / (self.nx - 1)
                    + self.T[-1, j] * i / (self.nx - 1)
                ) * 0.5
        
        source_term = np.zeros((self.nx, self.ny))
        if heat_source is not None:
            source_term = heat_source * (self.dx ** 2) / self.alpha
        
        for it in range(max_iter):
            T_old = self.T.copy()
            


            self.T[1:-1, 1:-1] = 0.25 * (
                T_old[:-2, 1:-1]
                + T_old[2:, 1:-1]
                + T_old[1:-1, :-2]
                + T_old[1:-1, 2:]
                + source_term[1:-1, 1:-1]
            )
            

            self.T[0, :] = T_boundary[0, :]
            self.T[-1, :] = T_boundary[-1, :]
            self.T[:, 0] = T_boundary[:, 0]
            self.T[:, -1] = T_boundary[:, -1]
            
            diff = np.max(np.abs(self.T - T_old))
            if diff < tol:
                break
        
        return self.T.copy()
    
    def effective_thermal_conductivity(self) -> float:
        return float(self.alpha * np.mean(self.T))
