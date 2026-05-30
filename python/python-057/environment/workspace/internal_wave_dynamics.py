
import numpy as np
from scipy.integrate import solve_ivp


class NonlinearInternalWave:
    
    def __init__(self, alpha=1.0, beta=5.0, gamma=8.0, delta=0.02,
                 omega=0.5, N=0.01, f=1.0e-4, depth=200.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.N = N
        self.f = f
        self.depth = depth
        

        self.alpha_eff = alpha + N**2
        

        self.state = np.array([1.0, 0.0, 0.5])
    
    def rhs(self, t, y):
        xi, xi_dot, E = y
        

        xi = np.clip(xi, -50.0, 50.0)
        xi_dot = np.clip(xi_dot, -10.0, 10.0)
        

        nonlinear_force = -self.beta * xi**3
        

        forcing = self.gamma * np.cos(self.omega * t)
        

        coriolis = -self.f * xi_dot
        

        damping = -self.delta * xi_dot
        

        dxi_dt = xi_dot
        dxi_dot_dt = damping - self.alpha_eff * xi + nonlinear_force + forcing + coriolis
        

        r = 2.0 * self.delta
        E_max = 0.5 * self.gamma**2 / self.alpha_eff
        epsilon_diss = self.delta * xi_dot**2
        
        dE_dt = r * E * (1.0 - E / E_max) - epsilon_diss
        

        E = np.clip(E, 0.0, E_max * 2.0)
        
        return np.array([dxi_dt, dxi_dot_dt, dE_dt])
    
    def solve(self, t_span=(0, 100), dt=0.1, method='RK45'):
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        
        sol = solve_ivp(
            fun=self.rhs,
            t_span=t_span,
            y0=self.state,
            t_eval=t_eval,
            method=method,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        t = sol.t
        xi = sol.y[0, :]
        xi_dot = sol.y[1, :]
        E = sol.y[2, :]
        

        xi = np.clip(xi, -100.0, 100.0)
        xi_dot = np.clip(xi_dot, -20.0, 20.0)
        E = np.clip(E, 0.0, None)
        
        return t, xi, xi_dot, E
    
    def compute_wave_action(self, t, xi, xi_dot):
        E_kin = 0.5 * xi_dot**2
        E_pot = 0.5 * self.alpha_eff * xi**2 + 0.25 * self.beta * xi**4
        E_total = E_kin + E_pot
        
        action = E_total / (self.omega + 1.0e-12)
        action = np.clip(action, 0.0, 1.0e6)
        return action


def kdv_internal_wave(xi0, c, alpha_kdv, beta_kdv, t_span=(0, 50), nx=256):
    L = 2000.0
    dx = L / nx
    dt = 0.5 * dx / (abs(c) + 1.0)
    nt = int((t_span[1] - t_span[0]) / dt) + 1
    
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(t_span[0], t_span[1], nt)
    

    eta = np.zeros((nt, nx))
    eta[0, :] = xi0 / np.cosh((x - L/2) / 100.0)**2
    

    k = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    ik = 1j * k
    ik3 = (1j * k)**3
    
    for n in range(nt - 1):
        eta_hat = np.fft.fft(eta[n, :])
        

        eta_hat = eta_hat * np.exp(-1j * k * c * dt)
        

        eta_nl = np.real(np.fft.ifft(eta_hat))
        

        d_eta_dx = np.real(np.fft.ifft(ik * np.fft.fft(eta_nl)))
        eta_nl = eta_nl - alpha_kdv * eta_nl * d_eta_dx * dt
        

        eta_hat = np.fft.fft(eta_nl)
        eta_hat = eta_hat * np.exp(-beta_kdv * ik3 * dt)
        
        eta[n+1, :] = np.real(np.fft.ifft(eta_hat))
        

        eta[n+1, :] = np.clip(eta[n+1, :], -50.0, 50.0)
    
    return x, t, eta
