
import numpy as np
from scipy.optimize import fsolve


EARTH_RADIUS = 6.371e6
OMEGA = 7.2921159e-5
GRAVITY = 9.81
RHO_AIR = 1.225


def rossby_parameter(latitude):
    phi = np.deg2rad(latitude)
    return 2.0 * OMEGA * np.cos(phi) / EARTH_RADIUS


def coriolis_f(latitude):
    phi = np.deg2rad(latitude)
    return 2.0 * OMEGA * np.sin(phi)


class TyphoonVortexParameters:
    def __init__(self):

        self.p_env = 1010.0
        self.p_min_initial = 990.0
        self.p_c = 870.0
        

        self.r_max_initial = 50.0
        self.r_max_eq = 30.0
        

        self.lambda_1 = 0.15
        self.lambda_2 = 0.08
        self.lambda_3 = 0.05
        

        self.mu_1 = 0.02
        self.mu_2 = 0.5
        

        self.sst_threshold = 299.15
        

        self.x0 = 125.0
        self.y0 = 18.0
        

        self.env_flow_factor = 0.7
        self.beta_drift_factor = 0.3


class TyphoonVortexODE:
    def __init__(self, params=None):
        if params is None:
            params = TyphoonVortexParameters()
        self.params = params
        self.time_history = []
        self.state_history = []
    
    def environment_flow(self, x, y, t):

        U0 = -3.0
        V0 = 1.0
        

        omega_mjo = 2.0 * np.pi / (45.0 * 86400.0)
        u_t = 2.0 * np.sin(omega_mjo * t)
        v_t = 1.5 * np.cos(omega_mjo * t)
        

        k_wave = 2.0 * np.pi / 30.0
        l_wave = 2.0 * np.pi / 20.0
        u_x = 1.5 * np.sin(k_wave * (x - 120.0))
        v_y = 1.0 * np.cos(l_wave * (y - 15.0))
        
        u_env = U0 + u_t + u_x
        v_env = V0 + v_t + v_y
        
        return u_env, v_env
    
    def beta_drift_velocity(self, p_min, r_max, latitude):
        f = coriolis_f(latitude)
        beta = rossby_parameter(latitude)
        

        f_safe = max(abs(f), 1e-8)
        

        dp = self.params.p_env - p_min
        dp = max(dp, 0.0)
        dp_pa = dp * 100.0
        

        v_max = np.sqrt(dp_pa / RHO_AIR) * 0.5
        

        u_beta = -beta * v_max**2 / (2.0 * f_safe**2)
        v_beta = beta * v_max**2 / (2.0 * f_safe**2) * 0.3
        

        u_beta = np.clip(u_beta, -5.0, 5.0)
        v_beta = np.clip(v_beta, -5.0, 5.0)
        
        return u_beta, v_beta
    
    def rhs(self, t, state):

        x, y, p_min, r_max = state
        params = self.params
        

        dxdt = 0.0
        dydt = 0.0
        

        dpdt = 0.0
        

        drdt = 0.0

        
        return np.array([dxdt, dydt, dpdt, drdt])
    
    def implicit_rk2_step(self, t, state, dt):

        rhs_val = self.rhs(t, state)
        y_guess = state + 0.5 * dt * rhs_val
        


        rhs_mid = self.rhs(t + 0.5 * dt, y_guess)
        state_explicit = state + dt * rhs_mid
        

        def residual(y_mid):
            return y_mid - state - 0.5 * dt * self.rhs(t + 0.5 * dt, y_mid)
        
        try:
            y_mid_imp, infodict, ier, mesg = fsolve(
                residual, y_guess, full_output=True, xtol=1e-8, maxfev=100
            )
            if ier == 1:

                state_new = 2.0 * y_mid_imp - state
            else:
                state_new = state_explicit
        except Exception:
            state_new = state_explicit
        

        state_new[2] = np.clip(state_new[2], self.params.p_c, self.params.p_env)
        state_new[3] = np.clip(state_new[3], 5.0, 200.0)
        
        return state_new
    
    def solve(self, t_span=(0.0, 72.0), n_steps=720):
        t0, tf = t_span
        dt_hours = (tf - t0) / n_steps
        dt_seconds = dt_hours * 3600.0
        
        t_array = np.zeros(n_steps + 1)
        states = np.zeros((n_steps + 1, 4))
        

        states[0, 0] = self.params.x0
        states[0, 1] = self.params.y0
        states[0, 2] = self.params.p_min_initial
        states[0, 3] = self.params.r_max_initial
        
        t_array[0] = t0
        
        for i in range(n_steps):

            t_sec = t_array[i] * 3600.0
            states[i + 1] = self.implicit_rk2_step(t_sec, states[i], dt_seconds)
            t_array[i + 1] = t_array[i] + dt_hours
        
        self.time_history = t_array
        self.state_history = states
        
        return t_array, states



params = TyphoonVortexParameters()
