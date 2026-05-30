
import numpy as np


FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
M_C = 12.011e-3


def corrosion_current_density(E, E_corr_0=0.207, j0_corr=1e-8, 
                               alpha_corr=0.5, T=353.15):
    eta_corr = E - E_corr_0
    
    if eta_corr <= 0:
        return 0.0
    
    exponent = alpha_corr * FARADAY * eta_corr / (GAS_CONSTANT * T)
    exponent = np.clip(exponent, -50, 50)
    
    j_corr = j0_corr * np.exp(exponent)
    

    max_j = 1e4
    return float(np.clip(j_corr, 0.0, max_j))


def carbon_mass_loss_rate(j_corr, A_carbon):
    if j_corr < 0 or A_carbon < 0:
        return 0.0
    
    rate = -M_C * j_corr * A_carbon / (4.0 * FARADAY)
    return rate


def numerical_flux_godunov(u_left, u_right, v=0.0):
    if v >= 0:
        return v * u_left
    else:
        return v * u_right


def solve_corrosion_propagation(u0, nx, nt, dx, dt, v_front, k_corr, theta_pore,
                                 method='godunov'):
    if nx < 2 or nt < 0 or dx <= 0 or dt <= 0:
        raise ValueError("网格参数无效")
    

    cfl = abs(v_front) * dt / dx
    if cfl > 1.0:

        dt = 0.9 * dx / max(abs(v_front), 1e-10)
        print(f"警告: CFL={cfl:.2f}>1, 已自动调整 dt={dt:.4e}")
    
    U = np.zeros((nt + 1, nx))
    u = np.array(u0, dtype=float)
    
    if len(u) != nx:
        raise ValueError("u0 长度必须与 nx 一致")
    
    U[0, :] = u
    
    for n_step in range(nt):
        unew = np.zeros(nx)
        
        if method == 'godunov':

            unew[0] = u[0] - dt * (numerical_flux_godunov(u[0], u[1], v_front) 
                                    - numerical_flux_godunov(u[-1], u[0], v_front)) / dx \
                        - dt * k_corr * u[0] * theta_pore
            
            for i in range(1, nx - 1):
                flux_right = numerical_flux_godunov(u[i], u[i + 1], v_front)
                flux_left = numerical_flux_godunov(u[i - 1], u[i], v_front)
                unew[i] = u[i] - dt * (flux_right - flux_left) / dx \
                           - dt * k_corr * u[i] * theta_pore
            
            unew[nx - 1] = u[nx - 1] - dt * (numerical_flux_godunov(u[nx - 1], u[0], v_front)
                                                - numerical_flux_godunov(u[nx - 2], u[nx - 1], v_front)) / dx \
                            - dt * k_corr * u[nx - 1] * theta_pore
        
        elif method == 'lax_wendroff':

            unew[0] = 0.5 * (u[1] + u[-1]) - 0.5 * dt / dx * (
                        v_front * u[1] - v_front * u[-1]) \
                        - dt * k_corr * u[0] * theta_pore
            
            for i in range(1, nx - 1):
                unew[i] = u[i] - 0.5 * dt / dx * (v_front * u[i + 1] - v_front * u[i - 1]) \
                           + 0.5 * (dt / dx) ** 2 * v_front ** 2 * (u[i + 1] - 2 * u[i] + u[i - 1]) \
                           - dt * k_corr * u[i] * theta_pore
            
            unew[nx - 1] = u[nx - 1] - 0.5 * dt / dx * (v_front * u[0] - v_front * u[nx - 2]) \
                            + 0.5 * (dt / dx) ** 2 * v_front ** 2 * (u[0] - 2 * u[nx - 1] + u[nx - 2]) \
                            - dt * k_corr * u[nx - 1] * theta_pore
        
        elif method == 'maccormack':

            us = np.zeros(nx)
            for i in range(nx - 1):
                us[i] = u[i] - dt / dx * (v_front * u[i + 1] - v_front * u[i]) \
                         - dt * k_corr * u[i] * theta_pore
            us[nx - 1] = u[nx - 1] - dt / dx * (v_front * u[0] - v_front * u[nx - 1]) \
                          - dt * k_corr * u[nx - 1] * theta_pore
            
            unew[0] = 0.5 * (u[0] + us[0]) - 0.5 * dt / dx * (
                        v_front * us[0] - v_front * us[-1]) \
                        - dt * k_corr * us[0] * theta_pore
            
            for i in range(1, nx):
                unew[i] = 0.5 * (u[i] + us[i]) - 0.5 * dt / dx * (
                            v_front * us[i] - v_front * us[i - 1]) \
                            - dt * k_corr * us[i] * theta_pore
        
        else:
            raise ValueError(f"未知方法: {method}")
        

        unew = np.clip(unew, 0.0, np.max(u0) * 1.5)
        
        u = unew
        U[n_step + 1, :] = u
    
    return U


def corrosion_front_velocity(E, T=353.15):
    A_prefactor = 1e-12
    Ea = 80000
    alpha = 0.3
    
    v = A_prefactor * np.exp(-Ea / (GAS_CONSTANT * T)) \
        * np.exp(alpha * FARADAY * E / (GAS_CONSTANT * T))
    
    return float(np.clip(v, 0.0, 1e-6))


def structural_integrity_loss(S_c_current, S_c_initial):
    if S_c_initial <= 0:
        return 0.0
    
    loss = 1.0 - S_c_current / S_c_initial
    return float(np.clip(loss, 0.0, 1.0))


if __name__ == "__main__":
    nx = 51
    L = 10e-6
    dx = L / (nx - 1)
    u0 = np.ones(nx) * 200.0
    
    E = 1.0
    v = corrosion_front_velocity(E)
    k = 1e-5
    theta = 0.4
    dt = 0.5 * dx / max(v, 1e-10)
    nt = 100
    
    U = solve_corrosion_propagation(u0, nx, nt, dx, dt, v, k, theta, method='godunov')
    print(f"碳腐蚀传播: 初始均值={np.mean(U[0]):.2f}, 最终均值={np.mean(U[-1]):.2f} m^2/g")
