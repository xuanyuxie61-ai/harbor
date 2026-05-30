
import numpy as np


def random_phase_superposition(n_modes=20, z=None, t=0.0,
                                N=0.01, f=1.0e-4):
    if z is None:
        z = np.linspace(-200, 0, 101)
    
    z = np.asarray(z)
    H = np.max(z) - np.min(z)
    

    m = np.arange(1, n_modes + 1)
    A_m = 0.5 / m
    

    theta_m = np.random.uniform(0, 2.0 * np.pi, n_modes)
    phi_m = np.random.uniform(0, 2.0 * np.pi, n_modes)
    

    k_h = 2.0 * np.pi / 1000.0
    k_m = m * np.pi / H




    raise NotImplementedError("待实现: 内波模态频率计算")
    

    u = np.zeros_like(z)
    for i in range(n_modes):
        u += A_m[i] * np.sin(k_m[i] * (z - np.min(z))) * \
             np.cos(omega_m[i] * t + theta_m[i])
    

    shear = np.zeros_like(z)
    for i in range(n_modes):
        shear += A_m[i] * k_m[i] * np.cos(k_m[i] * (z - np.min(z))) * \
                 np.cos(omega_m[i] * t + theta_m[i])
    

    N2 = N**2
    shear_sq = shear**2
    Ri = np.where(shear_sq > 1.0e-12, N2 / shear_sq, 1.0e6)
    Ri = np.clip(Ri, 0.0, 100.0)
    
    return u, shear, Ri


def monte_carlo_breaking_probability(n_realizations=1000,
                                      n_modes=20,
                                      n_depths=101,
                                      N=0.01):
    z = np.linspace(-200, 0, n_depths)
    n_break = np.zeros(n_depths)
    
    for _ in range(n_realizations):
        _, shear, Ri = random_phase_superposition(n_modes, z, t=0.0, N=N)
        

        breaking_mask = Ri < 0.25
        n_break += breaking_mask.astype(float)
    
    P_break_z = n_break / n_realizations
    P_break = np.mean(P_break_z)
    
    return P_break, P_break_z, z


def energy_cascade_simulation(E0=1.0, n_steps=1000,
                               growth_factor=1.05,
                               dissipation_factor=0.97):
    E_history = np.zeros(n_steps)
    E = E0
    E_history[0] = E
    

    E_critical = 5.0 * E0
    breaking_events = []
    

    p_growth = 0.45
    
    for n in range(1, n_steps):
        coin = np.random.rand()
        
        if coin < p_growth:
            E *= growth_factor
        else:
            E *= dissipation_factor
        

        E = max(E, 1.0e-6)
        
        E_history[n] = E
        

        if E > E_critical:
            breaking_events.append(n)

            E = E0 * 0.5
    
    return E_history, breaking_events


def mixing_patch_ifs(n_points=5000, n_iterations=10):

    A = [
        np.array([[0.5, 0.0], [0.0, 0.5]]),
        np.array([[0.5, 0.0], [0.0, 0.5]]),
        np.array([[0.4, 0.1], [-0.1, 0.4]]),
        np.array([[0.3, -0.2], [0.2, 0.3]]),
    ]
    
    b = [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.25, 0.5]),
        np.array([0.6, 0.4]),
    ]
    
    p = [0.3, 0.3, 0.25, 0.15]
    

    point = np.random.rand(2)
    

    for _ in range(n_iterations):
        idx = np.random.choice(4, p=p)
        point = A[idx] @ point + b[idx]
    

    points = np.zeros((n_points, 2))
    intensities = np.zeros(n_points)
    
    for i in range(n_points):
        idx = np.random.choice(4, p=p)
        point = A[idx] @ point + b[idx]
        points[i, :] = point
        

        intensities[i] = 0.5 + 0.5 * idx / 3.0
    

    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0)
    
    return points[:, 0], points[:, 1], intensities
