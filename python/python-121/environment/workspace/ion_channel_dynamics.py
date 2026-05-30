
import numpy as np
from math import exp, sqrt, pi, cos, sin






def generate_colored_noise(n_samples, q_d, alpha):
    if n_samples <= 0 or q_d < 0 or alpha < 0:
        return np.zeros(n_samples)
    
    q_d_sqrt = sqrt(q_d)
    

    hfa = np.zeros(2 * n_samples)
    hfa[0] = 1.0
    for i in range(1, n_samples):
        hfa[i] = hfa[i - 1] * (0.5 * alpha + (i - 2)) / (i - 1)
    

    wfa = np.zeros(2 * n_samples)
    wfa[:n_samples] = np.random.randn(n_samples) * q_d_sqrt
    

    H = np.fft.fft(hfa)
    W = np.fft.fft(wfa)
    X = np.fft.ifft(H * W)
    

    noise = np.real(X[:n_samples])
    
    return noise


def generate_ion_channel_noise(n_channels, dt, T, D_ion=0.01, alpha=1.5):
    n_steps = int(T / dt) + 1
    noise_array = np.zeros((n_steps, n_channels))
    
    for c in range(n_channels):

        q_d = D_ion * dt
        noise = generate_colored_noise(n_steps, q_d, alpha)
        noise_array[:, c] = noise
    
    return noise_array






def gate_alpha_beta(v, gate_type):
    v = float(v)
    
    if gate_type == 'm':
        alpha = 0.32 * (v + 47.13) / (1.0 - exp(-0.1 * (v + 47.13)))
        if abs(v + 47.13) < 1e-6:
            alpha = 3.2
        beta = 0.08 * exp(-v / 11.0)
    
    elif gate_type == 'h':
        alpha = 0.135 * exp(-(v + 80.0) / 6.8)
        beta = 3.56 / (1.0 + exp(-0.1 * (v + 40.0))) + 0.0075
    
    elif gate_type == 'j':
        alpha = (-1.2714e5 * exp(0.2444 * v) - 3.474e-5 * exp(-0.04391 * v)) * (v + 37.78) / (1.0 + exp(0.311 * (v + 79.23)))
        if abs(v + 37.78) < 1e-3:
            alpha = 0.0
        beta = 0.1212 * exp(-0.01052 * v) / (1.0 + exp(-0.1378 * (v + 40.14)))
    
    elif gate_type == 'd':
        alpha = 0.095 * exp(-(v - 5.0) / 13.0) / (1.0 + exp(-(v - 5.0) / 13.0))
        beta = 0.07 * exp(-(v + 44.0) / 20.0) / (1.0 + exp((v + 44.0) / 20.0))
    
    elif gate_type == 'f':
        alpha = 0.012 * exp(-(v + 28.0) / 30.0) / (1.0 + exp((v + 28.0) / 30.0))
        beta = 0.0065 * exp(-(v + 30.0) / 40.0) / (1.0 + exp(-(v + 30.0) / 40.0))
    
    elif gate_type == 'x':
        alpha = 0.0005 * exp(0.083 * (v + 50.0)) / (1.0 + exp(0.057 * (v + 50.0)))
        beta = 0.0013 * exp(-0.06 * (v + 20.0)) / (1.0 + exp(-0.04 * (v + 20.0)))
    
    else:
        alpha = 0.0
        beta = 1.0
    

    alpha = max(0.0, alpha)
    beta = max(0.0, beta)
    
    return alpha, beta


def update_gate(gate, v, gate_type, dt):
    alpha, beta = gate_alpha_beta(v, gate_type)
    tau = 1.0 / (alpha + beta + 1e-12)
    x_inf = alpha / (alpha + beta + 1e-12)
    

    gate_new = x_inf + (gate - x_inf) * exp(-dt / tau)
    

    gate_new = max(0.0, min(1.0, gate_new))
    
    return gate_new






def compute_ionic_currents(v, gates, ion_noise=None):

    E_Na = 54.4
    E_Ca = 130.0
    E_K = -87.0
    

    G_Na = 23.0
    G_Ca = 0.09
    G_K = 0.282
    G_K1 = 0.6047
    G_Kp = 0.0183
    G_b = 0.03921
    
    m = gates.get('m', 0.0)
    h = gates.get('h', 0.0)
    j = gates.get('j', 0.0)
    d = gates.get('d', 0.0)
    f = gates.get('f', 0.0)
    x = gates.get('x', 0.0)
    

    I_Na = G_Na * (m ** 3) * h * j * (v - E_Na)
    

    I_Ca = G_Ca * d * f * (v - E_Ca)
    


    xi = 1.0 / (1.0 + exp((v - 56.26) / 32.1))
    I_K = G_K * x * xi * (v - E_K)
    

    alpha_K1 = 1.02 / (1.0 + exp(0.2385 * (v - E_K - 59.215)))
    beta_K1 = (0.49124 * exp(0.08032 * (v - E_K + 5.476)) + exp(0.06175 * (v - E_K - 594.31))) / (1.0 + exp(-0.5143 * (v - E_K + 4.753)))
    x_K1 = alpha_K1 / (alpha_K1 + beta_K1 + 1e-12)
    I_K1 = G_K1 * x_K1 * (v - E_K)
    

    x_Kp = 1.0 / (1.0 + exp((7.488 - v) / 5.98))
    I_Kp = G_Kp * x_Kp * (v - E_K)
    

    E_b = -59.87
    I_b = G_b * (v - E_b)
    

    noise_factor = 1.0
    if ion_noise is not None:
        noise_factor = 1.0 + 0.05 * ion_noise
    
    I_total = (I_Na + I_Ca + I_K + I_K1 + I_Kp + I_b) * noise_factor
    
    currents = {
        'I_Na': I_Na,
        'I_Ca': I_Ca,
        'I_K': I_K,
        'I_K1': I_K1,
        'I_Kp': I_Kp,
        'I_b': I_b,
        'I_total': I_total
    }
    
    return currents, I_total






def aliev_panfilov_reaction(u, v, a=0.1, k=8.0, mu1=0.2, mu2=0.3, eps=0.002):






    raise NotImplementedError("Hole 1: aliev_panfilov_reaction 待实现")


def single_cell_ap_model(t_max=500.0, dt=0.01, stim_period=300.0):
    C_m = 1.0
    n_steps = int(t_max / dt) + 1
    
    t = np.linspace(0, t_max, n_steps)
    v = np.zeros(n_steps)
    

    v[0] = -86.2
    gates = {
        'm': 0.0,
        'h': 1.0,
        'j': 1.0,
        'd': 0.0,
        'f': 1.0,
        'x': 0.0
    }
    

    gate_history = {k: np.zeros(n_steps) for k in gates}
    for k in gates:
        gate_history[k][0] = gates[k]
    
    for i in range(1, n_steps):

        I_stim = 0.0
        if (t[i] % stim_period) < 2.0:
            I_stim = -80.0
        

        for g_name in gates:
            gates[g_name] = update_gate(gates[g_name], v[i - 1], g_name, dt)
            gate_history[g_name][i] = gates[g_name]
        

        _, I_ion = compute_ionic_currents(v[i - 1], gates)
        

        dv = (-I_ion + I_stim) / C_m
        v[i] = v[i - 1] + dt * dv
        

        v[i] = max(-100.0, min(60.0, v[i]))
    
    return t, v, gate_history






def squircle_ode_integrate(y0, t_span, s=4.0, n_steps=1000):
    t0, t1 = t_span
    dt = (t1 - t0) / n_steps
    
    t = np.linspace(t0, t1, n_steps + 1)
    u = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    H = np.zeros(n_steps + 1)
    
    u[0], v[0] = y0
    H[0] = (abs(u[0]) ** s + abs(v[0]) ** s) / s
    
    for i in range(n_steps):


        ui, vi = u[i], v[i]
        

        u_pow = np.sign(ui) * (abs(ui) ** (s - 1)) if ui != 0 else 0.0
        v_pow = np.sign(vi) * (abs(vi) ** (s - 1)) if vi != 0 else 0.0
        
        u[i + 1] = ui + dt * v_pow
        v[i + 1] = vi - dt * u_pow
        
        H[i + 1] = (abs(u[i + 1]) ** s + abs(v[i + 1]) ** s) / s
    
    return t, u, v, H
