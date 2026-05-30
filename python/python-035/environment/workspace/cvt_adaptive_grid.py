import numpy as np
from constants import TINY




def cvt_1d_lloyd(n_generators, density_func, domain=(0.0, 1.0),
                 max_iter=200, tol=1.0e-10, init_mode="random"):
    a, b = domain
    

    if init_mode == "uniform":
        generators = np.linspace(a, b, n_generators)
    else:
        generators = np.sort(np.random.uniform(a, b, n_generators))
    

    def integrate(f, lo, hi, n=100):
        if hi <= lo:
            return 0.0, 0.0
        h = (hi - lo) / n

        vals = np.array([f(lo + i * h) for i in range(n + 1)])
        total = vals[0] + vals[-1]
        total += 4.0 * np.sum(vals[1:-1:2])
        total += 2.0 * np.sum(vals[2:-1:2])
        return total * h / 3.0, np.sum(vals) * h
    
    def integrate_xf(f, lo, hi, n=100):
        if hi <= lo:
            return 0.0
        h = (hi - lo) / n
        total = 0.0
        for i in range(n + 1):
            x = lo + i * h
            fx = f(x)
            if i == 0 or i == n:
                w = 1.0
            elif i % 2 == 1:
                w = 4.0
            else:
                w = 2.0
            total += w * x * fx
        return total * h / 3.0
    
    energy_history = []
    for iteration in range(max_iter):
        generators = np.sort(generators)
        

        boundaries = np.zeros(n_generators + 1)
        boundaries[0] = a
        boundaries[-1] = b
        for j in range(1, n_generators):
            boundaries[j] = (generators[j - 1] + generators[j]) / 2.0
        

        new_generators = np.zeros(n_generators)
        for j in range(n_generators):
            lo = boundaries[j]
            hi = boundaries[j + 1]
            mass, _ = integrate(density_func, lo, hi, n=80)
            xmass = integrate_xf(density_func, lo, hi, n=80)
            if mass > TINY:
                new_generators[j] = xmass / mass
            else:
                new_generators[j] = (lo + hi) / 2.0
        

        new_generators = np.clip(new_generators, a, b)
        

        avg_motion = np.mean(np.abs(new_generators - generators))
        

        energy = 0.0
        for j in range(n_generators):
            lo = boundaries[j]
            hi = boundaries[j + 1]
            n_sample = 50
            h = (hi - lo) / n_sample
            for k in range(n_sample):
                x = lo + (k + 0.5) * h
                energy += density_func(x) * (x - generators[j]) ** 2 * h
        energy_history.append(energy)
        
        generators = new_generators
        
        if avg_motion < tol:
            return generators, energy, True, iteration + 1
    
    return generators, energy_history[-1], False, max_iter





def cvt_nd_product(n_per_dim, density_funcs, domains, max_iter=100, tol=1.0e-8):
    dim = len(n_per_dim)
    nodes_per_dim = []
    
    for d in range(dim):
        nodes, _, _, _ = cvt_1d_lloyd(
            n_per_dim[d], density_funcs[d], domains[d],
            max_iter=max_iter, tol=tol
        )
        nodes_per_dim.append(nodes)
    

    from itertools import product
    grid_points = []
    grid_weights = []
    
    for coords in product(*nodes_per_dim):
        pt = np.array(coords)

        w = 1.0
        for d in range(dim):
            rho_val = density_funcs[d](coords[d])
            w *= max(rho_val, TINY)
        grid_points.append(pt)
        grid_weights.append(w)
    
    grid_points = np.array(grid_points)
    grid_weights = np.array(grid_weights)
    

    if np.sum(grid_weights) > TINY:
        grid_weights = grid_weights / np.sum(grid_weights)
    
    return grid_points, grid_weights





def make_breit_wigner_density(m0, gamma, domain):
    def density(m):
        if m < domain[0] or m > domain[1]:
            return TINY
        bw = (1.0 / np.pi) * (m0 * gamma) / ((m ** 2 - m0 ** 2) ** 2 + (m0 * gamma) ** 2)
        return bw + TINY
    return density


def make_amplitude_density(amplitude_func, domain, n_sample=200):
    samples = np.linspace(domain[0], domain[1], n_sample)
    vals = np.array([max(amplitude_func(s), 0.0) for s in samples])
    max_val = np.max(vals)
    if max_val < TINY:
        max_val = 1.0
    
    def density(x):
        if x < domain[0] or x > domain[1]:
            return TINY

        idx = np.searchsorted(samples, x)
        if idx <= 0:
            v = vals[0]
        elif idx >= n_sample:
            v = vals[-1]
        else:
            frac = (x - samples[idx - 1]) / (samples[idx] - samples[idx - 1])
            v = vals[idx - 1] + frac * (vals[idx] - vals[idx - 1])
        return v / max_val + TINY
    
    return density





def adaptive_phase_space_grid(n_m1=20, n_m2=20, n_cos=10, n_phi=8):
    from matrix_element import matrix_element_squared_hzz4l
    from constants import M_HIGGS, M_Z, GAMMA_Z
    
    m_min = 0.001
    m_max = M_HIGGS - m_min
    

    rho_m1 = make_breit_wigner_density(M_Z, GAMMA_Z, (m_min, m_max))
    nodes_m1, _, _, _ = cvt_1d_lloyd(n_m1, rho_m1, (m_min, m_max), max_iter=150)
    

    rho_m2 = make_breit_wigner_density(M_Z, GAMMA_Z, (m_min, m_max))
    nodes_m2, _, _, _ = cvt_1d_lloyd(n_m2, rho_m2, (m_min, m_max), max_iter=150)
    

    nodes_cos = np.linspace(-1.0, 1.0, n_cos)
    

    nodes_phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    
    return {
        "m1": nodes_m1,
        "m2": nodes_m2,
        "cos_theta": nodes_cos,
        "phi": nodes_phi,
    }
