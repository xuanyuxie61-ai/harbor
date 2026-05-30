
import numpy as np






def shifted_legendre_polynomial(x, n_max):
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[0] if x.ndim > 0 else 1
    if n_max < 0:
        return np.zeros((m, 0))
    
    v = np.zeros((m, n_max + 1), dtype=np.float64)
    v[:, 0] = 1.0
    if n_max < 1:
        return v
    
    v[:, 1] = 2.0 * x - 1.0
    for i in range(2, n_max + 1):
        v[:, i] = ((2 * i - 1) * (2.0 * x - 1.0) * v[:, i - 1] - (i - 1) * v[:, i - 2]) / i
    
    return v


def spherical_harmonic_s2(l, m, theta, phi):
    if l < 2:
        raise ValueError("引力波球谐函数要求 l >= 2")
    if np.abs(m) > l:
        raise ValueError("|m| <= l 必须满足")
    

    legendre_vals = shifted_legendre_polynomial(np.cos(theta), l)
    P_l = legendre_vals[:, l] if legendre_vals.ndim > 1 else legendre_vals[l]
    
    real_part = P_l * np.cos(m * phi)
    imag_part = P_l * np.sin(m * phi)
    
    norm_factor = np.sqrt((2 * l + 1) / (4 * np.pi))
    return norm_factor * (real_part + 1j * imag_part)






def chebyshev2_rule(n, a=-1.0, b=1.0):
    if n < 1:
        raise ValueError("求积阶数 n 必须 >= 1")
    
    k = np.arange(1, n + 1)
    x_std = np.cos(k * np.pi / (n + 1))
    w_std = (np.pi / (n + 1)) * np.sin(k * np.pi / (n + 1))**2
    

    x = 0.5 * (b - a) * x_std + 0.5 * (b + a)
    w = 0.5 * (b - a) * w_std
    
    return x, w


def waveform_inner_product_chebyshev(h1_func, h2_func, f_min, f_max, n=63, Sn_func=None):
    if f_min >= f_max:
        raise ValueError("f_min 必须小于 f_max")
    
    x, w = chebyshev2_rule(n, a=f_min, b=f_max)
    
    if Sn_func is None:
        Sn_func = lambda f: 1.0
    
    integrand = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        fi = x[i]
        h1_val = h1_func(fi)
        h2_val = h2_func(fi)
        Sn_val = Sn_func(fi)
        if np.abs(Sn_val) < 1e-300:
            Sn_val = 1e-300
        integrand[i] = np.conj(h1_val) * h2_val / Sn_val
    
    inner_prod = 4.0 * np.real(np.sum(w * integrand))
    return inner_prod






def _patterson_abscissas_weights(order):
    valid_orders = [1, 3, 7, 15, 31, 63, 127, 255, 511]
    if order not in valid_orders:
        raise ValueError(f"Patterson 规则不支持阶数 {order}，可用: {valid_orders}")
    
    if order == 1:

        x = np.array([0.0])
        w = np.array([2.0])
        return x, w
    
    if order == 3:
        x = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
        return x, w
    



    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(order)
    return x, w


def patterson_quadrature(func, a, b, order=31):
    if a >= b:
        raise ValueError("积分下限必须小于上限")
    
    x_std, w = _patterson_abscissas_weights(order)
    

    x = 0.5 * (b - a) * x_std + 0.5 * (a + b)
    w_scaled = 0.5 * (b - a) * w
    
    fx = np.array([func(xi) for xi in x], dtype=np.float64)
    

    valid = np.isfinite(fx)
    if not np.all(valid):
        fx[~valid] = 0.0
    
    return np.sum(w_scaled * fx)


def adaptive_patterson_integral(func, a, b, tol=1e-10, max_level=5):
    orders = [1, 3, 7, 15, 31, 63]
    
    def recursive_integrate(f, left, right, level):
        if level >= max_level or level >= len(orders):
            return patterson_quadrature(f, left, right, orders[-1])
        
        order_low = orders[level]
        order_high = orders[min(level + 1, len(orders) - 1)]
        
        I_low = patterson_quadrature(f, left, right, order_low)
        I_high = patterson_quadrature(f, left, right, order_high)
        
        err = np.abs(I_high - I_low)
        if err < tol:
            return I_high
        
        mid = 0.5 * (left + right)
        I_left = recursive_integrate(f, left, mid, level + 1)
        I_right = recursive_integrate(f, mid, right, level + 1)
        return I_left + I_right
    
    return recursive_integrate(func, a, b, 0)






def post_newtonian_waveform(t, m1, m2, D_L, inclination=0.0, phi_c=0.0, t_c=None):
    t = np.asarray(t, dtype=np.float64)
    if t_c is None:
        t_c = t[-1]
    
    M = m1 + m2
    eta = m1 * m2 / (M**2)
    

    eta = np.clip(eta, 1e-6, 0.25)
    









    raise NotImplementedError("Hole 1: post_newtonian_waveform 核心计算待补全")


def ringdown_waveform(t, M, a, qnm_freqs, t_merge, amplitudes=None):
    t = np.asarray(t, dtype=np.float64)
    h = np.zeros_like(t, dtype=np.complex128)
    
    if amplitudes is None:
        amplitudes = {}
    
    for key, omega in qnm_freqs.items():
        l, m, n = key
        A = amplitudes.get(key, 1.0 / (l + 1))

        mask = t >= t_merge
        decay = np.exp(-1j * omega * (t - t_merge))
        decay[~mask] = 0.0
        h += A * decay
    
    return np.real(h), np.imag(h)


def full_imrphenom_waveform(t, m1, m2, D_L, inclination=0.0, M_final=None, a_final=None, qnm_freqs=None):
    t = np.asarray(t, dtype=np.float64)
    M = m1 + m2
    
    if M_final is None:
        M_final = M * 0.95
    if a_final is None:
        a_final = 0.0
    
    t_c = t[-1] * 0.7
    Delta_t = M * 10.0
    

    h_p_insp, h_c_insp, _, _ = post_newtonian_waveform(t, m1, m2, D_L, inclination, t_c=t_c)
    

    if qnm_freqs is None:
        from teukolsky import solve_qnm_frequencies
        qnm_freqs = solve_qnm_frequencies(l_max=3, n_overtones=1, M=M_final, a=a_final)
    
    h_p_ring, h_c_ring = ringdown_waveform(t, M_final, a_final, qnm_freqs, t_c)
    

    w_insp = 0.5 * (1.0 + np.tanh((t_c - t) / Delta_t))
    w_ring = 0.5 * (1.0 + np.tanh((t - t_c) / Delta_t))
    
    h_plus = h_p_insp * w_insp + h_p_ring * w_ring
    h_cross = h_c_insp * w_insp + h_c_ring * w_ring
    
    return h_plus, h_cross






def matched_filter_snr(template_func, signal_func, f_min, f_max, Sn_func, n_quad=127):
    def integrand(f):
        h_tilde = template_func(f)
        Sn = Sn_func(f)
        if np.abs(Sn) < 1e-300:
            Sn = 1e-300
        return np.abs(h_tilde)**2 / Sn
    
    rho_sq = 4.0 * adaptive_patterson_integral(integrand, f_min, f_max, tol=1e-8)
    rho = np.sqrt(max(rho_sq, 0.0))
    return rho
