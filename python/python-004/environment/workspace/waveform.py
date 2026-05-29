"""
waveform.py
引力波波形生成模块：基于后牛顿近似与数值相对论拟合的多极展开。

融合种子项目:
- 666_legendre_shifted_polynomial: 移位Legendre多项式 → 球谐展开与多极矩
- 167_chebyshev2_rule: Gauss-Chebyshev Type 2求积 → 波形内积计算
- 851_patterson_rule: Gauss-Patterson嵌套求积 → 高精度辐射反作用积分

核心公式:
1. 引力波多极展开:
   h_+ - i h_× = (1/D_L) Σ_{l=2}^∞ Σ_{m=-l}^l H_{lm}(t) _{-2}Y_{lm}(ι, φ)
   
   其中 H_{lm}(t) 为波形模式函数，_{-2}Y_{lm} 为自旋权重-2的球谐函数。

2. 后牛顿 inspiral 波形 (2PN 近似):
   H_{22}(t) = η M / D_L * (MΩ)^{2/3} * exp(-2iΦ(t))
   
   其中:
     Φ(t) = φ_c - (1/η) * [ (t_c - t) / (5M) ]^{3/8}
     Ω = dΦ/dt = (1/8M) * [η / (t_c - t)]^{3/8}
     η = μ/M = m1*m2 / (m1+m2)^2  (对称质量比)

3. Chebyshev-Gauss 求积 (Type 2):
   ∫_{-1}^1 f(x) sqrt(1-x^2) dx ≈ Σ_{k=1}^n w_k f(x_k)
   节点: x_k = cos(kπ / (n+1))
   权重: w_k = π/(n+1) * sin^2(kπ / (n+1))

4. Gauss-Patterson 嵌套求积:
   规则阶数为 1, 3, 7, 15, 31, 63, 127, 255, 511
   每级规则嵌套前一级节点，适合自适应积分。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 移位 Legendre 多项式 (源自 666_legendre_shifted_polynomial)
# ---------------------------------------------------------------------------

def shifted_legendre_polynomial(x, n_max):
    """
    计算移位 Legendre 多项式 P_n^*(x) = P_n(2x-1)，定义在 [0,1] 上。
    
    递推关系:
        P_0^*(x) = 1
        P_1^*(x) = 2x - 1
        (n+1) P_{n+1}^*(x) = (2n+1)(2x-1) P_n^*(x) - n P_{n-1}^*(x)
    
    在引力波理论中，用于多极矩展开:
        I_{lm} = ∫ ρ(r) r^l P_l^m(cosθ) d^3x
    """
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
    """
    自旋权重 s=-2 的球谐函数近似。
    用于引力波辐射模式在天空中的角分布。
    
    简化计算: 使用实函数近似
    """
    if l < 2:
        raise ValueError("引力波球谐函数要求 l >= 2")
    if np.abs(m) > l:
        raise ValueError("|m| <= l 必须满足")
    
    # 简化的实部近似
    legendre_vals = shifted_legendre_polynomial(np.cos(theta), l)
    P_l = legendre_vals[:, l] if legendre_vals.ndim > 1 else legendre_vals[l]
    
    real_part = P_l * np.cos(m * phi)
    imag_part = P_l * np.sin(m * phi)
    
    norm_factor = np.sqrt((2 * l + 1) / (4 * np.pi))
    return norm_factor * (real_part + 1j * imag_part)


# ---------------------------------------------------------------------------
# Gauss-Chebyshev Type 2 求积 (源自 167_chebyshev2_rule)
# ---------------------------------------------------------------------------

def chebyshev2_rule(n, a=-1.0, b=1.0):
    """
    生成 Gauss-Chebyshev Type 2 求积规则。
    
    积分公式:
        ∫_a^b f(x) sqrt((x-a)(b-x)) dx ≈ Σ_{k=1}^n w_k f(x_k)
    
    节点和权重 (标准区间 [-1,1]):
        x_k = cos(kπ / (n+1))
        w_k = π/(n+1) * sin^2(kπ / (n+1))
    
    引力波应用: 计算波形内积 (匹配滤波)
        ⟨h1|h2⟩ = 4 Re ∫_0^∞ H1*(f) H2(f) / S_n(f) df
    """
    if n < 1:
        raise ValueError("求积阶数 n 必须 >= 1")
    
    k = np.arange(1, n + 1)
    x_std = np.cos(k * np.pi / (n + 1))
    w_std = (np.pi / (n + 1)) * np.sin(k * np.pi / (n + 1))**2
    
    # 线性变换到 [a, b]
    x = 0.5 * (b - a) * x_std + 0.5 * (b + a)
    w = 0.5 * (b - a) * w_std
    
    return x, w


def waveform_inner_product_chebyshev(h1_func, h2_func, f_min, f_max, n=63, Sn_func=None):
    """
    使用 Chebyshev-Gauss 求积计算引力波内积。
    
    公式:
        ⟨h1|h2⟩ = 4 Re ∫_{f_min}^{f_max} h1*(f) h2(f) / S_n(f) df
    
    其中 S_n(f) 为噪声功率谱密度。
    """
    if f_min >= f_max:
        raise ValueError("f_min 必须小于 f_max")
    
    x, w = chebyshev2_rule(n, a=f_min, b=f_max)
    
    if Sn_func is None:
        Sn_func = lambda f: 1.0  # 白噪声近似
    
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


# ---------------------------------------------------------------------------
# Gauss-Patterson 嵌套求积 (源自 851_patterson_rule)
# ---------------------------------------------------------------------------

def _patterson_abscissas_weights(order):
    """
    返回 Gauss-Patterson 规则的节点和权重（标准区间 [-1,1]）。
    支持的阶数: 1, 3, 7, 15, 31, 63, 127, 255, 511
    """
    valid_orders = [1, 3, 7, 15, 31, 63, 127, 255, 511]
    if order not in valid_orders:
        raise ValueError(f"Patterson 规则不支持阶数 {order}，可用: {valid_orders}")
    
    if order == 1:
        # Gauss-Legendre 1点规则
        x = np.array([0.0])
        w = np.array([2.0])
        return x, w
    
    if order == 3:
        x = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
        w = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
        return x, w
    
    # 对于更高阶，使用近似值（基于文献中的 Patterson 规则数据）
    # 这里使用 Kronrod 扩展的近似
    # 由于完整数据量大，我们使用 Legendre-Gauss 节点近似
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(order)
    return x, w


def patterson_quadrature(func, a, b, order=31):
    """
    使用 Gauss-Patterson 规则计算定积分。
    
    公式:
        ∫_a^b f(x) dx ≈ (b-a)/2 * Σ_{k=1}^n w_k f((b-a)/2 * x_k + (a+b)/2)
    """
    if a >= b:
        raise ValueError("积分下限必须小于上限")
    
    x_std, w = _patterson_abscissas_weights(order)
    
    # 变换到 [a, b]
    x = 0.5 * (b - a) * x_std + 0.5 * (a + b)
    w_scaled = 0.5 * (b - a) * w
    
    fx = np.array([func(xi) for xi in x], dtype=np.float64)
    
    # 边界处理: 检查 NaN/Inf
    valid = np.isfinite(fx)
    if not np.all(valid):
        fx[~valid] = 0.0
    
    return np.sum(w_scaled * fx)


def adaptive_patterson_integral(func, a, b, tol=1e-10, max_level=5):
    """
    自适应 Gauss-Patterson 积分。
    使用嵌套性质逐级细化，直到误差 < tol。
    
    在引力波中用于高精度计算辐射反作用力:
        F_{RR}^i = (dE/dt) / v^i
    """
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


# ---------------------------------------------------------------------------
# 引力波波形生成
# ---------------------------------------------------------------------------

def post_newtonian_waveform(t, m1, m2, D_L, inclination=0.0, phi_c=0.0, t_c=None):
    """
    生成双黑洞并合的简化后牛顿 inspiral 波形。
    
    物理参数:
        m1, m2: 黑洞质量 (太阳质量)
        D_L: 光度距离 (Mpc)
        inclination: 轨道倾角 ι
        phi_c: 并合相位
        t_c: 并合时间
    
    公式 (2PN):
        chirp 质量: M_c = (m1*m2)^{3/5} / (m1+m2)^{1/5}
        总质量: M = m1 + m2
        对称质量比: η = m1*m2 / M^2
        
        引力波应变:
            h_+(t) = (4/D_L) * (G M_c / c^2)^{5/3} * (π f_{GW} D_L / c)^{2/3} * (1+cos^2ι)/2 * cos(2Φ)
            h_×(t) = (4/D_L) * (G M_c / c^2)^{5/3} * (π f_{GW} D_L / c)^{2/3} * cosι * sin(2Φ)
        
        这里使用几何单位制 G = c = 1 简化:
            h_+(t) ≈ (M η / D_L) * (M Ω)^{2/3} * (1+cos^2ι)/2 * cos(2Φ)
    """
    t = np.asarray(t, dtype=np.float64)
    if t_c is None:
        t_c = t[-1]
    
    M = m1 + m2
    eta = m1 * m2 / (M**2)
    
    # 确保 eta 在物理范围内 (0, 0.25]
    eta = np.clip(eta, 1e-6, 0.25)
    
    # TODO: 请补全此处的后牛顿波形核心计算代码
    # 需要计算: chirp时间尺度 tau, 轨道相位 Phi, 轨道角频率 Omega, 振幅 amp
    # 以及最终极化波形 h_plus 和 h_cross
    # 关键物理: 后牛顿展开参数 x = (eta * tau / (5M))^(-1/8)
    #          Phi = phi_c - x
    #          Omega = (1/(8M)) * (eta/tau)^(3/8)
    #          amp = (eta * M / D_L) * (M * Omega)^(2/3)
    #          h_plus = amp * (1+cos^2(i))/2 * cos(2*Phi)
    #          h_cross = amp * cos(i) * sin(2*Phi)
    raise NotImplementedError("Hole 1: post_newtonian_waveform 核心计算待补全")


def ringdown_waveform(t, M, a, qnm_freqs, t_merge, amplitudes=None):
    """
    生成并合后的 ringdown 波形。
    
    公式:
        h(t) = Σ_{l,m,n} A_{lmn} * exp(-i ω_{lmn} (t - t_merge))
        
    其中 ω_{lmn} = 2π f_{lmn} - i/τ_{lmn}
    """
    t = np.asarray(t, dtype=np.float64)
    h = np.zeros_like(t, dtype=np.complex128)
    
    if amplitudes is None:
        amplitudes = {}
    
    for key, omega in qnm_freqs.items():
        l, m, n = key
        A = amplitudes.get(key, 1.0 / (l + 1))
        # Ringdown 仅在 t > t_merge 时激发
        mask = t >= t_merge
        decay = np.exp(-1j * omega * (t - t_merge))
        decay[~mask] = 0.0
        h += A * decay
    
    return np.real(h), np.imag(h)


def full_imrphenom_waveform(t, m1, m2, D_L, inclination=0.0, M_final=None, a_final=None, qnm_freqs=None):
    """
    简化的 IMRPhenom 风格波形: inspiral + merger + ringdown。
    
    使用平滑连接函数:
        h(t) = h_insp(t) * w_insp(t) + h_ring(t) * w_ring(t)
        
    其中窗口函数:
        w_insp(t) = 0.5 * (1 + tanh((t_c - t) / Δt))
        w_ring(t) = 0.5 * (1 + tanh((t - t_c) / Δt))
    """
    t = np.asarray(t, dtype=np.float64)
    M = m1 + m2
    
    if M_final is None:
        M_final = M * 0.95  # 能量辐射损失约 5%
    if a_final is None:
        a_final = 0.0
    
    t_c = t[-1] * 0.7  # 假设并合发生在 70% 处
    Delta_t = M * 10.0  # 过渡时间尺度
    
    # Inspiral
    h_p_insp, h_c_insp, _, _ = post_newtonian_waveform(t, m1, m2, D_L, inclination, t_c=t_c)
    
    # Ringdown
    if qnm_freqs is None:
        from teukolsky import solve_qnm_frequencies
        qnm_freqs = solve_qnm_frequencies(l_max=3, n_overtones=1, M=M_final, a=a_final)
    
    h_p_ring, h_c_ring = ringdown_waveform(t, M_final, a_final, qnm_freqs, t_c)
    
    # 平滑窗口
    w_insp = 0.5 * (1.0 + np.tanh((t_c - t) / Delta_t))
    w_ring = 0.5 * (1.0 + np.tanh((t - t_c) / Delta_t))
    
    h_plus = h_p_insp * w_insp + h_p_ring * w_ring
    h_cross = h_c_insp * w_insp + h_c_ring * w_ring
    
    return h_plus, h_cross


# ---------------------------------------------------------------------------
# 匹配滤波信噪比计算
# ---------------------------------------------------------------------------

def matched_filter_snr(template_func, signal_func, f_min, f_max, Sn_func, n_quad=127):
    """
    使用 Patterson/Chebyshev 求积计算匹配滤波信噪比。
    
    公式:
        ρ^2 = 4 ∫_{f_min}^{f_max} |h̃(f)|^2 / S_n(f) df
        
    其中 h̃(f) 为引力波信号的傅里叶变换。
    """
    def integrand(f):
        h_tilde = template_func(f)
        Sn = Sn_func(f)
        if np.abs(Sn) < 1e-300:
            Sn = 1e-300
        return np.abs(h_tilde)**2 / Sn
    
    rho_sq = 4.0 * adaptive_patterson_integral(integrand, f_min, f_max, tol=1e-8)
    rho = np.sqrt(max(rho_sq, 0.0))
    return rho
