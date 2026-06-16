"""
stability_analysis.py
=====================
线性稳定性分析与混沌诊断模块。

科学背景:
  材料系统在高通量筛选中需要评估参数空间中各区域的稳定性。
  离子输运-热传导耦合系统可能出现:
    1. 扩散失稳 (Spinodal decomposition)
    2. 热失控 (Thermal runaway)
    3. 电化学振荡 (Limit cycles)

  本模块整合:
    1. Lorenz-96 混沌动力学 (源自 703_lorenz96_ode)
       - 作为材料相变动力学的类比模型
       - Lyapunov 指数计算

    2. Lindberg 精确解基准 (源自 674_lindberg_exact)
       - 用于验证稳定性分析的正确性
       - 环形振荡器的精确解

    3. 谱方法稳定性矩阵
       - 特征值分析 (Garden of Eden)
       - 中性曲线绘制

稳定性理论:
  线性稳定性: 对稳态解 x* 施加小扰动 delta_x:
    d(delta_x)/dt = J * delta_x
    J = df/dx |_{x*}  (Jacobian)

  稳定 iff 所有 Re(lambda_i) < 0
  Hopf 分岔: 一对共轭复特征值穿过虚轴

  Lorenz-96 模型:
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    周期边界: x_{i+N} = x_i

  Lyapunov 指数:
    lambda_1 = lim_{t->inf} (1/t) * log(||delta_x(t)|| / ||delta_x(0)||)

  材料相场稳定性 (Cahn-Hilliard):
    dphi/dt = M * laplacian(mu)
    mu = df/dphi - kappa * laplacian(phi)
  线性化: omega(k) = -M * k^2 * (f'' + kappa*k^2)
  失稳判据: f'' < 0 (spinodal 区域)
  最快生长波数: k_max = sqrt(-f'' / (2*kappa))

  von Neumann 稳定性分析 (对有限差分格式):
    放大因子 g(k) 满足 |g(k)| <= 1 + O(dt)
    对热方程 dT/dt = alpha * d^2T/dx^2 的 FTCS 格式:
      g = 1 - 4*r*sin^2(k*h/2), r = alpha*dt/h^2
    稳定性: r <= 1/2
"""

import numpy as np
from scipy.linalg import eig, svd
from material_constants import SMALL_NUMBER, MAX_STABILITY_ITER, CONVERGENCE_TOL


# ============================================================================
# Lorenz-96 模型 (源自 703)
# ============================================================================
def lorenz96_rhs(y, F, N):
    """
    Lorenz-96 右端函数:

    dy_i/dt = (y_{i+1} - y_{i-2}) * y_{i-1} - y_i + F

    其中下标取模 N (周期边界)。
    该模型用于研究大气可预测性, 在材料科学中类比
    多尺度相变动力学。

    参数:
        y: [N] 状态向量
        F: 外力参数 (分岔参数)
        N: 系统维度
    """
    dydt = np.zeros(N)
    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        dydt[i] = (y[ip1] - y[im2]) * y[im1] - y[i] + F
    return dydt


def lorenz96_jacobian(y, F, N):
    """
    Lorenz-96 Jacobian 矩阵:

    dF_i/dy_j:
      j = i: -1
      j = i-1: y_{i+1} - y_{i-2}
      j = i+1: y_{i-1}
      j = i-2: -y_{i-1}
      其他: 0
    """
    J = np.zeros((N, N))
    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        J[i, i] = -1.0
        J[i, im1] = y[ip1] - y[im2]
        J[i, ip1] = y[im1]
        J[i, im2] = -y[im1]
    return J


def lorenz96_integrate_rk4(y0, F, N, dt, n_steps):
    """
    Lorenz-96 经典 RK4 积分。

    RK4 公式:
      k1 = f(y_n)
      k2 = f(y_n + dt/2 * k1)
      k3 = f(y_n + dt/2 * k2)
      k4 = f(y_n + dt * k3)
      y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    y = np.asarray(y0, dtype=np.float64).copy()
    trajectory = np.zeros((n_steps + 1, N))
    trajectory[0] = y.copy()

    for step in range(n_steps):
        k1 = lorenz96_rhs(y, F, N)
        k2 = lorenz96_rhs(y + 0.5*dt*k1, F, N)
        k3 = lorenz96_rhs(y + 0.5*dt*k2, F, N)
        k4 = lorenz96_rhs(y + dt*k3, F, N)
        y = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory[step + 1] = y.copy()

    return trajectory


def lyapunov_exponent(y0, F, N, dt, n_steps, delta=1e-8, renorm_interval=10):
    """
    最大 Lyapunov 指数 (Benettin 算法简化版).

    算法:
      1. 主轨道 y(t), 影子轨道 y'(t) = y(t) + delta * v
      2. 同时积分 renorm_interval 步
      3. 测量距离 d = ||y' - y||
      4. lambda_inst = log(d / delta) / (renorm_interval * dt)
      5. 重新归一化 y' = y + delta * (y' - y) / d
      6. lambda = mean(lambda_inst)

    返回:
        lambda_max: 最大 Lyapunov 指数
        history: 瞬时 Lyapunov 指数历史
    """
    y = np.asarray(y0, dtype=np.float64).copy()
    y_shadow = y + delta * np.random.randn(N) / np.sqrt(N)

    history = []
    for cycle in range(n_steps // renorm_interval):
        for _ in range(renorm_interval):
            k1 = lorenz96_rhs(y, F, N)
            k2 = lorenz96_rhs(y + 0.5*dt*k1, F, N)
            k3 = lorenz96_rhs(y + 0.5*dt*k2, F, N)
            k4 = lorenz96_rhs(y + dt*k3, F, N)
            y = y + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

            k1s = lorenz96_rhs(y_shadow, F, N)
            k2s = lorenz96_rhs(y_shadow + 0.5*dt*k1s, F, N)
            k3s = lorenz96_rhs(y_shadow + 0.5*dt*k2s, F, N)
            k4s = lorenz96_rhs(y_shadow + dt*k3s, F, N)
            y_shadow = y_shadow + (dt/6.0) * (k1s + 2*k2s + 2*k3s + k4s)

        d = np.linalg.norm(y_shadow - y)
        if d < SMALL_NUMBER:
            d = SMALL_NUMBER
        lam_inst = np.log(d / delta) / (renorm_interval * dt)
        history.append(lam_inst)

        # 重归一化
        y_shadow = y + delta * (y_shadow - y) / d

    return np.mean(history), np.array(history)


def lorenz96_bifurcation_scan(N, F_range, dt=0.01, warmup=5000):
    """
    Lorenz-96 分岔扫描:
    对一系列 F 值, 计算:
      - 稳态/周期/混沌状态
      - 最大 Lyapunov 指数

    返回:
        results: list of dict {F, lambda_max, mean_y, std_y}
    """
    results = []
    for F in F_range:
        y0 = np.random.randn(N) * 0.1
        # 预热
        traj = lorenz96_integrate_rk4(y0, F, N, dt, warmup)
        y_warm = traj[-1]
        # 计算 Lyapunov
        lam, _ = lyapunov_exponent(y_warm, F, N, dt, n_steps=2000, renorm_interval=5)
        # 统计
        traj2 = lorenz96_integrate_rk4(y_warm, F, N, dt, 1000)
        mean_y = np.mean(traj2[-500:])
        std_y = np.std(traj2[-500:])
        results.append({
            'F': F,
            'lambda_max': lam,
            'mean_y': mean_y,
            'std_y': std_y,
        })
    return results


# ============================================================================
# Lindberg 环形振荡器精确解 (源自 674)
# ============================================================================
def lindberg_ring_oscillator(n_stages, gain_per_stage, tau=1.0):
    """
    Lindberg 环形振荡器模型。

    n 级环形振荡器, 每级传递函数:
      H(s) = -A / (1 + s*tau)^n

    起振条件 (Barkhausen):
      |H(jw)| = 1 且 phase(H(jw)) = 2*pi*k

    对 n 级 (奇数), 振荡频率:
      omega_0 = tan(pi/n) / tau
      临界增益: A_crit = 1 / cos(pi/n)^n

    返回:
        omega_0: 振荡角频率
        A_crit: 临界增益
        poles: 闭环极点
    """
    if n_stages < 3 or n_stages % 2 == 0:
        n_stages = 3

    omega_0 = np.tan(np.pi / n_stages) / tau
    A_crit = 1.0 / (np.cos(np.pi / n_stages) ** n_stages)

    # 开环极点: s_k = -1/tau + j*0 (n 重)
    # 闭环: (1 + s*tau)^n + A*gain = 0
    # s_k = (-1 + (A*gain)^(1/n) * exp(j*pi*(2k+1)/n)) / tau
    poles = []
    for k in range(n_stages):
        angle = np.pi * (2*k + 1) / n_stages
        root = (gain_per_stage * A_crit) ** (1.0/n_stages) * np.exp(1j * angle)
        s_k = (-1.0 + root) / tau
        poles.append(s_k)

    return omega_0, A_crit, np.array(poles)


def lindberg_exact_response(t, n_stages, A_input, tau=1.0):
    """
    Lindberg 精确时域响应 (小信号近似)。

    对输入阶跃 A_input, n 级级联响应:
      V_n(t) = A_input^n * (1 - exp(-t/tau) * sum_{k=0}^{n-1} (t/tau)^k / k!)

    这是 n 阶 Erlang 分布的 CDF 形式。
    """
    from math import factorial
    x = t / tau
    sum_terms = sum((x**k) / factorial(k) for k in range(n_stages))
    V = (A_input ** n_stages) * (1.0 - np.exp(-x) * sum_terms)
    return V


# ============================================================================
# 谱稳定性分析
# ============================================================================
def spectral_stability(J):
    """
    对 Jacobian 矩阵 J 进行谱分析。

    返回:
        eigenvalues: 复特征值
        growth_rates: Re(lambda) (增长率)
        frequencies: Im(lambda) (振荡频率)
        is_stable: True iff 所有 Re(lambda) < 0
        spectral_radius: max |lambda|
        dominant_mode: 最不稳定 (或最慢衰减) 的特征向量
    """
    eigenvalues = eig(J, left=False, right=True)[0] if J.shape[0] > 0 else np.array([])
    growth_rates = np.real(eigenvalues)
    frequencies = np.imag(eigenvalues)
    is_stable = np.all(growth_rates < -SMALL_NUMBER)
    spectral_radius = np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0.0

    if len(eigenvalues) > 0:
        eig_data = eig(J, left=False, right=True)
        dominant_idx = np.argmax(growth_rates)
        dominant_mode = np.real(eig_data[1][:, dominant_idx])
    else:
        dominant_mode = np.array([])

    return {
        'eigenvalues': eigenvalues,
        'growth_rates': growth_rates,
        'frequencies': frequencies,
        'is_stable': bool(is_stable),
        'spectral_radius': float(spectral_radius),
        'dominant_mode': dominant_mode,
        'max_growth_rate': float(np.max(growth_rates)) if len(growth_rates) > 0 else 0.0,
    }


def von_neumann_stability_heat(alpha, dt, h):
    """
    热方程 FTCS 格式的 von Neumann 稳定性分析:

    放大因子:
      g(k) = 1 - 4*r*sin^2(k*h/2), r = alpha*dt/h^2

    稳定性条件: r <= 1/2 (即 dt <= h^2 / (2*alpha))

    返回:
        r: 扩散数
        g_max: 最大放大因子
        g_min: 最小放大因子
        is_stable: |g| <= 1
        dt_critical: 临界时间步长
    """
    r = alpha * dt / max(h**2, SMALL_NUMBER)
    g_max = 1.0
    g_min = 1.0 - 4.0 * r
    is_stable = r <= 0.5
    dt_crit = h**2 / (2.0 * max(alpha, SMALL_NUMBER))
    return {
        'r': r,
        'g_max': g_max,
        'g_min': g_min,
        'is_stable': bool(is_stable),
        'dt_critical': dt_crit,
    }


def cahn_hilliard_dispersion(f_pp, kappa_grad, mobility, k_array):
    """
    Cahn-Hilliard 方程的色散关系:

    omega(k) = -M * k^2 * (f'' + kappa * k^2)

    失稳条件: f'' < 0 (spinodal 区域)
    最快生长波数: k_max = sqrt(-f'' / (2*kappa))
    临界波数: k_c = sqrt(-f'' / kappa)

    返回:
        omega: 增长率谱
        k_max: 最快生长波数
        k_critical: 临界波数
    """
    if f_pp >= 0:
        # 稳定区域
        omega = -mobility * k_array**2 * (f_pp + kappa_grad * k_array**2)
        return omega, 0.0, 0.0

    omega = -mobility * k_array**2 * (f_pp + kappa_grad * k_array**2)
    k_max = np.sqrt(-f_pp / (2.0 * max(kappa_grad, SMALL_NUMBER)))
    k_c = np.sqrt(-f_pp / max(kappa_grad, SMALL_NUMBER))
    return omega, k_max, k_c


def hopf_bifurcation_check(J, param_name='F', param_values=None):
    """
    检测 Hopf 分岔点: 一对复特征值穿过虚轴。

    条件:
      1. 存在 lambda = +- i*omega (纯虚特征值)
      2. transversality: d(Re(lambda))/d(param) != 0

    返回:
        hopf_detected: bool
        critical_param: 临界参数值
        omega_hopf: 振荡频率
    """
    if param_values is None:
        return False, None, None

    prev_eigs = None
    for p in param_values:
        J_p = J(p) if callable(J) else J
        eigs = eig(J_p, left=False, right=False)[0]
        growth_rates = np.real(eigs)

        if prev_eigs is not None:
            prev_growth = np.real(prev_eigs)
            # 检测符号变化
            for i in range(len(growth_rates)):
                if prev_growth[i] < 0 and growth_rates[i] >= 0:
                    if abs(np.imag(eigs[i])) > SMALL_NUMBER:
                        return True, p, abs(np.imag(eigs[i]))
        prev_eigs = eigs

    return False, None, None
