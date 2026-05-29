"""
orthogonal_fit.py
基于正交多项式的背景拟合与信号提取

基于 209_conte_deboor 项目重构:
  - ortpol: 正交多项式构造 (三项递推)
  - ortval: 正交多项式求值
  - cheb: Chebyshev 多项式 Clenshaw 求值

物理应用:
  在四轻子不变质量谱 m_4l 上，背景通常用低阶多项式描述:
    B(m) = sum_{k=0}^{N} c_k * P_k(m)
  其中 P_k 为关于背景权函数 w(m) 正交的多项式。
  
  使用正交多项式的优势:
    - 系数独立计算，条件数好
    - 避免 Runge 现象
    - 物理上截断到低阶即可描述平滑背景
"""
import numpy as np
from constants import TINY

# ============================================================
# 1. 正交多项式构造 (三项递推) (映射 209 ortpol)
# ============================================================
def orthopoly_construct(n, weight_func, a, b, n_quad=64):
    """
    构造关于权函数 w(x) 在 [a,b] 上的正交多项式系 {P_k}_{k=0}^{n}
    
    三项递推关系:
      P_{-1}(x) = 0
      P_0(x) = 1 / sqrt(int_a^b w(x) dx)
      P_{k+1}(x) = (x - alpha_k) * P_k(x) - beta_k * P_{k-1}(x)
    
    其中:
      alpha_k = <x*P_k, P_k> / <P_k, P_k>
      beta_k = <P_k, P_k> / <P_{k-1}, P_{k-1}>
    
    参数:
        n: 最高阶数
        weight_func: 权函数 w(x)
        a, b: 区间
        n_quad: 用于内积计算的 Gauss-Legendre 节点数
    返回:
        alphas: 递推系数 alpha_0, ..., alpha_{n-1}
        betas: 递推系数 beta_0, ..., beta_{n-1}
        norm0: P_0 的归一化常数
    """
    # 使用 Legendre-Gauss 节点计算内积
    from quadrature_engine import legendre_gauss_rule
    nodes, weights = legendre_gauss_rule(n_quad)
    
    # 映射到 [a,b]
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    phys_nodes = nodes * scale + shift
    phys_weights = weights * scale
    
    alphas = np.zeros(n)
    betas = np.zeros(n)
    
    # P_0
    p0_sq = np.sum(phys_weights * weight_func(phys_nodes))
    if p0_sq < TINY:
        p0_sq = TINY
    norm0 = 1.0 / np.sqrt(p0_sq)
    
    # 递推计算 alpha_k, beta_k
    # 使用 Stieltjes 过程
    p_prev = np.ones(n_quad) * norm0  # P_0
    p_curr = np.zeros(n_quad)         # P_1 (待计算)
    
    for k in range(n):
        # alpha_k = <x*P_k, P_k>
        num = np.sum(phys_weights * weight_func(phys_nodes) * phys_nodes * p_prev ** 2)
        den = np.sum(phys_weights * weight_func(phys_nodes) * p_prev ** 2)
        alphas[k] = num / max(den, TINY)
        
        # P_{k+1} = (x - alpha_k) * P_k - beta_k * P_{k-1}
        if k == 0:
            p_curr = (phys_nodes - alphas[k]) * p_prev
        else:
            p_curr = (phys_nodes - alphas[k]) * p_prev - betas[k - 1] * p_prev_prev
        
        # beta_{k+1} = <P_{k+1}, P_{k+1}> / <P_k, P_k>
        num_beta = np.sum(phys_weights * weight_func(phys_nodes) * p_curr ** 2)
        den_beta = np.sum(phys_weights * weight_func(phys_nodes) * p_prev ** 2)
        if k < n - 1:
            betas[k + 1] = num_beta / max(den_beta, TINY)
        
        p_prev_prev = p_prev.copy()
        p_prev = p_curr.copy()
    
    return alphas, betas, norm0


def orthopoly_eval(alphas, betas, norm0, x, k_max):
    """
    使用三项递推求值正交多项式 P_0(x), ..., P_{k_max}(x)
    
    参数:
        alphas, betas, norm0: 递推系数
        x: 求值点 (标量或数组)
        k_max: 最高阶
    返回:
        values: shape (len(x), k_max+1) 或 (k_max+1,)
    """
    x = np.asarray(x, dtype=float)
    scalar_input = (x.ndim == 0)
    x = np.atleast_1d(x)
    n = len(x)
    
    vals = np.zeros((n, k_max + 1))
    vals[:, 0] = norm0
    
    if k_max >= 1:
        vals[:, 1] = (x - alphas[0]) * vals[:, 0]
        for k in range(1, k_max):
            if k < len(alphas) and k < len(betas):
                vals[:, k + 1] = (x - alphas[k]) * vals[:, k] - betas[k] * vals[:, k - 1]
    
    if scalar_input:
        return vals[0, :]
    return vals


# ============================================================
# 2. Clenshaw 求值 Chebyshev 级数 (映射 209 cheb)
# ============================================================
def clenshaw_chebyshev(coeffs, x):
    """
    Clenshaw 算法求 Chebyshev 级数:
      f(x) = sum_{k=0}^n c_k * T_k(x)
    
    递推:
      b_{n+1} = b_n = 0
      b_k = 2*x*b_{k+1} - b_{k+2} + c_k,  k = n-1, ..., 0
      f(x) = b_0 - x*b_1
    
    参数:
        coeffs: [c0, c1, ..., cn]
        x: 求值点
    返回:
        f(x)
    """
    x = float(x)
    x = np.clip(x, -1.0, 1.0)
    n = len(coeffs) - 1
    
    b_prev2 = 0.0  # b_{k+2}
    b_prev1 = 0.0  # b_{k+1}
    
    for k in range(n, 0, -1):
        b = 2.0 * x * b_prev1 - b_prev2 + coeffs[k]
        b_prev2 = b_prev1
        b_prev1 = b
    
    return coeffs[0] + x * b_prev1 - b_prev2


# ============================================================
# 3. 背景拟合: 正交多项式最小二乘
# ============================================================
def orthogonal_background_fit(mass_bins, counts, degree=4, weight_func=None, a=80.0, b=170.0):
    """
    用正交多项式拟合 m_4l 谱的背景
    
    模型:
      N(m) = sum_{k=0}^{degree} c_k * P_k(m) + epsilon
    
    系数计算 (离散正交投影):
      c_k = sum_i w_i * N_i * P_k(m_i) / sum_i w_i * P_k(m_i)^2
    
    参数:
        mass_bins: 质量分箱中心
        counts: 每个分箱的计数
        degree: 多项式阶数
        weight_func: 权函数 (默认常数)
        a, b: 正交区间
    返回:
        coeffs: 拟合系数
        alphas, betas, norm0: 正交多项式参数
        fitted: 拟合值
    """
    mass_bins = np.asarray(mass_bins, dtype=float)
    counts = np.asarray(counts, dtype=float)
    
    if weight_func is None:
        weight_func = lambda x: np.ones_like(np.atleast_1d(x))
    
    alphas, betas, norm0 = orthopoly_construct(degree, weight_func, a, b)
    
    # 求值正交多项式在数据点上
    P_vals = orthopoly_eval(alphas, betas, norm0, mass_bins, degree)  # (N, degree+1)
    
    coeffs = np.zeros(degree + 1)
    for k in range(degree + 1):
        num = np.sum(counts * P_vals[:, k])
        den = np.sum(P_vals[:, k] ** 2)
        if den > TINY:
            coeffs[k] = num / den
    
    fitted = P_vals @ coeffs
    
    return coeffs, (alphas, betas, norm0), fitted


def predict_background(mass, coeffs, ortho_params):
    """
    使用拟合的正交多项式预测背景计数
    
    参数:
        mass: 质量值或数组
        coeffs: 拟合系数
        ortho_params: (alphas, betas, norm0)
    返回:
        预测背景
    """
    alphas, betas, norm0 = ortho_params
    P_vals = orthopoly_eval(alphas, betas, norm0, mass, len(coeffs) - 1)
    if P_vals.ndim == 1:
        return float(np.dot(P_vals, coeffs))
    return P_vals @ coeffs


# ============================================================
# 4. 信号提取: 多项式背景减除
# ============================================================
def extract_signal(mass_bins, counts, fitted_background):
    """
    背景减除法提取信号:
      S(m) = max(N(m) - B(m), 0)
    
    参数:
        mass_bins: 质量分箱
        counts: 观测计数
        fitted_background: 拟合背景
    返回:
        signal_estimate: 信号估计
        significance: 每个分箱的显著性 ~ S/sqrt(B)
    """
    signal = np.maximum(counts - fitted_background, 0.0)
    significance = np.zeros_like(signal)
    for i in range(len(signal)):
        if fitted_background[i] > 1.0:
            significance[i] = signal[i] / np.sqrt(fitted_background[i])
    return signal, significance


# ============================================================
# 5. 完整 m_4l 谱分析流程
# ============================================================
def analyze_mass_spectrum(mass_bins, counts, background_degree=4):
    """
    完整的四轻子不变质量谱分析
    
    步骤:
      1. 用正交多项式拟合背景
      2. 背景减除提取信号
      3. 计算信号显著性
      4. 定位希格斯峰位置
    
    参数:
        mass_bins: 质量分箱中心 [GeV]
        counts: 各分箱计数
        background_degree: 背景多项式阶数
    返回:
        dict: 分析结果
    """
    a = np.min(mass_bins)
    b = np.max(mass_bins)
    
    coeffs, ortho_params, fitted = orthogonal_background_fit(
        mass_bins, counts, degree=background_degree, a=a, b=b
    )
    
    signal, significance = extract_signal(mass_bins, counts, fitted)
    
    # 寻找显著性峰值
    peak_idx = np.argmax(significance)
    peak_mass = mass_bins[peak_idx]
    peak_significance = significance[peak_idx]
    
    # 总信号和总背景
    total_signal = np.sum(signal)
    total_background = np.sum(fitted)
    
    return {
        "mass_bins": mass_bins,
        "observed": counts,
        "background_fit": fitted,
        "signal": signal,
        "significance": significance,
        "peak_mass": peak_mass,
        "peak_significance": peak_significance,
        "total_signal": total_signal,
        "total_background": total_background,
        "s_over_sqrt_b": total_signal / np.sqrt(total_background) if total_background > 1.0 else 0.0,
        "polynomial_coeffs": coeffs,
        "ortho_params": ortho_params,
    }
