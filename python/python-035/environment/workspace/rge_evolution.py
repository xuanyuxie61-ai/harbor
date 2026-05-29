"""
rge_evolution.py
希格斯耦合常数的重整化群方程 (RGE) 演化

基于三个种子项目重构:
  - 1042_roessler_ode: 非线性ODE系统框架
  - 861_pendulum_nonlinear_ode: 能量守恒监测与精确解验证
  - 312_dosage_ode: 多室/compartment 参数结构

物理内容:
  标准模型中，希格斯自耦合 lambda、汤川耦合 y_t、y_b、y_tau 以及
  规范耦合 g_1, g_2, g_3 都随能标 mu 跑动。
  
  一圈图 RGE (MS-bar 方案):
    d g_i / dt = beta_i(g) / (16*pi^2),  t = ln(mu)
    
  希格斯自耦合 lambda 的 beta 函数:
    beta_lambda = 6*lambda^2 + 3*lambda*(y_t^2+y_b^2+y_tau^2) 
                  - (y_t^4+y_b^4+y_tau^4)
                  + (3/8)*(2*g_2^4 + (g_1^2+g_2^2)^2)
                  - 3*lambda*(3*g_2^2 + g_1^2)
"""
import numpy as np
from constants import G_F, TINY
from utils import rk2_integrate

# ============================================================
# 1. 标准模型耦合常数的 RGE (一圈图)
# ============================================================
def sm_rge_beta(t, y):
    """
    标准模型耦合常数的一圈图 beta 函数
    
    状态向量 y = [g1, g2, g3, yt, yb, ytau, lambda]
      g1: U(1)_Y 规范耦合 (GUT 归一化: g1 = sqrt(5/3) * g')
      g2: SU(2)_L 规范耦合
      g3: SU(3)_C 规范耦合
      yt: 顶夸克汤川耦合
      yb: 底夸克汤川耦合
      ytau: tau 轻子汤川耦合
      lambda: 希格斯自耦合
    
    t = ln(mu / m_Z)
    
    Beta 函数 (文献: Arxiv:1205.6497, S. Betheke et al.):
      beta_g1 = (41/10) * g1^3
      beta_g2 = (-19/6) * g2^3
      beta_g3 = (-7) * g3^3
      beta_yt = yt * (9/2*yt^2 + yb^2 - (17/20)*g1^2 - (9/4)*g2^2 - 8*g3^2)
      beta_yb = yb * (yt^2 + 9/2*yb^2 + ytau^2 - (1/4)*g1^2 - (9/4)*g2^2 - 8*g3^2)
      beta_ytau = ytau * (3*yb^2 + 5/2*ytau^2 - (9/4)*g1^2 - (9/4)*g2^2)
      beta_lambda = 6*lambda^2 + 2*lambda*(yt^2+yb^2+ytau^2) - (yt^4+yb^4+ytau^4)
                    + (3/8)*(2*g2^4 + (g1^2+g2^2)^2) - 3*lambda*(3*g2^2+g1^2)
    
    注意: beta_g = dg/dt = beta/(16*pi^2)
    """
    g1, g2, g3, yt, yb, ytau, lam = y
    
    # 防止耦合变负
    g1 = max(g1, 0.0)
    g2 = max(g2, 0.0)
    g3 = max(g3, 0.0)
    yt = max(yt, 0.0)
    yb = max(yb, 0.0)
    ytau = max(ytau, 0.0)
    
    factor = 1.0 / (16.0 * np.pi ** 2)
    
    beta_g1 = (41.0 / 10.0) * g1 ** 3
    beta_g2 = (-19.0 / 6.0) * g2 ** 3
    beta_g3 = (-7.0) * g3 ** 3
    
    beta_yt = yt * (4.5 * yt ** 2 + yb ** 2 - 0.85 * g1 ** 2 - 2.25 * g2 ** 2 - 8.0 * g3 ** 2)
    beta_yb = yb * (yt ** 2 + 4.5 * yb ** 2 + ytau ** 2 - 0.25 * g1 ** 2 - 2.25 * g2 ** 2 - 8.0 * g3 ** 2)
    beta_ytau = ytau * (3.0 * yb ** 2 + 2.5 * ytau ** 2 - 2.25 * g1 ** 2 - 2.25 * g2 ** 2)
    
    beta_lambda = (6.0 * lam ** 2 
                   + 2.0 * lam * (yt ** 2 + yb ** 2 + ytau ** 2)
                   - (yt ** 4 + yb ** 4 + ytau ** 4)
                   + 0.375 * (2.0 * g2 ** 4 + (g1 ** 2 + g2 ** 2) ** 2)
                   - 3.0 * lam * (3.0 * g2 ** 2 + g1 ** 2))
    
    return np.array([
        beta_g1 * factor,
        beta_g2 * factor,
        beta_g3 * factor,
        beta_yt * factor,
        beta_yb * factor,
        beta_ytau * factor,
        beta_lambda * factor
    ])


# ============================================================
# 2. 初始条件 (m_Z 能标处的实验值)
# ============================================================
def sm_initial_conditions():
    """
    在 mu = m_Z 处的标准模型耦合常数值 (PDG 近似)
    
    g1 (GUT归一化) = sqrt(5/3) * g' = sqrt(5/3) * e / cos(theta_W)
    g2 = e / sin(theta_W)
    g3 ~ 1.22 (alpha_s(m_Z) ~ 0.118)
    yt ~ sqrt(2)*m_t/v ~ 0.99
    yb ~ sqrt(2)*m_b/v ~ 0.016
    ytau ~ sqrt(2)*m_tau/v ~ 0.010
    lambda = m_H^2 / (2*v^2) ~ 0.13
    """
    alpha_em = 1.0 / 127.9  # 低能精细结构常数近似
    sin2w = 0.23121
    e = np.sqrt(4.0 * np.pi * alpha_em)
    
    g1 = np.sqrt(5.0 / 3.0) * e / np.sqrt(1.0 - sin2w)
    g2 = e / np.sqrt(sin2w)
    g3 = np.sqrt(4.0 * np.pi * 0.118)
    
    v = 1.0 / np.sqrt(np.sqrt(2.0) * G_F)
    m_t = 173.1
    m_b = 4.18
    m_tau = 1.777
    m_h = 125.1
    
    yt = np.sqrt(2.0) * m_t / v
    yb = np.sqrt(2.0) * m_b / v
    ytau = np.sqrt(2.0) * m_tau / v
    lam = m_h ** 2 / (2.0 * v ** 2)
    
    return np.array([g1, g2, g3, yt, yb, ytau, lam])


# ============================================================
# 3. 能量监测 (映射 861_pendulum_nonlinear_ode 的能量守恒思想)
# ============================================================
def effective_potential(y):
    """
    有效势 (类比能量):
      V_eff = lambda / 4 * v^4  (希格斯势)
    
    在 RGE 框架下，v(mu)^2 = m_H^2 / lambda(mu)
    """
    lam = max(y[6], TINY)
    m_h = 125.1
    v_sq = m_h ** 2 / lam
    return 0.25 * lam * v_sq ** 2


def check_rge_stability(t_array, y_array):
    """
    检查 RGE 演化数值稳定性
    
    指标:
      - 耦合是否发散或变负
      - 有效势是否单调 (热力学稳定性)
    """
    issues = []
    for i, y in enumerate(y_array):
        if np.any(y[:3] < 0):
            issues.append((i, t_array[i], "negative gauge coupling"))
        if y[3] > 2.0:  # yt  Landau pole 预警
            issues.append((i, t_array[i], "Yukawa Landau pole warning"))
    
    v_eff = [effective_potential(y) for y in y_array]
    # 检查有效势是否有非物理的剧烈振荡
    v_diff = np.diff(v_eff)
    if np.any(np.abs(v_diff) > 1.0e6):
        issues.append((-1, -1, "effective potential unstable"))
    
    return len(issues) == 0, issues


# ============================================================
# 4. 希格斯-矢量玻色子耦合的能标演化
# ============================================================
def higgs_vv_coupling_evolution(mu_values):
    """
    计算 g_{HVV}(mu) 的能标演化
    
    在 SM 中:
      g_{HWW} = g2 * m_W / (2 * m_W^2 / v) ... 简化为:
      g_{HZZ} = sqrt(g1^2 + g2^2) * v / 2  (在 tree level ~ m_Z)
    
    实际上随能标变化主要来自 v(mu) = sqrt(m_H^2 / (2*lambda(mu)))
    和混合角的变化。
    
    参数:
        mu_values: 能标数组 [GeV]
    返回:
        g_hzz_values: HZZ 耦合随能标的变化
        lambda_values: 希格斯自耦合
    """
    t_span = (0.0, np.log(np.max(mu_values) / 91.1876))
    y0 = sm_initial_conditions()
    n_steps = max(500, int(t_span[1] * 100))
    
    t_arr, y_arr = rk2_integrate(sm_rge_beta, t_span, y0, n_steps)
    
    # 插值到请求能标
    log_mu_request = np.log(mu_values / 91.1876)
    g_hzz = []
    lam_vals = []
    
    for lmu in log_mu_request:
        # 线性插值
        idx = np.searchsorted(t_arr, lmu)
        if idx <= 0:
            y = y_arr[0]
        elif idx >= len(t_arr):
            y = y_arr[-1]
        else:
            frac = (lmu - t_arr[idx - 1]) / (t_arr[idx] - t_arr[idx - 1] + TINY)
            y = y_arr[idx - 1] + frac * (y_arr[idx] - y_arr[idx - 1])
        
        g1, g2 = y[0], y[1]
        lam = max(y[6], TINY)
        v = 246.0 / np.sqrt(lam / 0.13)  # 近似标度
        g_hzz.append(np.sqrt(g1 ** 2 + g2 ** 2) * v / 2.0)
        lam_vals.append(lam)
    
    return np.array(g_hzz), np.array(lam_vals)


# ============================================================
# 5. 顶夸克汤川耦合的Landau Pole分析
# ============================================================
def landau_pole_estimate():
    """
    估算顶夸克汤川耦合的 Landau Pole 位置
    
    近似解析解 (忽略其他耦合):
      1/yt^2(mu) = 1/yt^2(m_Z) - (9/(8*pi^2)) * ln(mu/m_Z)
      
      Landau Pole: mu_LP = m_Z * exp(8*pi^2 / (9*yt^2(m_Z)))
    """
    y0 = sm_initial_conditions()
    yt0 = y0[3]
    if yt0 < TINY:
        return np.inf
    exponent = 8.0 * np.pi ** 2 / (9.0 * yt0 ** 2)
    return 91.1876 * np.exp(exponent)


# ============================================================
# 6. 完整 RGE 分析报告
# ============================================================
def rge_analysis_report(mu_high=1.0e4, n_steps=1000):
    """
    生成 RGE 演化的完整分析报告
    
    参数:
        mu_high: 最高能标 [GeV]
        n_steps: RK2 积分步数
    返回:
        dict: 包含演化数组、Landau Pole、稳定性判断
    """
    t_span = (0.0, np.log(mu_high / 91.1876))
    y0 = sm_initial_conditions()
    
    t_arr, y_arr = rk2_integrate(sm_rge_beta, t_span, y0, n_steps)
    mu_arr = 91.1876 * np.exp(t_arr)
    
    stable, issues = check_rge_stability(t_arr, y_arr)
    lp = landau_pole_estimate()
    
    g_hzz, lam_vals = higgs_vv_coupling_evolution(mu_arr)
    
    return {
        "mu": mu_arr,
        "t": t_arr,
        "y": y_arr,
        "g1": y_arr[:, 0],
        "g2": y_arr[:, 1],
        "g3": y_arr[:, 2],
        "yt": y_arr[:, 3],
        "yb": y_arr[:, 4],
        "ytau": y_arr[:, 5],
        "lambda": y_arr[:, 6],
        "g_hzz": g_hzz,
        "lambda_values": lam_vals,
        "stable": stable,
        "issues": issues,
        "landau_pole_gev": lp,
    }
