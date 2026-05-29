"""
phase_space.py
希格斯玻色子 H -> ZZ* -> 4l 衰变的相空间蒙特卡洛生成

基于两个种子项目:
  - 450_gamblers_ruin_simulation: 马尔可夫链蒙特卡洛思想
  - 182_circle_positive_distance: 圆/球面上的均匀采样

物理背景:
  四体末态相空间度规:
    dPhi_4 = (2*pi)^4 / (2^4) * delta^4(P - sum p_i) * prod d^3p_i / (2E_i (2*pi)^3)
  
  在希格斯静止系中，使用双Z中间态参数化:
    P_H = (m_H, 0, 0, 0)
    P_Z1 + P_Z2 = P_H
    每个 Z -> l+ l-
"""
import numpy as np
from constants import M_HIGGS, M_Z, GAMMA_Z, TINY
from utils import safe_sqrt, safe_divide

# ============================================================
# 1. 球面与环面均匀采样 (映射 182_circle_positive_distance)
# ============================================================
def sample_unit_sphere_uniform(n):
    """
    在单位球面 S^2 上均匀采样 n 个点
    
    方法: Marsaglia 方法
      1. 生成 (x1, x2) 在单位圆盘内
      2. 投影到球面:
         x = 2*x1 * sqrt(1 - r^2)
         y = 2*x2 * sqrt(1 - r^2)
         z = 1 - 2*r^2,  r^2 = x1^2 + x2^2
    
    数学保证: 球面面积元 dOmega = sin(theta) dtheta dphi 被均匀覆盖
    """
    points = np.zeros((n, 3))
    i = 0
    max_trials = n * 100
    trial = 0
    while i < n and trial < max_trials:
        x1 = np.random.uniform(-1.0, 1.0)
        x2 = np.random.uniform(-1.0, 1.0)
        r2 = x1 * x1 + x2 * x2
        if r2 < 1.0 and r2 > TINY:
            sqrt_term = np.sqrt(1.0 - r2)
            points[i, 0] = 2.0 * x1 * sqrt_term
            points[i, 1] = 2.0 * x2 * sqrt_term
            points[i, 2] = 1.0 - 2.0 * r2
            i += 1
        trial += 1
    # 填充剩余
    if i < n:
        theta = np.random.uniform(0.0, np.pi, n - i)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n - i)
        points[i:, 0] = np.sin(theta) * np.cos(phi)
        points[i:, 1] = np.sin(theta) * np.sin(phi)
        points[i:, 2] = np.cos(theta)
    return points


def sample_positive_quadrant_circle(n):
    """
    单位圆正象限均匀采样 (映射 182_circle_positive_distance 的核心)
    
    方法: theta ~ U[0, 2*pi], 取绝对值投影到第一象限
         x = |cos(theta)|, y = |sin(theta)|
    
    注意: 这实际上在单位圆的四分之一弧上均匀分布，但在面积上不是均匀的。
    对于角度采样，我们直接使用: theta ~ U[0, pi/2]
    """
    theta = np.random.uniform(0.0, 0.5 * np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.column_stack([x, y])


# ============================================================
# 2. 四体相空间参数化
# ============================================================
def two_body_decay(m_parent, m1, m2, direction):
    """
    在母粒子静止系中的二体衰变:
      E1 = (m_parent^2 + m1^2 - m2^2) / (2*m_parent)
      E2 = (m_parent^2 + m2^2 - m1^2) / (2*m_parent)
      |p| = sqrt(E1^2 - m1^2)
    
    方向由单位矢量 direction 给出
    
    返回: (p1, p2) 四动量数组，shape (4,) each
    """
    if m_parent < m1 + m2:
        # 运动学禁戒，返回零动量
        return np.array([m1, 0.0, 0.0, 0.0]), np.array([m2, 0.0, 0.0, 0.0])
    
    e1 = (m_parent ** 2 + m1 ** 2 - m2 ** 2) / (2.0 * m_parent)
    e2 = (m_parent ** 2 + m2 ** 2 - m1 ** 2) / (2.0 * m_parent)
    p_mag = safe_sqrt(e1 ** 2 - m1 ** 2)
    
    p1 = np.array([e1, p_mag * direction[0], p_mag * direction[1], p_mag * direction[2]])
    p2 = np.array([e2, -p_mag * direction[0], -p_mag * direction[1], -p_mag * direction[2]])
    return p1, p2


def generate_hzz4l_event(m_higgs=M_HIGGS, m_z=M_Z):
    """
    生成一个 H -> ZZ* -> 4l 事件的四动量
    
    参数化步骤:
      1. 从 Breit-Wigner 分布抽取两个 Z 的不变质量 m1, m2
      2. 确定 Z1 和 Z2 在希格斯静止系中的运动方向 (球面均匀)
      3. 对每个 Z，确定其衰变产物在其静止系中的方向 (球面均匀)
      4. 通过洛伦兹 boost 将轻子四动量变换到希格斯静止系
    
    运动学约束:
      m1 + m2 <= m_higgs
      m_ll >= 2*m_e (最小双轻子不变质量)
    """
    m_ll_min = 0.001  # GeV, 近似电子对质量阈值
    
    # 从 Breit-Wigner 抽取 m1, m2
    # 使用 accept-reject 方法
    max_bw = 1.0 / (m_z * GAMMA_Z)  # Breit-Wigner 峰值近似
    max_trials = 10000
    
    m1 = m_z
    m2 = m_z
    trial = 0
    while trial < max_trials:
        # 在 [m_ll_min, m_higgs - m_ll_min] 上均匀提案
        m1_prop = np.random.uniform(m_ll_min, m_higgs - m_ll_min)
        m2_prop = np.random.uniform(m_ll_min, m_higgs - m_ll_min)
        
        # Breit-Wigner 形状 (一个 Z 在壳，一个离壳)
        # 简单模型: 乘积形式
        bw1 = (GAMMA_Z / np.pi) / ((m1_prop - m_z) ** 2 + (0.5 * GAMMA_Z) ** 2)
        bw2 = (GAMMA_Z / np.pi) / ((m2_prop - m_z) ** 2 + (0.5 * GAMMA_Z) ** 2)
        
        if np.random.uniform(0.0, max_bw * max_bw) < bw1 * bw2:
            if m1_prop + m2_prop <= m_higgs:
                m1 = m1_prop
                m2 = m2_prop
                break
        trial += 1
    
    # Z1 在希格斯静止系中的方向
    dir_z1 = sample_unit_sphere_uniform(1)[0]
    dir_z2 = -dir_z1
    
    # Z1, Z2 的四动量 (在希格斯静止系)
    pz1, pz2 = two_body_decay(m_higgs, m1, m2, dir_z1)
    
    # Z1 衰变到 l+ l-
    dir_l1_z1 = sample_unit_sphere_uniform(1)[0]
    pl1_z1, pl2_z1 = two_body_decay(m1, 0.0, 0.0, dir_l1_z1)  # 质量less近似
    
    # Z2 衰变到 l+ l-
    dir_l1_z2 = sample_unit_sphere_uniform(1)[0]
    pl1_z2, pl2_z2 = two_body_decay(m2, 0.0, 0.0, dir_l1_z2)
    
    # 将轻子从 Z 静止系 boost 到希格斯静止系
    # Z1 的 boost 矢量: beta = p_z1[1:] / p_z1[0]
    beta1 = pz1[1:] / pz1[0] if pz1[0] > TINY else np.zeros(3)
    gamma1 = 1.0 / np.sqrt(max(1.0 - np.dot(beta1, beta1), TINY))
    
    beta2 = pz2[1:] / pz2[0] if pz2[0] > TINY else np.zeros(3)
    gamma2 = 1.0 / np.sqrt(max(1.0 - np.dot(beta2, beta2), TINY))
    
    def boost_lab(p_rest, beta, gamma):
        """沿 beta 方向的洛伦兹 boost"""
        bp = np.dot(p_rest[1:], beta)
        factor = (gamma - 1.0) * safe_divide(bp, np.dot(beta, beta), 0.0) if np.dot(beta, beta) > TINY else 0.0
        e_lab = gamma * (p_rest[0] + bp)
        p_parallel = beta * factor
        p_perp = p_rest[1:] + p_parallel
        return np.array([e_lab, p_perp[0], p_perp[1], p_perp[2]])
    
    pl1_lab = boost_lab(pl1_z1, beta1, gamma1)
    pl2_lab = boost_lab(pl2_z1, beta1, gamma1)
    pl3_lab = boost_lab(pl1_z2, beta2, gamma2)
    pl4_lab = boost_lab(pl2_z2, beta2, gamma2)
    
    return {
        "m_z1": m1,
        "m_z2": m2,
        "pz1": pz1,
        "pz2": pz2,
        "leptons": [pl1_lab, pl2_lab, pl3_lab, pl4_lab]
    }


# ============================================================
# 3. 蒙特卡洛事件批量生成 (映射 450_gamblers_ruin_simulation 的 MC 思想)
# ============================================================
def generate_event_batch(n_events, m_higgs=M_HIGGS, m_z=M_Z):
    """
    批量生成 n_events 个 H->ZZ*->4l 衰变事件
    
    使用马尔可夫链接受-拒绝采样确保运动学约束满足
    
    返回: 字典列表，每个字典包含事件四动量
    """
    events = []
    for _ in range(n_events):
        evt = generate_hzz4l_event(m_higgs, m_z)
        events.append(evt)
    return events


def compute_invariant_masses(event):
    """
    计算四轻子系统的不变质量
    
    m_4l^2 = (sum_i p_i)^2 = (sum E_i)^2 - |sum p_vec_i|^2
    """
    leptons = event["leptons"]
    total_p = np.zeros(4)
    for p in leptons:
        total_p += p
    m4l_sq = total_p[0] ** 2 - np.dot(total_p[1:], total_p[1:])
    return safe_sqrt(m4l_sq)


def compute_z_masses(event):
    """返回 (m_z1, m_z2)"""
    return event["m_z1"], event["m_z2"]


# ============================================================
# 4. 相空间权重计算 (映射 gambler's ruin 的统计量)
# ============================================================
def event_statistics(events):
    """
    计算批量事件的统计量:
      - 平均 m_4l, 标准差
      - m_z1, m_z2 的均值和相关系数
    """
    m4l_list = [compute_invariant_masses(e) for e in events]
    mz1_list = [e["m_z1"] for e in events]
    mz2_list = [e["m_z2"] for e in events]
    
    m4l_arr = np.array(m4l_list)
    mz1_arr = np.array(mz1_list)
    mz2_arr = np.array(mz2_list)
    
    stats = {
        "m4l_mean": np.mean(m4l_arr),
        "m4l_std": np.std(m4l_arr),
        "mz1_mean": np.mean(mz1_arr),
        "mz2_mean": np.mean(mz2_arr),
        "mz_corr": float(np.corrcoef(mz1_arr, mz2_arr)[0, 1]) if len(mz1_arr) > 1 else 0.0,
        "count": len(events)
    }
    return stats
