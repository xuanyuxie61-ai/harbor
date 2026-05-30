import numpy as np
from constants import M_HIGGS, M_Z, GAMMA_Z, TINY
from utils import safe_sqrt, safe_divide




def sample_unit_sphere_uniform(n):
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

    if i < n:
        theta = np.random.uniform(0.0, np.pi, n - i)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n - i)
        points[i:, 0] = np.sin(theta) * np.cos(phi)
        points[i:, 1] = np.sin(theta) * np.sin(phi)
        points[i:, 2] = np.cos(theta)
    return points


def sample_positive_quadrant_circle(n):
    theta = np.random.uniform(0.0, 0.5 * np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    return np.column_stack([x, y])





def two_body_decay(m_parent, m1, m2, direction):
    if m_parent < m1 + m2:

        return np.array([m1, 0.0, 0.0, 0.0]), np.array([m2, 0.0, 0.0, 0.0])
    
    e1 = (m_parent ** 2 + m1 ** 2 - m2 ** 2) / (2.0 * m_parent)
    e2 = (m_parent ** 2 + m2 ** 2 - m1 ** 2) / (2.0 * m_parent)
    p_mag = safe_sqrt(e1 ** 2 - m1 ** 2)
    
    p1 = np.array([e1, p_mag * direction[0], p_mag * direction[1], p_mag * direction[2]])
    p2 = np.array([e2, -p_mag * direction[0], -p_mag * direction[1], -p_mag * direction[2]])
    return p1, p2


def generate_hzz4l_event(m_higgs=M_HIGGS, m_z=M_Z):
    m_ll_min = 0.001
    


    max_bw = 1.0 / (m_z * GAMMA_Z)
    max_trials = 10000
    
    m1 = m_z
    m2 = m_z
    trial = 0
    while trial < max_trials:

        m1_prop = np.random.uniform(m_ll_min, m_higgs - m_ll_min)
        m2_prop = np.random.uniform(m_ll_min, m_higgs - m_ll_min)
        


        bw1 = (GAMMA_Z / np.pi) / ((m1_prop - m_z) ** 2 + (0.5 * GAMMA_Z) ** 2)
        bw2 = (GAMMA_Z / np.pi) / ((m2_prop - m_z) ** 2 + (0.5 * GAMMA_Z) ** 2)
        
        if np.random.uniform(0.0, max_bw * max_bw) < bw1 * bw2:
            if m1_prop + m2_prop <= m_higgs:
                m1 = m1_prop
                m2 = m2_prop
                break
        trial += 1
    

    dir_z1 = sample_unit_sphere_uniform(1)[0]
    dir_z2 = -dir_z1
    

    pz1, pz2 = two_body_decay(m_higgs, m1, m2, dir_z1)
    

    dir_l1_z1 = sample_unit_sphere_uniform(1)[0]
    pl1_z1, pl2_z1 = two_body_decay(m1, 0.0, 0.0, dir_l1_z1)
    

    dir_l1_z2 = sample_unit_sphere_uniform(1)[0]
    pl1_z2, pl2_z2 = two_body_decay(m2, 0.0, 0.0, dir_l1_z2)
    


    beta1 = pz1[1:] / pz1[0] if pz1[0] > TINY else np.zeros(3)
    gamma1 = 1.0 / np.sqrt(max(1.0 - np.dot(beta1, beta1), TINY))
    
    beta2 = pz2[1:] / pz2[0] if pz2[0] > TINY else np.zeros(3)
    gamma2 = 1.0 / np.sqrt(max(1.0 - np.dot(beta2, beta2), TINY))
    
    def boost_lab(p_rest, beta, gamma):
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





def generate_event_batch(n_events, m_higgs=M_HIGGS, m_z=M_Z):
    events = []
    for _ in range(n_events):
        evt = generate_hzz4l_event(m_higgs, m_z)
        events.append(evt)
    return events


def compute_invariant_masses(event):
    leptons = event["leptons"]
    total_p = np.zeros(4)
    for p in leptons:
        total_p += p
    m4l_sq = total_p[0] ** 2 - np.dot(total_p[1:], total_p[1:])
    return safe_sqrt(m4l_sq)


def compute_z_masses(event):
    return event["m_z1"], event["m_z2"]





def event_statistics(events):
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
