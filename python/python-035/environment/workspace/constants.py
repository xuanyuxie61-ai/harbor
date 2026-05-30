import numpy as np




M_HIGGS = 125.10
M_Z = 91.1876
GAMMA_Z = 2.4952
GAMMA_H = 4.07e-3
ALPHA_EM = 1.0 / 137.036
G_F = 1.1663787e-5
SIN2THETA_W = 0.23121
E_CHARGE = np.sqrt(4.0 * np.pi * ALPHA_EM)




def g_weak():
    v = 1.0 / np.sqrt(np.sqrt(2.0) * G_F)
    return 2.0 * (M_Z * np.sqrt(1.0 - SIN2THETA_W)) / v

def yukawa_tau():
    m_tau = 1.77686
    v = 1.0 / np.sqrt(np.sqrt(2.0) * G_F)
    return m_tau / v




def kine_bounds():
    m_l = 0.000511
    m_ll_min = 2.0 * m_l
    m_ll_max = M_HIGGS - 2.0 * m_l
    m_4l_min = 4.0 * m_l
    m_4l_max = M_HIGGS
    return {
        "m_l": m_l,
        "m_ll_min": m_ll_min,
        "m_ll_max": m_ll_max,
        "m_4l_min": m_4l_min,
        "m_4l_max": m_4l_max,
    }




def breit_wigner_params():
    return {
        "m_z": M_Z,
        "gamma_z": GAMMA_Z,
        "m_h": M_HIGGS,
        "gamma_h": GAMMA_H,
    }




TINY = 1.0e-15
EPS = np.finfo(float).eps
MAX_ITER = 10000
