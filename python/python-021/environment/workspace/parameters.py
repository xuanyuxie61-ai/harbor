
import numpy as np




MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854187817e-12
KB = 1.380649e-23
QE = 1.602176634e-19
ME = 9.10938356e-31
MP = 1.6726219e-27
MD = 3.343583719e-27
MT = 5.006412e-27
HE3_MASS = 5.006412e-27 * 1.5
C_LIGHT = 2.99792458e8




R0 = 6.2
a_minor = 2.0
B0 = 5.3
KAPPA = 1.7
DELTA = 0.33
q0 = 1.0
q_edge = 3.0




N_E_AXIS = 1.0e20
N_E_PED = 0.4e20
T_E_AXIS = 20.0e3
T_E_PED = 3.0e3
T_I_AXIS = 20.0e3
Z_EFF = 1.8




DT_ENERGY_FUS = 17.6e6
DT_CROSS_PEAK = 5.0e-22




NR_EQUIL = 129
NTHETA_EQUIL = 129
N_DDE_STEPS = 2000
N_FFT = 1024
N_FEKETE = 16
N_GAUSS = 64




TAU_TRANSPORT = 0.15
BETA_TRANSPORT = 2.0
gamma_transport = 1.0
N_TRANSPORT = 9.65




I1_DRIFT = 1.6
I2_DRIFT = 1.0
I3_DRIFT = 2.0 / 3.0




MM_TITLE_DEFAULT = "Tokamak MHD Stiffness Matrix"
MM_KEY_DEFAULT = "TOKAMAK1"
MM_TYPE_DEFAULT = "RUA"
MM_IFMT_DEFAULT = 8
MM_JOB_DEFAULT = 2


def get_equilibrium_params():
    return {
        "R0": R0,
        "a_minor": a_minor,
        "B0": B0,
        "kappa": KAPPA,
        "delta": DELTA,
        "q0": q0,
        "q_edge": q_edge,
        "nr": NR_EQUIL,
        "ntheta": NTHETA_EQUIL,
    }


def get_transport_params():
    return {
        "gamma": gamma_transport,
        "beta": BETA_TRANSPORT,
        "n": N_TRANSPORT,
        "tau": TAU_TRANSPORT,
        "t0": 0.0,
        "y0": np.array([0.5]),
        "tstop": 10.0,
    }


def get_fusion_params():
    return {
        "k": 1.0e-18,
        "t0": 0.0,
        "y0": np.array([1.0e20, 1.0e20, 0.0]),
        "tstop": 100.0,
    }


def get_drift_params():
    return {
        "i1": I1_DRIFT,
        "i2": I2_DRIFT,
        "i3": I3_DRIFT,
        "t0": 0.0,
        "y0": np.array([np.cos(0.9), 0.0, np.sin(0.9)]),
        "tstop": 50.0,
    }
