
import numpy as np


class BinaryBlackHole:
    

    G_SI = 6.67430e-11
    C_SI = 2.99792458e8
    MSUN_SI = 1.98847e30
    MPC_SI = 3.08567758e22
    
    def __init__(self, m1_msun=30.0, m2_msun=25.0, a1=0.0, a2=0.0,
                 D_L_mpc=400.0, inclination=0.0, phi_c=0.0, psi=0.0, t_c=0.0):
        self.m1_msun = float(m1_msun)
        self.m2_msun = float(m2_msun)
        self.a1 = float(np.clip(a1, -0.999, 0.999))
        self.a2 = float(np.clip(a2, -0.999, 0.999))
        self.D_L_mpc = float(D_L_mpc)
        self.inclination = float(inclination)
        self.phi_c = float(phi_c)
        self.psi = float(psi)
        self.t_c = float(t_c)
        

        self.m1 = self.m1_msun * self.MSUN_SI * self.G_SI / self.C_SI**3
        self.m2 = self.m2_msun * self.MSUN_SI * self.G_SI / self.C_SI**3
        self.M = self.m1 + self.m2
        self.eta = self.m1 * self.m2 / self.M**2
        self.eta = np.clip(self.eta, 1e-6, 0.25)
        
        self.M_c = self.M * self.eta**(3.0 / 5.0)
        self.D_L = self.D_L_mpc * self.MPC_SI / self.C_SI
        

        self.f_isco = self._compute_isco_frequency()
        

        self.t_inspiral = self._compute_inspiral_time()
    
    def _compute_isco_frequency(self):
        return 1.0 / (6.0**1.5 * np.pi * self.M)
    
    def _compute_inspiral_time(self):



        raise NotImplementedError("Hole 2: _compute_inspiral_time 核心公式待补全")
    
    def orbital_separation(self, f_gw):
        f_gw = np.asarray(f_gw, dtype=np.float64)
        f_gw = np.where(f_gw <= 0, 1e-10, f_gw)
        return (self.M / (np.pi * f_gw)**2)**(1.0 / 3.0)
    
    def orbital_velocity(self, f_gw):
        f_gw = np.asarray(f_gw, dtype=np.float64)
        return (np.pi * self.M * f_gw)**(1.0 / 3.0)
    
    def energy_flux(self, f_gw):
        f_gw = np.asarray(f_gw, dtype=np.float64)
        x = np.pi * self.M * f_gw
        return (32.0 / 5.0) * self.eta**2 * x**(10.0 / 3.0)
    
    def strain_amplitude(self, f_gw):
        f_gw = np.asarray(f_gw, dtype=np.float64)
        return (self.M_c**(5.0 / 3.0) / self.D_L) * (np.pi * f_gw)**(2.0 / 3.0)
    
    def to_parameter_dict(self):
        return {
            'm1': self.m1_msun,
            'm2': self.m2_msun,
            'a1': self.a1,
            'a2': self.a2,
            'D_L': self.D_L_mpc,
            'inclination': self.inclination,
            'phi_c': self.phi_c,
            'psi': self.psi,
            't_c': self.t_c
        }
    
    @classmethod
    def from_parameter_dict(cls, params):
        return cls(
            m1_msun=params.get('m1', 30.0),
            m2_msun=params.get('m2', 25.0),
            a1=params.get('a1', 0.0),
            a2=params.get('a2', 0.0),
            D_L_mpc=params.get('D_L', 400.0),
            inclination=params.get('inclination', 0.0),
            phi_c=params.get('phi_c', 0.0),
            psi=params.get('psi', 0.0),
            t_c=params.get('t_c', 0.0)
        )
    
    def info(self):
        return {
            'm1_msun': self.m1_msun,
            'm2_msun': self.m2_msun,
            'total_mass_msun': self.m1_msun + self.m2_msun,
            'chirp_mass_msun': (self.m1_msun * self.m2_msun)**(3.0/5.0) / (self.m1_msun + self.m2_msun)**(1.0/5.0),
            'symmetric_mass_ratio': self.eta,
            'spin1': self.a1,
            'spin2': self.a2,
            'distance_mpc': self.D_L_mpc,
            'f_isco_hz': self.f_isco,
            't_inspiral_s': self.t_inspiral,
        }


def mass_ratio_to_components(q, M_total):
    if q < 1.0:
        q = 1.0 / q
    m1 = q * M_total / (1.0 + q)
    m2 = M_total / (1.0 + q)
    return m1, m2


def effective_spin(m1, m2, a1, a2):
    return (m1 * a1 + m2 * a2) / (m1 + m2)


def precession_spin(m1, m2, a1, a2, theta1, theta2):
    q = m1 / m2
    term1 = np.abs(a1 * np.sin(theta1))
    term2 = q * (4.0 * q + 3.0) / (4.0 + 3.0 * q) * np.abs(a2 * np.sin(theta2))
    return max(term1, term2)
