
import numpy as np




class PhysicalConstants:
    BOLTZMANN: float = 1.380649e-23
    ELECTRON_MASS: float = 9.10938356e-31
    PROTON_MASS: float = 1.6726219e-27
    NEUTRON_MASS: float = 1.6749275e-27
    ELEMENTARY_CHARGE: float = 1.602176634e-19
    SPEED_OF_LIGHT: float = 2.99792458e8
    VACUUM_PERMITTIVITY: float = 8.854187817e-12
    PLANCK: float = 6.62607015e-34
    AVOGADRO: float = 6.02214076e23
    STEFAN_BOLTZMANN: float = 5.670374419e-8





class TargetParameters:

    R_ABLATION: float = 1.1e-3
    R_DT_ICE: float = 0.95e-3
    R_GAS: float = 0.4e-3


    RHO_CH: float = 1350.0
    RHO_DT: float = 250.0
    RHO_GAS: float = 0.3


    A_C: float = 12.0
    A_H: float = 1.0
    A_D: float = 2.0
    A_T: float = 3.0


    X_C: float = 1.0
    Y_H: float = 1.5

    @property
    def ablator_average_atomic_mass(self) -> float:
        return (self.X_C * self.A_C + self.Y_H * self.A_H) / (self.X_C + self.Y_H)

    @property
    def ablator_atomic_number(self) -> float:
        return (6.0 * self.X_C + 1.0 * self.Y_H) / (self.X_C + self.Y_H)





class LaserParameters:
    NUM_BEAMS: int = 192
    WAVELENGTH: float = 351.0e-9
    TOTAL_ENERGY: float = 1.8e6
    PULSE_DURATION: float = 15.0e-9
    POWER_PEAK: float = 350.0e12


    @staticmethod
    def power_profile(t: float, t0: float = 7.5e-9, sigma: float = 3.0e-9) -> float:
        if t < 0.0 or t > 20.0e-9:
            return 0.0
        p = LaserParameters.POWER_PEAK * np.exp(-(t - t0)**2 / (2.0 * sigma**2))
        return max(p, 0.0)





class NumericalParameters:
    N_RADIAL: int = 200
    T_MAX: float = 20.0e-9
    CFL: float = 0.3
    MAX_DT: float = 1.0e-12
    MIN_DT: float = 1.0e-16
    ADAPTIVE_TOL: float = 1.0e-6


    FLUX_LIMITER: float = 0.06
    MAX_FLUX_MULTIPLIER: float = 5.0


    PERTURBATION_MODE: int = 12
    PERTURBATION_AMPLITUDE: float = 1.0e-7


    MC_NEUTRON_SAMPLES: int = 5000





class EOSParameters:
    GAMMA_IDEAL: float = 5.0 / 3.0
    DEGENERACY_COEFF: float = 2.0
    COULOMB_CORRECTION: float = 0.3





class FusionParameters:
    Q_ALPHA: float = 3.5e6 * PhysicalConstants.ELEMENTARY_CHARGE
    Q_NEUTRON: float = 14.1e6 * PhysicalConstants.ELEMENTARY_CHARGE
    REACTIVITY_COEFF: np.ndarray = np.array([
        6.6610e-21, 2.4120e-14, 1.0290e-11,
        1.5630e-10, 1.6900e-9, 1.0200e-8,
        2.9750e-8, 4.7680e-8, 3.6970e-8,
        1.0400e-8
    ])

    @staticmethod
    def reactivity_dt(T_ion_kev: float) -> float:
        if T_ion_kev <= 0.0:
            return 0.0
        theta = T_ion_kev / (1.0 - (T_ion_kev * (0.0642 + 0.0149 * T_ion_kev))
                             / (1.0 + 0.0642 * T_ion_kev + 0.0149 * T_ion_kev**2))
        xi = (0.2396 * theta)**(1.0 / 3.0)

        if xi > 50.0:
            return 0.0
        sigma_v = 1.0e-6 * 1.17302e-9 * theta * np.sqrt(xi / (0.2396 * T_ion_kev**3)) \
            * np.exp(-3.0 * xi)
        return max(sigma_v, 0.0)





PC = PhysicalConstants()
TP = TargetParameters()
LP = LaserParameters()
NP = NumericalParameters()
EOS = EOSParameters()
FP = FusionParameters()
