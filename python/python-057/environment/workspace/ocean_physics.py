
import numpy as np


G = 9.81
RHO0 = 1025.0
NU = 1.0e-6
KAPPA = 1.4e-7
OMEGA_EARTH = 7.2921159e-5
F_CORIOLIS = 1.0e-4


def density_profile(z, rho0=RHO0, drho_dz=-0.01):
    z = np.asarray(z)
    rho = rho0 + drho_dz * z

    rho = np.clip(rho, 1020.0, 1030.0)
    return rho


def buoyancy_frequency(z, rho0=RHO0, drho_dz=-0.01):
    z = np.asarray(z)
    N2 = -(G / rho0) * drho_dz
    N2 = np.maximum(N2, 1.0e-8)
    N = np.sqrt(N2)
    return N


def richardson_number(dudz, dvdz, N):
    shear_squared = dudz**2 + dvdz**2
    N2 = N**2

    Ri = np.where(shear_squared > 1.0e-12,
                  N2 / shear_squared,
                  1.0e6)
    return Ri


def internal_wave_dispersion(kh, m, N, f=F_CORIOLIS):



    raise NotImplementedError("待实现: 内波色散关系")


def group_velocity(kh, m, N, f=F_CORIOLIS):
    kh = np.asarray(kh)
    m = np.asarray(m)
    omega = internal_wave_dispersion(kh, m, N, f)
    denom = (kh**2 + m**2)**2
    denom = np.where(denom < 1.0e-12, 1.0e-12, denom)
    
    factor = (N**2 - f**2) / denom
    cgx = (kh / omega) * factor * m**2
    cgz = -(m / omega) * factor * kh**2
    return cgx, cgz


def turbulent_dissipation_rate(Ri, shear_squared, nu=NU, mixing_efficiency=0.2):
    Ri = np.asarray(Ri)
    Ri_critical = 0.25
    

    mix_func = np.maximum(0.0, 1.0 - Ri / Ri_critical)
    epsilon = nu * shear_squared * mix_func
    


    N2 = np.where(Ri > 1.0e-6, shear_squared * Ri, 1.0e-8)
    Kz = mixing_efficiency * epsilon / N2
    Kz = np.clip(Kz, 1.0e-7, 1.0e-1)
    
    return epsilon, Kz


def breaking_criterion(amplitude, wavelength, N, depth):
    kh = 2.0 * np.pi / wavelength
    steepness = amplitude * kh
    


    critical_steepness = 0.2 * np.sqrt(N**2 * depth / G)
    critical_steepness = np.clip(critical_steepness, 0.05, 0.5)
    
    is_breaking = steepness > critical_steepness
    return is_breaking, steepness, critical_steepness


def thope_internal_wave_spectrum(kh, N, f=F_CORIOLIS, E0=6.3e-5):
    kh = np.asarray(kh)
    kh = np.where(kh < 1.0e-8, 1.0e-8, kh)
    
    k_star = 2.0 * np.pi / 1000.0
    

    spectrum = E0 * (kh / k_star)**(-2.0)
    

    omega_min = f
    omega_max = N
    omega = internal_wave_dispersion(kh, 2.0 * np.pi / 200.0, N, f)
    
    freq_factor = np.ones_like(omega)
    freq_factor = np.where((omega < omega_min) | (omega > omega_max), 0.0, freq_factor)
    
    spectrum = spectrum * freq_factor
    return spectrum
