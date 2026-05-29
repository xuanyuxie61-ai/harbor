"""
yukawa_physics.py
=================
Dusty plasma interaction physics synthesized from seed projects:
  - 281_diff2_center (centered finite difference for 2nd derivative)
  - 323_e_spigot (spigot algorithm for computing e)

Core physics:
  - Screened Coulomb (Yukawa) potential and force
  - Debye shielding length calculation
  - Coupling parameter Gamma (determines phase state)
  - Centered finite differences for potential derivatives
  - High-precision spigot computation of e for exponential terms
"""

import numpy as np


def diff2_center(f_func, x, h=1e-5):
    """
    Second derivative via centered finite differences.
    
    Based on seed 281_diff2_center.
    Formula:
      f''(x) = [ f(x+h) - 2f(x) + f(x-h) ] / h^2  +  O(h^2)
    
    For smooth functions, the truncation error is proportional to h^2.
    """
    x = float(x)
    h = float(h)
    if h <= 0.0:
        raise ValueError("Step size h must be positive")
    return (f_func(x + h) - 2.0 * f_func(x) + f_func(x - h)) / (h * h)


def e_spigot(n_digits):
    """
    Compute decimal digits of Euler's number e = 2.71828... using spigot algorithm.
    
    Based on seed 323_e_spigot.
    The algorithm uses only integer arithmetic on a fixed-size array.
    
    Mathematical basis:
      e = 2 + sum_{k=1}^{infinity} 1/k!
    
    Algorithm:
      Initialize array a[0..n] = 1.
      For each digit:
        1. Multiply every element by 10
        2. Propagate carries from right to left:
             q = floor( a[i] / (i+2) )
             a[i] = a[i] mod (i+2)
        3. The carry q becomes the next decimal digit.
    """
    if n_digits <= 0:
        return [2]
    n = n_digits + 2
    a = np.ones(n, dtype=np.int64)
    digits = [2]  # leading digit
    
    for _ in range(n_digits):
        a *= 10
        carry = 0
        for i in range(n - 1, -1, -1):
            q = a[i] // (i + 2)
            a[i] = a[i] % (i + 2)
            if i > 0:
                a[i - 1] += q
            else:
                carry = q
        digits.append(int(carry))
    return digits[:n_digits + 1]


def yukawa_potential(r, Q_eff, lambda_D):
    """
    Screened Coulomb (Yukawa) potential between two charged dust grains.
    
    U(r) = (Q_eff^2 / (4 * pi * eps0 * r)) * exp(-r / lambda_D)
    
    Parameters:
      r        : inter-particle distance [m]
      Q_eff    : effective dust charge [C]
      lambda_D : Debye shielding length [m]
    
    Physical interpretation:
      The potential reduces to bare Coulomb at r << lambda_D,
      and is exponentially screened at r >> lambda_D.
    """
    eps0 = 8.854187817e-12  # vacuum permittivity [F/m]
    if r <= 0.0:
        return 0.0
    return (Q_eff**2 / (4.0 * np.pi * eps0 * r)) * np.exp(-r / lambda_D)


def yukawa_force_magnitude(r, Q_eff, lambda_D):
    """
    Magnitude of the Yukawa force: F = -dU/dr.
    
    F(r) = (Q_eff^2 / (4*pi*eps0)) * exp(-r/lambda_D) * (1/r^2 + 1/(r*lambda_D))
    
    The force is repulsive for like charges and always points along the
    inter-particle separation vector.
    """
    # TODO: Implement the Yukawa force magnitude formula.
    # HINT: The force is the negative derivative of the Yukawa potential U(r).
    #       F(r) = (Q_eff^2 / (4*pi*eps0)) * exp(-r/lambda_D) * (1/r^2 + 1/(r*lambda_D))
    #       Return 0.0 for r < 1e-15 to avoid singularity.
    raise NotImplementedError("Hole 1: yukawa_force_magnitude is not implemented.")


def yukawa_force_vector(r_vec, Q_eff, lambda_D):
    """
    Vector Yukawa force exerted on particle 1 by particle 2.
    
    F_12 = -dU/dr * (r_vec / |r_vec|)
         = -F(r) * r_hat
    
    where r_vec = r_1 - r_2 and r_hat = r_vec / |r_vec|.
    """
    r = np.linalg.norm(r_vec)
    if r < 1e-15:
        return np.zeros_like(r_vec)
    fm = yukawa_force_magnitude(r, Q_eff, lambda_D)
    return -fm * (r_vec / r)


def debye_length(n_e, T_e):
    """
    Electron Debye shielding length.
    
    lambda_D = sqrt( eps0 * k_B * T_e / (n_e * e^2) )
    
    Parameters:
      n_e : electron number density [m^-3]
      T_e : electron temperature [K]
    """
    eps0 = 8.854187817e-12
    k_B = 1.380649e-23   # Boltzmann constant [J/K]
    e = 1.602176634e-19  # elementary charge [C]
    return np.sqrt(eps0 * k_B * T_e / (n_e * e**2))


def coupling_parameter(Q_eff, n_dust, T_dust, lambda_D):
    """
    Coupling parameter Gamma for a dusty plasma.
    
    Gamma = (Q_eff^2 / (4*pi*eps0 * a_WS * k_B * T_dust)) * exp(-kappa)
    
    where:
      a_WS  = (3 / (4*pi*n_dust))^(1/3)  is the Wigner-Seitz radius
      kappa = a_WS / lambda_D
    
    Phase transition to the crystalline state occurs at Gamma_c ~ 170
    for 3D isotropic Yukawa systems (Ikezi criterion).
    For Gamma >> Gamma_c: crystalline (plasma crystal)
    For Gamma << Gamma_c: gaseous/liquid
    """
    eps0 = 8.854187817e-12
    k_B = 1.380649e-23
    a_ws = (3.0 / (4.0 * np.pi * n_dust)) ** (1.0 / 3.0)
    kappa = a_ws / lambda_D
    gamma = (Q_eff**2 / (4.0 * np.pi * eps0 * a_ws * k_B * T_dust)) * np.exp(-kappa)
    return gamma


def wigner_seitz_radius(n_dust):
    """
    Wigner-Seitz radius for a 3D system.
    
    a_WS = (3 / (4 * pi * n_dust))^(1/3)
    """
    return (3.0 / (4.0 * np.pi * n_dust)) ** (1.0 / 3.0)
