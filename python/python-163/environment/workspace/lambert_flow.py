"""
lambert_flow.py
===============
Wellbore flow model using the Lambert W function for nonlinear
pressure-temperature relationships in geothermal production.

Incorporates algorithms from:
  - 644_lambert_w: accurate approximation of the Lambert W function

Mathematical formulation:
The Lambert W function W(z) is defined as the inverse of f(w) = w e^w,
i.e., W(z) e^{W(z)} = z.

For wellbore flow, the Darcy-Weisbach equation with temperature-dependent
viscosity leads to transcendental equations solvable via Lambert W.

Consider pressure drop in a geothermal production well:
  \Delta p = f \frac{L}{D} \frac{\rho v^2}{2}

With temperature-dependent viscosity \mu(T), the friction factor f
in laminar flow is:
  f = \frac{64}{Re} = \frac{64 \mu}{\rho v D}

The mass flow rate \dot{m} = \rho v A is related to the pressure drop
through a nonlinear equation. For a simplified model with
\mu(T) = \mu_0 \exp[-\alpha (T - T_0)], we obtain:

  \dot{m} = \frac{\pi D^3 \Delta p}{128 \mu_0 L}
    \exp\left[\alpha (T - T_0)\right]

For more complex coupled equations, the Lambert W function appears
when solving for pressure in equations of the form:

  p \exp(\beta p) = C

which yields:
  p = \frac{1}{\beta} W(\beta C)

Another application: the Forchheimer equation for non-Darcy flow:
  -\nabla p = \frac{\mu}{k} \mathbf{v} + \beta_F \rho |\mathbf{v}| \mathbf{v}

For 1D steady flow with |v| = v:
  \frac{dp}{dx} = -\frac{\mu}{k} v - \beta_F \rho v^2

Integrating with compressibility gives a transcendental equation
solved by Lambert W.
"""

import numpy as np


def lambert_w_approx(x, branch=0, n_iter=1):
    """
    Approximate the Lambert W function for real arguments.

    Parameters
    ----------
    x : float or np.ndarray
        Argument.
    branch : int
        0 for principal branch W_0 (x >= -1/e),
        nonzero for lower branch W_{-1} (-1/e < x < 0).
    n_iter : int
        Number of Halley iterations.

    Returns
    -------
    w : float or np.ndarray
        Approximate W(x).
    """
    x = np.asarray(x, dtype=np.float64)
    scalar = (x.ndim == 0)
    x = np.atleast_1d(x)
    w = np.full_like(x, np.nan)

    em = -np.exp(-1.0)
    em9 = -np.exp(-9.0)
    c13 = 1.0 / 3.0
    c23 = 2.0 * c13
    em2 = 2.0 / em
    d12 = -em2
    tb = 0.5 ** 52
    tb2 = np.sqrt(tb)
    x0 = tb ** (1.0 / 6.0) * 0.5
    x1 = (1.0 - 17.0 * tb ** (2.0 / 7.0)) * em
    an3 = 8.0 / 3.0
    an4 = 135.0 / 83.0
    an5 = 166.0 / 39.0
    an6 = 3167.0 / 3549.0
    s2 = np.sqrt(2.0)
    s21 = 2.0 * s2 - 3.0
    s22 = 4.0 - 3.0 * s2
    s23 = s2 - 2.0

    mask_valid = x >= em if branch == 0 else ((x > em) & (x < 0))
    xx = x[mask_valid].copy()
    delx = xx - em

    if branch == 0:
        # Principal branch W_0
        m1 = np.abs(xx) <= x0
        m2 = (~m1) & (xx <= x1)
        m3 = (~m1) & (~m2) & (xx <= 20.0)
        m4 = (~m1) & (~m2) & (~m3)

        w_sub = np.zeros_like(xx)
        if np.any(m1):
            xm = xx[m1]
            w_sub[m1] = xm / (1.0 + xm / (1.0 + xm / (2.0 + xm / (0.6 + 0.34 * xm))))
        if np.any(m2):
            xm = xx[m2]
            reta = np.sqrt(d12 * (xm - em))
            w_sub[m2] = reta / (1.0 + reta / (3.0 + reta / (reta / (an4 + reta / (reta * an6 + an5)) + an3))) - 1.0
        if np.any(m3):
            xm = xx[m3]
            reta = s2 * np.sqrt(1.0 - xm / em)
            an2 = 4.612634277343749 * np.sqrt(np.sqrt(reta + 1.09556884765625))
            w_sub[m3] = reta / (1.0 + reta / (3.0 + (s21 * an2 + s22) * reta / (s23 * (an2 + reta)))) - 1.0
        if np.any(m4):
            xm = xx[m4]
            zl = np.log(xm)
            w_sub[m4] = np.log(xm / np.log(xm / zl ** np.exp(-1.124491989777808 / (0.4225028202459761 + zl))))
    else:
        # Lower branch W_{-1}
        m1 = xx <= x1
        m2 = (~m1) & (xx <= em9)
        m3 = (~m1) & (~m2)

        w_sub = np.zeros_like(xx)
        if np.any(m1):
            xm = xx[m1]
            reta = np.sqrt(d12 * (xm - em))
            w_sub[m1] = reta / (reta / (3.0 + reta / (reta / (an4 + reta / (reta * an6 - an5)) - an3)) - 1.0) - 1.0
        if np.any(m2):
            xm = xx[m2]
            zl = np.log(-xm)
            t = -1.0 - zl
            ts = np.sqrt(t)
            w_sub[m2] = zl - (2.0 * ts) / (s2 + (c13 - t / (270.0 + ts * 127.0471381349219)) * ts)
        if np.any(m3):
            xm = xx[m3]
            zl = np.log(-xm)
            eta = 2.0 - em2 * xm
            w_sub[m3] = np.log(xm / np.log(-xm / ((1.0 - 0.5043921323068457 * (zl + 1.0)) * (np.sqrt(eta) + eta / 3.0) + 1.0)))

    # Halley iteration
    for _ in range(n_iter):
        zn = np.log(xx / (w_sub + 1.0e-30)) - w_sub
        temp = 1.0 + w_sub
        temp2 = temp + c23 * zn
        temp2 = 2.0 * temp * temp2
        w_sub = w_sub * (1.0 + (zn / temp) * (temp2 - zn) / (temp2 - 2.0 * zn))

    w[mask_valid] = w_sub
    w[x == em] = -1.0

    if scalar:
        return float(w[0])
    return w


def wellbore_pressure_drop_lambert(m_dot, D_well, L_well, T_well, rho_f, mu_ref,
                                   alpha_visc=0.02, beta_forch=1.0e8):
    """
    Compute wellbore pressure drop using a Lambert-W-based solution
    to the coupled Forchheimer-Darcy equation with temperature-dependent viscosity.

    Parameters
    ----------
    m_dot : float
        Mass flow rate (kg/s).
    D_well : float
        Well diameter (m).
    L_well : float
        Well length (m).
    T_well : float
        Well temperature (K).
    rho_f : float
        Fluid density (kg/m^3).
    mu_ref : float
        Reference viscosity (Pa·s).
    alpha_visc : float
        Viscosity temperature coefficient.
    beta_forch : float
        Forchheimer coefficient (1/m).

    Returns
    -------
    dp : float
        Pressure drop (Pa).
    """
    if D_well <= 0 or L_well <= 0 or rho_f <= 0:
        raise ValueError("Physical parameters must be positive.")

    A_cross = np.pi * (D_well / 2.0) ** 2
    v = m_dot / (rho_f * A_cross)

    mu = mu_ref * np.exp(-alpha_visc * (T_well - 373.15))
    mu = max(mu, 1.0e-6)

    # Darcy-Forchheimer equation in form:
    # dp = a v + b v^2 where a = (mu/k_eff) * L, b = beta_F * rho * L
    # For a wellbore, use pipe-flow analogy:
    Re = rho_f * v * D_well / mu
    if Re < 2300.0:
        # Laminar: f = 64/Re
        f = 64.0 / max(Re, 1.0)
    else:
        # Blasius correlation for turbulent smooth pipes
        f = 0.3164 / (Re ** 0.25)

    dp_darcy = f * (L_well / D_well) * (rho_f * v ** 2) / 2.0

    # Forchheimer non-Darcy contribution
    dp_forch = beta_forch * rho_f * v ** 2 * L_well

    # Cap Forchheimer term to physically reasonable values
    dp_forch_capped = min(dp_forch, 10.0e6)  # cap at 10 MPa
    dp_total = dp_darcy + dp_forch_capped
    return dp_total


def solve_transcendental_pressure(C, beta):
    """
    Solve p * exp(beta * p) = C for p using Lambert W.
    Solution: p = W(beta * C) / beta

    Parameters
    ----------
    C : float
        Right-hand side constant.
    beta : float
        Exponential coefficient.

    Returns
    -------
    p : float
        Solution.
    """
    if beta == 0:
        raise ValueError("beta cannot be zero.")
    if C < 0 and beta > 0:
        raise ValueError("No real solution for C < 0, beta > 0.")

    arg = beta * C
    if arg < -np.exp(-1.0):
        # No real solution
        return np.nan

    W_val = lambert_w_approx(arg, branch=0)
    p = W_val / beta
    return p


def injection_well_pressure(m_dot, k_inj, mu_inj, h_inj, r_e, r_w, p_res):
    """
    Steady-state injection well pressure using the Peaceman equivalent radius.
    The pressure relationship involves:

    p_inj - p_res = \frac{m_dot \mu}{2 \pi k h} \ln\left(\frac{r_e}{r_w}\right)

    For compressible flow with linearized compressibility:
    The equation can be rearranged to a form involving Lambert W
    when incorporating non-linear skin effects.

    Parameters
    ----------
    m_dot : float
        Mass injection rate (kg/s).
    k_inj : float
        Permeability near well (m^2).
    mu_inj : float
        Injection fluid viscosity (Pa·s).
    h_inj : float
        Injection interval thickness (m).
    r_e : float
        Drainage radius (m).
    r_w : float
        Wellbore radius (m).
    p_res : float
        Reservoir pressure (Pa).

    Returns
    -------
    p_inj : float
        Injection pressure (Pa).
    """
    if k_inj <= 0 or mu_inj <= 0 or h_inj <= 0 or r_w <= 0 or r_e <= r_w:
        raise ValueError("Physical parameters must be positive and consistent.")

    log_ratio = np.log(r_e / r_w)
    dp = (m_dot * mu_inj * log_ratio) / (2.0 * np.pi * k_inj * h_inj)
    p_inj = p_res + dp
    return p_inj
