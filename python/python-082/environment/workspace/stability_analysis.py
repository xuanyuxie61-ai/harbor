"""
stability_analysis.py
=====================
Absolute stability analysis of time integration schemes for damage evolution.

Incorporates core algorithms from:
- 104_boundary_locus : Stability region computation for ODE solvers
    applied to the test equation y' = lambda*y.

Scientific role:
    Analyzes the stability of numerical time integrators used for
    damage evolution ODEs. For nonlinear damage accumulation, the
    local linearization leads to the test equation:
        d(d)/dt = lambda_eff * d
    where lambda_eff = d(dot{d})/dd is the local eigenvalue of the
    damage evolution law. The stability region of the integrator
    must contain lambda_eff * dt for numerical stability.

Key formulas:
-----------
1. Test equation for stability:
    y' = lambda * y,   z = h * lambda
    The numerical method gives y_{n+1} = R(z) * y_n.
    Stability requires |R(z)| <= 1.

2. Stability function for common methods:
   - Forward Euler: R(z) = 1 + z
   - Backward Euler: R(z) = 1 / (1 - z)
   - Trapezoidal: R(z) = (1 + z/2) / (1 - z/2)
   - RK4: R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24

3. Boundary locus method:
    The stability boundary is the set of z such that |R(z)| = 1.
    For A-stable methods, the entire left half-plane is stable.

4. Local eigenvalue of damage ODE:
    For dd/dt = f(d), linearize: lambda_eff = df/dd |_{d=d_n}
    For the van der Pol-type model:
        lambda_eff = -1/epsilon * (3*d^2 - a)

5. CFL-like condition for damage:
    dt <= dt_crit = C / |lambda_eff|_max
    where C depends on the integrator (C=2 for RK4, C=1 for FE).
"""

import numpy as np


def stability_function(z, method='rk4'):
    """
    Evaluate the stability function R(z) for various ODE methods.

    Parameters
    ----------
    z : complex or ndarray
        z = h * lambda.
    method : str
        'fe', 'be', 'trapezoidal', 'rk4', 'ab2', 'bdf2'

    Returns
    -------
    R : complex or ndarray
    """
    z = np.asarray(z, dtype=complex)
    method = method.lower()

    if method == 'fe':
        return 1.0 + z
    elif method == 'be':
        return 1.0 / (1.0 - z)
    elif method == 'trapezoidal':
        return (1.0 + 0.5 * z) / (1.0 - 0.5 * z)
    elif method == 'rk4':
        return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0
    elif method == 'ab2':
        # Adams-Bashforth 2-step
        return np.ones_like(z)  # Simplified
    elif method == 'bdf2':
        # BDF2: (3-4R+R^2) = 2zR => R = [2 - sqrt(4-2z)] / (1 - 2z/3)
        # Simplified approximation
        return 1.0 / (1.0 - z + 0.5 * z ** 2)
    else:
        raise ValueError(f"Unknown method: {method}")


def stability_region_grid(method='rk4', xlim=(-5, 5), ylim=(-5, 5),
                          npts=401):
    """
    Compute |R(z)| on a fine grid in the complex plane.

    Parameters
    ----------
    method : str
    xlim, ylim : tuple
        Real and imaginary ranges.
    npts : int
        Grid resolution.

    Returns
    -------
    X, Y, Rabs : ndarray
        Grid coordinates and |R(z)| values.
    """
    x = np.linspace(xlim[0], xlim[1], npts)
    y = np.linspace(ylim[0], ylim[1], npts)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    Rval = stability_function(Z, method)
    Rabs = np.abs(Rval)
    return X, Y, Rabs


def is_stable(method, z):
    """Check if z = h*lambda lies in the stability region."""
    Rabs = np.abs(stability_function(z, method))
    return Rabs <= 1.0 + 1e-10


def compute_cfl_damage(damage_model, d_state, dt, method='rk4'):
    """
    Compute the local CFL number for damage evolution.

    Parameters
    ----------
    damage_model : CyclicDamageModel
    d_state : ndarray
        Current damage state [d_f, d_m].
    dt : float
        Time step.
    method : str

    Returns
    -------
    cfl : float
        CFL = dt * |lambda_eff|.
    stable : bool
    """
    d_f = d_state[0]
    # Local eigenvalue of the van der Pol-type damage ODE
    lambda_eff = -(1.0 / damage_model.epsilon) * (3.0 * d_f ** 2 - damage_model.a_param)
    z = dt * lambda_eff
    cfl = abs(z)
    stable = is_stable(method, z)
    return cfl, stable


def recommend_timestep(damage_model, d_state, method='rk4', safety=0.8):
    """
    Recommend a stable time step for damage evolution.

    Parameters
    ----------
    damage_model : CyclicDamageModel
    d_state : ndarray
    method : str
    safety : float
        Safety factor.

    Returns
    -------
    dt_max : float
        Maximum stable time step.
    """
    d_f = d_state[0]
    lambda_eff = -(1.0 / damage_model.epsilon) * (3.0 * d_f ** 2 - damage_model.a_param)
    lambda_max = abs(lambda_eff)

    if method == 'fe':
        C = 1.0
    elif method == 'rk4':
        C = 2.78  # Approximate stability limit on real axis
    elif method == 'be':
        C = 1e12  # A-stable
    elif method == 'trapezoidal':
        C = 1e12  # A-stable
    else:
        C = 2.0

    if lambda_max < 1e-14:
        return 1.0
    dt_max = safety * C / lambda_max
    return dt_max


def a_stability_test(method, n_test=1000):
    """
    Test A-stability by evaluating |R(z)| for z in left half-plane.

    A method is A-stable if |R(z)| <= 1 for all Re(z) < 0.

    Parameters
    ----------
    method : str
    n_test : int
        Number of random test points.

    Returns
    -------
    is_a_stable : bool
    max_amplification : float
    """
    # Sample random points in left half-plane
    re_z = -np.random.rand(n_test) * 10.0
    im_z = (np.random.rand(n_test) - 0.5) * 20.0
    z = re_z + 1j * im_z
    Rabs = np.abs(stability_function(z, method))
    max_amp = np.max(Rabs)
    return max_amp <= 1.0 + 1e-6, max_amp


def stability_diagnostic(damage_model, t_span, n_steps, method='rk4'):
    """
    Run a full stability diagnostic for a damage evolution simulation.

    Returns statistics on CFL numbers and stability violations.
    """
    y0 = np.array([0.01, 0.01])
    t_array, y_array = damage_model.rk4_integrate(y0, t_span[0], t_span[1], n_steps)
    dt = t_array[1] - t_array[0]

    cfl_values = []
    violations = 0
    for y in y_array:
        cfl, stable = compute_cfl_damage(damage_model, y, dt, method)
        cfl_values.append(cfl)
        if not stable:
            violations += 1

    return {
        'dt': dt,
        'cfl_mean': np.mean(cfl_values),
        'cfl_max': np.max(cfl_values),
        'violations': violations,
        'violation_ratio': violations / len(cfl_values)
    }
