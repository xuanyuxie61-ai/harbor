"""
Parton Shower Monte Carlo
==========================
Derived from 134_california_migration (discrete Markov chain / transfer matrix)
and 1086_sir_ode (ODE system evolution).

Implements a simplified angular-ordered parton shower where:
- The emission probability follows a continuous-time Markov process
  governed by the Sudakov form factor (from 134 migration's transfer matrix).
- The parton multiplicity evolves like an ODE system (from 1086 SIR model).

Physics model:
    Each parton can emit a softer parton with probability per unit
    evolution variable t = ln(Q^2/k_t^2):
        dP = (α_s / 2π) Σ_j ∫ dz P_{ij}(z) dt
    The no-emission probability is the Sudakov factor Δ(t).
"""

import numpy as np
from scipy.integrate import solve_ivp

from special_functions_qcd import (
    alpha_s_1loop, sudakov_quark, p_qq_lo, p_gq_lo, p_gg_lo,
    CF, CA, N_F
)
from cubature_integrator import integrate_adaptive_1d


class Parton:
    """Represents a single parton with 4-momentum and flavor."""
    
    def __init__(self, px, py, pz, E, flavor, pid=0):
        self.p = np.array([px, py, pz, E], dtype=float)
        self.flavor = flavor  # 'q', 'g', or 'qbar'
        self.pid = pid
        self.children = []
    
    @property
    def pt(self):
        return float(np.sqrt(self.p[0]**2 + self.p[1]**2))
    
    @property
    def mass(self):
        m2 = self.p[3]**2 - np.sum(self.p[:3]**2)
        return float(np.sqrt(max(m2, 0.0)))
    
    @property
    def eta(self):
        """Pseudorapidity."""
        p = self.p
        pt = self.pt
        if pt < 1e-12:
            return 1e6 if p[2] > 0 else -1e6
        theta = np.arctan2(pt, p[2])
        return -np.log(np.tan(theta / 2.0))
    
    @property
    def phi(self):
        return float(np.arctan2(self.p[1], self.p[0]))
    
    def scale(self):
        """Return a virtuality scale Q^2 = E^2 - |p|^2 (approx m^2)."""
        return float(max(self.mass**2, (0.5 * self.pt)**2 + 0.01))


def generate_hard_process(E_cm=14000.0, pt_hard=100.0, seed=42):
    """
    Generate a simplified 2→2 hard scattering event:
    q + qbar → q + qbar at central rapidity.
    
    Parameters
    ----------
    E_cm : float
        Center-of-mass energy in GeV.
    pt_hard : float
        Transverse momentum scale of hard scattering.
    seed : int
        RNG seed.
    
    Returns
    -------
    list of Parton
        Final-state partons from hard scattering.
    """
    rng = np.random.default_rng(seed)
    
    phi = rng.uniform(0.0, 2.0 * np.pi)
    eta = rng.uniform(-1.0, 1.0)
    
    # Parton 1
    px1 = pt_hard * np.cos(phi)
    py1 = pt_hard * np.sin(phi)
    pz1 = pt_hard * np.sinh(eta)
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2)
    
    # Parton 2 (back-to-back)
    px2 = -px1
    py2 = -py1
    pz2 = -pz1
    E2 = E1
    
    p1 = Parton(px1, py1, pz1, E1, 'q', pid=1)
    p2 = Parton(px2, py2, pz2, E2, 'qbar', pid=2)
    return [p1, p2]


def sample_z_and_phi(rng, flavor='q', zmin=0.01, zmax=0.99):
    """
    Sample splitting variable z and azimuthal angle φ from LO splitting
    functions using the acceptance-rejection method.
    
    Parameters
    ----------
    rng : Generator
        NumPy RNG instance.
    flavor : str
        'q' for quark→quark+g, 'g' for gluon→gg or g→qq̄.
    zmin, zmax : float
        Kinematic bounds.
    
    Returns
    -------
    z : float
        Momentum fraction carried by emitted parton.
    phi : float
        Azimuthal angle of emission.
    """
    if flavor == 'q':
        f_max = CF * (1.0 + zmax**2) / (1.0 - zmax)
    elif flavor == 'g':
        f_max = 2.0 * CA * (zmax / (1.0 - zmax) + (1.0 - zmax) / zmax + zmax * (1.0 - zmax))
    else:
        f_max = 1.0
    
    f_max = max(f_max, 1e-3)
    
    for _ in range(10000):
        z = rng.uniform(zmin, zmax)
        if flavor == 'q':
            f_val = CF * (1.0 + z**2) / (1.0 - z)
        elif flavor == 'g':
            f_val = 2.0 * CA * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))
        else:
            f_val = 1.0
        
        if rng.uniform(0.0, f_max) <= f_val:
            phi = rng.uniform(0.0, 2.0 * np.pi)
            return z, phi
    
    # Fallback
    return 0.5 * (zmin + zmax), rng.uniform(0.0, 2.0 * np.pi)


def run_parton_shower(initial_partons, Q_cut=1.0, z_cut=0.05,
                      max_multiplicity=200, seed=42):
    """
    Run a Monte Carlo parton shower starting from hard-scattering partons.
    
    Evolution variable: t = ln(Q^2 / Q_cut^2)
    Each step, a parton is chosen to emit; emission probability comes from
    the differential rate dP = (α_s/2π) P(z) dz dt.
    The no-emission probability is enforced by the Sudakov veto algorithm.
    
    Parameters
    ----------
    initial_partons : list of Parton
    Q_cut : float
        Infrared cutoff scale in GeV.
    z_cut : float
        Minimum momentum fraction for emission.
    max_multiplicity : int
        Safety cap on parton multiplicity.
    seed : int
        RNG seed.
    
    Returns
    -------
    final_partons : list of Parton
        All partons after shower termination.
    history : list
        Shower history log.
    """
    rng = np.random.default_rng(seed)
    partons = list(initial_partons)
    history = []
    next_pid = max([p.pid for p in partons]) + 1 if partons else 1
    
    # Sudakov veto shower algorithm
    iteration = 0
    while len(partons) < max_multiplicity and iteration < 5000:
        iteration += 1
        
        # Select a parton to potentially emit (prefer high-scale partons)
        scales = np.array([p.scale() for p in partons])
        alive = scales > Q_cut**2
        if not np.any(alive):
            break
        
        idx = rng.choice(np.where(alive)[0])
        emitter = partons[idx]
        Q2_emit = emitter.scale()
        
        # Generate a trial emission scale
        # Use exponential distribution in t = ln(Q^2)
        t_current = np.log(Q2_emit / (Q_cut**2))
        if t_current <= 0:
            continue
        
        # Trial emission: sample a smaller scale
        # The probability density for the next emission is ~ exp(-∫ dP)
        # We approximate by direct sampling
        alpha_s_val = alpha_s_1loop(Q2_emit) / (2.0 * np.pi)
        
        # Estimate average splitting probability
        if emitter.flavor in ('q', 'qbar'):
            p_avg = integrate_adaptive_1d(
                lambda z: (p_qq_lo(z) + p_gq_lo(z)) / (z_cut * (1.0 - z_cut)),
                z_cut, 1.0 - z_cut, tol=1e-3
            )
        else:
            p_avg = integrate_adaptive_1d(
                lambda z: (p_gg_lo(z) + 2.0 * N_F * p_qq_lo(z)) / (z_cut * (1.0 - z_cut)),
                z_cut, 1.0 - z_cut, tol=1e-3
            )
        
        rate = alpha_s_val * p_avg
        if rate <= 0:
            continue
        
        # Sample next t from exponential distribution P(t) ~ rate * exp(-rate * t)
        dt_trial = rng.exponential(1.0 / rate)
        t_next = t_current - dt_trial
        
        if t_next <= 0:
            # No emission before cutoff
            continue
        
        Q2_next = Q_cut**2 * np.exp(t_next)
        
        # TODO: Implement Sudakov veto algorithm and emission acceptance logic
        pass
        
        # Kinematics: soft/collinear approximation
        # Emitted parton gets fraction z of energy, transverse kick k_t
        kt = np.sqrt(Q2_next)
        E_emit = emitter.p[3]
        pz_emit = emitter.p[2]
        
        # Split in the plane perpendicular to z-axis with azimuth φ
        # Child 1: carries z fraction (leading)
        # Child 2: emitted soft parton
        E1 = z * E_emit
        E2 = (1.0 - z) * E_emit
        
        px1 = emitter.p[0] * z
        py1 = emitter.p[1] * z
        pz1 = pz_emit * z
        
        px2 = emitter.p[0] * (1.0 - z) + kt * np.cos(phi)
        py2 = emitter.p[1] * (1.0 - z) + kt * np.sin(phi)
        pz2 = pz_emit * (1.0 - z)
        
        # Energy conservation correction (simple rescaling)
        sum_E = E1 + E2
        if sum_E > 0:
            E1 *= E_emit / sum_E
            E2 *= E_emit / sum_E
        
        flavor1 = emitter.flavor
        flavor2 = 'g' if emitter.flavor in ('q', 'qbar') else ('q' if rng.random() < 0.5 else 'qbar')
        
        child1 = Parton(px1, py1, pz1, E1, flavor1, pid=next_pid)
        next_pid += 1
        child2 = Parton(px2, py2, pz2, E2, flavor2, pid=next_pid)
        next_pid += 1
        
        emitter.children = [child1, child2]
        
        # Replace emitter with children in the event list
        partons[idx] = child1
        partons.append(child2)
        
        history.append({
            'iteration': iteration,
            'emitter_pid': emitter.pid,
            'children_pid': [child1.pid, child2.pid],
            'Q2': Q2_next,
            'z': z,
            'multiplicity': len(partons)
        })
    
    return partons, history


def shower_multiplicity_ode(t_span, N0, alpha=0.3, beta=0.1):
    """
    ODE model for mean parton multiplicity evolution (inspired by SIR model).
    
    Let M(t) = mean multiplicity at evolution time t = ln(Q^2/Q_0^2).
    The emission rate is proportional to the number of active emitters,
    but saturation occurs due to phase-space restrictions:
        dM/dt = α M - β M^2
    
    This is a logistic-like equation with analytic solution:
        M(t) = α M0 exp(α t) / [α + β M0 (exp(α t) - 1)]
    
    Parameters
    ----------
    t_span : (t0, tf)
    N0 : float
        Initial multiplicity.
    alpha, beta : float
        Growth and saturation coefficients.
    
    Returns
    -------
    sol : OdeSolution
        Scipy IVP solution object.
    """
    def rhs(t, y):
        M = y[0]
        if M < 1e-12:
            return [alpha * 1e-12]
        return [alpha * M - beta * M**2]
    
    sol = solve_ivp(rhs, t_span, [N0], method='RK45',
                    dense_output=True, max_step=0.5,
                    rtol=1e-8, atol=1e-10)
    return sol


def test_parton_shower():
    """Validate parton shower generation."""
    hard = generate_hard_process(E_cm=14000.0, pt_hard=50.0, seed=42)
    assert len(hard) == 2
    assert abs(hard[0].pt - 50.0) < 1e-6
    
    final, hist = run_parton_shower(hard, Q_cut=1.0, z_cut=0.05, max_multiplicity=100, seed=42)
    assert len(final) >= 2
    assert all(p.pt >= 0 for p in final)
    assert all(p.p[3] > 0 for p in final)
    
    # ODE multiplicity model
    sol = shower_multiplicity_ode((0.0, 5.0), 2.0, alpha=0.5, beta=0.05)
    assert sol.success
    M_final = sol.y[0, -1]
    assert M_final > 2.0, "Multiplicity should grow"
    
    return True


if __name__ == "__main__":
    test_parton_shower()
    print("Parton shower tests passed.")
