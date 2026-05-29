"""
inference_engine.py

Complete Bayesian inference pipeline for the spatially-coupled
reaction-diffusion system on an annular domain.

Combines:
  - Forward models (FHN + Helmholtz)
  - FEM basis spatial discretization
  - Periodic tridiagonal GMRF prior (r83p)
  - Polynomial surrogate acceleration
  - Latin-hypercube initialization
  - Unicycle rotation proposals
  - Bayesian quadrature for model evidence
  - Calendar indexing for synthetic observations
"""
import math
import numpy as np
from scipy.special import jv

from prng_asa183 import WichmannHill
from periodic_solver import r83p_fa, r83p_sl
from spatial_domain import (
    annulus_grid_fibonacci, fem_basis_2d,
    circle_distance_exact_mean, circle_distance_exact_variance
)
from utils import latin_edge, jed_to_nyt, unicycle_next, circle_distance_stats
from forward_models import fhn_stationary_voltage, besselzero
from surrogate import build_fhn_surrogate, surrogate_predict
from bayesian_quadrature import integrate_1d, integrate_square, integrate_triangle


def _build_gmrf_precision(n: int, tau: float, delta: float = 1.0):
    """Build periodic tridiagonal precision matrix for circular GMRF.
    delta is a small nugget to ensure positive definiteness."""
    # TODO Hole 4: Build the n x n precision matrix Q for a circular GMRF.
    # The matrix should encode nearest-neighbor coupling with strength tau on a ring,
    # plus a diagonal nugget delta for positive definiteness.
    raise NotImplementedError("Hole 4: GMRF precision matrix construction not implemented")


def _compute_gmrf_covariance_via_r83p(n: int, tau: float, delta: float = 1.0):
    """
    Compute the exact covariance matrix Sigma = Q^{-1} of a circular GMRF
    by solving Q * Sigma[:,j] = e_j for each j using the periodic tridiagonal
    solver r83p. A small nugget delta ensures positive definiteness.
    """
    # TODO Hole 3: Pack the periodic tridiagonal precision matrix Q into R83P format.
    # R83P storage layout (3 x n):
    #   a[0, 0]  = lower-left wrap  A(N,1)
    #   a[0, 1:] = superdiagonal    A(j, j+1)
    #   a[1, :]  = diagonal
    #   a[2, :-1] = subdiagonal     A(j+1, j)
    #   a[2, -1] = upper-right wrap A(1,N)
    # This packing must exactly match the assumptions in periodic_solver.r83p_fa.
    a = None

    a_lu, work2, work3, work4, info = r83p_fa(n, a)
    if info != 0:
        raise RuntimeError(f"r83p_fa failed with info={info}")

    Sigma = np.zeros((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        col = r83p_sl(n, a_lu, e, 0, work2, work3, work4)
        Sigma[:, j] = col
    return Sigma


def generate_synthetic_data(rng):
    """
    Generate synthetic observations on an annular domain.
    """
    n_sectors = 4
    r = 0.75
    thetas = np.linspace(0.0, 2.0 * math.pi, n_sectors, endpoint=False)
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)

    # True parameters
    a_true = 0.7
    b_true = 0.8
    c_true = 3.0
    gamma_true = 0.2
    d0_true = 0.0
    c_field = np.array([0.02, -0.01, 0.01, -0.02], dtype=float)
    sigma_true = 0.1

    # FEM basis weights at mid-edge of reference triangle (0.5, 0.5)
    # This uses the seed project's fem_basis_2d algorithm
    w1 = fem_basis_2d(1, 0, 0, 0.5, 0.5)
    w2 = fem_basis_2d(0, 1, 0, 0.5, 0.5)
    w3 = fem_basis_2d(0, 0, 1, 0.5, 0.5)

    # Helmholtz exact solution (stationary wave field on disk)
    a_disk = 1.0
    rho10 = besselzero(0, 1, 1)[0]
    Z = gamma_true * jv(0, rho10 * r / a_disk) * np.ones(n_sectors)

    # Forward model: FHN at each sector
    v = np.empty(n_sectors, dtype=float)
    for j in range(n_sectors):
        d_j = d0_true + w1 * c_field[j] + w2 * c_field[(j + 1) % n_sectors] + w3 * c_field[(j - 1) % n_sectors]
        v[j] = fhn_stationary_voltage(a_true, b_true, c_true, d_j)

    obs = v + Z + rng.normals(n_sectors, 0.0, sigma_true)

    # Synthetic observation batch ID via calendar_nyt conversion
    jed_epoch = 2450000.5  # synthetic Julian date
    vol, issue = jed_to_nyt(jed_epoch)

    # Circle distance statistics for spatial validation
    mu_est, var_est = circle_distance_stats(200, rng)
    mu_exact = circle_distance_exact_mean()
    var_exact = circle_distance_exact_variance()

    return {
        'x': x, 'y': y, 'thetas': thetas, 'obs': obs,
        'n_sectors': n_sectors,
        'fem_weights': (w1, w2, w3),
        'batch_volume': vol, 'batch_issue': issue,
        'circle_mu_est': mu_est, 'circle_var_est': var_est,
        'circle_mu_exact': mu_exact, 'circle_var_exact': var_exact,
        'true_params': {
            'a': a_true, 'b': b_true, 'gamma': gamma_true,
            'd0': d0_true, 'c': c_field, 'sigma': sigma_true,
            'rho10': rho10
        }
    }


def log_prior(params: np.ndarray, Q: np.ndarray):
    """
    Log-prior density with informative Gaussian priors for physically
    interpretable parameters and a GMRF prior for the spatial field.

    a     ~ N(0.7, 0.08^2)  truncated [0.2, 1.2]
    b     ~ N(0.8, 0.08^2)  truncated [0.2, 1.2]
    gamma ~ N(0.2, 0.12^2)  truncated [-0.5, 0.5]
    d0    ~ N(0.0, 0.08^2)  truncated [-0.3, 0.3]
    c     ~ N(0, Q^{-1})    GMRF on circular graph
    log_sigma: Jeffreys prior
    """
    a, b, gamma, d0, c0, c1, c2, c3, log_sigma = params
    if not (0.2 <= a <= 1.2):
        return -np.inf
    if not (0.2 <= b <= 1.2):
        return -np.inf
    if not (-0.5 <= gamma <= 0.5):
        return -np.inf
    if not (-0.3 <= d0 <= 0.3):
        return -np.inf
    sigma = math.exp(log_sigma)
    if sigma <= 1e-12 or sigma > 10.0:
        return -np.inf

    c = np.array([c0, c1, c2, c3], dtype=float)
    logp_c = -0.5 * float(np.dot(c, Q.dot(c)))
    logp_a = -0.5 * ((a - 0.7) / 0.08) ** 2
    logp_b = -0.5 * ((b - 0.8) / 0.08) ** 2
    logp_gamma = -0.5 * ((gamma - 0.2) / 0.12) ** 2
    logp_d0 = -0.5 * (d0 / 0.08) ** 2
    return logp_c + logp_a + logp_b + logp_gamma + logp_d0


def log_likelihood(params: np.ndarray, data: dict, surrogate: dict):
    """
    Gaussian log-likelihood for the spatial observations.
    """
    a, b, gamma, d0, c0, c1, c2, c3, log_sigma = params
    sigma = math.exp(log_sigma)
    if sigma <= 1e-12:
        return -np.inf

    c = np.array([c0, c1, c2, c3], dtype=float)
    n = data['n_sectors']
    w1, w2, w3 = data['fem_weights']

    # Use surrogate when near the training point (a=0.7, b=0.8)
    use_surrogate = (abs(a - 0.7) < 0.15) and (abs(b - 0.8) < 0.15)

    v_pred = np.empty(n, dtype=float)
    for j in range(n):
        d_j = d0 + w1 * c[j] + w2 * c[(j + 1) % n] + w3 * c[(j - 1) % n]
        if use_surrogate:
            v_pred[j] = surrogate_predict(surrogate, d_j)
        else:
            v_pred[j] = fhn_stationary_voltage(a, b, 3.0, d_j)

    a_disk = 1.0
    rho10 = data['true_params']['rho10']
    Z = gamma * jv(0, rho10 * 0.75 / a_disk)
    pred = v_pred + Z

    residuals = data['obs'] - pred
    ll = -0.5 * n * log_sigma - 0.5 * np.sum(residuals ** 2) / (sigma ** 2)
    # Add normalization constant (omitted if comparing posteriors, but needed for evidence)
    ll -= 0.5 * n * math.log(2.0 * math.pi)
    return ll


def log_posterior(params: np.ndarray, data: dict, Q: np.ndarray, surrogate: dict):
    lp = log_prior(params, Q)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, data, surrogate)
    return lp + ll


def run_bayesian_inference():
    """
    Main pipeline: data generation, MCMC inference, and Bayesian quadrature.
    """
    rng = WichmannHill(12345, 30306, 13579)

    # --- Synthetic data ---
    data = generate_synthetic_data(rng)
    print(f"[Data] NYT Batch: Volume {data['batch_volume']}, Issue {data['batch_issue']}")
    print(f"[Data] Circle distance MC: mu={data['circle_mu_est']:.4f} (exact={data['circle_mu_exact']:.4f}), "
          f"var={data['circle_var_est']:.4f} (exact={data['circle_var_exact']:.4f})")

    # --- GMRF prior for spatial coefficients ---
    n_c = 4
    tau = 100.0
    Q = _build_gmrf_precision(n_c, tau)
    Sigma = _compute_gmrf_covariance_via_r83p(n_c, tau)
    print(f"[Prior] GMRF covariance trace via r83p: {np.trace(Sigma):.6f}")
    # Validate by comparing with numpy inverse on a small test
    Q_test = _build_gmrf_precision(n_c, tau)
    Sigma_np = np.linalg.inv(Q_test)
    print(f"[Prior] GMRF covariance trace via numpy: {np.trace(Sigma_np):.6f}")

    # --- Build polynomial surrogate ---
    surrogate = build_fhn_surrogate(a_fixed=0.7, b_fixed=0.8, c_fixed=3.0,
                                    degree=6, n_train=15, d_min=-0.3, d_max=0.3)
    print("[Surrogate] FHN voltage surrogate built (degree=6, n_train=15)")

    # --- Latin hypercube initialization for 4 core parameters ---
    lhs = latin_edge(4, 5, rng)
    # Map LHS samples [0,1] to parameter bounds
    # a,b in [0.2,1.2], gamma in [-0.5,0.5], d0 in [-0.3,0.3]
    lhs_mapped = np.zeros_like(lhs)
    lhs_mapped[0, :] = 0.2 + lhs[0, :] * 1.0
    lhs_mapped[1, :] = 0.2 + lhs[1, :] * 1.0
    lhs_mapped[2, :] = -0.5 + lhs[2, :] * 1.0
    lhs_mapped[3, :] = -0.3 + lhs[3, :] * 0.6
    # Pick the sample closest to true values for initialization
    target = np.array([0.7, 0.8, 0.2, 0.0])
    dists = np.sum((lhs_mapped - target[:, None]) ** 2, axis=0)
    best = np.argmin(dists)
    init = np.zeros(9, dtype=float)
    init[:4] = lhs_mapped[:, best]
    init[4:8] = 0.0
    init[8] = math.log(0.1)
    print(f"[MCMC] Latin hypercube init: a={init[0]:.3f}, b={init[1]:.3f}, gamma={init[2]:.3f}, d0={init[3]:.3f}")

    # --- Run MCMC ---
    from mcmc_sampler import run_adaptive_mcmc
    lp_func = lambda p: log_posterior(p, data, Q, surrogate)
    chain, logpost_trace, accept_rate = run_adaptive_mcmc(
        lp_func, init, rng, n_iter=500, proposal_scale=0.05, rotation_period=10
    )
    print(f"[MCMC] Acceptance rate (incl. Gibbs & rotations): {accept_rate:.3f}")

    # Burn-in and posterior summaries
    burn = 150
    post_chain = chain[burn:, :]
    post_logpost = logpost_trace[burn:]
    mean_params = np.mean(post_chain, axis=0)
    std_params = np.std(post_chain, axis=0)

    # --- Unicycle permutation diagnostics ---
    u = np.arange(1, n_c + 1)
    rank = -1
    perm_sequence = []
    for _ in range(5):
        u, rank = unicycle_next(n_c, u, rank)
        perm_sequence.append(u.copy())
    print(f"[Diagnostics] First 5 unicycle permutations: {perm_sequence}")

    # --- Bayesian Quadrature: model evidence estimates ---
    # 1D marginal over gamma using ergodic line sampling
    def integrand_gamma(g):
        if isinstance(g, np.ndarray):
            return np.array([integrand_gamma(val) for val in g], dtype=float)
        p = mean_params.copy()
        p[2] = g
        return math.exp(log_posterior(p, data, Q, surrogate))
    ev_gamma, se_gamma = integrate_1d(integrand_gamma, n=80, method="ergodic", shift=0.1)
    print(f"[Quadrature] 1D evidence slice over gamma: {ev_gamma:.4e} (SE={se_gamma:.4e})")

    # 2D marginal over (a,b) using square random sampling
    def integrand_square(pts):
        vals = np.empty(pts.shape[1])
        for i in range(pts.shape[1]):
            p = mean_params.copy()
            a_val = 0.2 + pts[0, i] * 1.0  # map [0,1] -> [0.2, 1.2]
            b_val = 0.2 + pts[1, i] * 1.0
            p[0] = a_val
            p[1] = b_val
            vals[i] = math.exp(log_posterior(p, data, Q, surrogate))
        return vals
    ev_sq, se_sq = integrate_square(integrand_square, n=60, method="random", rng=rng)
    print(f"[Quadrature] 2D evidence slice over (a,b): {ev_sq:.4e} (SE={se_sq:.4e})")

    # Triangle integration over a dummy simplex weight for demonstration
    def integrand_triangle(pts):
        vals = np.empty(pts.shape[1])
        for i in range(pts.shape[1]):
            # pts are inside reference triangle; use barycentric coords as weights
            w1, w2 = pts[0, i], pts[1, i]
            w3 = 1.0 - w1 - w2
            # Mixture of three candidate stimulus profiles
            d_mix = 0.0 * w1 + 0.05 * w2 + (-0.05) * w3
            p = mean_params.copy()
            p[3] = d_mix
            vals[i] = math.exp(log_posterior(p, data, Q, surrogate))
        return vals
    ev_tri, se_tri = integrate_triangle(integrand_triangle, n=60, method="random", rng=rng)
    print(f"[Quadrature] Triangle evidence slice over simplex weights: {ev_tri:.4e} (SE={se_tri:.4e})")

    results = {
        'data': data,
        'chain': chain,
        'logpost': logpost_trace,
        'posterior_mean': mean_params,
        'posterior_std': std_params,
        'accept_rate': accept_rate,
        'evidence_gamma': ev_gamma,
        'evidence_ab': ev_sq,
        'evidence_triangle': ev_tri,
    }
    return results


if __name__ == "__main__":
    res = run_bayesian_inference()
    print("\n=== Posterior Summaries ===")
    names = ['a', 'b', 'gamma', 'd0', 'c0', 'c1', 'c2', 'c3', 'log_sigma']
    for i, name in enumerate(names):
        print(f"  {name:12s}: mean={res['posterior_mean'][i]:.4f}, std={res['posterior_std'][i]:.4f}")
    true = res['data']['true_params']
    print(f"\nTrue values: a={true['a']}, b={true['b']}, gamma={true['gamma']}, d0={true['d0']}, sigma={true['sigma']}")
