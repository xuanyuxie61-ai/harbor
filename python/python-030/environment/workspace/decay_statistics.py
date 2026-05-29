# -*- coding: utf-8 -*-
"""
decay_statistics.py
===================
Statistical description of radioactive decay chains, β-decay branching
ratios, and noncentral-beta-distributed uncertainties.

This module fuses:
* **beta_noncentral_cdf** — evaluation of the noncentral Beta CDF for
  Bayesian inference on decay probabilities.
* **discrete_pdf_sample_2d** — discrete inverse-CDF sampling for
  statistical decay-chain Monte Carlo.

Physical model
--------------
For a parent nucleus :math:`(Z,A)` the :math:`\beta`-decay Q-value is

.. math::
    Q_{\beta} = \bigl[M(Z,A) - M(Z+1,A)\bigr]c^2 + (m_e - m_{\bar{\nu}})c^2
    \approx \Delta M_{np} - \Delta E_{\text{Coul}} \;.

The allowed :math:`\beta`-decay half-life is estimated by

.. math::
    T_{1/2} = \frac{D}{f(Z, Q_{\beta})\,B_{\text{gt}}} \;,

where :math:`D\approx 6147` s is the comparative half-life constant,
:math:`f(Z,Q)` is the Fermi integral, and :math:`B_{\text{gt}}` is the
Gamow-Teller transition strength.

The Fermi integral for allowed decays (approximate):

.. math::
    f(Z, Q) \approx \frac{(Q + m_e c^2)^5}{30\,(\hbar c)^3\,(\alpha Z)^2}
    \;\bigl[1 - \frac{\pi\alpha Z}{p_0}\bigr] \;.

Uncertainty propagation
-----------------------
When the branching ratio :math:`B` is inferred from a finite sample of
:math:`n` decays with :math:`k` observed in a given channel, the posterior
distribution of :math:`B` is modelled as a **noncentral Beta distribution**:

.. math::
    B \sim \text{Beta}_{\text{nc}}(a, b, \lambda) \;,

with shape parameters :math:`a = k + 1`, :math:`b = n - k + 1`, and
noncentrality :math:`\lambda` encoding systematic-detection-efficiency bias.

The CDF is computed by the series (Posten 1993):

.. math::
    F(x; a, b, \lambda) = \sum_{i=0}^{\infty} p_i(\lambda)\,I_x(a+i, b) \;,

where :math:`p_i(\lambda) = e^{-\lambda/2}(\lambda/2)^i / i!` and
:math:`I_x` is the regularised incomplete beta function.
"""

import numpy as np
from scipy.special import betainc, gammaln


def _log_gamma(x):
    """Log-gamma via gammaln (scipy)."""
    return gammaln(x)


def noncentral_beta_cdf(x, a, b, lam, error_max=1e-12):
    r"""
    Noncentral Beta cumulative distribution function.

    .. math::
        F(x; a, b, \lambda)
        = \sum_{i=0}^{\infty} p_i(\lambda)\,I_x(a+i, b)

    with :math:`p_i(\lambda) = e^{-\lambda/2}(\lambda/2)^i / i!`.

    Parameters
    ----------
    x : float
        Argument :math:`x \in [0,1]`.
    a, b : float
        Shape parameters (must be > 0).
    lam : float
        Noncentrality parameter :math:`\lambda \ge 0`.
    error_max : float
        Truncation error tolerance.

    Returns
    -------
    cdf : float
        Value of the noncentral Beta CDF.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0 or b <= 0:
        raise ValueError("Shape parameters a and b must be positive.")
    if lam < 0:
        raise ValueError("Noncentrality lambda must be non-negative.")

    # Compute using series expansion (Posten algorithm)
    half_lam = lam / 2.0
    pi_val = np.exp(-half_lam)
    if pi_val == 0.0:
        # Very large lambda: approximate by shifting a
        return float(betainc(a + half_lam, b, x))

    # Log of beta function
    beta_log = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)

    # Initial incomplete beta I_x(a,b)
    bi = float(betainc(a, b, x))
    si = np.exp(a * np.log(x) + b * np.log(1.0 - x) - beta_log - np.log(a))

    p_sum = pi_val
    pb_sum = pi_val * bi
    i = 0

    while p_sum < 1.0 - error_max and i < 10000:
        i += 1
        pi_val = half_lam * pi_val / i
        bi = bi - si
        si = x * (a + b + i - 1) * si / (a + i)
        p_sum += pi_val
        pb_sum += pi_val * bi
        # Guard against divergence
        if not np.isfinite(pb_sum):
            break

    return float(np.clip(pb_sum, 0.0, 1.0))


def noncentral_beta_pdf(x, a, b, lam, dx=1e-6):
    r"""
    Noncentral Beta probability density function via numerical derivative of CDF.

    .. math::
        f(x) = \frac{d}{dx}F(x; a, b, \lambda)

    Parameters
    ----------
    x : float
        Evaluation point.
    a, b : float
        Shape parameters.
    lam : float
        Noncentrality.
    dx : float
        Step size for derivative.

    Returns
    -------
    pdf : float
        Approximate PDF value.
    """
    x = np.clip(x, dx, 1.0 - dx)
    return (noncentral_beta_cdf(x + dx, a, b, lam) -
            noncentral_beta_cdf(x - dx, a, b, lam)) / (2.0 * dx)


def decay_chain_simulation(initial_state, transition_matrix, n_steps,
                           n_samples=1000, seed=None):
    r"""
    Monte-Carlo simulation of a discrete decay chain using inverse-CDF sampling.

    The transition matrix :math:`T_{ij}` gives the probability that nucleus
    :math:`i` decays to nucleus :math:`j` in one step.  At each step a random
    draw from the discrete CDF determines the daughter nucleus.

    Parameters
    ----------
    initial_state : int
        Index of the initial nucleus.
    transition_matrix : ndarray, shape (n_states, n_states)
        Row-stochastic transition matrix.
    n_steps : int
        Number of decay steps to simulate.
    n_samples : int
        Number of independent Monte-Carlo trajectories.
    seed : int, optional
        Random seed.

    Returns
    -------
    populations : ndarray, shape (n_steps+1, n_states)
        Mean population distribution at each step.
    trajectories : ndarray, shape (n_samples, n_steps+1)
        Individual state-index trajectories.
    """
    if seed is not None:
        np.random.seed(seed)

    T = np.asarray(transition_matrix, dtype=float)
    n_states = T.shape[0]
    # Validate stochastic rows
    row_sums = T.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] <= 0:
            T[i, i] = 1.0
        else:
            T[i, :] /= row_sums[i]

    # Build CDF rows
    cdf = np.cumsum(T, axis=1)
    cdf[:, -1] = 1.0  # Ensure exactly 1

    trajectories = np.empty((n_samples, n_steps + 1), dtype=int)
    trajectories[:, 0] = initial_state

    for step in range(1, n_steps + 1):
        u = np.random.rand(n_samples)
        for samp in range(n_samples):
            state = trajectories[samp, step - 1]
            # Inverse CDF: find first index where cdf[state, j] >= u[samp]
            next_state = np.searchsorted(cdf[state, :], u[samp])
            next_state = min(next_state, n_states - 1)
            trajectories[samp, step] = next_state

    populations = np.zeros((n_steps + 1, n_states))
    for step in range(n_steps + 1):
        for s in range(n_states):
            populations[step, s] = np.mean(trajectories[:, step] == s)

    return populations, trajectories


def beta_decay_halflife(Z, Q_mev, Bgt=1.0, ft_const=6147.0):
    r"""
    Estimate allowed :math:`\beta`-decay half-life.

    .. math::
        T_{1/2} = \frac{f_t}{f(Z, Q)\,B_{\text{gt}}}

    with an approximate Fermi integral

    .. math::
        f(Z, Q) \approx \frac{Q^5}{30}\Bigl(1 - 2\pi\alpha Z\Bigr) \;.

    Parameters
    ----------
    Z : int
        Daughter proton number.
    Q_mev : float
        Q-value in MeV.
    Bgt : float
        Gamow-Teller strength (dimensionless).
    ft_const : float
        Comparative half-life constant :math:`f t_{1/2}` in seconds.

    Returns
    -------
    T12 : float
        Estimated half-life in seconds.
    """
    if Q_mev <= 0:
        return np.inf
    alpha = 1.0 / 137.035999084
    # Simplified Fermi integral (allowed, non-relativistic)
    f_approx = (Q_mev ** 5) / 30.0 * max(1.0 - 2.0 * np.pi * alpha * Z, 0.1)
    if f_approx <= 0:
        return np.inf
    return ft_const / (f_approx * Bgt)


def q_value_beta_decay(M_parent, M_daughter, Q_ec=0.0):
    r"""
    :math:`\beta`-decay Q-value from atomic masses.

    .. math::
        Q_{\beta^-} = \bigl[M_{\text{parent}} - M_{\text{daughter}}\bigr]c^2

    (neglecting atomic electron binding differences).

    Parameters
    ----------
    M_parent, M_daughter : float
        Atomic masses in MeV/c².
    Q_ec : float
        Additional electron-capture correction (MeV).

    Returns
    -------
    Q : float
        Q-value in MeV.
    """
    return M_parent - M_daughter + Q_ec


def neutron_drip_line_uncertainty(N_obs, Z, confidence=0.95, eff_bias=0.05):
    r"""
    Compute a Bayesian credible interval for the neutron-drip existence
    probability using the noncentral Beta distribution.

    The observation of :math:`N_{\text{obs}}` bound neutrons out of
    :math:`N_{\text{max}}` attempted syntheses is modelled as
    :math:`p \sim \text{Beta}_{\text{nc}}(a, b, \lambda)` with
    :math:`a = N_{\text{obs}} + 1`, :math:`b = N_{\text{max}} - N_{\text{obs}} + 1`,
    and :math:`\lambda` proportional to the detection-efficiency bias.

    Parameters
    ----------
    N_obs : int
        Observed bound-nucleus events.
    Z : int
        Proton number (used to scale systematic bias).
    confidence : float
        Desired credible level.
    eff_bias : float
        Efficiency bias parameter.

    Returns
    -------
    lower, upper : float
        Credible interval bounds.
    mean : float
        Posterior mean.
    """
    N_max = max(N_obs + 10, 20)
    a = N_obs + 1.0
    b = N_max - N_obs + 1.0
    lam = eff_bias * Z

    # Search for quantiles by bisection on the CDF
    def find_quantile(p_target):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if noncentral_beta_cdf(mid, a, b, lam) < p_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    alpha = 0.5 * (1.0 - confidence)
    lower = find_quantile(alpha)
    upper = find_quantile(1.0 - alpha)
    # Approximate mean: a/(a+b) shifted by lambda
    mean = (a + 0.5 * lam) / (a + b + lam)
    mean = np.clip(mean, 0.0, 1.0)
    return lower, upper, mean
