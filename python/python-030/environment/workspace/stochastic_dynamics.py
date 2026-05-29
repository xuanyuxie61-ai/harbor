# -*- coding: utf-8 -*-
r"""
stochastic_dynamics.py
======================
Langevin dynamics of nucleons in a time-dependent nuclear mean field,
based on the **Brownian-motion simulation** concepts of
*brownian_motion_simulation*.

Physical model
--------------
A nucleon of mass :math:`m` moving in a potential :math:`V(\mathbf{r})`
obeys the overdamped Langevin equation

.. math::
    \gamma\,\frac{d\mathbf{r}}{dt}
    = -\nabla V(\mathbf{r}) + \boldsymbol{\xi}(t) \;,

where :math:`\gamma` is the friction coefficient and the random force
satisfies the fluctuation-dissipation theorem:

.. math::
    \langle \xi_i(t)\,\xi_j(t') \rangle
    = 2\gamma k_B T\,\delta_{ij}\,\delta(t-t') \;.

The corresponding Smoluchowski (diffusion) equation for the probability
density :math:`P(\mathbf{r},t)` is

.. math::
    \frac{\partial P}{\partial t}
    = \frac{1}{\gamma}\nabla\!
      \cdot\!\bigl[P\,\nabla V\bigr]
    + \frac{k_B T}{\gamma}\nabla^2 P \;.

In nuclear context :math:`k_B T \approx \sqrt{E^*/a}` with level-density
parameter :math:`a = A/8` MeV⁻¹ and excitation energy :math:`E^*`.

The Euler-Maruyama discretisation with time step :math:`\Delta t` is

.. math::
    \mathbf{r}_{n+1} = \mathbf{r}_n
    - \frac{\Delta t}{\gamma}\nabla V(\mathbf{r}_n)
    + \sqrt{\frac{2 k_B T \Delta t}{\gamma}}\,\boldsymbol{\eta}_n \;,

where :math:`\boldsymbol{\eta}_n \sim \mathcal{N}(0, I)`.
"""

import numpy as np


def langevin_step(r, force_func, gamma, temperature, dt, dim=3):
    r"""
    Single Euler-Maruyama step for overdamped Langevin dynamics.

    Parameters
    ----------
    r : ndarray, shape (dim,)
        Current position in fm.
    force_func : callable
        Force function :math:`F(r)` in MeV/fm.
    gamma : float
        Friction coefficient in MeV·fs/fm².
    temperature : float
        Temperature in MeV.
    dt : float
        Time step in fs.
    dim : int
        Spatial dimension.

    Returns
    -------
    r_new : ndarray
        Updated position.
    """
    r = np.asarray(r, dtype=float)
    F = np.asarray(force_func(r), dtype=float)
    # Fluctuation-dissipation amplitude
    sigma = np.sqrt(2.0 * temperature * dt / gamma)
    noise = sigma * np.random.randn(dim)
    drift = (dt / gamma) * F
    return r + drift + noise


def run_langevin_trajectory(r0, force_func, gamma, temperature, dt,
                            n_steps, dim=3, seed=None):
    r"""
    Run a Langevin trajectory.

    Parameters
    ----------
    r0 : ndarray
        Initial position.
    force_func : callable
        Force :math:`F(r)`.
    gamma : float
        Friction coefficient.
    temperature : float
        Temperature in MeV.
    dt : float
        Time step in fs.
    n_steps : int
        Number of steps.
    dim : int
        Spatial dimension.
    seed : int, optional
        Random seed.

    Returns
    -------
    traj : ndarray, shape (n_steps+1, dim)
        Trajectory positions.
    """
    if seed is not None:
        np.random.seed(seed)
    traj = np.zeros((n_steps + 1, dim))
    traj[0, :] = r0
    for n in range(n_steps):
        traj[n + 1, :] = langevin_step(traj[n, :], force_func, gamma,
                                       temperature, dt, dim)
    return traj


def ensemble_langevin(r0_list, force_func, gamma, temperature, dt,
                      n_steps, dim=3, seed=None):
    r"""
    Run an ensemble of Langevin trajectories and compute mean-squared
    displacement (MSD).

    The MSD is defined as

    .. math::
        \text{MSD}(t) = \frac{1}{N_{\text{ens}}}\sum_{i=1}^{N_{\text{ens}}}
        |\mathbf{r}_i(t) - \mathbf{r}_i(0)|^2 \;.

    Parameters
    ----------
    r0_list : ndarray, shape (n_ens, dim)
        Initial positions.
    force_func : callable
        Force function.
    gamma : float
        Friction coefficient.
    temperature : float
        Temperature in MeV.
    dt : float
        Time step in fs.
    n_steps : int
        Number of steps.
    dim : int
        Spatial dimension.
    seed : int, optional

    Returns
    -------
    msd : ndarray, shape (n_steps+1,)
        Mean-squared displacement in fm².
    mean_traj : ndarray, shape (n_steps+1, dim)
        Ensemble-averaged position.
    """
    if seed is not None:
        np.random.seed(seed)
    n_ens = r0_list.shape[0]
    all_traj = np.zeros((n_ens, n_steps + 1, dim))
    for i in range(n_ens):
        all_traj[i, :, :] = run_langevin_trajectory(
            r0_list[i, :], force_func, gamma, temperature, dt, n_steps, dim)

    msd = np.mean(np.sum((all_traj - all_traj[:, 0:1, :]) ** 2, axis=2), axis=0)
    mean_traj = np.mean(all_traj, axis=0)
    return msd, mean_traj


def diffusion_coefficient_from_msd(msd, dt):
    r"""
    Extract diffusion coefficient from long-time MSD slope.

    For normal diffusion :math:`\text{MSD}(t) = 2 d D t`, where :math:`d`
    is the spatial dimension.

    Parameters
    ----------
    msd : ndarray
        MSD time series.
    dt : float
        Time step.

    Returns
    -------
    D : float
        Diffusion coefficient in fm²/fs.
    """
    t = np.arange(msd.size) * dt
    # Linear fit on the second half to avoid ballistic short-time regime
    start = msd.size // 2
    if start < 2:
        start = 1
    coef = np.polyfit(t[start:], msd[start:], 1)
    slope = coef[0]
    # D = slope / (2 * dim)
    return slope / 6.0  # assume 3D


def nuclear_temperature(excitation_energy, A):
    r"""
    Nuclear temperature from the Fermi-gas relation.

    .. math::
        T = \sqrt{\frac{E^*}{a}} \;,
        \qquad a = \frac{A}{8}\;\text{MeV}^{-1}

    Parameters
    ----------
    excitation_energy : float
        Excitation energy :math:`E^*` in MeV.
    A : int
        Mass number.

    Returns
    -------
    T : float
        Temperature in MeV.
    """
    a = A / 8.0
    if excitation_energy <= 0 or a <= 0:
        return 0.0
    return np.sqrt(excitation_energy / a)


def evaporative_decay_rate(A, Z, T, separation_energy):
    r"""
    Weisskopf evaporation rate for a nucleon.

    .. math::
        \Gamma = \frac{g\,m\,\sigma}{\pi^2\hbar^3}\,(k_B T)^2
        \exp\!\left(-\frac{S_n}{k_B T}\right)

    Parameters
    ----------
    A, Z : int
        Residue mass and charge.
    T : float
        Temperature in MeV.
    separation_energy : float
        Neutron or proton separation energy in MeV.

    Returns
    -------
    rate : float
        Decay rate in fs⁻¹.
    """
    if T <= 0 or separation_energy <= 0:
        return 0.0
    # Simplified: g=2 (spin), sigma ~ geometric cross section
    g = 2.0
    sigma = np.pi * (1.2 * (A ** (1.0 / 3.0))) ** 2  # fm²
    # Constant prefactor in natural units (approximate)
    prefactor = 1.0e-4  # empirical scaling to fs⁻¹
    rate = prefactor * g * sigma * (T ** 2) * np.exp(-separation_energy / T)
    return rate
